"""Thin locus layer over the upstream mlx-vlm server.

Upstream (mlx_vlm.server) owns generation: continuous batching, structured output, thinking budget, APC prefix caching, speculative decode, vision feature cache.
This module boots it behind the operational guarantees a resident model demands — memory limits from the sysctl (main), one model and the routes its clients use and nothing else (ServedOnlyMiddleware), admission (AdmissionMiddleware: one request resident at a time, first come first served; the line waits and nothing is refused; the slot frees only once upstream's GPU loop has dropped the request, install_gpu_tracking), the liveness guard (serve_guarded), memory telemetry (TelemetryMiddleware), the rendered-prompt record (install_prompt_recording), a lock over upstream's caller-thread preprocessing (install_preprocess_serializer), the pixel arrays released at encode (install_pixel_release), and two math-preserving patches (the vision-encode chunking, the fused attention). Every upstream-keyed patch verifies the source it stands in for at boot and refuses to serve on drift (patching.py).
README.md says what each is for.

Known upstream gaps, version-keyed — re-validate on every mlx-vlm upgrade:
- min_p is accepted per-request but never reaches the batched sampler; the cards therefore do not claim it.
- a request's images encode in one batch, so the vision-encode transient scales with the request's total pixels and no token-denominated guard bounds it — a large enough image batch aborts the server even when it fits the token context (MAX_KV_SIZE counts expanded vision tokens, so its guard doesn't close this).
  Patched here: vision_chunking.install_vision_encode_chunking encodes in pixel-bounded chunks, output-identical on the pinned upstream.
- on GPUs without neural accelerators, mlx reads a multi-token chunk of a head-dim-256 model by materializing its whole score table, so prefill memory grows with the prompt (16 GiB per 2,048-token chunk at 256k keys).
  Patched here: fused_attention.install_fused_attention has Qwen's attention ask mlx for its fused kernel, which never holds the table.
- the GPU loop keeps a request's pixel arrays referenced after encoding them, through the request's prefill and decode and until the next request arrives (over 1 GiB for 47k image tokens).
  Patched here: install_pixel_release drops them from the request's inputs once they are encoded.
- a caller that disconnects ends its HTTP response at once, while the GPU loop drops the request only at its next step, or never hears the cancel and runs it to its last token.
  Patched here: install_gpu_tracking counts the requests the loop holds, and admission frees its slot only once that count is zero.
- the chat-completions handler compacts a conversation unless the request sends `context_management: []`: it re-renders and preprocesses the request to count it, clamps max_tokens to the context left, and past the context limit replaces the conversation's head with a model-written summary where it would otherwise refuse the request. Every locus client sends `[]` (model/openai_compat.py, chat.py).
- thinking_budget is rejected in combination with speculative decoding, so no draft model can be configured while the cards send a budget (they always do — it absorbs the thinking-runaway).

Usage: locus-mlx-serve <model-slug>   (slugs are the models.toml keys)
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import tomllib

from . import require_extra
from .paths import PIDFILE

MODELS_TOML = Path(__file__).parent / "models.toml"


# The analysis card owns which repo a slug serves (`model`), the context it holds (`context_tokens` → MAX_KV_SIZE), and the endpoint analysis dials (`base_url` → bind host/port); the server derives all three at boot, so what it serves, guards, and where it listens cannot drift from what analysis requests and prices.
# The card is read as a file from config/cards/ under the deployment root, never through the conversation layer.
def card_file(slug: str) -> Path:
    from locus.analysis.model.cards import CARDS_DIR
    from locus.evidence.deployment import CONFIG_DIR, deployment_root

    return deployment_root() / CONFIG_DIR / CARDS_DIR / f"{slug}.toml"


def _card(slug: str) -> dict | None:
    path = card_file(slug)
    return tomllib.loads(path.read_text()) if path.exists() else None


def bind_from_card(card: dict) -> tuple[str, int]:
    """Host and port the server binds, derived from the card's base_url — the same endpoint analysis dials."""
    url = card.get("base_url")
    if not isinstance(url, str) or not url:
        raise ValueError(
            "card declares no base_url — the endpoint analysis dials "
            "(and this server binds) is the card's; declare it in the card's "
            "file in config/cards/ under the deployment root"
        )
    parsed = urlparse(url)
    if not parsed.hostname or parsed.port is None:
        raise ValueError(
            f"card base_url {url!r} must be an absolute URL with an explicit "
            f"host and port so analysis and the server agree on one endpoint"
        )
    return parsed.hostname, parsed.port


def bind_for(slug: str) -> tuple[str, int]:
    """Bind host/port for a models.toml slug, read from that slug's analysis card."""
    card = _card(slug)
    if card is None:
        raise ValueError(f"no card for slug {slug!r} at {card_file(slug)}")
    return bind_from_card(card)


def sole_bind() -> tuple[str, int]:
    """The unique host:port among models.toml-bootable cards.

    Probe tools and readiness checks use this when no slug is in hand: one resident model means one bind, and the card is still the owner of the number.
    """
    models = tomllib.loads(MODELS_TOML.read_text())
    binds: set[tuple[str, int]] = set()
    for slug in models:
        card = _card(slug)
        if card is None:
            raise ValueError(
                f"models.toml slug {slug!r} has no card at {card_file(slug)}"
            )
        binds.add(bind_from_card(card))
    if len(binds) != 1:
        raise ValueError(
            f"bootable cards must share one bind address so probe tools can "
            f"find the server without restating a port; found "
            f"{sorted(binds) if binds else 'none'}"
        )
    return next(iter(binds))


# The routes this server answers, each at the paths upstream serves it on: chat completions, which
# reaches upstream's GPU generator (runtime.response_generator.generate) and so passes admission
# and is what the memory telemetry brackets, and the two reads a client of it makes — the model
# list and /health. Every other route is refused (ServedOnlyMiddleware).
GENERATING_ROUTES = frozenset({"/v1/chat/completions", "/chat/completions"})
MODEL_LIST_ROUTES = frozenset({"/v1/models", "/models"})
SERVED_ROUTES = GENERATING_ROUTES | MODEL_LIST_ROUTES | {"/health"}

logger = logging.getLogger("locus.analysis.mlx.serve")


def route(scope) -> str:
    return scope.get("path", "").rstrip("/")


def generating(scope) -> bool:
    return scope["type"] == "http" and route(scope) in GENERATING_ROUTES


def install_log_handler() -> None:
    handler = logging.StreamHandler(sys.stderr)
    # Log stamps are full UTC instants with a Z, so a serve-log line diffs
    # cleanly against memory.jsonl's epoch stamps and every other locus clock —
    # a reader is never handed a zone-less wall-clock to guess at.
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)


def serve_guarded(app, port: int, host: str = "127.0.0.1") -> None:
    """Refuse to boot while another server process is alive, then launch uvicorn.

    Liveness is the pidfile plus signal 0, not the port: uvicorn's graceful shutdown closes the listening sockets and only then waits for in-flight connections (Server.shutdown in uvicorn/server.py — https://github.com/encode/uvicorn — "Stop accepting new connections" via server.close(), then _wait_tasks_to_complete), so the port reads free while a prior server still holds tens of GiB wired. Two resident model instances on one machine is the state it does not recover from. The port check catches only foreign (non-locus) processes squatting on the port. Host and port are the card's (bind_from_card); this function only enforces liveness and launches.
    """
    if PIDFILE.exists():
        prior = int(PIDFILE.read_text().strip())
        alive = (
            "Another MLX server (pid {pid}) is {state} — its in-flight generation can "
            "outlive its listening socket, so the port being free proves nothing. One "
            "resident model per machine. Stop it with scripts/stop-server.sh (SIGTERM, "
            "bounded wait, SIGKILL, verified by process death), then boot."
        )
        try:
            os.kill(prior, 0)
        except ProcessLookupError:
            print(f"Removing stale pidfile for dead pid {prior}.", flush=True)
            PIDFILE.unlink()
        except PermissionError:
            # EPERM is not "no such process": it is the kernel confirming the process exists and
            # refusing us the signal — what a server booted under sudo looks like to a later
            # unprivileged boot. Reading it as death would remove the pidfile and boot a second
            # resident model.
            print(
                alive.format(
                    pid=prior,
                    state="alive and owned by another user (the kernel refused us the signal)",
                ),
                flush=True,
            )
            sys.exit(1)
        else:
            print(alive.format(pid=prior, state="still alive"), flush=True)
            sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_use = sock.connect_ex((host, port)) == 0
    sock.close()
    if in_use:
        print(
            f"{host}:{port} is held by a non-locus process (no live pidfile). "
            f"All MLX servers share the card's endpoint by design to eliminate "
            f"resource contention.",
            flush=True,
        )
        sys.exit(1)

    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    try:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
    finally:
        if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
            PIDFILE.unlink()


def install_heif_decode() -> None:
    """Register HEIC/HEIF decoding with PIL, process-wide. The server is handed bare file paths and loads them with PIL, which cannot open HEIF containers on its own; registration makes them decode like any other image format."""
    from pillow_heif import register_heif_opener

    register_heif_opener()


def _wired_cap_bytes() -> int:
    """The operator's dial, read from the machine — a failed read is an error, never a zero the boot would take for a memory ceiling."""
    read = subprocess.run(
        ["sysctl", "-n", "iogpu.wired_limit_mb"],
        capture_output=True,
        text=True,
        check=False,
    )
    if read.returncode != 0:
        raise SystemExit(
            f"could not read iogpu.wired_limit_mb ({read.stderr.strip() or f'sysctl exited {read.returncode}'}). The sysctl is the operator's dial — set it to the machine's measured safe maximum, e.g.: sudo sysctl iogpu.wired_limit_mb=<mb>"
        )
    cap_mb = int(read.stdout.strip() or 0)
    if cap_mb <= 0:
        raise SystemExit(
            f"iogpu.wired_limit_mb is {cap_mb}. The sysctl is the operator's dial — set it to the machine's measured safe maximum, e.g.: sudo sysctl iogpu.wired_limit_mb=<mb>"
        )
    return cap_mb * 2**20


async def _buffered_response(asgi_app, scope, receive) -> tuple[dict, bytes]:
    """The app's whole response to a request — its start message and its body — held instead of sent, for a small non-streamed body that is rewritten before it goes out."""
    start = {}
    body = bytearray()

    async def capture(message):
        if message["type"] == "http.response.start":
            start.update(message)
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))

    await asgi_app(scope, receive, capture)
    return start, bytes(body)


async def _send_json(send, status: int, payload, headers=()) -> None:
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                *((k, v) for k, v in headers if k.lower() != b"content-length"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class ServedOnlyMiddleware:
    """The server answers only for the model it booted, and only on the routes a client of that model uses (SERVED_ROUTES).

    Upstream is a multi-model server. A chat request naming another model makes it unload the resident weights and load that one in their place — fetching it from the Hub when it is not on disk — and one carrying an adapter path loads the adapter onto them, so what answers is no longer what the card prices and guards; the routes for other model kinds (embeddings, reranking, audio, image) load theirs beside the resident weights. Its other routes — other protocols, settings, cache and offload control, unload — have no caller here. So every route outside SERVED_ROUTES is refused, websockets included, and a chat request naming another model or carrying an adapter path is refused before it waits in line for admission; each refusal names what is served. The model list lists the served model alone, since upstream's also lists every model it finds on disk. The allowlist is the whole rule, so a route a later upstream adds is refused with nothing to update.

    A chat request's body is read whole to check it, then handed on as one message. A body that is not a JSON object passes through to upstream, whose own validation answers it."""

    def __init__(self, asgi_app, served: str):
        self.asgi_app = asgi_app
        self.served = served

    def _what_is_served(self) -> str:
        return f"this server serves only {self.served}, at /v1/chat/completions, with /v1/models and /health beside it"

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.asgi_app(scope, receive, send)
        if scope["type"] != "http" or route(scope) not in SERVED_ROUTES:
            refusal = f"{self._what_is_served()}; {scope.get('path', '')} is not served"
            if scope["type"] == "websocket":
                return await send(
                    {"type": "websocket.close", "code": 1008, "reason": refusal}
                )
            return await _send_json(send, 404, {"detail": refusal})
        if route(scope) in MODEL_LIST_ROUTES:
            return await self._models(scope, receive, send)
        if generating(scope):
            return await self._chat(scope, receive, send)
        return await self.asgi_app(scope, receive, send)

    async def _models(self, scope, receive, send):
        start, body = await _buffered_response(self.asgi_app, scope, receive)
        try:
            listing = json.loads(body)
        except ValueError:
            listing = None
        if not (isinstance(listing, dict) and isinstance(listing.get("data"), list)):
            await send(start)
            return await send({"type": "http.response.body", "body": body})
        listing["data"] = [
            m
            for m in listing["data"]
            if isinstance(m, dict) and m.get("id") == self.served
        ]
        await _send_json(
            send, start.get("status", 200), listing, start.get("headers", [])
        )

    async def _chat(self, scope, receive, send):
        chunks = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        body = b"".join(chunks)
        try:
            request = json.loads(body)
        except ValueError:
            request = None
        if isinstance(request, dict):
            model = request.get("model")
            if model != self.served:
                return await _send_json(
                    send,
                    400,
                    {
                        "detail": f"{self._what_is_served()}; the request names model {model!r}"
                    },
                )
            adapter = request.get("adapter_path")
            if adapter is not None:
                return await _send_json(
                    send,
                    400,
                    {
                        "detail": f"{self._what_is_served()}, with no adapter; the request carries adapter_path {adapter!r}"
                    },
                )
        del request
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        return await self.asgi_app(scope, replay, send)


class Admission:
    """What admission holds: the request being served and the line behind it, for /health — and how many requests upstream's GPU loop still holds, which is what the slot waits out before it frees.

    A request's HTTP response can end while the GPU loop is still working on it: a caller that disconnects mid-prefill ends its response at once, and the loop drops the request only at its next step (or, if upstream never hears the cancel, at its last token). The GPU count is fed from upstream's own generation path (install_gpu_tracking): a request is counted from the moment upstream starts preparing it for the GPU until the loop sends it its terminal message — None, or the exception it failed with — which the loop sends only once it has removed the request from its batch."""

    def __init__(self):
        import threading

        self.running = 0
        self.waiting = 0
        self._in_gpu = 0
        self._lock = threading.Lock()
        self._wakers: list = []

    def state(self) -> dict:
        return {"running": self.running, "waiting": self.waiting}

    def entered_gpu(self) -> None:
        with self._lock:
            self._in_gpu += 1

    def left_gpu(self) -> None:
        with self._lock:
            self._in_gpu -= 1
            if self._in_gpu:
                return
            wakers, self._wakers = self._wakers, []
        for loop, event in wakers:
            loop.call_soon_threadsafe(event.set)

    async def gpu_drained(self) -> None:
        """Return once upstream's GPU loop holds no request."""
        import asyncio

        event = asyncio.Event()
        with self._lock:
            if not self._in_gpu:
                return
            self._wakers.append((asyncio.get_running_loop(), event))
        await event.wait()


class AdmissionMiddleware:
    """Admission to the generating routes: one request resident at a time, first come first served, the line as long as it is.

    A request holds the slot from admission until its response is done, whatever ended it, and upstream's GPU loop has dropped it (Admission.gpu_drained) — a caller that disconnects ends its response while the loop is still prefilling or decoding for it. One resident request is the whole memory policy: the machine holds the weights, one cache, and one prefill's working set, never a batch, and whether that fits under the dial is not projectable from the request — memory is a function of upstream's forward pass over the request's exact shape, not of its token count — so nothing is refused for memory; a request that does not fit dies in Metal, and the cut is made after the fact (analysis/README.md). It is also the whole throughput policy on this backend: a prefill saturates the GPU and starves every decode sharing it, and a batched decode over rows padded to the longest reads more cache per step than it saves in weights, so a second resident request costs more than it returns. The line behind the slot is as long as it is — a waiting caller holds its own request body in this process's heap and nothing else (the body is read off the socket while it waits, which is how its disconnect is seen, and a caller that disconnects leaves the line) — and /health says how long."""

    def __init__(self, asgi_app, admission):
        import asyncio

        self.asgi_app = asgi_app
        self.admission = admission
        self.slot = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if not generating(scope):
            return await self.asgi_app(scope, receive, send)
        import asyncio

        # A waiter's socket is watched for disconnect, and a disconnected waiter is never
        # admitted: a prefill for a dead socket is minutes of GPU the line behind it pays
        # for. What the watch read off the socket before admission (the request body) is
        # replayed to the app, which sees the request whole.
        buffered: list = []
        gone = asyncio.Event()

        async def watch():
            while True:
                message = await receive()
                buffered.append(message)
                if message["type"] == "http.disconnect":
                    gone.set()
                    return

        watcher = asyncio.create_task(watch())
        acquire = asyncio.create_task(self.slot.acquire())
        left = asyncio.create_task(gone.wait())

        async def settle(*tasks):
            """Every helper ended and awaited, whatever it was doing, and the slot handed back if the acquire got it — on every exit but the admitting one, admission is left exactly as found."""
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if (
                acquire.done()
                and not acquire.cancelled()
                and acquire.exception() is None
            ):
                self.slot.release()

        self.admission.waiting += 1
        try:
            await asyncio.wait({acquire, left}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # The caller's own task is being cancelled (the server shutting down): the
            # wait does not cancel what it waited on, so an orphaned acquire would take
            # the slot later with nobody to release it, and nothing would ever be admitted again.
            await settle(watcher, left, acquire)
            raise
        finally:
            self.admission.waiting -= 1
        if gone.is_set():
            await settle(watcher, left, acquire)
            return
        left.cancel()
        watcher.cancel()
        await asyncio.gather(left, watcher, return_exceptions=True)

        async def replay():
            if buffered:
                return buffered.pop(0)
            return await receive()

        self.admission.running += 1
        try:
            return await self.asgi_app(scope, replay, send)
        finally:
            # The response ending is not the request leaving the GPU: a caller that
            # disconnects ends its response while the loop still prefills or decodes it,
            # so the slot is held until the loop has dropped every request it holds.
            try:
                await self.admission.gpu_drained()
            finally:
                self.admission.running -= 1
                self.slot.release()


class TelemetryMiddleware:
    """Pure ASGI (BaseHTTPMiddleware buffers and breaks SSE streaming): brackets each generating request with the full-signal memory sampler and logs the summary when the response — streamed or not — finishes sending. Then releases the allocator's buffer cache, which is wired memory the OS cannot reclaim while the model idles holding it. The clear runs after the end snapshot, so the telemetry still records what the request held.

    The bracket opens by resetting mlx's peak, so the peak every reader of this request sees — the serve-log line, the memory.jsonl rows, the `timings` upstream returns — is this request's own. It sits inside admission, which opens only once the GPU loop holds nothing, so no other request's work lands in it. mlx restarts the peak from zero and raises it at each allocation, so a reading before the request first allocates — its start row — is 0. The peak counts mlx's live arrays only — the weights, the KV cache, the vision feature cache, the request's working set — never the allocator's buffer cache or the machine's wired memory, which the snapshot reads beside it.

    The request id keying the telemetry rides back on the response as the `x-locus-request` header — it is minted here, so this is the only place the caller can learn which memory.jsonl rows and serve-log line are its own."""

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if not generating(scope):
            return await self.asgi_app(scope, receive, send)
        import mlx.core as mx

        from .telemetry import MemoryTelemetry

        mx.reset_peak_memory()
        telemetry = MemoryTelemetry(uuid.uuid4().hex[:12])
        started = time.time()

        async def sending(message):
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        (b"x-locus-request", telemetry.request_id.encode()),
                    ],
                }
            await send(message)

        try:
            await self.asgi_app(scope, receive, sending)
        finally:
            summary = telemetry.end()
            logger.info(
                f"request {telemetry.request_id} done in {time.time() - started:.1f}s: wired max {summary['max_system_wired_gib']} GiB, end {summary['end']}"
            )
            mx.clear_cache()


class HealthMemoryMiddleware:
    """Add the live memory snapshot to upstream's /health under `memory`, and the admission state under `admission` — the request being served and how many wait — so a caller in line can read where it stands. The memory read is best-effort — on failure `memory` carries the reason. /health is a small non-streamed JSON body, so it is buffered and rewritten, under upstream's status; a body upstream sends that is not a JSON object — an error page — is carried whole, stripped, as `upstream_error`."""

    def __init__(self, asgi_app, admission=None):
        self.asgi_app = asgi_app
        self.admission = admission

    async def __call__(self, scope, receive, send):
        if not (
            scope["type"] == "http"
            and scope.get("method") == "GET"
            and route(scope) == "/health"
        ):
            return await self.asgi_app(scope, receive, send)

        start, body = await _buffered_response(self.asgi_app, scope, receive)

        from .telemetry import memory_snapshot

        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            payload = {"upstream_error": body.decode(errors="replace").strip()}
        payload["current_timestamp"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        try:
            payload["memory"] = memory_snapshot()
        # The snapshot reads the machine through subprocesses, a regex, and dict lookups, so its
        # failure modes are open-ended; /health is what a caller asks whether the server is up at
        # all, and it answering outranks every one of them.
        except Exception as exc:  # noqa: BLE001
            payload["memory"] = {"error": f"{type(exc).__name__}: {exc}"}
        if self.admission is not None:
            payload["admission"] = self.admission.state()
        await _send_json(
            send, start.get("status", 200), payload, start.get("headers", [])
        )


def install_prompt_recording() -> None:
    """Record each request's rendered prompt — the exact templated string upstream hands the tokenizer — to data/mlx/rendered_prompt.txt, latest request winning. Upstream exposes the rendering nowhere (its response and debug log carry only counts), and the rendering is the one place image placement is checkable against the wire, so the server writes it where the integration suite and a reader can open it."""
    from mlx_vlm.server import openai as server_openai

    from .paths import rendered_prompt

    render = server_openai.apply_chat_template

    def recording(*args, **kwargs):
        rendered = render(*args, **kwargs)
        if isinstance(rendered, str):
            path = rendered_prompt()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
        return rendered

    server_openai.apply_chat_template = recording


def install_preprocess_serializer() -> None:
    """Serialize upstream's caller-thread CPU preprocessing.

    Upstream runs _cpu_preprocess (prepare_inputs: Rust tokenizer encode + image load/resize) on each request's own thread against ONE shared processor instance (server/generation.py: _cpu_preprocess runs on the caller thread before the request is queued to the GPU thread), and the HF fast tokenizer's Rust core raises "RuntimeError: Already borrowed" on concurrent use — a PyO3 RefCell double-borrow on one shared tokenizer (huggingface/tokenizers#537, huggingface/transformers#12658). Upstream holds its own _tokenizer_lock around the generate path's preprocessing, but validate_context_budget — run on the caller thread before every streaming response opens — preprocesses outside it. One process-wide lock around _cpu_preprocess covers every path, serializing CPU-side preprocessing; GPU batching is untouched."""
    from threading import Lock

    from mlx_vlm.server import generation

    lock = Lock()
    original = generation.ResponseGenerator._cpu_preprocess

    def serialized(self, *args, **kwargs):
        with lock:
            return original(self, *args, **kwargs)

    generation.ResponseGenerator._cpu_preprocess = serialized


EXPECTED_GPU_EMBED_SHA256 = (
    "ad945605ec505fa9975ff9a45d958fb03a6343f1f2ef8d8743aca3774d385ceb"
)


def install_pixel_release() -> None:
    """Release a request's pixel arrays the moment its images are encoded.

    Upstream's GPU loop binds each admitted request's preprocessed inputs — pixel_values among them, an mlx array of every image's patches — to loop locals before encoding them in _gpu_embed, and those locals outlive the loop body: the arrays stay referenced through the request's whole prefill and decode, and past it, until the next request rebinds the names. Nothing reads them after _gpu_embed (its output carries the encoded embeddings, and the vision cache keeps its own encoded copy), so dropping them from the inputs once it returns frees them at the encode. It is keyed to _gpu_embed's exact source, the step whose contract that rests on (patching.py owns the policy)."""
    from mlx_vlm.server import generation

    from .patching import expect_source

    expect_source(
        "pixel release",
        "ResponseGenerator._gpu_embed",
        generation.ResponseGenerator._gpu_embed,
        EXPECTED_GPU_EMBED_SHA256,
        "The release rests on _gpu_embed reading the pixel arrays only to encode them and "
        "returning nothing that references them — re-verify that against the installed "
        "version, then re-pin.",
    )
    original = generation.ResponseGenerator._gpu_embed

    def releasing(self, raw_inputs, *args, **kwargs):
        encoded = original(self, raw_inputs, *args, **kwargs)
        for key in [k for k in raw_inputs if k.startswith("pixel_values")]:
            del raw_inputs[key]
        return encoded

    generation.ResponseGenerator._gpu_embed = releasing


EXPECTED_GPU_LIFECYCLE_SHA256 = {
    "QueuedGenerationRequest": "690821fb0c0336df699a52c3ea9b64efb5deb016bf3dfc91b03c7c32c35a2a26",
    "_notify_queues": "6556e1c7ea3bc865a0c5d77a55c4e0f1a315a904c39f3adfb9e1b66afb2cd714",
    "ResponseGenerator.generate": "3f8cdfc9e6dd715bbcedcb044af5ba84968dbb661923f7febb5a4c1a99b0fab2",
    "ResponseGenerator._run_impl": "024ace0b29d1ad6683ed5d573ce6835a1b1267421d41cfbe60197ebb4958395e",
    "ResponseGenerator._step": "d7abfe591a2ab6cdd0fe4b577c29786198e00722f1f7499b53b1af41a1df657a",
    "ResponseGenerator._apply_cancellations": "7c1d8a8391d8550b961ab912b2753d656f615f3c7dfa21774f4f5a662dedce0f",
}


def install_gpu_tracking(admission: Admission) -> None:
    """Count the requests upstream's GPU loop holds, so admission frees its slot only once the loop has dropped them.

    ResponseGenerator.generate runs on a worker thread the caller's handler awaits: it preprocesses the request (tokenizing, loading and resizing images — seconds for a large image set), builds one QueuedGenerationRequest, and hands it to the GPU loop's queue; the loop answers on the request's own rqueue and ends every request it took with a terminal message — None when it finishes or is cancelled (sent after the request is out of the batch), an exception when inserting or stepping it fails. A caller that disconnects while generate is still preprocessing ends its response at once, but the worker thread runs on and queues the request anyway. So a request is counted from the moment generate starts until the first terminal message put on its rqueue — or until generate raises before building one, which queues nothing. Every path to those messages lives in the pinned functions, so the count is keyed to their exact source and the server refuses to boot on drift (patching.py owns the policy)."""
    from mlx_vlm.server import generation

    from .patching import expect_source

    pinned = {
        "QueuedGenerationRequest": generation.QueuedGenerationRequest,
        "_notify_queues": generation._notify_queues,
        "ResponseGenerator.generate": generation.ResponseGenerator.generate,
        "ResponseGenerator._run_impl": generation.ResponseGenerator._run_impl,
        "ResponseGenerator._step": generation.ResponseGenerator._step,
        "ResponseGenerator._apply_cancellations": generation.ResponseGenerator._apply_cancellations,
    }
    for name, fn in pinned.items():
        expect_source(
            "GPU tracking",
            name,
            fn,
            EXPECTED_GPU_LIFECYCLE_SHA256[name],
            "The count rests on every request the GPU loop takes ending with exactly one "
            "terminal message on its rqueue (None, or an exception) sent once it is out of "
            "the batch — re-verify every path to those messages against the installed "
            "version, then re-pin.",
        )

    import threading

    queued = generation.QueuedGenerationRequest
    built = threading.local()

    class TrackedRequest(queued):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.request = True
            put = self.rqueue.put
            ended = threading.Event()

            def tracking(item, *put_args, **put_kwargs):
                put(item, *put_args, **put_kwargs)
                if (
                    item is None or isinstance(item, BaseException)
                ) and not ended.is_set():
                    ended.set()
                    admission.left_gpu()

            self.rqueue.put = tracking

    generate = generation.ResponseGenerator.generate

    def tracked_generate(self, *args, **kwargs):
        built.request = False
        admission.entered_gpu()
        try:
            return generate(self, *args, **kwargs)
        except BaseException:
            if not built.request:
                admission.left_gpu()
            raise

    generation.QueuedGenerationRequest = TrackedRequest
    generation.ResponseGenerator.generate = tracked_generate


def main() -> None:
    require_extra("mlx.core", "mlx_vlm", "uvicorn", "pillow_heif")
    install_log_handler()
    models = tomllib.loads(MODELS_TOML.read_text())
    if len(sys.argv) != 2 or sys.argv[1] not in models:
        raise SystemExit(
            f"usage: locus-mlx-serve <model-slug>; available: {sorted(models)}"
        )
    slug = sys.argv[1]
    config = models[slug]

    cap = _wired_cap_bytes()
    import mlx.core as mx

    mx.set_wired_limit(cap)
    mx.set_memory_limit(cap)

    card = _card(slug)
    if card is None:
        raise SystemExit(
            f"no card for slug {slug!r} at {card_file(slug)} — the card owns which repo a slug serves (`model`), the context it holds (`context_tokens`), and the endpoint it dials (`base_url`); declare it there, beside this entry's serving env."
        )
    hf_repo = card["model"]
    try:
        host, port = bind_from_card(card)
    except ValueError as exc:
        raise SystemExit(exc) from exc

    os.environ["MLX_VLM_PRELOAD_MODEL"] = hf_repo
    # The declarations — models.toml's env and the card's KV budget — beat an inherited shell
    # value, loudly. An override is applied by editing the declaration.
    for key, value in {
        **config["env"],
        "MAX_KV_SIZE": str(card["context_tokens"]),
    }.items():
        inherited = os.environ.get(key)
        if inherited is not None and inherited != value:
            logger.warning(
                f"{key}={inherited!r} in the environment; the declaration says {value!r} and wins"
            )
        os.environ[key] = value

    from .telemetry import memory_snapshot

    logger.info(
        f"Wired and memory limits set to {cap // 2**20} MB (the iogpu.wired_limit_mb sysctl — the operator's dial). The wired limit keeps that many bytes resident (mlx.core.set_wired_limit). The memory limit matches it so the allocator reclaims its own cache at the dial instead of growing past it toward MLX's default of 1.5x the device's recommended working set; past the limit MLX keeps allocating while the machine still has RAM or swap and raises when it has neither (mlx.core.set_memory_limit)."
    )
    logger.info(f"Boot memory snapshot: {memory_snapshot()}")
    logger.info(
        f"Serving {hf_repo} ({slug}) via upstream mlx_vlm.server on {host}:{port} (card base_url)"
    )

    install_heif_decode()
    install_preprocess_serializer()
    install_pixel_release()

    from .vision_chunking import install_vision_encode_chunking

    install_vision_encode_chunking()
    logger.info(
        "vision-encode chunking installed: a request's images encode in pixel-bounded chunks (LOCUS_MLX_ENCODE_CHUNK_MPX)"
    )

    from .fused_attention import install_fused_attention

    install_fused_attention()
    logger.info(
        "fused attention installed: Qwen's attention reads every chunk through mlx's fused kernel, never materializing the score table"
    )

    from mlx_vlm.server import app

    install_prompt_recording()

    admission = Admission()
    install_gpu_tracking(admission)
    app.add_middleware(HealthMemoryMiddleware, admission=admission)
    app.add_middleware(TelemetryMiddleware)
    app.add_middleware(AdmissionMiddleware, admission=admission)
    app.add_middleware(ServedOnlyMiddleware, served=hf_repo)

    serve_guarded(app, port, host=host)


if __name__ == "__main__":
    main()
