"""Thin locus layer over the upstream mlx-vlm server.

Upstream (mlx_vlm.server) owns generation: continuous batching, structured output, thinking budget, APC prefix caching, speculative decode, vision feature cache.
This module boots it behind the operational guarantees a resident model demands — memory limits from the sysctl (main), admission (AdmissionMiddleware: one request resident at a time, first come first served; the line waits and nothing is refused), the liveness guard (serve_guarded), memory telemetry (TelemetryMiddleware), the rendered-prompt record (install_prompt_recording), a lock over upstream's caller-thread preprocessing (install_preprocess_serializer), two math-preserving patches (the GQA decode kernel, the vision-encode chunking), and one that renders a conversation's images in the message, and at the moment, that carried them (media_placement). Every upstream-keyed patch verifies the source it stands in for at boot and refuses to serve on drift (patching.py).
README.md says what each is for.

Known upstream gaps, version-keyed — re-validate on every mlx-vlm upgrade:
- top_k/min_p are accepted per-request but dropped by the batched sampler; the cards therefore do not claim top_k.
- mlx's decode-attention kernel halves its bandwidth on high-GQA shapes; gqa_decode.install_gqa_decode routes eligible calls to a shared-tile kernel.
- a request's images encode in one batch, so the vision-encode transient scales with the request's total pixels and no token-denominated guard bounds it — a large enough image batch aborts the server even when it fits the token context (MAX_KV_SIZE counts expanded vision tokens, so its guard doesn't close this).
  Patched here: vision_chunking.install_vision_encode_chunking encodes in pixel-bounded chunks, output-identical on the pinned upstream.
- the chat-completions handler flattens every message's media away before templating: a conversation's images all land on its newest user message, and a message's own text/image interleaving collapses to an image block ahead of its text. media_placement.install_media_placement renders each image where its message put it.
- thinking_budget is rejected in combination with speculative decoding, so no draft model can be configured while the cards send a budget (they always do — it absorbs the thinking-runaway).
- a prefill batch of several prompts with a large length spread faults get_rope_index and 500s every in-flight request (upstream issue https://github.com/Blaizzy/mlx-vlm/issues/1346); admission serves one request at a time, so its prefill batch is always one row and the fault has no path.

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


# Upstream's routes that submit work to its single GPU generator: chat completions and responses
# (server/openai.py) and the Anthropic messages endpoint (server/anthropic.py) all reach
# runtime.response_generator.generate, so all three pass admission, and all three are what
# the memory telemetry brackets. Every other upstream route
# is a metadata, tokenizer, or control read that never reaches the generator. The `/v1` prefix is
# optional on all of them.
GENERATING_ROUTES = frozenset({"/chat/completions", "/responses", "/messages"})

logger = logging.getLogger("locus.analysis.mlx.serve")


def generating(scope) -> bool:
    if scope["type"] != "http":
        return False
    path = scope.get("path", "").rstrip("/")
    return path.removeprefix("/v1") in GENERATING_ROUTES


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


class Admission:
    """What admission holds: the request being served and the line behind it, for /health."""

    def __init__(self):
        self.running = 0
        self.waiting = 0

    def state(self) -> dict:
        return {"running": self.running, "waiting": self.waiting}


class AdmissionMiddleware:
    """Admission to the generating routes: one request resident at a time, first come first served, the line as long as it is.

    A request holds the slot from admission until its response is done, whatever ended it. One resident request is the whole memory policy: the machine holds the weights, one cache, and one prefill's working set, never a batch, and whether that fits under the dial is not projectable from the request — memory is a function of upstream's forward pass over the request's exact shape, not of its token count — so nothing is refused for memory; a request that does not fit dies in Metal, and the cut is made after the fact (analysis/README.md). It is also the whole throughput policy on this backend: a prefill saturates the GPU and starves every decode sharing it, and a batched decode over rows padded to the longest reads more cache per step than it saves in weights, so a second resident request costs more than it returns. Serving one at a time also keeps upstream's prefill batch at one row (the get_rope_index fault in the module docstring). The line behind the slot is as long as it is — a waiting caller holds its own request body in this process's heap and nothing else (the body is read off the socket while it waits, which is how its disconnect is seen, and a caller that disconnects leaves the line) — and /health says how long."""

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
            self.admission.running -= 1
            self.slot.release()


class TelemetryMiddleware:
    """Pure ASGI (BaseHTTPMiddleware buffers and breaks SSE streaming): brackets each generating request with the full-signal memory sampler and logs the summary when the response — streamed or not — finishes sending. Then releases the allocator's buffer cache, which is wired memory the OS cannot reclaim while the model idles holding it. The clear runs after the end snapshot, so the telemetry still records what the request held.

    The request id keying the telemetry rides back on the response as the `x-locus-request` header — it is minted here, so this is the only place the caller can learn which memory.jsonl rows and serve-log line are its own."""

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if not generating(scope):
            return await self.asgi_app(scope, receive, send)
        import mlx.core as mx

        from .telemetry import MemoryTelemetry

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
    """Add the live memory snapshot to upstream's /health under `memory`, and the admission state under `admission` — the request being served and how many wait — so a caller in line can read where it stands. The memory read is best-effort — on failure `memory` carries the reason. /health is a small non-streamed JSON body, so it is buffered and rewritten."""

    def __init__(self, asgi_app, admission=None):
        self.asgi_app = asgi_app
        self.admission = admission

    async def __call__(self, scope, receive, send):
        if not (
            scope["type"] == "http"
            and scope.get("method") == "GET"
            and scope.get("path", "").rstrip("/") == "/health"
        ):
            return await self.asgi_app(scope, receive, send)

        start = {}
        body = bytearray()

        async def capture(message):
            if message["type"] == "http.response.start":
                start.update(message)
            elif message["type"] == "http.response.body":
                body.extend(message.get("body", b""))

        await self.asgi_app(scope, receive, capture)

        from .telemetry import memory_snapshot

        try:
            payload = json.loads(bytes(body) or b"{}")
        except ValueError:
            payload = {}
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
        merged = json.dumps(payload).encode()

        headers = [
            (k, v)
            for k, v in start.get("headers", [])
            if k.lower() != b"content-length"
        ]
        headers.append((b"content-length", str(len(merged)).encode()))
        await send(
            {
                "type": "http.response.start",
                "status": start.get("status", 200),
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": merged})


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

    from .gqa_decode import install_gqa_decode

    install_gqa_decode()
    logger.info(
        "gqa decode kernel installed: eligible qwen3_5 decode-attention calls route to the shared-tile kernel"
    )

    from .vision_chunking import install_vision_encode_chunking

    install_vision_encode_chunking()
    logger.info(
        "vision-encode chunking installed: a request's images encode in pixel-bounded chunks (LOCUS_MLX_ENCODE_CHUNK_MPX)"
    )

    from .media_placement import install_media_placement

    install_media_placement()
    logger.info(
        "media placement installed: a request's images render in the message, and at the moment, that carried them"
    )

    from mlx_vlm.server import app

    install_prompt_recording()

    admission = Admission()
    app.add_middleware(HealthMemoryMiddleware, admission=admission)
    app.add_middleware(TelemetryMiddleware)
    app.add_middleware(AdmissionMiddleware, admission=admission)

    serve_guarded(app, port, host=host)


if __name__ == "__main__":
    main()
