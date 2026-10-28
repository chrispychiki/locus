"""Unit tests for the thin layer — no model, no server, no GPU."""

import asyncio
import os
import socket
import subprocess
from typing import ClassVar

import pytest
from locus.analysis.mlx import serve
from locus.analysis.mlx.serve import (
    Admission,
    AdmissionMiddleware,
    HealthMemoryMiddleware,
    TelemetryMiddleware,
)

# A generating route — what the middlewares gate.
CHAT = {"type": "http", "path": "/v1/chat/completions"}


def run_asgi(middleware, scope, messages):
    """Drive an ASGI app with canned receive messages; returns sent events."""
    sent = []
    queue = list(messages)

    async def receive():
        if queue:
            return queue.pop(0)
        return {"type": "http.disconnect", "sentinel": True}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def http_scope(method="POST", path="/v1/chat/completions", headers=None):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [(b"content-length", b"0")],
    }


class FakeTelemetry:
    instances: ClassVar[list] = []

    def __init__(self, request_id):
        self.request_id = request_id
        self.ended = False
        FakeTelemetry.instances.append(self)

    def end(self):
        self.ended = True
        return {"max_system_wired_gib": 1.0, "end": {}}


class TestTelemetryMiddleware:
    @pytest.fixture(autouse=True)
    def patch_telemetry(self, monkeypatch):
        FakeTelemetry.instances = []
        import locus.analysis.mlx.telemetry

        monkeypatch.setattr(
            locus.analysis.mlx.telemetry, "MemoryTelemetry", FakeTelemetry
        )

    def test_brackets_chat_request(self):
        order = []

        async def inner(scope, receive, send):
            order.append("inner")

        run_asgi(TelemetryMiddleware(inner), http_scope(), [])
        (telemetry,) = FakeTelemetry.instances
        assert order == ["inner"]
        assert telemetry.ended

    def test_ends_even_when_inner_app_raises(self):
        async def inner(scope, receive, send):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            run_asgi(TelemetryMiddleware(inner), http_scope(), [])
        (telemetry,) = FakeTelemetry.instances
        assert telemetry.ended

    def test_non_chat_traffic_skipped(self):
        async def inner(scope, receive, send):
            pass

        run_asgi(TelemetryMiddleware(inner), http_scope(path="/health"), [])
        assert FakeTelemetry.instances == []

    def test_cache_cleared_after_the_end_snapshot(self, monkeypatch):
        """The allocator's buffer cache is wired memory the OS cannot reclaim while the model idles holding it, so every generating request must end with a clear — taken after the end snapshot, so the telemetry still records what the request held."""
        import mlx.core as mx

        order = []
        monkeypatch.setattr(mx, "clear_cache", lambda: order.append("clear"))
        original_end = FakeTelemetry.end

        def spying_end(self):
            order.append("end")
            return original_end(self)

        monkeypatch.setattr(FakeTelemetry, "end", spying_end)

        async def inner(scope, receive, send):
            pass

        run_asgi(TelemetryMiddleware(inner), http_scope(), [])
        assert order == ["end", "clear"]

    def test_the_peak_is_reset_before_the_bracket_opens(self, monkeypatch):
        """Every reader of a request's peak — its start snapshot, the serve-log line, upstream's timings — reads mlx's peak since this reset, so the peak is the request's own."""
        import mlx.core as mx

        order = []
        monkeypatch.setattr(mx, "reset_peak_memory", lambda: order.append("reset"))
        original_init = FakeTelemetry.__init__

        def spying_init(self, request_id):
            order.append("start")
            original_init(self, request_id)

        monkeypatch.setattr(FakeTelemetry, "__init__", spying_init)

        async def inner(scope, receive, send):
            order.append("inner")

        run_asgi(TelemetryMiddleware(inner), http_scope(), [])
        assert order == ["reset", "start", "inner"]

    def test_response_carries_the_telemetry_request_id(self):
        async def inner(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b"{}"})

        sent = run_asgi(TelemetryMiddleware(inner), http_scope(), [])
        (telemetry,) = FakeTelemetry.instances
        start = next(m for m in sent if m["type"] == "http.response.start")
        headers = dict(start["headers"])
        assert headers[b"x-locus-request"] == telemetry.request_id.encode()


class TestHealthMemoryMiddleware:
    @pytest.fixture(autouse=True)
    def patch_snapshot(self, monkeypatch):
        import locus.analysis.mlx.telemetry

        monkeypatch.setattr(
            locus.analysis.mlx.telemetry,
            "memory_snapshot",
            lambda: {"mlx_active_gib": 27.0},
        )

    def test_merges_memory_into_health(self):
        async def inner(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", b"1"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b'{"status": "healthy"}'})

        sent = run_asgi(
            HealthMemoryMiddleware(inner), http_scope(method="GET", path="/health"), []
        )
        import json

        body = next(m for m in sent if m["type"] == "http.response.body")["body"]
        payload = json.loads(body)
        assert payload["status"] == "healthy"
        assert payload["memory"] == {"mlx_active_gib": 27.0}
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert dict(start["headers"])[b"content-length"] == str(len(body)).encode()

    def test_memory_read_failure_degrades_without_failing_health(self, monkeypatch):
        import locus.analysis.mlx.telemetry

        def boom():
            raise RuntimeError("vm_stat unavailable")

        monkeypatch.setattr(locus.analysis.mlx.telemetry, "memory_snapshot", boom)

        async def inner(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": b'{"status": "healthy"}'})

        sent = run_asgi(
            HealthMemoryMiddleware(inner), http_scope(method="GET", path="/health"), []
        )
        import json

        start = next(m for m in sent if m["type"] == "http.response.start")
        assert start["status"] == 200
        payload = json.loads(
            next(m for m in sent if m["type"] == "http.response.body")["body"]
        )
        assert payload["status"] == "healthy"
        assert "vm_stat unavailable" in payload["memory"]["error"]

    @pytest.mark.parametrize(
        ("body", "carried"),
        [
            (b"  Internal Server Error\n", "Internal Server Error"),
            (b'["not", "an object"]', '["not", "an object"]'),
        ],
    )
    def test_an_upstream_body_that_is_not_a_json_object_is_carried_under_its_status(
        self, body, carried
    ):
        async def inner(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": body})

        sent = run_asgi(
            HealthMemoryMiddleware(inner, admission=Admission()),
            http_scope(method="GET", path="/health"),
            [],
        )
        status, payload = response_of(sent)
        assert status == 500
        assert payload["upstream_error"] == carried
        assert payload["memory"] == {"mlx_active_gib": 27.0}
        assert payload["admission"] == {"running": 0, "waiting": 0}

    def test_non_health_passes_through_untouched(self):
        seen = []

        async def inner(scope, receive, send):
            seen.append("inner")
            await send({"type": "http.response.body", "body": b"raw"})

        sent = run_asgi(
            HealthMemoryMiddleware(inner),
            http_scope(method="POST", path="/v1/chat/completions"),
            [],
        )
        assert seen == ["inner"]
        assert sent == [{"type": "http.response.body", "body": b"raw"}]


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestRunPortGuarded:
    @pytest.fixture(autouse=True)
    def isolate(self, monkeypatch, tmp_path):
        self.pidfile = tmp_path / "server.pid"
        monkeypatch.setattr(serve, "PIDFILE", self.pidfile)
        self.launches = []
        import uvicorn

        monkeypatch.setattr(
            uvicorn, "run", lambda app, **kw: self.launches.append((app, kw))
        )

    def test_alive_prior_pid_refuses_boot(self, capsys):
        self.pidfile.write_text(str(os.getpid()))
        with pytest.raises(SystemExit):
            serve.serve_guarded(object(), free_port())
        assert "still alive" in capsys.readouterr().out
        assert self.launches == []
        assert self.pidfile.exists()

    def test_stale_pidfile_removed_then_boots(self):
        dead = subprocess.Popen(["true"])
        dead.wait()
        self.pidfile.write_text(str(dead.pid))
        app = object()
        port = free_port()
        serve.serve_guarded(app, port)
        assert self.launches == [(app, {"host": "127.0.0.1", "port": port})]
        assert not self.pidfile.exists()

    def test_a_prior_server_we_may_not_signal_is_alive_not_dead(
        self, capsys, monkeypatch
    ):
        """EPERM from kill(pid, 0) is the kernel confirming the process exists and refusing us the
        signal — which is exactly what a server booted under sudo looks like to a later unprivileged
        boot, and the README's own first step is a sudo sysctl. Read as death it removes the pidfile
        and boots a second resident model, and two resident models is the state the machine does not
        come back from."""

        def refuse(pid, sig):
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(serve.os, "kill", refuse)
        self.pidfile.write_text("4242")

        with pytest.raises(SystemExit):
            serve.serve_guarded(object(), free_port())

        assert "alive" in capsys.readouterr().out
        assert self.launches == []
        assert self.pidfile.exists()  # never removed as stale

    def test_foreign_port_holder_refuses_boot(self, capsys):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        try:
            with pytest.raises(SystemExit):
                serve.serve_guarded(object(), port)
        finally:
            sock.close()
        assert "non-locus process" in capsys.readouterr().out
        assert self.launches == []

    def test_pidfile_written_during_run_and_cleaned_after(self, monkeypatch):
        observed = {}
        import uvicorn

        monkeypatch.setattr(
            uvicorn,
            "run",
            lambda app, **kw: observed.update(pid=self.pidfile.read_text().strip()),
        )
        serve.serve_guarded(object(), free_port())
        assert observed["pid"] == str(os.getpid())
        assert not self.pidfile.exists()

    def test_exit_never_clobbers_a_successors_pidfile(self, monkeypatch):
        """A dying server may only remove its own claim: stop-server.sh can have cleared the pidfile and a successor claimed it while this process winds down, and unlinking the successor's claim would license a third boot beside it — two resident models."""
        import uvicorn

        monkeypatch.setattr(
            uvicorn, "run", lambda app, **kw: self.pidfile.write_text("31337")
        )
        serve.serve_guarded(object(), free_port())
        assert self.pidfile.read_text() == "31337"


class TestWiredCap:
    """The sysctl half of the boot guard: iogpu.wired_limit_mb is the operator's dial and the only memory ceiling the boot accepts — a failed or unset read must abort, never read as a zero the boot would take for a limit."""

    def fake_sysctl(self, monkeypatch, returncode=0, stdout="", stderr=""):
        result = subprocess.CompletedProcess(
            [], returncode, stdout=stdout, stderr=stderr
        )
        monkeypatch.setattr(serve.subprocess, "run", lambda *a, **kw: result)

    def test_reads_the_dial_in_bytes(self, monkeypatch):
        self.fake_sysctl(monkeypatch, stdout="24576\n")
        assert serve._wired_cap_bytes() == 24576 * 2**20

    def test_failed_read_aborts_the_boot(self, monkeypatch):
        self.fake_sysctl(monkeypatch, returncode=1, stderr="unknown oid")
        with pytest.raises(SystemExit, match="iogpu.wired_limit_mb"):
            serve._wired_cap_bytes()

    @pytest.mark.parametrize("stdout", ["0\n", "", "-1\n"])
    def test_unset_dial_aborts_the_boot(self, monkeypatch, stdout):
        self.fake_sysctl(monkeypatch, stdout=stdout)
        with pytest.raises(SystemExit, match="operator's dial"):
            serve._wired_cap_bytes()


class TestGeneratingRoutes:
    """Admission covers chat completions, the one served route that reaches upstream's generator, and leaves the reads beside it free: a /health that queued behind a prefill could not tell a caller in line where it stands."""

    @pytest.mark.parametrize(
        "path", ["/v1/chat/completions", "/chat/completions", "/v1/chat/completions/"]
    )
    def test_chat_completions_gated(self, path):
        assert serve.generating(http_scope(path=path))

    @pytest.mark.parametrize("path", ["/health", "/v1/models", "/models"])
    def test_served_reads_free(self, path):
        assert not serve.generating(http_scope(path=path))

    def test_lifespan_scope_is_not_a_request(self):
        assert not serve.generating({"type": "lifespan"})


SERVED = "mlx-community/Served-Model"


def json_response(status, payload):
    """An inner app answering with one JSON body, the way upstream's handlers do."""
    import json

    async def inner(scope, receive, send):
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return inner


def response_of(sent):
    import json

    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert dict(start["headers"])[b"content-length"] == str(len(body)).encode()
    return start["status"], json.loads(body)


def chat_body(**fields):
    import json

    return json.dumps({"model": SERVED, "messages": [], **fields}).encode()


class TestServedOnly:
    """Upstream swaps its resident weights for whatever model a request names and serves routes nothing here calls, so the server answers only for the model it booted, on the routes its clients use."""

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/responses",
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/embeddings",
            "/v1/rerank",
            "/metrics",
            "/v1/cache/reset",
            "/v1/settings",
            "/unload",
            "/v1/images/defaults",
            "/a/route/upstream/adds/later",
            "/",
            "/v1/health",
            "/v1/v1/chat/completions",
            "/v1/chat/completions/extra",
        ],
    )
    def test_every_other_route_is_refused_naming_what_is_served(self, path):
        reached = []

        async def inner(scope, receive, send):
            reached.append(scope["path"])

        sent = run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED), http_scope(path=path), []
        )
        status, payload = response_of(sent)
        assert reached == []
        assert status == 404
        assert SERVED in payload["detail"] and path in payload["detail"]
        assert "/v1/chat/completions" in payload["detail"]

    def test_a_websocket_is_refused(self):
        reached = []

        async def inner(scope, receive, send):
            reached.append(scope)

        sent = run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED),
            {"type": "websocket", "path": "/v1/realtime"},
            [],
        )
        assert reached == []
        ((close),) = sent
        assert close["type"] == "websocket.close" and SERVED in close["reason"]

    def test_lifespan_passes(self):
        reached = []

        async def inner(scope, receive, send):
            reached.append(scope["type"])

        run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED), {"type": "lifespan"}, []
        )
        assert reached == ["lifespan"]

    @pytest.mark.parametrize("path", ["/health", "/v1/chat/completions"])
    def test_served_routes_reach_upstream(self, path):
        sent = run_asgi(
            serve.ServedOnlyMiddleware(json_response(200, {"ok": True}), served=SERVED),
            http_scope(path=path),
            [{"type": "http.request", "body": chat_body()}],
        )
        assert response_of(sent) == (200, {"ok": True})

    def test_a_chat_request_reaches_upstream_whole_and_still_hears_a_disconnect(self):
        """The body is read whole to check it, then handed on as one message; what follows on the socket — the disconnect admission watches for — still reaches the app."""
        heard = []

        async def inner(scope, receive, send):
            heard.append(await receive())
            heard.append(await receive())

        body = chat_body(max_tokens=4)
        run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED),
            http_scope(),
            [
                {"type": "http.request", "body": body[:10], "more_body": True},
                {"type": "http.request", "body": body[10:], "more_body": False},
                {"type": "http.disconnect"},
            ],
        )
        assert heard == [
            {"type": "http.request", "body": body, "more_body": False},
            {"type": "http.disconnect"},
        ]

    @pytest.mark.parametrize(
        "fields",
        [{"model": "mlx-community/Other"}, {"model": None}, {"model": SERVED.lower()}],
    )
    def test_a_request_for_another_model_is_refused(self, fields):
        reached = []

        async def inner(scope, receive, send):
            reached.append(True)

        sent = run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED),
            http_scope(),
            [{"type": "http.request", "body": chat_body(**fields)}],
        )
        status, payload = response_of(sent)
        assert reached == []
        assert status == 400
        assert (
            SERVED in payload["detail"] and repr(fields["model"]) in payload["detail"]
        )

    def test_a_request_without_a_model_is_refused(self):
        import json

        sent = run_asgi(
            serve.ServedOnlyMiddleware(json_response(200, {}), served=SERVED),
            http_scope(),
            [{"type": "http.request", "body": json.dumps({"messages": []}).encode()}],
        )
        status, payload = response_of(sent)
        assert status == 400 and SERVED in payload["detail"]

    def test_an_adapter_is_refused(self):
        reached = []

        async def inner(scope, receive, send):
            reached.append(True)

        sent = run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED),
            http_scope(),
            [{"type": "http.request", "body": chat_body(adapter_path="/tmp/lora")}],
        )
        status, payload = response_of(sent)
        assert reached == []
        assert status == 400
        assert SERVED in payload["detail"] and "/tmp/lora" in payload["detail"]

    def test_an_explicit_null_adapter_is_the_served_model(self):
        sent = run_asgi(
            serve.ServedOnlyMiddleware(json_response(200, {"ok": True}), served=SERVED),
            http_scope(),
            [{"type": "http.request", "body": chat_body(adapter_path=None)}],
        )
        assert response_of(sent) == (200, {"ok": True})

    def test_a_body_that_is_not_a_json_object_is_upstreams_to_answer(self):
        heard = []

        async def inner(scope, receive, send):
            heard.append((await receive())["body"])

        run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED),
            http_scope(),
            [{"type": "http.request", "body": b"not json"}],
        )
        assert heard == [b"not json"]

    def test_a_caller_gone_before_its_body_arrived_reaches_nothing(self):
        reached = []

        async def inner(scope, receive, send):
            reached.append(True)

        sent = run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED),
            http_scope(),
            [
                {"type": "http.request", "body": b'{"mod', "more_body": True},
                {"type": "http.disconnect"},
            ],
        )
        assert reached == [] and sent == []

    @pytest.mark.parametrize("path", ["/v1/models", "/models"])
    def test_the_model_list_lists_the_served_model_alone(self, path):
        listing = {
            "object": "list",
            "data": [
                {"id": "mlx-community/Other", "object": "model"},
                {"id": SERVED, "object": "model", "loaded": True},
                {"id": "some/local/dir", "object": "model"},
            ],
        }
        sent = run_asgi(
            serve.ServedOnlyMiddleware(json_response(200, listing), served=SERVED),
            http_scope(method="GET", path=path),
            [],
        )
        assert response_of(sent) == (
            200,
            {
                "object": "list",
                "data": [{"id": SERVED, "object": "model", "loaded": True}],
            },
        )

    def test_a_model_list_error_passes_through(self):
        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 500, "headers": []})
            await send({"type": "http.response.body", "body": b"Internal Server Error"})

        sent = run_asgi(
            serve.ServedOnlyMiddleware(inner, served=SERVED),
            http_scope(method="GET", path="/v1/models"),
            [],
        )
        assert sent == [
            {"type": "http.response.start", "status": 500, "headers": []},
            {"type": "http.response.body", "body": b"Internal Server Error"},
        ]


async def _stays():
    """A caller's receive that never speaks: the body is in and the socket stays open."""
    await asyncio.Event().wait()


class TestAdmission:
    def test_one_request_at_a_time_in_arrival_order_and_nothing_refused(self):
        """Six requests at once: each runs alone, from admission to the end of its response, in the order it arrived; none is refused."""
        events = []

        async def inner_app(scope, receive, send):
            events.append((scope["name"], "enter"))
            await send({"type": "http.response.start", "status": 200})
            await send(
                {"type": "http.response.body", "body": b"tok", "more_body": True}
            )
            await asyncio.sleep(0.01)
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            events.append((scope["name"], "exit"))

        middleware = AdmissionMiddleware(inner_app, Admission())

        async def noop_send(message):
            pass

        async def storm():
            await asyncio.gather(
                *(
                    middleware({**CHAT, "name": f"r{i}"}, _stays, noop_send)
                    for i in range(6)
                )
            )

        asyncio.run(storm())
        assert events == [
            (f"r{i}", step) for i in range(6) for step in ("enter", "exit")
        ], "one resident request, first come first served"
        assert middleware.admission.state() == {"running": 0, "waiting": 0}

    def test_the_state_counts_the_one_served_and_the_line_behind_it(self):
        gate = asyncio.Event()

        async def inner_app(scope, receive, send):
            await send(
                {"type": "http.response.body", "body": b"tok", "more_body": True}
            )
            await gate.wait()

        middleware = AdmissionMiddleware(inner_app, Admission())

        async def noop_send(message):
            pass

        async def storm():
            tasks = [
                asyncio.create_task(
                    middleware({**CHAT, "name": str(i)}, _stays, noop_send)
                )
                for i in range(3)
            ]
            await asyncio.sleep(0.05)
            assert middleware.admission.state() == {"running": 1, "waiting": 2}
            gate.set()
            await asyncio.gather(*tasks)

        asyncio.run(storm())
        assert middleware.admission.state() == {"running": 0, "waiting": 0}

    def test_the_slot_is_released_when_a_request_dies(self):
        ran = []

        async def inner(scope, receive, send):
            if scope["name"] == "a":
                raise RuntimeError("died in prefill")
            ran.append(scope["name"])

        middleware = AdmissionMiddleware(inner, Admission())

        async def noop_send(message):
            pass

        async def storm():
            results = await asyncio.gather(
                middleware({**CHAT, "name": "a"}, _stays, noop_send),
                middleware({**CHAT, "name": "b"}, _stays, noop_send),
                return_exceptions=True,
            )
            assert isinstance(results[0], RuntimeError)

        asyncio.run(storm())
        assert ran == ["b"]
        assert middleware.admission.state() == {"running": 0, "waiting": 0}

    def test_a_caller_that_disconnects_while_waiting_leaves_the_line(self):
        """A waiter whose socket closes is never admitted: the line shortens at once, the app never runs for it, and the next caller is served in its place — a prefill for a dead socket is minutes the line behind it would pay for."""
        gate = asyncio.Event()
        ran = []

        async def inner(scope, receive, send):
            ran.append(scope["name"])
            body = await receive()
            ran.append(body["body"])
            await gate.wait()

        middleware = AdmissionMiddleware(inner, Admission())

        async def noop_send(message):
            pass

        def receiving(*messages):
            queue = list(messages)

            async def receive():
                if queue:
                    return queue.pop(0)
                await asyncio.Event().wait()

            return receive

        async def storm():
            first = asyncio.create_task(
                middleware(
                    {**CHAT, "name": "first"},
                    receiving({"type": "http.request", "body": b"one"}),
                    noop_send,
                )
            )
            await asyncio.sleep(0.01)
            leaver = asyncio.create_task(
                middleware(
                    {**CHAT, "name": "leaver"},
                    receiving(
                        {"type": "http.request", "body": b"two"},
                        {"type": "http.disconnect"},
                    ),
                    noop_send,
                )
            )
            stayer = asyncio.create_task(
                middleware(
                    {**CHAT, "name": "stayer"},
                    receiving({"type": "http.request", "body": b"three"}),
                    noop_send,
                )
            )
            await asyncio.sleep(0.05)
            assert leaver.done(), "the leaver is gone from the line at once"
            assert middleware.admission.state() == {"running": 1, "waiting": 1}
            gate.set()
            await asyncio.gather(first, stayer)

        asyncio.run(storm())
        assert ran == ["first", b"one", "stayer", b"three"], (
            "the app never ran for the leaver, and each admitted caller's body was replayed whole"
        )
        assert middleware.admission.state() == {"running": 0, "waiting": 0}

    def test_a_waiter_cancelled_by_its_server_leaves_admission_as_it_found_it(self):
        """The server cancelling a waiting request's task (its own shutdown) must not orphan the acquire: an orphan would take the slot later with nobody to release it, and admission would read idle on /health while every request hung. After the cancel the line is shorter by one, and the next caller is admitted once the holder finishes."""
        gate = asyncio.Event()
        ran = []

        async def inner(scope, receive, send):
            ran.append(scope["name"])
            await gate.wait()

        middleware = AdmissionMiddleware(inner, Admission())

        async def noop_send(message):
            pass

        async def storm():
            first = asyncio.create_task(
                middleware({**CHAT, "name": "first"}, _stays, noop_send)
            )
            await asyncio.sleep(0.01)
            waiter = asyncio.create_task(
                middleware({**CHAT, "name": "waiter"}, _stays, noop_send)
            )
            await asyncio.sleep(0.01)
            assert middleware.admission.state() == {"running": 1, "waiting": 1}
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert middleware.admission.state() == {"running": 1, "waiting": 0}
            gate.set()
            await first
            assert not middleware.slot.locked(), (
                "the cancelled waiter's acquire never took the slot"
            )
            later = asyncio.create_task(
                middleware({**CHAT, "name": "later"}, _stays, noop_send)
            )
            await asyncio.sleep(0.01)
            await later

        asyncio.run(storm())
        assert ran == ["first", "later"]
        assert middleware.admission.state() == {"running": 0, "waiting": 0}

    def test_non_chat_traffic_passes_admission_untouched(self):
        in_flight = {"n": 0, "max": 0}

        async def inner_app(scope, receive, send):
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
            await asyncio.sleep(0.02)
            in_flight["n"] -= 1

        middleware = AdmissionMiddleware(inner_app, Admission())
        models = {"type": "http", "path": "/v1/models"}

        async def storm():
            await asyncio.gather(*(middleware(models, None, None) for _ in range(6)))

        asyncio.run(storm())
        assert in_flight["max"] > 2


class TestHealthAdmission:
    @pytest.fixture(autouse=True)
    def patch_snapshot(self, monkeypatch):
        import locus.analysis.mlx.telemetry

        monkeypatch.setattr(locus.analysis.mlx.telemetry, "memory_snapshot", dict)

    def test_health_carries_the_admission_state(self):
        admission = Admission()
        admission.running, admission.waiting = 1, 4

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"status": "ok"}'})

        sent = run_asgi(
            HealthMemoryMiddleware(inner, admission=admission),
            http_scope(method="GET", path="/health"),
            [],
        )
        import json

        payload = json.loads(
            next(m for m in sent if m["type"] == "http.response.body")["body"]
        )
        assert payload["admission"] == {"running": 1, "waiting": 4}


class TestPreprocessSerializer:
    def test_concurrent_preprocess_is_serialized(self, monkeypatch):
        """Upstream runs _cpu_preprocess on each request's own thread against
        one shared processor; the HF tokenizers Rust core forbids concurrent
        use. The boot-time serializer must reduce caller-thread concurrency
        to 1.
        """
        import time
        from threading import Thread

        from locus.analysis.mlx.serve import install_preprocess_serializer
        from mlx_vlm.server import generation

        in_flight = {"n": 0, "max": 0}

        def slow_preprocess(self, *args, **kwargs):
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
            time.sleep(0.02)
            in_flight["n"] -= 1
            return {"input_ids": [1]}

        monkeypatch.setattr(
            generation.ResponseGenerator, "_cpu_preprocess", slow_preprocess
        )
        dummy = object.__new__(generation.ResponseGenerator)

        def storm():
            threads = [
                Thread(
                    target=generation.ResponseGenerator._cpu_preprocess, args=(dummy,)
                )
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        storm()
        assert in_flight["max"] > 1, "test harness failed to produce overlap"

        in_flight["max"] = 0
        install_preprocess_serializer()
        storm()
        assert in_flight["max"] == 1


class TestPixelRelease:
    def test_pixel_arrays_leave_the_inputs_once_encoded(self, monkeypatch):
        import mlx.core as mx
        from locus.analysis.mlx import patching
        from locus.analysis.mlx.serve import install_pixel_release
        from mlx_vlm.server import generation

        seen = {}

        def encode(self, raw_inputs, images=None, apc_semantic_hash=None):
            seen["pixels_at_encode"] = "pixel_values" in raw_inputs
            return raw_inputs["input_ids"], {"inputs_embeds": mx.zeros((1, 2, 4))}

        # The stand-in encode has its own source, so the pin is lifted for it; the pin
        # itself is the drift test's subject.
        monkeypatch.setattr(patching, "expect_source", lambda *a, **k: None)
        monkeypatch.setattr(generation.ResponseGenerator, "_gpu_embed", encode)
        install_pixel_release()

        raw_inputs = {
            "input_ids": mx.array([[1, 2]]),
            "pixel_values": mx.zeros((8, 4)),
            "pixel_values_videos": mx.zeros((8, 4)),
            "image_grid_thw": mx.array([[1, 2, 2]]),
        }
        dummy = object.__new__(generation.ResponseGenerator)
        _, gen_kwargs = generation.ResponseGenerator._gpu_embed(dummy, raw_inputs, None)

        assert seen["pixels_at_encode"]
        assert set(raw_inputs) == {"input_ids", "image_grid_thw"}
        assert "inputs_embeds" in gen_kwargs

    def test_refuses_boot_on_upstream_source_drift(self, monkeypatch):
        from locus.analysis.mlx import serve
        from mlx_vlm.server import generation

        monkeypatch.setattr(serve, "EXPECTED_GPU_EMBED_SHA256", "0" * 64)
        before = generation.ResponseGenerator._gpu_embed
        with pytest.raises(SystemExit, match="upstream drift"):
            serve.install_pixel_release()
        assert generation.ResponseGenerator._gpu_embed is before


class TestGpuTracking:
    @pytest.fixture(autouse=True)
    def restore_upstream(self):
        from mlx_vlm.server import generation

        request_class = generation.QueuedGenerationRequest
        generate = generation.ResponseGenerator.generate
        yield
        generation.QueuedGenerationRequest = request_class
        generation.ResponseGenerator.generate = generate

    @staticmethod
    def _install(monkeypatch, admission, generate):
        """Track a stand-in for upstream's generate: it has its own source, so the pin is lifted for it; the pin itself is the drift test's subject."""
        from locus.analysis.mlx import patching
        from mlx_vlm.server import generation

        monkeypatch.setattr(patching, "expect_source", lambda *a, **k: None)
        monkeypatch.setattr(generation.ResponseGenerator, "generate", generate)
        serve.install_gpu_tracking(admission)
        return object.__new__(generation.ResponseGenerator)

    def test_a_request_counts_from_generate_to_its_first_terminal_message(
        self, monkeypatch
    ):
        from queue import Queue

        from mlx_vlm.server import generation

        admission = Admission()
        seen = {}

        def generate(self, prompt):
            seen["while_preprocessing"] = admission._in_gpu
            rqueue = Queue()
            seen["request"] = generation.QueuedGenerationRequest(
                rqueue=rqueue, raw_inputs={}, prompt_tokens=1, args=None
            )
            return rqueue

        worker = self._install(monkeypatch, admission, generate)
        rqueue = generation.ResponseGenerator.generate(worker, "prompt")
        assert seen["while_preprocessing"] == 1, (
            "counted from the start of generate, before any request is built"
        )
        assert admission._in_gpu == 1, "still counted once generate has queued it"

        request = seen["request"]
        request.rqueue.put(generation.GenerationContext(uid=0, prompt_tokens=1))
        request.rqueue.put("a token")
        assert admission._in_gpu == 1, "context and tokens are not the end"

        request.rqueue.put(None)
        assert admission._in_gpu == 0
        request.rqueue.put(None)
        assert admission._in_gpu == 0, "a second terminal message is not a second drop"
        assert [rqueue.get_nowait() for _ in range(4)][-2:] == [None, None], (
            "every message still reaches the caller"
        )

    def test_generate_raising_before_it_builds_a_request_counts_nothing(
        self, monkeypatch
    ):
        admission = Admission()

        def generate(self, prompt):
            raise ValueError("prompt too long")

        worker = self._install(monkeypatch, admission, generate)
        from mlx_vlm.server import generation

        with pytest.raises(ValueError, match="prompt too long"):
            generation.ResponseGenerator.generate(worker, "prompt")
        assert admission._in_gpu == 0

    def test_a_failure_the_loop_reports_drops_the_request_once(self, monkeypatch):
        """The loop's insert failure puts the exception on the rqueue, and generate re-raises it to the caller: the request leaves the count once, by its terminal message."""
        from queue import Queue

        from mlx_vlm.server import generation

        admission = Admission()

        def generate(self, prompt):
            rqueue = Queue()
            request = generation.QueuedGenerationRequest(
                rqueue=rqueue, raw_inputs={}, prompt_tokens=1, args=None
            )
            request.rqueue.put(RuntimeError("insert failed"))
            raise rqueue.get_nowait()

        worker = self._install(monkeypatch, admission, generate)
        with pytest.raises(RuntimeError, match="insert failed"):
            generation.ResponseGenerator.generate(worker, "prompt")
        assert admission._in_gpu == 0

    def test_the_loops_error_path_drops_the_request_once(self, monkeypatch):
        from queue import Queue

        from mlx_vlm.server import generation

        admission = Admission()
        built = []

        def generate(self, prompt):
            built.append(
                generation.QueuedGenerationRequest(
                    rqueue=Queue(), raw_inputs={}, prompt_tokens=1, args=None
                )
            )

        worker = self._install(monkeypatch, admission, generate)
        generation.ResponseGenerator.generate(worker, "a")
        generation.ResponseGenerator.generate(worker, "b")
        assert admission._in_gpu == 2

        first, second = built
        first.rqueue.put(RuntimeError("step failed"))
        first.rqueue.put(None)
        assert admission._in_gpu == 1, "the error path sends the exception, then None"
        second.rqueue.put(None)
        assert admission._in_gpu == 0

    def test_refuses_boot_on_upstream_source_drift(self, monkeypatch):
        from mlx_vlm.server import generation

        monkeypatch.setitem(
            serve.EXPECTED_GPU_LIFECYCLE_SHA256, "ResponseGenerator._step", "0" * 64
        )
        before = generation.QueuedGenerationRequest
        with pytest.raises(SystemExit, match="upstream drift"):
            serve.install_gpu_tracking(Admission())
        assert generation.QueuedGenerationRequest is before

    def test_the_slot_waits_out_a_request_the_gpu_still_holds(self):
        """The response ends — a caller disconnected — while the GPU loop still works on the request: the next request is admitted only when the loop drops it, signalled from the loop's own thread."""
        import threading

        admission = Admission()
        events = []

        async def inner(scope, receive, send):
            events.append((scope["name"], "enter"))
            if scope["name"] == "dead":
                admission.entered_gpu()

        middleware = AdmissionMiddleware(inner, admission)

        async def noop_send(message):
            pass

        async def storm():
            dead = asyncio.create_task(
                middleware({**CHAT, "name": "dead"}, _stays, noop_send)
            )
            await asyncio.sleep(0.01)
            nxt = asyncio.create_task(
                middleware({**CHAT, "name": "next"}, _stays, noop_send)
            )
            await asyncio.sleep(0.05)
            assert events == [("dead", "enter")], "the next request is not admitted"
            assert not dead.done(), "the dead request's slot is held"
            assert admission.state() == {"running": 1, "waiting": 1}
            gpu = threading.Thread(target=admission.left_gpu)
            gpu.start()
            gpu.join()
            await asyncio.gather(dead, nxt)

        asyncio.run(storm())
        assert events == [("dead", "enter"), ("next", "enter")]
        assert admission.state() == {"running": 0, "waiting": 0}

    def test_the_slot_frees_at_once_when_the_gpu_holds_nothing(self):
        admission = Admission()
        middleware = AdmissionMiddleware(lambda s, r, x: asyncio.sleep(0), admission)

        async def noop_send(message):
            pass

        async def run():
            await asyncio.wait_for(middleware(CHAT, _stays, noop_send), 1)

        asyncio.run(run())
        assert not middleware.slot.locked()

    def test_a_cancelled_drain_still_frees_the_slot(self):
        """The server shutting down cancels a request whose slot is waiting on the GPU: the slot is released, never orphaned."""
        admission = Admission()
        admission.entered_gpu()

        async def inner(scope, receive, send):
            pass

        middleware = AdmissionMiddleware(inner, admission)

        async def noop_send(message):
            pass

        async def run():
            task = asyncio.create_task(middleware(CHAT, _stays, noop_send))
            await asyncio.sleep(0.02)
            assert middleware.slot.locked()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not middleware.slot.locked()

        asyncio.run(run())
        assert admission.state() == {"running": 0, "waiting": 0}


class TestUpstreamSamplerTripwire:
    def test_batched_sampler_draws_from_exactly_the_claimed_fields(self):
        """The cards may claim top_k and not min_p because upstream's batched sampler is built from temperature/top_p/top_k/seed alone. When this fails, re-verify SUPPORTED_SAMPLING (cards.py) against the new sampler."""
        import inspect

        from mlx_vlm.server.generation import _PositionedTargetSampler

        params = set(
            inspect.signature(_PositionedTargetSampler.__init__).parameters
        ) - {"self"}
        assert params == {"temperature", "top_p", "top_k", "seed"}


class TestThinkingBudgetDecodePipeline:
    """A thinking budget must not cost the decode-ahead pipeline. mlx-vlm 0.6.4 through 0.6.14 materialized the just-dispatched forward on every token whenever a budget was armed — an unconditional per-token mx.eval that collapsed decode to lockstep, ~3ms/token at every context depth — and the pinned release handles the forced `\\n</think>` close lazily (mx.where + async_eval) instead. Every analysis request arms a budget, so a regression here is a silent decode slowdown; these tests pin the property on upstream's own GenerationBatch.next so it surfaces at upgrade time."""

    class FakeTokenizer:
        VOCAB: ClassVar[dict] = {"\n": [10], "</think>": [99], "<think>": [98]}

        def encode(self, text, add_special_tokens=False):
            return self.VOCAB[text]

    @staticmethod
    def make_batch(planned):
        import mlx.core as mx
        from mlx_vlm.generate.ar import GenerationBatch
        from mlx_vlm.utils import ThinkingBudgetCriteria

        batch = object.__new__(GenerationBatch)
        batch.uids = [0]
        batch._num_tokens = [0]
        batch.max_tokens = [10_000]
        batch.stop_criteria = lambda tok: False
        batch.thinking_budget_criteria = [
            ThinkingBudgetCriteria(
                tokenizer=TestThinkingBudgetDecodePipeline.FakeTokenizer(),
                thinking_budget=2,
                thinking_start_token="<think>",
                thinking_end_token="</think>",
                enable_thinking=True,
            )
        ]
        batch._next_tokens = mx.array([planned[0]], dtype=mx.int32)
        feed = iter(planned[1:] + [0])

        def step():
            tokens = batch._next_tokens.tolist()
            batch._next_tokens = mx.array([next(feed)], dtype=mx.int32)
            return tokens, None, None, None

        batch._step = step
        batch.filter = lambda keep: None
        return batch

    def test_never_syncs_and_the_forced_close_lands(self, monkeypatch):
        """One walk of the whole budget lifecycle: `<think>` opens thinking, the budget is spent, and the forced `\\n</think>` close must overwrite the next two sampled tokens — with mx.eval never called at any step, forced close included."""
        import mlx.core as mx
        from mlx_vlm.generate.ar import GenerationBatch

        evals = []
        real_eval = mx.eval

        def spying_eval(*args, **kwargs):
            evals.append(args)
            return real_eval(*args, **kwargs)

        monkeypatch.setattr(mx, "eval", spying_eval)
        # Budget 2: 98 opens thinking, 5 and 6 spend the budget, 7 exceeds it,
        # and the forced 10, 99 close must overwrite the sampled 41 and 42.
        batch = self.make_batch([98, 5, 6, 7, 41, 42, 43])
        seen = [GenerationBatch.next(batch)[0].token for _ in range(7)]
        assert seen == [98, 5, 6, 7, 10, 99, 43]
        assert evals == [], (
            "upstream GenerationBatch.next synced the decode-ahead pipeline while a "
            "thinking budget was armed — the per-token lockstep stall is back; "
            "re-validate before shipping this mlx-vlm version"
        )


class TestHeifDecode:
    def test_pil_opens_heic_after_install(self, tmp_path):
        """The server is handed bare file paths and loads them with PIL, which cannot open
        HEIC/HEIF on its own; install_heif_decode registers the codec process-wide, so a
        round-trip through a real .heic file proves the wiring."""
        from locus.analysis.mlx.serve import install_heif_decode
        from PIL import Image

        install_heif_decode()
        import pillow_heif

        heic = tmp_path / "image.heic"
        pillow_heif.from_pillow(Image.new("RGB", (32, 24), "red")).save(heic)
        with Image.open(heic) as image:
            assert image.size == (32, 24)


class TestCardDerivation:
    def test_every_serveable_slug_derives_its_card(self):
        """The analysis card owns which repo a slug serves, the context it
        holds, and the endpoint it dials; models.toml carries only serving env.
        A slug with env but no card cannot boot, and a restated repo, KV budget,
        or port here would be the drift this split exists to make impossible."""
        import tomllib
        from locus.analysis.mlx.serve import MODELS_TOML, _card, bind_from_card

        models = tomllib.loads(MODELS_TOML.read_text())
        assert models, "no serveable slug declared"
        for slug, entry in models.items():
            card = _card(slug)
            assert card is not None, (
                f"{slug} declares serving env but no card owns its facts"
            )
            assert card["model"] and card["context_tokens"] > 0
            host, port = bind_from_card(card)
            assert host and port > 0
            assert "hf_repo" not in entry, f"{slug} restates the repo the card owns"
            assert "MAX_KV_SIZE" not in entry.get("env", {}), (
                f"{slug} restates the KV budget the card owns"
            )
            assert "LOCUS_MLX_ADMISSION_CAP" not in entry.get("env", {}), (
                f"{slug} declares a request count as admission; admission is the prefill slot and the measured KV budget"
            )

    def test_bind_from_card_parses_host_and_port(self):
        from locus.analysis.mlx.serve import bind_from_card

        assert bind_from_card({"base_url": "http://127.0.0.1:9999/v1"}) == (
            "127.0.0.1",
            9999,
        )

    def test_bind_from_card_refuses_missing_or_portless_url(self):
        from locus.analysis.mlx.serve import bind_from_card

        with pytest.raises(ValueError, match="base_url"):
            bind_from_card({})
        with pytest.raises(ValueError, match="host and port"):
            bind_from_card({"base_url": "http://127.0.0.1/v1"})

    def test_sole_bind_matches_every_bootable_card(self):
        import tomllib
        from locus.analysis.mlx.serve import MODELS_TOML, bind_for, sole_bind

        sole = sole_bind()
        for slug in tomllib.loads(MODELS_TOML.read_text()):
            assert bind_for(slug) == sole


def test_the_boot_refuses_a_count_setting_only_its_shell_holds(monkeypatch):
    """A count-affecting variable the server would inherit but models.toml does not declare refuses the boot before anything is loaded: pricing counts with the declared serving env alone, so the server would count differently from every price. A declared one is the declaration's, and boots."""
    from locus.analysis.mlx import serve

    slug = "qwen3.6-35b-a3b-6bit"
    monkeypatch.setenv("MLX_VLM_ENABLE_THINKING", "0")
    assert serve.undeclared_count_env(slug) == [], "declared: the declaration wins"

    monkeypatch.setenv("MLX_VLM_MAX_PIXELS", "65536")
    monkeypatch.setattr(serve, "install_log_handler", lambda: None)

    def loaded(*args, **kwargs):
        raise AssertionError("the boot went past the refusal")

    monkeypatch.setattr(serve, "_wired_cap_bytes", loaded)
    monkeypatch.setattr(serve.sys, "argv", ["locus-mlx-serve", slug])
    with pytest.raises(SystemExit, match="MLX_VLM_MAX_PIXELS set in the environment"):
        serve.main()
