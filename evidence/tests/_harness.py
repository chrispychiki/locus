"""Browser-integration scaffolding shared by the recorder tests: a throwaway
HTTP server over a directory and a headless Chromium page. Each test owns its
own site, sink, drive, and assertions — this is only the plumbing they all
stand up identically.
"""

import http.server
import threading
from contextlib import contextmanager
from functools import partial


def serve_dir(directory, routes=None, on_put=None):
    """Serve `directory` on a random loopback port, returning (server, origin).

    `routes` maps a request path to (body_bytes, content_type) served verbatim
    ahead of the static files; `on_put`, when given, is called with (path, body)
    on each PUT/POST — the local-sink capture path. The caller owns
    `server.shutdown()`.
    """
    routes = routes or {}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            route = routes.get(self.path)
            if route is None:
                return super().do_GET()
            body, content_type = route
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        if on_put is not None:

            def do_PUT(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                on_put(self.path, body)
                self.send_response(200)
                self.end_headers()

            do_POST = do_PUT

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(directory))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@contextmanager
def chromium_page(viewport=None):
    """A headless Chromium page in a fresh context, closed on exit."""
    viewport = viewport or {"width": 1280, "height": 800}

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            yield browser.new_context(viewport=viewport).new_page()
        finally:
            browser.close()
