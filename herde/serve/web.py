"""The HTTP side of `herde serve`: workers' calls (protocol 2), the page's API, and the page itself.

Workers prove themselves by their key (X-Worker-Key). The page and the API by the
page's token in X-Herde-Token: any web page in the same browser could send a
request to this address, and none can read the token. On this computer only
(the default), a request naming another host is refused as well: a rebound DNS
name would otherwise reach the page.

Standard library only.
"""

from __future__ import annotations

import email.parser
import email.policy
import json
import logging
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from .project import Answer, Project

log = logging.getLogger(__name__)

JSON_MAX = 1024 * 1024
FORM_MAX = 30 * 1024 * 1024
LOCAL_NAMES = ("127.0.0.1", "localhost", "[::1]")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src blob:; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
WORKER_PATH = re.compile(r"/api/worker/v2/(join|poll|tasks/(\d{1,12})/(claim|result|fail))")
ADMIN_PATH = re.compile(r"/api/(state|invites|jobs|invites/(\d{1,12})|jobs/(\d{1,12})|jobs/(\d{1,12})/audio|workers/(\d{1,12}))")


class Server:
    def __init__(self, project: Project, host: str = "127.0.0.1", port: int = 8740, public_url: str = ""):
        self.project = project
        self.local_only = host in ("127.0.0.1", "localhost", "::1")
        self.public_url = public_url.rstrip("/")
        server = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "herde"
            sys_version = ""

            def log_message(self, fmt: str, *args) -> None:
                log.debug("serve: " + fmt, *args)

            def do_GET(self) -> None:
                server.handle(self, "GET")

            def do_POST(self) -> None:
                server.handle(self, "POST")

            def do_PATCH(self) -> None:
                server.handle(self, "PATCH")

            def do_DELETE(self) -> None:
                server.handle(self, "DELETE")

        self.http = ThreadingHTTPServer((host, port), Handler)
        self.http.daemon_threads = True
        self.thread = threading.Thread(target=self.http.serve_forever, name="serve", daemon=True)

    @property
    def port(self) -> int:
        return int(self.http.server_address[1])

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.http.shutdown()
        self.http.server_close()

    # --- routing -------------------------------------------------------------------------------------

    def handle(self, req: BaseHTTPRequestHandler, method: str) -> None:
        path = req.path.split("?", 1)[0]
        try:
            if m := WORKER_PATH.fullmatch(path):
                if method != "POST":
                    raise Answer(405, "method_not_allowed")
                return self._send_json(req, 200, self._worker(req, m))
            if m := ADMIN_PATH.fullmatch(path):
                self._local(req)
                if not secrets.compare_digest(req.headers.get("X-Herde-Token", ""), self.project.token):
                    raise Answer(401, "bad_token")
                return self._admin(req, method, m)
            if method == "GET" and path in STATIC:
                self._local(req)
                name, kind = STATIC[path]
                return self._send(req, 200, kind, resources.files("herde.serve").joinpath("static", name).read_bytes(), page=True)
            raise Answer(404, "not_found")
        except Answer as a:
            self._send_json(req, a.status, a.body)
        except Exception:
            log.exception("serve: %s %s failed", method, path)
            self._send_json(req, 500, {"error": "server_error"})

    def _worker(self, req: BaseHTTPRequestHandler, m: re.Match) -> dict:
        p = self.project
        call = m.group(1)
        if call == "join":
            return p.join(self._json(req), req.client_address[0])
        w = p.authenticate(req.headers.get("X-Worker-Key", "").strip())
        if call == "poll":
            return p.poll(w, self._json(req))
        job_id, action = int(m.group(2)), m.group(3)
        if action == "claim":
            return p.claim(w, job_id, self._json(req))
        if action == "fail":
            return p.fail(w, job_id, self._json(req))
        if (req.headers.get("Content-Type") or "").startswith("multipart/"):
            fields = self._form(req)
            ms = fields.get("ms", b"0")
            return p.result(w, job_id, fields.get("lease", b"").decode("utf-8", "replace"), fields.get("audio"),
                            int(ms) if ms.isdigit() else 0, None)
        body = self._json(req)
        return p.result(w, job_id, str(body.get("lease", "")) if isinstance(body, dict) else "", None, 0,
                        body.get("output") if isinstance(body, dict) else None)

    def _admin(self, req: BaseHTTPRequestHandler, method: str, m: re.Match) -> None:
        p = self.project
        name = m.group(1)
        if name == "state" and method == "GET":
            return self._send_json(req, 200, p.state() | {"address": self._address(req)})
        if name == "invites" and method == "POST":
            body = self._json(req)
            invite = p.invite(str(body.get("name", "")) if isinstance(body, dict) else "")
            return self._send_json(req, 200, invite | {"join": f"herde join {self._address(req)} {invite['code']}"})
        if m.group(2) and method == "DELETE":
            p.withdraw(int(m.group(2)))
            return self._send_json(req, 200, {"ok": True})
        if name == "jobs" and method == "POST":
            return self._send_json(req, 200, {"id": p.submit(self._json(req))})
        if m.group(3):
            if method == "GET":
                return self._send_json(req, 200, p.job(int(m.group(3))))
            if method == "DELETE":
                p.cancel(int(m.group(3)))
                return self._send_json(req, 200, {"ok": True})
        if m.group(4) and method == "GET":
            return self._send(req, 200, "audio/mpeg", p.audio(int(m.group(4))).read_bytes())
        if m.group(5):
            if method == "PATCH":
                body = self._json(req)
                p.set_active(int(m.group(5)), bool(body.get("active")) if isinstance(body, dict) else False)
                return self._send_json(req, 200, {"ok": True})
            if method == "DELETE":
                p.remove_worker(int(m.group(5)))
                return self._send_json(req, 200, {"ok": True})
        raise Answer(405, "method_not_allowed")

    # --- requests and answers ----------------------------------------------------------------------

    def _address(self, req: BaseHTTPRequestHandler) -> str:
        """Where workers join: the address given at the start, else the one this page was opened at."""
        base = self.public_url or f"http://{req.headers.get('Host') or '127.0.0.1'}"
        return base + "/api/worker/v2"

    def _local(self, req: BaseHTTPRequestHandler) -> None:
        if not self.local_only:
            return
        host = (req.headers.get("Host") or "").strip().lower()
        name = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        if name not in LOCAL_NAMES:
            raise Answer(403, "only_from_this_computer")

    def _body(self, req: BaseHTTPRequestHandler, limit: int) -> bytes:
        length = int(req.headers.get("Content-Length") or 0)
        if length > limit:
            raise Answer(413, "too_large")
        return req.rfile.read(length) if length > 0 else b""

    def _json(self, req: BaseHTTPRequestHandler) -> Any:
        raw = self._body(req, JSON_MAX)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            raise Answer(400, "bad_json") from None

    def _form(self, req: BaseHTTPRequestHandler) -> dict[str, bytes]:
        """multipart/form-data, read with the standard library's e-mail parser (Python 3.13 has no cgi module)."""
        raw = self._body(req, FORM_MAX)
        head = f"Content-Type: {req.headers.get('Content-Type')}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(head + raw)
        fields: dict[str, bytes] = {}
        for part in message.iter_parts() if message.is_multipart() else []:
            name = part.get_param("name", header="content-disposition")
            if isinstance(name, str):
                fields[name] = part.get_payload(decode=True) or b""
        return fields

    def _send_json(self, req: BaseHTTPRequestHandler, status: int, data: Any) -> None:
        self._send(req, status, "application/json", json.dumps(data, ensure_ascii=False).encode())

    def _send(self, req: BaseHTTPRequestHandler, status: int, kind: str, body: bytes, page: bool = False) -> None:
        req.send_response(status)
        req.send_header("Content-Type", kind)
        req.send_header("Content-Length", str(len(body)))
        req.send_header("Cache-Control", "no-store")
        req.send_header("X-Content-Type-Options", "nosniff")
        if page:
            req.send_header("Content-Security-Policy", CSP)
            req.send_header("Referrer-Policy", "no-referrer")
        req.end_headers()
        req.wfile.write(body)
