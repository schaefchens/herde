"""A small page on this computer (127.0.0.1 only): what the worker does, and a Pause button.

For lenders who do not use a terminal: a game to play, a film to cut, and the
GPU should be theirs for a while. Standard library only, plain HTML, no script.

Any web page the lender visits could post a form to 127.0.0.1: every change
needs the page's own token (another site cannot read it), and requests for a
host other than this computer are refused (a rebound DNS name would otherwise
reach the page).
"""

from __future__ import annotations

import html
import json
import logging
import secrets
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .state import State

log = logging.getLogger(__name__)

LOCAL_NAMES = ("127.0.0.1", "localhost", "[::1]", "::1")
MAX_FORM = 4096


class StatusPage:
    def __init__(self, host: str, port: int, state: State, status: Callable[[], dict], wake: Callable[[], None]):
        self.state = state
        self.status = status
        self.wake = wake
        self.token = secrets.token_urlsafe(24)
        page = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "herde"
            sys_version = ""

            def log_message(self, fmt: str, *args) -> None:
                log.debug("page: " + fmt, *args)

            def do_GET(self) -> None:
                if not self._local():
                    return
                if self.path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", page.render().encode())
                elif self.path == "/status.json":
                    self._send(200, "application/json", json.dumps(page.status()).encode())
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")

            def do_POST(self) -> None:
                if not self._local():
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_FORM:
                    self._send(413, "text/plain; charset=utf-8", b"too large")
                    return
                form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
                if not secrets.compare_digest((form.get("token") or [""])[0], page.token):
                    self._send(403, "text/plain; charset=utf-8", b"reload the page and try again")
                    return
                if self.path == "/pause":
                    hours = (form.get("hours") or ["0"])[0]
                    page.state.pause(time.time() + float(hours) * 3600 if hours in ("1", "3", "8") else None)
                elif self.path == "/resume":
                    page.state.resume()
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")
                    return
                page.wake()
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()

            def _local(self) -> bool:
                host = (self.headers.get("Host") or "").strip().lower()
                name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
                if name not in LOCAL_NAMES:
                    self._send(403, "text/plain; charset=utf-8", b"only from this computer")
                    return False
                return True

            def _send(self, code: int, kind: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, name="page", daemon=True)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def render(self) -> str:
        s = self.status() or {}
        e = html.escape
        paused_file, until = self.state.paused()
        if paused_file:
            state = "Paused " + (f"until {time.strftime('%H:%M', time.localtime(until))}" if until else "until you resume")
            buttons = self._form("/resume", "Resume")
        elif s.get("paused"):
            state = "Outside its hours" + (f" (back in {int(s['resume_in']) // 60} min)" if s.get("resume_in") else "")
            buttons = self._pause_forms()
        else:
            state = "Working"
            buttons = self._pause_forms()
        engines = "".join(
            f"<li>{e(str(x.get('kind')))}: {'ready' if x.get('ready') else ('loading…' if x.get('loading') else 'not ready')}"
            + (f" — {e(x.get('backend', ''))}, {x.get('voices', 0)} voices" if x.get("kind") == "tts" else "")
            + (" — " + ", ".join(e(m["name"]) + (" (in memory)" if m.get("loaded") else "") for m in x.get("models", []) if m.get("available")) if x.get("kind") == "text" else "")
            + "</li>"
            for x in s.get("engines", [])
        ) or "<li>none</li>"
        projects = "".join(
            f"<tr><td>{e(p['name'])}</td><td>{'connected' if p.get('ok') else e(p.get('last_error') or 'connecting…')}</td>"
            f"<td>{len(p.get('running', []))}</td><td>{p.get('done', 0)}</td><td>{p.get('failed', 0)}</td></tr>"
            for p in s.get("projects", [])
        )
        return (
            "<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<meta http-equiv=refresh content=5><title>herde</title>"
            "<style>body{font:15px system-ui,sans-serif;margin:24px;max-width:720px;color:#222;background:#fafafa}"
            "h1{font-size:20px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:6px;border-bottom:1px solid #ddd}"
            "form{display:inline-block;margin:4px 8px 4px 0}button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #888;background:#fff}"
            ".state{font-size:18px;margin:12px 0}@media (prefers-color-scheme:dark){body{background:#161616;color:#eee}button{background:#222;color:#eee}td,th{border-color:#333}}</style>"
            f"<h1>herde on {e(str(s.get('name', '')))}</h1><p class=state>{e(state)}</p><p>{buttons}</p>"
            "<p>Pausing lets running work finish, then takes nothing new until the time is up.</p>"
            f"<h2>Engines</h2><ul>{engines}</ul>"
            f"<h2>Projects</h2><table><tr><th>Project</th><th>State</th><th>Running</th><th>Done</th><th>Failed</th></tr>{projects}</table>"
            f"<p><small>herde {e(str(s.get('version', '')))}</small></p></html>"
        )

    def _pause_forms(self) -> str:
        return "".join(self._form("/pause", label, hours=h) for h, label in (("1", "Pause 1 hour"), ("3", "Pause 3 hours"), ("0", "Pause until I resume")))

    def _form(self, action: str, label: str, **fields: str) -> str:
        hidden = "".join(f"<input type=hidden name={k} value={html.escape(v)}>" for k, v in fields.items())
        return f"<form method=post action={action}><input type=hidden name=token value={html.escape(self.token)}>{hidden}<button>{html.escape(label)}</button></form>"
