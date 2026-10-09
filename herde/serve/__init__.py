"""`herde serve`: a little project of its own, to try herde on one computer and to submit jobs to your workers.

It is the project side of the protocol (project.py: a queue in SQLite) with a
page and a small JSON API (web.py). Workers join it with an invite like any
project; the page shows them, takes speech and text jobs, and plays or shows
what came back.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from pathlib import Path

from .project import Project
from .web import Server

log = logging.getLogger("herde")

DEFAULT_PORT = 8740


def default_data() -> Path:
    return Path(os.environ.get("HERDE_SERVE_DATA") or Path.home() / ".local" / "share" / "herde-serve").expanduser()


def serve_command(host: str, port: int, data: Path, name: str, public_url: str) -> int:
    """Run until stopped (Ctrl-C, or SIGTERM in Docker); the page's address, with its token, is printed once."""
    project = Project(data, name)
    try:
        server = Server(project, host, port, public_url)
    except OSError as e:
        log.error("Cannot listen on %s:%d: %s", host, port, e)
        return 1
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    server.start()
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    page = f"http://{shown}:{server.port}"
    print(f"herde serve: open {page}/#token={project.token}", flush=True)
    print(f"Workers join at {public_url.rstrip('/') or page}/api/worker/v2 (invite them on the page). Data in {data}. Ctrl-C stops.", flush=True)
    try:
        while not stop.wait(5):
            project.maintain()
    finally:
        server.stop()
    return 0


__all__ = ["DEFAULT_PORT", "Project", "Server", "default_data", "serve_command"]
