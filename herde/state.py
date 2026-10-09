"""What the worker keeps between runs, next to its config: a pause, measured speeds, its status.

Each is a small JSON file of its own, written whole (a temp file renamed over
it), so `herde pause` from another process — or from `docker compose exec` —
never meets a half-written file, and the worker never loses a pause to its own
speed updates.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_DURATION = re.compile(r"(\d+)\s*(h|m|min|s)?")
_UNTIL = re.compile(r"(?:until\s+)?(\d{1,2}):(\d{2})")


class PauseError(ValueError):
    pass


def parse_pause(spec: str, now: datetime) -> datetime | None:
    """`2h`, `30m`, `until 18:00` (today, else tomorrow) or nothing (until resumed: None)."""
    spec = " ".join(spec.lower().split())
    if spec in ("", "forever", "until resumed"):
        return None
    m = _UNTIL.fullmatch(spec)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            raise PauseError(f"{spec!r}: no such time")
        at = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        return at if at > now else at + timedelta(days=1)
    m = _DURATION.fullmatch(spec)
    if m:
        n, unit = int(m.group(1)), m.group(2) or "m"
        seconds = n * {"h": 3600, "m": 60, "min": 60, "s": 1}[unit]
        if not 0 < seconds <= 30 * 86400:
            raise PauseError(f"{spec!r}: between a second and 30 days")
        return now + timedelta(seconds=seconds)
    raise PauseError(f"{spec!r}: say 2h, 30m, until 18:00, or nothing (until resumed)")


class State:
    def __init__(self, folder: Path):
        self.folder = folder
        self.pause_path = folder / "pause.json"
        self.speeds_path = folder / "speeds.json"
        self.status_path = folder / "status.json"

    # --- pause (written by `herde pause`, the local page; read by the running worker) -----------

    def pause(self, until: float | None) -> None:
        """Paused until a Unix time, or until resumed (None)."""
        self._write(self.pause_path, {"paused": True, "until": until, "since": time.time()})

    def resume(self) -> None:
        self._write(self.pause_path, {"paused": False})

    def paused(self, now: float | None = None) -> tuple[bool, float | None]:
        """Whether it is paused now, and until when (None: until resumed)."""
        data = self._read_pause()
        if not data or data.get("paused") is not True:
            return False, None
        until = data.get("until")
        if until is None:
            return True, None
        if not isinstance(until, (int, float)) or isinstance(until, bool):
            return False, None
        return (until > (time.time() if now is None else now)), float(until)

    def _read_pause(self) -> dict | None:
        # Read every time: a tiny file, and a cached copy could miss a pause written within the same instant.
        return self._read(self.pause_path)

    # --- measured speeds and the status page's data ----------------------------------------------

    def speeds(self) -> dict[str, Any]:
        return self._read(self.speeds_path) or {}

    def save_speeds(self, speeds: dict[str, Any]) -> None:
        self._write(self.speeds_path, speeds)

    def status(self) -> dict | None:
        return self._read(self.status_path)

    def save_status(self, status: dict) -> None:
        self._write(self.status_path, status)

    # --- files ---------------------------------------------------------------------------------------

    def _read(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _write(self, path: Path, data: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.folder, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
