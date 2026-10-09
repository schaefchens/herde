"""What kind of computer this is, for the caps: projects show it to their admins, nothing decides by it."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def platform_info() -> dict:
    info: dict = {"os": {"darwin": "darwin", "linux": "linux", "win32": "windows"}.get(sys.platform, sys.platform)[:40], "arch": (platform.machine() or "unknown")[:40]}
    accelerator = _accelerator()
    if accelerator:
        info["accelerator"] = accelerator[:120]
    memory = _memory_gb()
    if memory:
        info["memory_gb"] = memory
    return info


def _run(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _accelerator() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    gpus = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return gpus.splitlines()[0].strip() if gpus else ""


def _memory_gb() -> int:
    try:
        if sys.platform == "darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except (ValueError, OSError, AttributeError):
        return 0
