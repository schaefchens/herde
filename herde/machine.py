"""What kind of computer this is, for the caps: projects show it to their admins, nothing decides by it."""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# The GPUs the kernel knows, and AMD's names for its own by PCI device and revision: libdrm's list, on the host
# and in the images (the base image's knows no GPU after 2022).
DRM = Path("/sys/class/drm")
AMDGPU_IDS = Path("/usr/share/libdrm/amdgpu.ids")


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
    return gpus.splitlines()[0].strip() if gpus else amd_gpu()


def amd_gpu(drm: Path = DRM, ids: tuple[Path, ...] | None = None) -> str:
    """The AMD GPU with the most memory of its own — a Radeon card before the APU beside it — by AMD's name for it."""
    cards = []
    for card in sorted(drm.glob("card*")):
        pci = card / "device"
        # card1-DP-1 and the like are a card's connectors.
        if re.fullmatch(r"card\d+", card.name) and _read(pci / "vendor") == "0x1002":
            vram = _read(pci / "mem_info_vram_total")
            cards.append((int(vram) if vram.isdigit() else 0, _read(pci / "device")[2:].upper(), _read(pci / "revision")[2:].upper()))
    if not cards:
        return ""
    _, device, revision = max(cards, key=lambda c: c[0])
    for path in _amdgpu_ids() if ids is None else ids:
        for line in _read(path).splitlines():
            # "744C,\tC8,\tAMD Radeon RX 7900 XTX": the same device in another revision can be another product.
            parts = [p.strip() for p in line.split(",", 2)]
            if len(parts) == 3 and parts[0].upper() == device and parts[1].upper() == revision:
                return parts[2]
    return f"AMD GPU {device}"


def _amdgpu_ids() -> tuple[Path, ...]:
    """The newer list first: the one ROCm's PyTorch brings along (the rocm image), found without importing torch."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return (AMDGPU_IDS,)
    return (Path(next(iter(spec.submodule_search_locations))) / "share" / "libdrm" / "amdgpu.ids", AMDGPU_IDS)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _memory_gb() -> int:
    try:
        if sys.platform == "darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except (ValueError, OSError, AttributeError):
        return 0
