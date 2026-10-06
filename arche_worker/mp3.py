"""16-bit PCM to MP3 through ffmpeg: the station plays MP3 and has no encoder of its own."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

# A launchd job's PATH lacks Homebrew: look there too.
HOMEBREW_FFMPEG = Path("/opt/homebrew/bin/ffmpeg")


class EncodeError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    if HOMEBREW_FFMPEG.is_file():
        return str(HOMEBREW_FFMPEG)
    raise EncodeError("ffmpeg not found; install it with: brew install ffmpeg")


def command(ffmpeg: str, sample_rate: int, bitrate: str) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", "pipe:1",
    ]


def encode(
    pcm16: bytes,
    sample_rate: int,
    bitrate: str = "96k",
    ffmpeg: str | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[bytes, int]:
    """The MP3 bytes and the clip's length in ms (from the samples, not the encoder's padding)."""
    if not pcm16 or sample_rate <= 0:
        raise EncodeError("no audio to encode")
    proc = run(command(ffmpeg or find_ffmpeg(), sample_rate, bitrate), input=pcm16, capture_output=True, timeout=120, check=False)
    if proc.returncode != 0 or not proc.stdout:
        detail = (proc.stderr or b"").decode(errors="replace").strip()[:200]
        raise EncodeError(f"ffmpeg failed ({proc.returncode}): {detail}")
    return proc.stdout, len(pcm16) // 2 * 1000 // sample_rate
