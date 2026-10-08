"""Voices of our own: a recording per language that Qwen3-TTS's Base model
clones in every take (station decision 2026-10-08: Faith's deep radio voice,
made with Qwen's VoiceDesign and kept as one take — VoiceDesign alone made a
slightly different woman in each take, the clone stays one).

A voice is two files per language in the voices folder ([clone] voices):
<Name>.<lang>.wav, the recording, and <Name>.<lang>.txt, exactly what it says
(the model reads the recording together with its words). The worker offers
the voice by its name next to Qwen's presets, and the station picks it like
one. The clone takes no direction: a moment's delivery does not reach it —
the words, and how the recording sounds, carry the feeling.

Standard library only: the tests run without the model's packages.
"""

from __future__ import annotations

import logging
import re
import wave
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# What the station takes as a worker voice's id (Host\Workers::poll).
NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")
# Enough of a voice to carry it, and short: every take reads the whole recording first.
MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
TEXT_MAX = 1000


@dataclass(frozen=True)
class Reference:
    path: Path
    seconds: float
    text: str = field(repr=False)


@dataclass(frozen=True)
class OwnVoice:
    name: str
    refs: dict[str, Reference]

    @property
    def languages(self) -> list[str]:
        return sorted(self.refs)

    def reference(self, lang: str) -> Reference:
        """The recording in the take's language, else another: Qwen carries a voice across languages."""
        return self.refs.get(lang.strip().lower()) or self.refs[self.languages[0]]


def discover(folder: Path, languages: Iterable[str], presets: Iterable[str]) -> dict[str, OwnVoice]:
    """The voices in `folder`, by lower-case name. A file that does not fit is left out with a warning:
    a voice that cannot be spoken must not be offered to a station."""
    if not folder.is_dir():
        return {}
    langs = {lang.lower() for lang in languages}
    taken = {p.lower() for p in presets}
    refs: dict[str, dict[str, Reference]] = {}
    names: dict[str, str] = {}
    for wav in sorted(folder.glob("*.wav")):
        name, _, lang = wav.name[: -len(".wav")].rpartition(".")
        lang = lang.lower()
        if not NAME.fullmatch(name) or lang not in langs:
            log.warning("voices: %s is not named <Name>.<language>.wav; left out", wav.name)
            continue
        # A preset's name would be two voices behind one id at the station.
        if name.lower() in taken:
            log.warning("voices: %s is the name of one of Qwen's presets; left out", wav.name)
            continue
        if names.setdefault(name.lower(), name) != name:
            log.warning("voices: %s differs from %s only in case; left out", wav.name, names[name.lower()])
            continue
        text = _words(wav.with_suffix(".txt"))
        seconds = _seconds(wav)
        if text is None or seconds is None:
            continue
        refs.setdefault(name.lower(), {})[lang] = Reference(path=wav, seconds=seconds, text=text)
    return {key: OwnVoice(name=names[key], refs=by_lang) for key, by_lang in refs.items()}


def _words(path: Path) -> str | None:
    if not path.is_file():
        log.warning("voices: %s is missing (what the recording says, word for word); %s left out", path.name, path.with_suffix(".wav").name)
        return None
    text = " ".join(path.read_text(encoding="utf-8").split())
    if not text or len(text) > TEXT_MAX:
        log.warning("voices: %s must hold the recording's words (1 to %d characters); left out", path.name, TEXT_MAX)
        return None
    return text


def _seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            seconds = w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        log.warning("voices: %s is no PCM WAV file; left out", path.name)
        return None
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        log.warning("voices: %s runs %.1f s (%d to %d s fit); left out", path.name, seconds, MIN_SECONDS, MAX_SECONDS)
        return None
    return seconds
