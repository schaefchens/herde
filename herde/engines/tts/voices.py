"""Voices of our own: a recording per language that Qwen3-TTS's Base model
clones in every take (ARCHE's decision 2026-10-08: Faith's deep radio voice,
made with Qwen's VoiceDesign and kept as one take — VoiceDesign alone made a
slightly different woman in each take, the clone stays one).

A voice is two files per language in the voices folder ([tts] voices):
<Name>.<lang>.wav, the recording, and <Name>.<lang>.txt, exactly what it says
(the model reads the recording together with its words). The worker offers
the voice by its name next to Qwen's presets, and a project picks it like
one. The clone takes no direction: a moment's delivery does not reach it —
the words, and how the recording sounds, carry the feeling.

A project's profile brings voices of its own the same way (profile.py): its
recordings are checked here exactly like the owner's.

Standard library only: the tests run without the model's packages.
"""

from __future__ import annotations

import hashlib
import logging
import re
import wave
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# A voice id as the protocol allows it (schema.json VoiceId).
NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")
# Enough of a voice to carry it, and short: every take reads the whole recording first.
MIN_SECONDS = 3.0
MAX_SECONDS = 30.0
TEXT_MAX = 1000


class BadRecording(ValueError):
    """A recording that cannot be cloned from; the message says why."""


@dataclass(frozen=True)
class Reference:
    path: Path
    seconds: float
    text: str = field(repr=False)
    # What a project's profile names the recording by, and what the caps' setup reports.
    sha256: str = ""


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
    a voice that cannot be spoken must not be offered to a project."""
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
        # A preset's name would be two voices behind one id at a project.
        if name.lower() in taken:
            log.warning("voices: %s is the name of one of Qwen's presets; left out", wav.name)
            continue
        if names.setdefault(name.lower(), name) != name:
            log.warning("voices: %s differs from %s only in case; left out", wav.name, names[name.lower()])
            continue
        words = wav.with_suffix(".txt")
        if not words.is_file():
            log.warning("voices: %s is missing (what the recording says, word for word); %s left out", words.name, wav.name)
            continue
        text = " ".join(words.read_text(encoding="utf-8").split())
        if not text or len(text) > TEXT_MAX:
            log.warning("voices: %s must hold the recording's words (1 to %d characters); left out", words.name, TEXT_MAX)
            continue
        try:
            ref = reference(wav, text)
        except BadRecording as e:
            log.warning("voices: %s %s; left out", wav.name, e)
            continue
        refs.setdefault(name.lower(), {})[lang] = ref
    return {key: OwnVoice(name=names[key], refs=by_lang) for key, by_lang in refs.items()}


def reference(path: Path, text: str, sha256: str | None = None) -> Reference:
    """A recording fit to clone from, with its words — or BadRecording. `sha256`, when known, is checked too."""
    words = " ".join(text.split())
    if not words or len(words) > TEXT_MAX:
        raise BadRecording(f"needs the recording's words (1 to {TEXT_MAX} characters)")
    try:
        data = path.read_bytes()
    except OSError:
        raise BadRecording("cannot be read") from None
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise BadRecording("is not the file its SHA-256 names")
    try:
        with wave.open(str(path), "rb") as w:
            if w.getcomptype() != "NONE" or w.getsampwidth() not in (1, 2, 3, 4):
                raise BadRecording("is no PCM WAV file")
            seconds = w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError, OSError, ZeroDivisionError):
        raise BadRecording("is no PCM WAV file") from None
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise BadRecording(f"runs {seconds:.1f} s ({MIN_SECONDS:.0f} to {MAX_SECONDS:.0f} s fit)")
    return Reference(path=path, seconds=seconds, text=words, sha256=digest)
