"""Qwen3-TTS for any backend: CustomVoice for Qwen's presets, and Base for voices of
our own, cloned from a recording (engines.tts.voices).

A text is spoken in pieces (`chunk_chars`), each a take of its own whose
delivery starts afresh; between two pieces a project's cancel or the
worker's stop ends the task. The backends (mlx on a Mac, torch on NVIDIA or AMD,
a silent stub for tests) only speak one piece: everything a project can hear —
where a text is cut, the pause between pieces, the length check's inputs — is
decided here, the same on every computer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .voices import OwnVoice, Reference

# The CustomVoice presets. None is a native German speaker; ARCHE's owner chose Sohee for German.
VOICES = ("Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee")
# Projects send language codes; the models want the names.
LANGUAGES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
}
# What projects ask for: the same name on every worker, whichever checkpoint speaks a voice.
MODEL_NAME = "qwen3-tts-1.7b"
# The version of this speech code: everything besides the model that a listener can hear — where a
# text is cut (split_text), the pause between pieces, how sampling is set, how long a take may run.
# A project's profile names the one it expects (PROTOCOL.md, Profiles): raise it with any such change,
# or computers on two herde versions speak alike in name only.
CODE = 1
# Each piece of a text is a take of its own, and its delivery starts afresh:
# ARCHE's owner heard the host "drift" across pieces. The size is the config's
# `chunk_chars` ([tts]); `herde lab` compares sizes by ear.
CHUNK_CHARS = 300
CHUNK_CHARS_MAX = 2000
PAUSE_MS = 150
# Measured on an M1 Max: German speech runs about 13 characters a second, and
# the model makes 12.5 audio tokens a second.
CHARS_PER_SECOND = 13
TOKENS_PER_SECOND = 12.5
REPETITION_PENALTY = 1.05
# A full stop after these is no sentence end: "z. B. Gott" must not be cut after "z.".
ABBREVIATIONS = {
    "bzw.", "ca.", "dr.", "evtl.", "ggf.", "usw.", "etc.", "vgl.", "nr.", "st.", "hl.", "prof.",
    "inkl.", "bspw.", "mr.", "mrs.", "ms.", "jr.", "sr.", "vs.", "e.g.", "i.e.", "z.b.", "d.h.", "u.a.",
}


class InvalidTask(ValueError):
    """A task this worker can never speak (an unknown voice or language, no text)."""


class TooLarge(ValueError):
    """Over this worker's limits: another worker may take it."""


class Cancelled(Exception):
    """Stopped between two takes: the project cancelled it, or the worker is stopping."""


@dataclass(frozen=True)
class Speech:
    pcm16: bytes
    sample_rate: int

    @property
    def ms(self) -> int:
        return len(self.pcm16) // 2 * 1000 // self.sample_rate


@dataclass(frozen=True)
class Sampling:
    temperature: float
    max_tokens: int
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = REPETITION_PENALTY


@dataclass(frozen=True)
class Settings:
    """What shapes every take besides its words, voice and temperature: the owner's config, or a project's profile."""

    chunk_chars: int = CHUNK_CHARS
    pause_ms: int = PAUSE_MS
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = REPETITION_PENALTY


class Backend(Protocol):
    """Speaks one piece of text: 16-bit mono PCM and its sample rate."""

    name: str

    @property
    def ready(self) -> bool: ...

    @property
    def dtype(self) -> str:
        """The number format the weights run in once loaded: bf16, fp16 or fp32 ("" before, and for the stub)."""
        ...

    def pins(self) -> tuple[str, str]:
        """The checkpoints it loads, or would load for an own voice, as repo@revision: what a profile is compared with."""
        ...

    def checkpoints(self) -> tuple[str, str]:
        """The checkpoints loaded, as repo@revision: CustomVoice, and Base once a recording needed it ("" until then)."""
        ...

    def load(self, refs: list[Reference]) -> None: ...

    def fetch_clone(self) -> None:
        """The Base model's files, downloaded without the GPU: a profile's first own voice must not stall the takes."""
        ...

    def add(self, refs: list[Reference]) -> None:
        """Recordings learnt while the engine runs (the Base model loaded first, if this is the first)."""
        ...

    def seed(self, seed: int) -> None: ...

    def preset(self, text: str, speaker: str, language: str, instruct: str, sampling: Sampling) -> tuple[bytes, int]: ...

    def clone(self, text: str, ref: Reference, language: str, sampling: Sampling) -> tuple[bytes, int]: ...


def at(repo: str, revision: str) -> str:
    """A checkpoint as setups and profiles name it: repo@revision (the repo alone when not pinned)."""
    return f"{repo}@{revision}" if repo and revision else repo


def language_name(code: str) -> str:
    name = LANGUAGES.get(code.strip().lower())
    if name is None:
        raise InvalidTask(f"unknown language {code[:8]!r}")
    return name


def voice_name(voice: str) -> str:
    for v in VOICES:
        if v.lower() == voice.strip().lower():
            return v
    raise InvalidTask(f"unknown voice {voice[:40]!r}")


def max_tokens(chunk: str) -> int:
    """Room for the chunk at a slow reading pace, and no more: a take that runs away ends here."""
    expected = len(chunk) / CHARS_PER_SECOND * TOKENS_PER_SECOND
    return max(64, min(8192, math.ceil(expected * 2.5) + 64))


def _sentence_end(chunk: str) -> bool:
    if re.search(r"[!?。！？]$", chunk):
        return True
    if not chunk.endswith("."):
        return False
    last = chunk.split()[-1].lower()
    # An abbreviation, an initial ("B.") or an ordinal ("3. Oktober") is no sentence end.
    return last not in ABBREVIATIONS and not re.fullmatch(r"\w\.|\d+\.", last)


def split_text(text: str, limit: int = CHUNK_CHARS, every_sentence: bool = False) -> list[str]:
    """Chunks of at most `limit` characters, cut at a sentence end once half full (the demo's rule) —
    or, `every_sentence`, at every sentence end."""
    chunks: list[str] = []
    for paragraph in text.splitlines():
        current = ""
        for word in paragraph.split():
            # A word longer than the limit (or text without spaces) is cut hard.
            while len(word) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(word[:limit])
                word = word[limit:]
            if not word:
                continue
            candidate = f"{current} {word}".strip()
            if len(candidate) > limit:
                chunks.append(current)
                current = word
            else:
                current = candidate
            if (every_sentence or len(current) >= limit // 2) and _sentence_end(current):
                chunks.append(current)
                current = ""
        if current:
            chunks.append(current)
    return chunks


class TtsEngine:
    kind = "tts"

    def __init__(self, backend: Backend, own: dict[str, OwnVoice] | None = None, chunk_chars: int = CHUNK_CHARS, max_chars: int = 2000):
        self.backend = backend
        # The owner's own voices by lower-case name, for every project without a profile; the Base model is loaded only for own voices.
        self.own = own or {}
        # A profile's own voices, per project: for that project only, in place of the owner's (PROTOCOL.md, Profiles).
        self.projects: dict[str, dict[str, OwnVoice]] = {}
        self.settings = Settings(chunk_chars=chunk_chars)
        self.max_chars = max_chars

    @property
    def ready(self) -> bool:
        return self.backend.ready

    @property
    def chunk_chars(self) -> int:
        return self.settings.chunk_chars

    def own_for(self, project: str | None) -> dict[str, OwnVoice]:
        """The own voices a project may ask for: its profile's, else the owner's."""
        return self.projects[project] if project is not None and project in self.projects else self.own

    def refs(self) -> list[Reference]:
        """Every recording the Base model must know: the owner's, and every profile's."""
        found: dict = {}
        # A snapshot: the loop may set another project's voices while the speech thread learns.
        for voices in (self.own, *tuple(self.projects.values())):
            for voice in voices.values():
                for ref in voice.refs.values():
                    found.setdefault(ref.path, ref)
        return list(found.values())

    def load(self) -> None:
        self.backend.load(self.refs())

    def learn(self, project: str, voices: dict[str, OwnVoice]) -> None:
        """A profile's voices for its project. In the speech thread: the Base model may have to load first,
        and the voices are offered only once their recordings are learnt."""
        if self.backend.ready:
            known = {ref.path for ref in self.refs()}
            new = [ref for voice in voices.values() for ref in voice.refs.values() if ref.path not in known]
            if new:
                self.backend.add(new)
        self.projects[project] = voices

    def forget(self, project: str) -> None:
        """The project dropped its profile: the owner's voices again."""
        self.projects.pop(project, None)

    def voices(self, project: str | None = None) -> list[dict]:
        """As the caps list them: Qwen's presets in every language, then the own voices in the languages they were recorded in."""
        presets = [{"id": v, "label": v.replace("_", " "), "langs": sorted(LANGUAGES), "instruct": True} for v in VOICES]
        own = [{"id": o.name, "label": f"{o.name} (own voice)", "langs": o.languages, "instruct": False, "own": True} for o in self.own_for(project).values()]
        return presets + own

    def voice_files(self, project: str | None = None) -> dict[str, dict[str, str]]:
        """The own voices' recordings by SHA-256, as the caps' setup reports them."""
        return {o.name: {lang: ref.sha256 for lang, ref in sorted(o.refs.items())} for o in self.own_for(project).values()}

    def caps(self, chars_per_s: float, project: str | None = None) -> dict:
        return {
            "kind": "tts", "model": MODEL_NAME, "location": "local", "voices": self.voices(project),
            "max_chars": self.max_chars, "speed": {"chars_per_s": max(1, round(chars_per_s))},
        }

    def can(self, model: str, voice: str, lang: str, project: str | None = None) -> bool:
        """Whether an offer is one this worker speaks — exactly what its caps say."""
        if model != MODEL_NAME:
            return False
        own = self.own_for(project).get(voice.lower())
        if own is not None:
            return lang in own.refs
        return lang in LANGUAGES and any(v.lower() == voice.lower() for v in VOICES)

    def route(self, voice: str, lang: str, project: str | None = None) -> Reference | str:
        """What speaks a task: an own voice's recording for its language, or the name of Qwen's preset."""
        own = self.own_for(project).get(voice.strip().lower())
        return own.reference(lang) if own is not None else voice_name(voice)

    def synthesize(
        self, text: str, voice: str, lang: str, instruct: str = "", temperature: float = 0.7, seed: int = 0,
        chunk_chars: int | None = None, every_sentence: bool = False,
        cancelled: Callable[[], bool] = lambda: False, progress: Callable[[float], None] | None = None,
        project: str | None = None, settings: Settings | None = None,
    ) -> Speech:
        if not self.backend.ready:
            raise RuntimeError("model not loaded")
        if len(text) > self.max_chars:
            raise TooLarge(f"{len(text)} characters, this worker speaks at most {self.max_chars}")
        s = settings or self.settings
        chunks = split_text(text, chunk_chars or s.chunk_chars, every_sentence)
        if not chunks:
            raise InvalidTask("no text")
        target, language = self.route(voice, lang, project), language_name(lang)
        self.backend.seed(max(0, min(4294967295, int(seed))))
        parts: list[bytes] = []
        sample_rate: int | None = None
        for i, chunk in enumerate(chunks):
            if cancelled():
                raise Cancelled()
            if parts and sample_rate is not None:
                # A short breath between chunks instead of words run together.
                parts.append(b"\x00\x00" * (sample_rate * s.pause_ms // 1000))
            sampling = Sampling(
                temperature=max(0.0, min(2.0, float(temperature))), max_tokens=max_tokens(chunk),
                top_k=s.top_k, top_p=s.top_p, repetition_penalty=s.repetition_penalty,
            )
            if isinstance(target, Reference):
                # The clone reads its recording with the recording's words, then speaks on in that voice.
                # It takes no direction: the moment's delivery stays unused.
                pcm, rate = self.backend.clone(chunk, target, language, sampling)
            else:
                pcm, rate = self.backend.preset(chunk, target, language, instruct, sampling)
            if sample_rate is not None and rate != sample_rate:
                raise RuntimeError("the model changed its sample rate between chunks")
            sample_rate = rate
            parts.append(pcm)
            if progress is not None:
                progress((i + 1) / len(chunks))
        if sample_rate is None or not any(parts):
            raise RuntimeError("the model returned no audio")
        return Speech(pcm16=b"".join(parts), sample_rate=sample_rate)


def to_pcm16(audio: object) -> bytes:
    """Float samples (numpy, any shape) as 16-bit little-endian PCM; NaN or infinity is the model's failure."""
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return b""
    if not np.isfinite(samples).all():
        raise RuntimeError("the model returned invalid samples")
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
