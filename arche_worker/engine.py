"""Qwen3-TTS through mlx-audio on the Mac's GPU: CustomVoice for Qwen's
presets, and Base for voices of our own, cloned from a recording
(arche_worker.voices).

mlx-audio is imported only when the models load: the tests, and `check`,
run without it.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .voices import OwnVoice, Reference

log = logging.getLogger(__name__)

# The CustomVoice presets. None is a native German speaker; the station's owner chose Sohee for German.
VOICES = ("Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee")
# The station sends language codes; mlx-audio wants the names (it lowercases them).
LANGUAGES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
}
# What the station calls this model (Host\Workers matches a host's model against it).
MODEL_NAME = "qwen3-tts-1.7b-customvoice"
# Each piece of a text is a take of its own, and its delivery starts afresh:
# the owner heard the host "drift" across pieces. The size is the config's
# `chunk_chars` ([engine]); `lab` compares sizes by ear.
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


@dataclass(frozen=True)
class Speech:
    pcm16: bytes
    sample_rate: int

    @property
    def ms(self) -> int:
        return len(self.pcm16) // 2 * 1000 // self.sample_rate


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


class Engine:
    def __init__(
        self, model_id: str, revision: str, hf_home: Path | None = None, chunk_chars: int = CHUNK_CHARS,
        own: dict[str, OwnVoice] | None = None, clone_model: str = "", clone_revision: str = "",
    ):
        self.model_id = model_id
        self.revision = revision
        self.hf_home = hf_home
        self.chunk_chars = chunk_chars
        # Our own voices by lower-case name; the Base model is loaded only for them.
        self.own = own or {}
        self.clone_model = clone_model
        self.clone_revision = clone_revision
        self._model = None
        self._clone = None
        self._refs: dict[Path, object] = {}

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        # Read when huggingface_hub is imported: set before mlx-audio comes in.
        if self.hf_home is not None:
            os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from mlx_audio.tts.utils import load_model

        model = load_model(self.model_id, revision=self.revision)
        if model.config.tts_model_type != "custom_voice":
            raise ValueError(f"{self.model_id} is no Qwen3-TTS CustomVoice model")
        if self.own:
            from mlx_audio.utils import load_audio

            clone = load_model(self.clone_model, revision=self.clone_revision or None)
            if clone.config.tts_model_type != "base":
                raise ValueError(f"{self.clone_model} is no Qwen3-TTS Base model")
            # Read once: mlx-audio keeps a recording's encoding for the takes after the first.
            self._refs = {ref.path: load_audio(str(ref.path), sample_rate=clone.sample_rate)
                          for voice in self.own.values() for ref in voice.refs.values()}
            self._clone = clone
        # Last: ready means every model a task may name is there.
        self._model = model

    def route(self, voice: str, lang: str) -> Reference | str:
        """What speaks a task: our own voice's recording for its language, or the name of Qwen's preset."""
        own = self.own.get(voice.strip().lower())
        return own.reference(lang) if own is not None else voice_name(voice)

    def synthesize(
        self, text: str, voice: str, lang: str, instruct: str = "", temperature: float = 0.7, seed: int = 0,
        chunk_chars: int | None = None, every_sentence: bool = False,
    ) -> Speech:
        if self._model is None:
            raise RuntimeError("model not loaded")
        chunks = split_text(text, chunk_chars or self.chunk_chars, every_sentence)
        if not chunks:
            raise InvalidTask("no text")
        target, language = self.route(voice, lang), language_name(lang)
        import mlx.core as mx
        import numpy as np

        mx.random.seed(max(0, min(4294967295, int(seed))))
        parts: list = []
        sample_rate: int | None = None
        for chunk in chunks:
            if parts and sample_rate is not None:
                # A short breath between chunks instead of words run together.
                parts.append(np.zeros(sample_rate * PAUSE_MS // 1000, dtype=np.float32))
            sampling = {
                "temperature": max(0.0, min(2.0, float(temperature))), "max_tokens": max_tokens(chunk),
                "top_k": 50, "top_p": 1.0, "repetition_penalty": REPETITION_PENALTY, "stream": False, "verbose": False,
            }
            if isinstance(target, Reference):
                # The clone reads its recording with the recording's words, then speaks on in that voice.
                # It takes no direction: the moment's delivery stays unused.
                takes = self._clone.generate(
                    text=chunk, ref_audio=self._refs[target.path], ref_text=target.text, lang_code=language.lower(), **sampling,
                )
            else:
                takes = self._model.generate_custom_voice(text=chunk, speaker=target, language=language, instruct=instruct or None, **sampling)
            for result in takes:
                if sample_rate is not None and result.sample_rate != sample_rate:
                    raise RuntimeError("the model changed its sample rate between chunks")
                sample_rate = int(result.sample_rate)
                parts.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
        if sample_rate is None or not parts or not any(p.size for p in parts):
            raise RuntimeError("the model returned no audio")
        audio = np.concatenate(parts)
        if not np.isfinite(audio).all():
            raise RuntimeError("the model returned invalid samples")
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return Speech(pcm16=pcm16, sample_rate=sample_rate)
