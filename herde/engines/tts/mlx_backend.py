"""Qwen3-TTS through mlx-audio on Apple's GPU (a Mac's worker runs natively: Docker cannot reach Metal).

mlx-audio is imported only when the models load: the tests, and `herde check`,
run without it.
"""

from __future__ import annotations

import os
from pathlib import Path

from .core import Sampling, to_pcm16
from .voices import Reference


class MlxBackend:
    name = "mlx"

    def __init__(self, model_id: str, revision: str, clone_model: str = "", clone_revision: str = "", hf_home: Path | None = None):
        self.model_id = model_id
        self.revision = revision
        self.clone_model = clone_model
        self.clone_revision = clone_revision
        self.hf_home = hf_home
        self._model = None
        self._clone = None
        self._refs: dict[Path, object] = {}

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self, refs: list[Reference]) -> None:
        # Read when huggingface_hub is imported: set before mlx-audio comes in.
        if self.hf_home is not None:
            os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from mlx_audio.tts.utils import load_model

        model = load_model(self.model_id, revision=self.revision or None)
        if model.config.tts_model_type != "custom_voice":
            raise ValueError(f"{self.model_id} is no Qwen3-TTS CustomVoice model")
        if refs:
            from mlx_audio.utils import load_audio

            clone = load_model(self.clone_model, revision=self.clone_revision or None)
            if clone.config.tts_model_type != "base":
                raise ValueError(f"{self.clone_model} is no Qwen3-TTS Base model")
            # Read once: mlx-audio keeps a recording's encoding for the takes after the first.
            self._refs = {ref.path: load_audio(str(ref.path), sample_rate=clone.sample_rate) for ref in refs}
            self._clone = clone
        # Last: ready means every model a task may name is there.
        self._model = model

    def seed(self, seed: int) -> None:
        import mlx.core as mx

        mx.random.seed(seed)

    def preset(self, text: str, speaker: str, language: str, instruct: str, sampling: Sampling) -> tuple[bytes, int]:
        return self._collect(self._model.generate_custom_voice(text=text, speaker=speaker, language=language, instruct=instruct or None, **_kwargs(sampling)))

    def clone(self, text: str, ref: Reference, language: str, sampling: Sampling) -> tuple[bytes, int]:
        return self._collect(self._clone.generate(text=text, ref_audio=self._refs[ref.path], ref_text=ref.text, lang_code=language.lower(), **_kwargs(sampling)))

    @staticmethod
    def _collect(takes) -> tuple[bytes, int]:
        pcm: list[bytes] = []
        rate: int | None = None
        for result in takes:
            if rate is not None and result.sample_rate != rate:
                raise RuntimeError("the model changed its sample rate within a take")
            rate = int(result.sample_rate)
            pcm.append(to_pcm16(result.audio))
        if rate is None:
            raise RuntimeError("the model returned no audio")
        return b"".join(pcm), rate


def _kwargs(s: Sampling) -> dict:
    return {
        "temperature": s.temperature, "max_tokens": s.max_tokens, "top_k": s.top_k, "top_p": s.top_p,
        "repetition_penalty": s.repetition_penalty, "stream": False, "verbose": False,
    }
