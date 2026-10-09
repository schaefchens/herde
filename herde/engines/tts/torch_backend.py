"""Qwen3-TTS through the official `qwen-tts` package (PyTorch) on an NVIDIA GPU — the `cuda` image.

The same checkpoints as on a Mac, so a host sounds alike wherever it is
spoken: CustomVoice for the presets (with direction), Base for our own voices
(with a clone prompt made once per recording). torch and qwen_tts are imported
only when the models load: the tests run without them.
"""

from __future__ import annotations

import os
from pathlib import Path

from .core import Sampling, to_pcm16
from .voices import Reference


class TorchBackend:
    name = "torch"

    def __init__(
        self, model_id: str, revision: str = "", clone_model: str = "", clone_revision: str = "", hf_home: Path | None = None, device: str = "cuda:0",
    ):
        self.model_id = model_id
        self.revision = revision
        self.clone_model = clone_model
        self.clone_revision = clone_revision
        self.hf_home = hf_home
        self.device = device
        self._model = None
        self._clone = None
        self._prompts: dict[Path, object] = {}

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self, refs: list[Reference]) -> None:
        if self.hf_home is not None:
            os.environ.setdefault("HF_HOME", str(self.hf_home))
        import torch
        from qwen_tts import Qwen3TTSModel

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU visible: start the container with the NVIDIA runtime (compose.nvidia.yaml)")
        # bf16 where the GPU has it (Ampere and newer); fp16 on older cards; fp32 on a CPU.
        if self.device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        def load_one(model_id: str, revision: str):
            kwargs: dict = {"device_map": self.device, "dtype": dtype}
            if revision:
                kwargs["revision"] = revision
            return Qwen3TTSModel.from_pretrained(model_id, **kwargs)

        model = load_one(self.model_id, self.revision)
        if refs:
            clone = load_one(self.clone_model, self.clone_revision)
            # Made once: the recording's features would otherwise be computed again for every take.
            self._prompts = {ref.path: clone.create_voice_clone_prompt(ref_audio=str(ref.path), ref_text=ref.text) for ref in refs}
            self._clone = clone
        # Last: ready means every model a task may name is there.
        self._model = model

    def seed(self, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def preset(self, text: str, speaker: str, language: str, instruct: str, sampling: Sampling) -> tuple[bytes, int]:
        wavs, rate = self._model.generate_custom_voice(text=text, speaker=speaker, language=language, instruct=instruct or None, **_kwargs(sampling))
        return to_pcm16(wavs[0]), int(rate)

    def clone(self, text: str, ref: Reference, language: str, sampling: Sampling) -> tuple[bytes, int]:
        wavs, rate = self._clone.generate_voice_clone(text=text, language=language, voice_clone_prompt=self._prompts[ref.path], **_kwargs(sampling))
        return to_pcm16(wavs[0]), int(rate)


def _kwargs(s: Sampling) -> dict:
    # qwen-tts hands these to transformers' generate (its _merge_generate_kwargs).
    return {
        "do_sample": True, "temperature": s.temperature, "top_k": s.top_k, "top_p": s.top_p,
        "repetition_penalty": s.repetition_penalty, "max_new_tokens": s.max_tokens,
    }
