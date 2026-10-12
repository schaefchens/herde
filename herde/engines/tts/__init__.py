"""Speech: Qwen3-TTS on a backend that fits the machine (mlx on a Mac, torch on NVIDIA or AMD, a silent stub)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import LANGUAGES, MODEL_NAME, VOICES, TtsEngine
from .voices import discover

if TYPE_CHECKING:
    # Only for the annotation: herde.config imports this package's core, so a real import here would be a cycle.
    from ...config import TtsConfig

__all__ = ["MODEL_NAME", "TtsEngine", "build"]


def build(config: TtsConfig, max_chars: int) -> TtsEngine | None:
    """The engine the config asks for (None without a backend). Loading its models is a separate step."""
    if not config.backend:
        return None
    own = discover(config.voices_dir, LANGUAGES, VOICES)
    if config.backend == "mlx":
        from .mlx_backend import MlxBackend

        backend = MlxBackend(config.model, config.revision, config.clone_model, config.clone_revision, config.hf_home)
    elif config.backend == "torch":
        from .torch_backend import TorchBackend

        backend = TorchBackend(config.model, config.revision, config.clone_model, config.clone_revision, config.hf_home, config.device)
    else:
        from .stub_backend import StubBackend

        backend = StubBackend()
    return TtsEngine(backend, own, config.chunk_chars, max_chars)
