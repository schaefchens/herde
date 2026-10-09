"""Text: a local model through Ollama (or a stub that fills a schema), for tasks of kind `text`."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..tts.core import Cancelled, InvalidTask, TooLarge

if TYPE_CHECKING:
    from ...config import TextConfig

__all__ = ["Cancelled", "EngineError", "InvalidTask", "TooLarge", "build"]


class EngineError(RuntimeError):
    """The model failed (or ran astray); another try, or another worker, may do better."""


Progress = Callable[[float], None]


def build(config: TextConfig, max_tokens: int):
    """The engine the config asks for (None without a backend)."""
    if not config.backend:
        return None
    if config.backend == "stub":
        from .stub import StubText

        return StubText(config.models, max_tokens)
    from .ollama import OllamaEngine

    return OllamaEngine(config.ollama, config.models, config.keep_alive, max_tokens)
