"""Silence as long as the words would take (13 characters a second): for tests and for a project's own
end-to-end runs, so its length check passes without a model or a GPU."""

from __future__ import annotations

from .core import CHARS_PER_SECOND, Sampling
from .voices import Reference

RATE = 24000


class StubBackend:
    name = "stub"
    # No weights: no number format, no checkpoints.
    dtype = ""

    def __init__(self) -> None:
        self._ready = False
        self.learnt: list[Reference] = []

    @property
    def ready(self) -> bool:
        return self._ready

    def pins(self) -> tuple[str, str]:
        return "", ""

    def checkpoints(self) -> tuple[str, str]:
        return "", ""

    def load(self, refs: list[Reference]) -> None:
        self.learnt = list(refs)
        self._ready = True

    def fetch_clone(self) -> None:
        pass

    def add(self, refs: list[Reference]) -> None:
        self.learnt.extend(refs)

    def seed(self, seed: int) -> None:
        pass

    def preset(self, text: str, speaker: str, language: str, instruct: str, sampling: Sampling) -> tuple[bytes, int]:
        return self._silence(text)

    def clone(self, text: str, ref: Reference, language: str, sampling: Sampling) -> tuple[bytes, int]:
        return self._silence(text)

    @staticmethod
    def _silence(text: str) -> tuple[bytes, int]:
        return b"\x00\x00" * max(RATE // 10, int(len(text) / CHARS_PER_SECOND * RATE)), RATE
