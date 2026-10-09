"""Text without a model: the schema's required fields filled with short words — for tests and for a
project's own end-to-end runs. Never offered by a worker that has a real backend."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ...config import TextModel
from ...protocol import TextInput, TextOutput
from . import InvalidTask, Progress, TooLarge


class StubText:
    kind = "text"

    def __init__(self, models: tuple[TextModel, ...], max_tokens: int = 4096):
        self.models = {m.name: m for m in models} or {"stub": TextModel(name="stub", tag="stub", context=8192)}
        self.max_tokens = max_tokens
        self.available = set(self.models)
        self.loaded = set(self.models)
        self.reachable = True

    @property
    def ready(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def refresh(self) -> None:
        pass

    def can(self, model: str) -> bool:
        return model in self.available

    def caps(self, speeds: Callable[[str], dict]) -> list[dict]:
        return [
            {"kind": "text", "model": name, "location": "local", "json_schema": True, "context": m.context,
             "max_tokens": self.max_tokens, "loaded": True, "speed": speeds(name)}
            for name, m in sorted(self.models.items())
        ]

    async def generate(self, inp: TextInput, cancelled: Callable[[], bool], progress: Progress | None = None) -> TextOutput:
        if inp.model not in self.available:
            raise InvalidTask(f"no model {inp.model!r} here")
        if inp.max_tokens > self.max_tokens:
            raise TooLarge(f"max_tokens {inp.max_tokens}, this worker writes at most {self.max_tokens}")
        content = json.dumps(fill(inp.json_schema), ensure_ascii=False) if inp.json_schema is not None else "Stub answer."
        return TextOutput(content=content, finish="stop", input_tokens=inp.chars // 4, output_tokens=max(1, len(content) // 4))


def fill(schema: Any) -> Any:
    """The least that fits: required fields only, the first of an enum, the minimum of a number."""
    if not isinstance(schema, dict):
        return None
    if "const" in schema:
        return schema["const"]
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind == "object" or (kind is None and "properties" in schema):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        return {k: fill(props.get(k, {"type": "string"})) for k in required}
    if kind == "array":
        n = schema.get("minItems", 0) if isinstance(schema.get("minItems"), int) else 0
        return [fill(schema.get("items", {"type": "string"})) for _ in range(n)]
    if kind == "string":
        return "Stub text."
    if kind in ("integer", "number"):
        low = schema.get("minimum", 0)
        return low if isinstance(low, (int, float)) else 0
    if kind == "boolean":
        return False
    return None
