"""Text through Ollama: a model the owner listed, by its short name, answering in a task's JSON schema.

Never pulls: a task may only name what the owner put in [[text.models]] and
Ollama already has (`/api/tags`) — otherwise a project could fill a lender's
disk. Streamed, so a cancel or a stop closes the connection (Ollama then stops
writing), and a model that only writes blank space — Gemma once wrote spaces
up to its token limit — is stopped after 200 of them instead of minutes later.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import httpx

from ...config import TextModel
from ...protocol import TextInput, TextOutput
from . import Cancelled, EngineError, InvalidTask, Progress, TooLarge

log = logging.getLogger(__name__)

BLANK_RUN = 200


class OllamaEngine:
    kind = "text"

    def __init__(self, url: str, models: tuple[TextModel, ...], keep_alive: str = "60m", max_tokens: int = 4096,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.url = url
        self.models = {m.name: m for m in models}
        self.keep_alive = keep_alive
        self.max_tokens = max_tokens
        # A long prompt is read before the first word comes: up to minutes on a cold model.
        self._client = httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(300.0, connect=5.0), follow_redirects=False, transport=transport)
        # Names whose tag Ollama has, and those it holds in memory now.
        self.available: set[str] = set()
        self.loaded: set[str] = set()
        self.reachable = False

    @property
    def ready(self) -> bool:
        return self.reachable and bool(self.available)

    async def close(self) -> None:
        await self._client.aclose()

    async def refresh(self) -> None:
        """What Ollama has and holds: asked now and then, never pulled."""
        try:
            tags = await self._get("/api/tags")
            ps = await self._get("/api/ps")
        except EngineError as e:
            if self.reachable:
                log.warning("Ollama at %s: %s; no text tasks until it answers", self.url, e)
            self.reachable, self.available, self.loaded = False, set(), set()
            return
        have = _names(tags)
        held = _names(ps)
        self.available = {n for n, m in self.models.items() if _tagged(m.tag, have)}
        self.loaded = {n for n, m in self.models.items() if _tagged(m.tag, held)}
        missing = sorted(set(self.models) - self.available)
        if missing and (not self.reachable or missing != getattr(self, "_missing", None)):
            log.warning("Ollama has no %s: offered only once it does (ollama pull <tag>)", ", ".join(f"{n} ({self.models[n].tag})" for n in missing))
        self._missing = missing
        self.reachable = True

    def can(self, model: str) -> bool:
        return model in self.available

    def caps(self, speeds: Callable[[str], dict]) -> list[dict]:
        return [
            {"kind": "text", "model": name, "location": "local", "json_schema": True, "context": self.models[name].context,
             "max_tokens": self.max_tokens, "loaded": name in self.loaded, "speed": speeds(name)}
            for name in sorted(self.available)
        ]

    async def generate(self, inp: TextInput, cancelled: Callable[[], bool], progress: Progress | None = None) -> TextOutput:
        model = self.models.get(inp.model)
        if model is None or inp.model not in self.available:
            raise InvalidTask(f"no model {inp.model!r} here")
        if inp.max_tokens > self.max_tokens:
            raise TooLarge(f"max_tokens {inp.max_tokens}, this worker writes at most {self.max_tokens}")
        messages = ([{"role": "system", "content": inp.system}] if inp.system else []) + [{"role": m.role, "content": m.content} for m in inp.messages]
        options: dict = {"num_predict": inp.max_tokens, "num_ctx": model.context}
        for k, v in (("temperature", inp.temperature), ("top_p", inp.top_p), ("top_k", inp.top_k)):
            if v is not None:
                options[k] = v
        body: dict = {"model": model.tag, "messages": messages, "stream": True, "think": False, "keep_alive": self.keep_alive, "options": options}
        if inp.json_schema is not None:
            body["format"] = inp.json_schema
        parts: list[str] = []
        tail = ""
        written = 0
        final: dict | None = None
        try:
            async with self._client.stream("POST", "/api/chat", json=body) as response:
                if response.status_code == 404:
                    raise InvalidTask(f"Ollama has no {model.tag}")
                if response.status_code != 200:
                    await response.aread()
                    raise EngineError(f"Ollama answered HTTP {response.status_code}")
                async for line in response.aiter_lines():
                    if cancelled():
                        raise Cancelled()
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        raise EngineError("Ollama sent a line that is not JSON") from None
                    if not isinstance(data, dict):
                        continue
                    if "error" in data:
                        # Its message may quote the prompt: only that there was one.
                        raise EngineError("Ollama reported an error")
                    piece = (data.get("message") or {}).get("content") or ""
                    if piece:
                        parts.append(piece)
                        written += len(piece)
                        tail = (tail + piece)[-BLANK_RUN:]
                        if written >= BLANK_RUN and not tail.strip():
                            raise EngineError(f"the model wrote only blank space ({BLANK_RUN} characters in a row)")
                        if progress is not None:
                            progress(min(0.99, written / 4 / max(1, inp.max_tokens)))
                    if data.get("done"):
                        final = data
                        break
        except httpx.HTTPError as e:
            raise EngineError(f"Ollama: {type(e).__name__}") from None
        if final is None:
            raise EngineError("Ollama ended without finishing")
        self.loaded.add(inp.model)
        return TextOutput(
            content="".join(parts),
            finish="length" if final.get("done_reason") == "length" else "stop",
            input_tokens=_count(final.get("prompt_eval_count")),
            output_tokens=_count(final.get("eval_count")),
            load_ms=_count(final.get("load_duration")) // 1_000_000,
            read_ms=_count(final.get("prompt_eval_duration")) // 1_000_000,
            write_ms=_count(final.get("eval_duration")) // 1_000_000,
        )

    async def _get(self, path: str) -> dict:
        try:
            r = await self._client.get(path)
        except httpx.HTTPError as e:
            raise EngineError(type(e).__name__) from None
        if r.status_code != 200:
            raise EngineError(f"HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError:
            raise EngineError("not JSON") from None
        return data if isinstance(data, dict) else {}


def _names(data: dict) -> set[str]:
    out = set()
    for m in data.get("models") or []:
        if isinstance(m, dict):
            for k in ("name", "model"):
                if isinstance(m.get(k), str):
                    out.add(m[k])
    return out


def _tagged(tag: str, names: set[str]) -> bool:
    # "llama3" is Ollama's "llama3:latest".
    return tag in names or (":" not in tag and f"{tag}:latest" in names)


def _count(v: object) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0
