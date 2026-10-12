"""The messages of the herde protocol (protocol/PROTOCOL.md), as the worker reads and writes them.

A project's answer is checked before anything acts on it: an id that does not
match its pattern never reaches a model, a file name or a log line. An offer
that does not fit is left out; a claim or poll answer that does not fit is the
project's error (the worker backs off as from a 5xx).

Times arrive as seconds from now (a lent computer's clock may be off) and are
turned into this machine's monotonic clock at once.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from . import PROTOCOL

LEASE = re.compile(r"[A-Za-z0-9-]{8,64}")
VOICE_ID = re.compile(r"[A-Za-z0-9_-]{1,40}")
MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
LANG = re.compile(r"[a-z]{2}")
ERROR_CODE = re.compile(r"[a-z_]{2,40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
# A Hugging Face repository at a commit, as profiles and setups name a checkpoint (empty: none).
CHECKPOINT = re.compile(r"|[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?(@[0-9a-f]{40})?")
KINDS = ("tts", "text")
# The kinds and kind versions this worker speaks.
KIND_VERSIONS: dict[str, list[int]] = {"tts": [1], "text": [1]}
CLASSES = ("live", "interactive", "background")
FAIL_REASONS = ("shutdown", "cancelled", "invalid", "too_large", "engine", "timeout")
# A project's retry_after is kept within these bounds whatever it says.
RETRY_MIN = 1.0
RETRY_MAX = 60.0


class ProtocolError(ValueError):
    """A project's answer that does not follow the protocol."""


@dataclass(frozen=True)
class Offer:
    """Open work a project offers this worker: metadata only, never its input."""

    project: str
    task: int
    kind: str
    cls: str
    model: str
    voice: str = ""
    lang: str = ""
    chars: int = 0
    input_tokens: int = 0
    max_tokens: int = 0
    # This machine's monotonic clock: when nobody should start it, when its result is useless.
    claim_by: float | None = None
    due_by: float | None = None
    received: float = 0.0


@dataclass(frozen=True)
class PollAnswer:
    now: int
    offers: tuple[Offer, ...]
    cancel: tuple[tuple[int, str], ...]
    need_caps: bool
    retry_after: float
    # The SHA-256 of this worker's profile at the project ("": it has none).
    profile: str = ""


@dataclass(frozen=True)
class TtsInput:
    lang: str
    voice: str
    model: str
    temperature: float
    seed: int
    # What listeners sent may be in it, and the host's style: never in a log line.
    text: str = field(repr=False)
    instruct: str = field(repr=False, default="")


@dataclass(frozen=True)
class Message:
    role: str
    content: str = field(repr=False)


@dataclass(frozen=True)
class TextInput:
    model: str
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    system: str = field(repr=False, default="")
    messages: tuple[Message, ...] = field(repr=False, default=())
    json_schema: dict[str, Any] | None = field(repr=False, default=None)

    @property
    def chars(self) -> int:
        return len(self.system) + sum(len(m.content) for m in self.messages)


@dataclass(frozen=True)
class Task:
    """A claimed task: ours while our polls list it with its lease."""

    project: str
    id: int
    kind: str
    cls: str
    lease: str
    input: TtsInput | TextInput = field(repr=False)
    due_by: float | None = None
    claimed: float = 0.0


@dataclass(frozen=True)
class TextOutput:
    finish: str
    input_tokens: int
    output_tokens: int
    load_ms: int = 0
    read_ms: int = 0
    write_ms: int = 0
    content: str = field(repr=False, default="")

    def body(self, lease: str) -> dict:
        return {
            "lease": lease,
            "output": {
                "content": self.content,
                "finish": self.finish,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
                "timings": {"load_ms": self.load_ms, "read_ms": self.read_ms, "write_ms": self.write_ms},
            },
        }


@dataclass(frozen=True)
class Joined:
    project: str
    worker_id: int
    worker_name: str
    trust: str
    key: str = field(repr=False)


def canonical_hash(value: object) -> str:
    """SHA-256 of canonical JSON: keys sorted, no spaces, UTF-8 (caps and profiles alike)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def caps_hash(caps: dict) -> str:
    return canonical_hash(caps)


def poll_body(
    state: str, caps: dict, send_caps: bool, running: list[dict], free: dict[str, int], kinds: dict[str, list[int]], version: str,
    resume_in: int | None = None, profile: dict | None = None,
) -> dict:
    body: dict[str, Any] = {
        "protocol": PROTOCOL,
        "version": version,
        "kinds": kinds,
        "state": state,
        "caps_hash": caps_hash(caps),
        "running": running,
        "free": free,
    }
    if resume_in is not None:
        body["resume_in"] = max(0, int(resume_in))
    if send_caps:
        body["caps"] = caps
    if profile is not None:
        body["profile"] = profile
    return body


def _int(v: Any, minimum: int = 0) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        return None
    return v


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _str(v: Any, pattern: re.Pattern[str]) -> str | None:
    return v if isinstance(v, str) and pattern.fullmatch(v) else None


def parse_poll(project: str, data: Any, received: float) -> PollAnswer:
    if not isinstance(data, dict):
        raise ProtocolError("the poll answer is not an object")
    retry = _num(data.get("retry_after"))
    if retry is None:
        raise ProtocolError("the poll answer has no retry_after")
    offers = []
    for raw in data.get("offers") or []:
        offer = _offer(project, raw, received)
        if offer is not None:
            offers.append(offer)
    cancel = []
    for raw in data.get("cancel") or []:
        if isinstance(raw, dict):
            task, lease = _int(raw.get("task"), 1), _str(raw.get("lease"), LEASE)
            if task is not None and lease is not None:
                cancel.append((task, lease))
    now = _int(data.get("now")) or 0
    return PollAnswer(
        now=now, offers=tuple(offers), cancel=tuple(cancel), need_caps=data.get("need_caps") is True,
        retry_after=max(RETRY_MIN, min(RETRY_MAX, retry)), profile=_str(data.get("profile"), SHA256) or "",
    )


def _offer(project: str, raw: Any, received: float) -> Offer | None:
    """An offer as the scheduler sees it, or None when it does not fit the protocol (it is left out)."""
    if not isinstance(raw, dict):
        return None
    task, kind, cls, model = _int(raw.get("task"), 1), raw.get("kind"), raw.get("class"), _str(raw.get("model"), MODEL)
    if task is None or kind not in KINDS or cls not in CLASSES or model is None:
        return None
    voice = lang = ""
    if kind == "tts":
        voice, lang = _str(raw.get("voice"), VOICE_ID) or "", _str(raw.get("lang"), LANG) or ""
        if not voice or not lang:
            return None
    size = raw.get("size") if isinstance(raw.get("size"), dict) else {}
    claim_in, due_in = _int(raw.get("claim_in")), _int(raw.get("due_in"))
    return Offer(
        project=project, task=task, kind=kind, cls=cls, model=model, voice=voice, lang=lang,
        chars=_int(size.get("chars")) or 0, input_tokens=_int(size.get("input_tokens")) or 0, max_tokens=_int(size.get("max_tokens")) or 0,
        claim_by=received + claim_in if claim_in is not None else None,
        due_by=received + due_in if due_in is not None else None,
        received=received,
    )


def parse_claim(project: str, task_id: int, lease: str, data: Any, received: float) -> Task:
    if not isinstance(data, dict) or not isinstance(data.get("task"), dict):
        raise ProtocolError(f"task {task_id}: the claim answer has no task")
    raw = data["task"]
    if raw.get("id") != task_id:
        raise ProtocolError(f"task {task_id}: the claim answer is for another task")
    kind, cls = raw.get("kind"), raw.get("class")
    if kind not in KINDS or cls not in CLASSES:
        raise ProtocolError(f"task {task_id}: unknown kind or class")
    inp = raw.get("input")
    if not isinstance(inp, dict):
        raise ProtocolError(f"task {task_id}: no input")
    due_in = _int(raw.get("due_in"))
    parsed = _tts_input(task_id, inp) if kind == "tts" else _text_input(task_id, inp)
    return Task(project=project, id=task_id, kind=kind, cls=cls, lease=lease, input=parsed,
                due_by=received + due_in if due_in is not None else None, claimed=received)


def _tts_input(task_id: int, inp: dict) -> TtsInput:
    text, lang, voice, model = inp.get("text"), _str(inp.get("lang"), LANG), _str(inp.get("voice"), VOICE_ID), _str(inp.get("model"), MODEL)
    if not isinstance(text, str) or not text.strip() or lang is None or voice is None or model is None:
        raise ProtocolError(f"task {task_id}: a speech task needs text, lang, voice and model")
    temperature = _num(inp.get("temperature"))
    seed = _int(inp.get("seed")) or 0
    instruct = inp.get("instruct") if isinstance(inp.get("instruct"), str) else ""
    return TtsInput(
        lang=lang, voice=voice, model=model,
        temperature=max(0.1, min(1.2, temperature)) if temperature is not None else 0.7,
        seed=min(seed, 4294967295), text=text, instruct=instruct,
    )


def _text_input(task_id: int, inp: dict) -> TextInput:
    model, max_tokens = _str(inp.get("model"), MODEL), _int(inp.get("max_tokens"), 1)
    raw_messages = inp.get("messages")
    if model is None or max_tokens is None or not isinstance(raw_messages, list) or not raw_messages:
        raise ProtocolError(f"task {task_id}: a text task needs model, messages and max_tokens")
    messages = []
    for m in raw_messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
            raise ProtocolError(f"task {task_id}: a message needs a role (user, assistant) and content")
        messages.append(Message(role=m["role"], content=m["content"]))
    schema = inp.get("json_schema")
    if schema is not None and not isinstance(schema, dict):
        raise ProtocolError(f"task {task_id}: json_schema must be an object")
    top_k = _int(inp.get("top_k"), 1)
    return TextInput(
        model=model, max_tokens=max_tokens, temperature=_num(inp.get("temperature")), top_p=_num(inp.get("top_p")), top_k=top_k,
        system=inp.get("system") if isinstance(inp.get("system"), str) else "", messages=tuple(messages), json_schema=schema,
    )


def parse_join(data: Any) -> Joined:
    if not isinstance(data, dict):
        raise ProtocolError("the join answer is not an object")
    key, worker, project = data.get("key"), data.get("worker"), data.get("project")
    if not isinstance(key, str) or not re.fullmatch(r"\S{32,200}", key) or not isinstance(worker, dict) or not isinstance(project, dict):
        raise ProtocolError("the join answer has no key")
    if project.get("protocol") != PROTOCOL:
        raise ProtocolError(f"the project speaks protocol {project.get('protocol')!r}, this worker {PROTOCOL}")
    return Joined(
        project=str(project.get("name") or "project")[:80], worker_id=_int(worker.get("id"), 1) or 0,
        worker_name=str(worker.get("name") or "")[:80], trust=str(worker.get("trust") or ""), key=key,
    )


def error_code(data: Any) -> str:
    """The `error` of a project's error answer, if it is one that fits the protocol."""
    if isinstance(data, dict):
        code = _str(data.get("error"), ERROR_CODE)
        if code is not None:
            return code
    return "unknown"
