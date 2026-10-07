"""The listening lab: the same words spoken in several ways, to choose by ear.

`run-arche-worker.sh lab cases.json` speaks every case of a cases file
(the station's bin/replay-show.php writes one: each host moment as it aired
and as it is written now) in each variant — how the text is cut into takes,
and the temperature — and writes the MP3s with an index.html to listen to
them side by side. Nothing is sent anywhere.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .engine import CHUNK_CHARS_MAX, InvalidTask, Speech

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Case:
    id: str
    lang: str
    voice: str
    text: str
    instruct: str


@dataclass(frozen=True)
class Variant:
    """How a text is cut into takes: at most `chunk_chars` each, or at every sentence end."""

    name: str
    chunk_chars: int
    every_sentence: bool
    temperature: float


class LabError(ValueError):
    pass


def variants(chunks: str, temperatures: str) -> list[Variant]:
    """`--chunks 300,whole,sentence` × `--temperatures 0.7,0.8`."""
    out = []
    for c in [c.strip() for c in chunks.split(",") if c.strip()]:
        if c == "whole":
            size, every = CHUNK_CHARS_MAX, False
        elif c == "sentence":
            size, every = CHUNK_CHARS_MAX, True
        elif c.isdigit() and 50 <= int(c) <= CHUNK_CHARS_MAX:
            size, every = int(c), False
        else:
            raise LabError(f"--chunks: {c!r} is no size (50–{CHUNK_CHARS_MAX}), 'whole' or 'sentence'")
        for t in [t.strip() for t in temperatures.split(",") if t.strip()]:
            try:
                temp = float(t)
            except ValueError:
                raise LabError(f"--temperatures: {t!r} is no number") from None
            if not 0.1 <= temp <= 1.2:
                raise LabError("--temperatures: each between 0.1 and 1.2")
            out.append(Variant(name=f"{c}-t{temp:g}", chunk_chars=size, every_sentence=every, temperature=temp))
    if not out:
        raise LabError("no variant to speak")
    return out


def load_cases(path: Path, only: str = "") -> list[Case]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LabError(f"cannot read {path}: {e}") from None
    raw = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise LabError(f"{path} has no list of cases")
    cases = []
    for r in raw:
        if not isinstance(r, dict) or not str(r.get("text", "")).strip():
            continue
        case = Case(id=str(r.get("id", len(cases))), lang=str(r.get("lang", "de")), voice=str(r.get("voice", "Sohee")),
                    text=str(r["text"]), instruct=str(r.get("instruct", "")))
        if only in case.id:
            cases.append(case)
    if not cases:
        raise LabError(f"{path}: no case" + (f" with {only!r} in its id" if only else ""))
    return cases


def run(
    synthesize: Callable[..., Speech],
    encode: Callable[[bytes, int], tuple[bytes, int]],
    cases: list[Case],
    chosen: list[Variant],
    out: Path,
    stopped: Callable[[], bool] = lambda: False,
) -> Path:
    """Speaks every case in every variant into `out`; returns its index.html."""
    out.mkdir(parents=True, exist_ok=True)
    files: dict[tuple[str, str], tuple[str, int, float]] = {}
    for case in cases:
        for v in chosen:
            if stopped():
                break
            started = time.monotonic()
            try:
                speech = synthesize(case.text, case.voice, case.lang, case.instruct, v.temperature, 0,
                                    chunk_chars=v.chunk_chars, every_sentence=v.every_sentence)
                mp3, ms = encode(speech.pcm16, speech.sample_rate)
            except InvalidTask as e:
                log.warning("%s: %s", case.id, e)
                continue
            name = f"{_safe(case.id)}__{_safe(v.name)}.mp3"
            (out / name).write_bytes(mp3)
            files[(case.id, v.name)] = (name, ms, time.monotonic() - started)
            log.info("%s %s: %.1f s of speech in %.1f s", case.id, v.name, ms / 1000, time.monotonic() - started)
        _write_index(out, cases, chosen, files)
    return out / "index.html"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:80]


def _write_index(out: Path, cases: list[Case], chosen: list[Variant], files: dict[tuple[str, str], tuple[str, int, float]]) -> None:
    head = "".join(f"<th>{html.escape(v.name)}</th>" for v in chosen)
    rows = []
    for c in cases:
        cells = []
        for v in chosen:
            f = files.get((c.id, v.name))
            cells.append(
                f'<td><audio controls preload="none" src="{html.escape(f[0])}"></audio><br><small>{f[1] / 1000:.1f} s, made in {f[2]:.1f} s</small></td>'
                if f else "<td>—</td>"
            )
        rows.append(
            f"<tr><td><b>{html.escape(c.id)}</b> <small>{html.escape(c.voice)}, {html.escape(c.lang)}</small><br>{html.escape(c.text)}"
            f"<br><small><i>{html.escape(c.instruct)}</i></small></td>{''.join(cells)}</tr>"
        )
    (out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>ARCHE voice lab</title>"
        "<style>body{font:14px system-ui;margin:16px}td,th{border-bottom:1px solid #ccc;padding:8px;vertical-align:top;text-align:left}"
        "td:first-child{max-width:520px}</style>"
        f"<h1>ARCHE voice lab</h1><table><tr><th>What is said, and how it is directed</th>{head}</tr>{''.join(rows)}</table>",
        encoding="utf-8",
    )
