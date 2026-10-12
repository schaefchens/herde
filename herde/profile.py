"""A project's profile (protocol/PROTOCOL.md, Profiles): the setup every worker that speaks for it must run.

A station that lets many computers speak in one voice cannot let each set up
its own: another recording, checkpoint or setting is another speaker, and a
hundred computers would be a hundred hosts. So the project holds the setup,
and the worker takes it from there — the settings of every take, and the
project's own voices with their recordings — keeps it under its state folder
(`projects/<name>/`), and proves with every poll that it still runs it: what
it verified here, and how its check takes sounded (`judge`).

A profile never makes the worker download a model: its checkpoints must be
the ones this herde pins (`unmet`). Standard library only.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .engines.tts.core import CODE, LANGUAGES, MODEL_NAME, VOICES, Settings
from .engines.tts.voices import TEXT_MAX, BadRecording, OwnVoice, reference
from .protocol import CHECKPOINT, LANG, MODEL, SHA256, VOICE_ID, canonical_hash

BACKENDS = ("mlx", "torch", "stub")
DTYPES = ("bf16", "fp16", "fp32")
# What a project may hand a worker: a recording at most 10 MB, a profile's JSON at most 256 KB.
FILE_MAX = 10 * 1024 * 1024
DOC_MAX = 256 * 1024
VOICES_MAX = 20
TAKES_MAX = 40
TAKE_TEXT_MAX = 500
EVERY_MIN = 600
# The worker verifies its setup this often, tries a failed check again after this long, and a failed fetch too (s).
VERIFY_EVERY = 600.0
RETRY_CHECK = 600.0
RETRY_SYNC = 60.0
# A check take must run as long as its words need — ARCHE holds every result to the same (Tasks::plausible).
CPS_MIN, CPS_MAX = 6.0, 30.0
# Quieter than this 16-bit RMS (about -50 dBFS) is silence; more than this share of samples at full scale is clipping.
SILENT_RMS = 100.0
CLIPPED_SHARE = 0.01
CHECK_TEMPERATURE = 0.7
CHECK_SEED = 1
# A take is sampled: one may run long or short by chance, which is why a project asks again for a result
# that does not fit its words. A check takes another seed the same way, and fails only when every try does.
CHECK_TRIES = 3


class ProfileError(ValueError):
    """A profile, or one of its files, that does not fit the protocol: the message says what (never a listener's words)."""


@dataclass(frozen=True)
class Recording:
    file: str
    bytes: int
    text: str = field(repr=False)


@dataclass(frozen=True)
class CheckTake:
    voice: str
    lang: str
    text: str = field(repr=False)


@dataclass(frozen=True)
class Profile:
    hash: str
    model: str
    code: int
    # Per backend: the checkpoints it must have loaded, CustomVoice and Base, as repo@revision.
    backends: dict[str, tuple[str, str]]
    dtypes: tuple[str, ...]
    settings: Settings
    mp3_kbps: int
    # The project's own voices by name as it spells them, per language.
    voices: dict[str, dict[str, Recording]]
    every: int
    takes: tuple[CheckTake, ...]
    document: dict = field(repr=False, compare=False)

    def files(self) -> dict[str, int]:
        """Every recording it names, by SHA-256, with its size."""
        return {r.file: r.bytes for langs in self.voices.values() for r in langs.values()}


def parse(data: object) -> Profile:
    """A project's `profile` answer, checked field by field — or ProfileError."""
    if not isinstance(data, dict):
        raise ProfileError("the profile is not an object")
    rest = {k: v for k, v in data.items() if k != "profile"}
    digest = data.get("profile")
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ProfileError("the profile has no hash")
    if canonical_hash(rest) != digest:
        raise ProfileError("the profile's hash does not match its content")
    tts = _obj(rest.get("tts"), "tts")
    model = tts.get("model")
    if not isinstance(model, str) or not MODEL.fullmatch(model):
        raise ProfileError("tts.model is not a model name")
    backends: dict[str, tuple[str, str]] = {}
    for name, pins in _obj(tts.get("backends"), "tts.backends").items():
        pins = _obj(pins, f"tts.backends.{name}")
        if name not in BACKENDS or not all(isinstance(pins.get(k), str) and CHECKPOINT.fullmatch(pins[k]) for k in ("model", "clone_model")):
            raise ProfileError(f"tts.backends: {str(name)[:20]!r} is no backend with two checkpoints")
        backends[name] = (pins["model"], pins["clone_model"])
    if not backends:
        raise ProfileError("tts.backends names none")
    dtypes = tts.get("dtypes")
    if not isinstance(dtypes, list) or not dtypes or any(d not in DTYPES for d in dtypes):
        raise ProfileError(f"tts.dtypes must name some of {', '.join(DTYPES)}")
    speech = _obj(tts.get("speech"), "tts.speech")
    settings = Settings(
        chunk_chars=_int(speech, "chunk_chars", 50, 2000), pause_ms=_int(speech, "pause_ms", 0, 2000), top_k=_int(speech, "top_k", 1, 1000),
        top_p=_num(speech, "top_p", 0.0, 1.0, low_open=True), repetition_penalty=_num(speech, "repetition_penalty", 1.0, 2.0),
    )
    voices = _voices(tts.get("voices"))
    checks = _obj(tts.get("checks"), "tts.checks")
    every = checks.get("every")
    if isinstance(every, bool) or not isinstance(every, int) or every < EVERY_MIN:
        raise ProfileError(f"tts.checks.every must be at least {EVERY_MIN} seconds")
    return Profile(
        hash=digest, model=model, code=_int(tts, "code", 1, 1_000_000), backends=backends, dtypes=tuple(dtypes), settings=settings,
        mp3_kbps=_int(speech, "mp3_kbps", 32, 320), voices=voices, every=every, takes=_takes(checks.get("takes"), voices), document=data,
    )


def _voices(raw: object) -> dict[str, dict[str, Recording]]:
    if not isinstance(raw, list) or len(raw) > VOICES_MAX:
        raise ProfileError(f"tts.voices must be a list of at most {VOICES_MAX}")
    presets = {v.lower() for v in VOICES}
    voices: dict[str, dict[str, Recording]] = {}
    for v in raw:
        v = _obj(v, "a voice")
        name = v.get("id")
        if not isinstance(name, str) or not VOICE_ID.fullmatch(name):
            raise ProfileError("a voice's id does not fit the protocol")
        # A preset's name would be two voices behind one id; two that differ in case would be one at the project.
        if name.lower() in presets or name.lower() in {n.lower() for n in voices}:
            raise ProfileError(f"the voice {name} is named like another")
        langs = _obj(v.get("langs"), f"the voice {name}'s langs")
        if not 1 <= len(langs) <= len(LANGUAGES):
            raise ProfileError(f"the voice {name} needs a recording in 1 to {len(LANGUAGES)} languages")
        by_lang: dict[str, Recording] = {}
        for lang, rec in langs.items():
            rec = _obj(rec, f"{name}.{lang}")
            if not isinstance(lang, str) or not LANG.fullmatch(lang) or lang not in LANGUAGES:
                raise ProfileError(f"the voice {name}: {str(lang)[:8]!r} is no language Qwen speaks")
            file, size, text = rec.get("file"), rec.get("bytes"), rec.get("text")
            if not isinstance(file, str) or not SHA256.fullmatch(file):
                raise ProfileError(f"{name}.{lang}: its file is no SHA-256")
            if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= FILE_MAX:
                raise ProfileError(f"{name}.{lang}: its size must be 1 to {FILE_MAX} bytes")
            words = " ".join(text.split()) if isinstance(text, str) else ""
            if not words or len(words) > TEXT_MAX:
                raise ProfileError(f"{name}.{lang}: its words must be 1 to {TEXT_MAX} characters")
            by_lang[lang] = Recording(file=file, bytes=size, text=words)
        voices[name] = by_lang
    return voices


def _takes(raw: object, voices: dict[str, dict[str, Recording]]) -> tuple[CheckTake, ...]:
    if not isinstance(raw, list) or len(raw) > TAKES_MAX:
        raise ProfileError(f"tts.checks.takes must be a list of at most {TAKES_MAX}")
    known = {v.lower() for v in VOICES} | {v.lower() for v in voices}
    takes = []
    for t in raw:
        t = _obj(t, "a check take")
        voice, lang, text = t.get("voice"), t.get("lang"), t.get("text")
        if not isinstance(voice, str) or voice.lower() not in known:
            raise ProfileError(f"a check take names {str(voice)[:40]!r}, neither a preset nor one of the profile's voices")
        if not isinstance(lang, str) or lang not in LANGUAGES:
            raise ProfileError(f"a check take of {voice} is in no language Qwen speaks")
        if not isinstance(text, str) or not text.strip() or len(text) > TAKE_TEXT_MAX:
            raise ProfileError(f"a check take of {voice} must say 1 to {TAKE_TEXT_MAX} characters")
        takes.append(CheckTake(voice=voice, lang=lang, text=text.strip()))
    return tuple(takes)


def _obj(v: object, what: str) -> dict:
    if not isinstance(v, dict):
        raise ProfileError(f"{what} is not an object")
    return v


def _int(d: dict, key: str, low: int, high: int) -> int:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high:
        raise ProfileError(f"{key} must be a whole number from {low} to {high}")
    return v


def _num(d: dict, key: str, low: float, high: float, low_open: bool = False) -> float:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (low < v if low_open else low <= v) or v > high:
        raise ProfileError(f"{key} must be a number from {low} to {high}")
    return float(v)


def unmet(profile: Profile, backend: str, model: str, clone_model: str, dtype: str) -> str:
    """Why this worker cannot run the profile ("": it can). Its own pins and config decide what it loads, never the
    profile: `model` and `clone_model` are the checkpoints it loads, or would load for an own voice."""
    if profile.model != MODEL_NAME:
        return f"the profile is for {profile.model}; this worker speaks {MODEL_NAME}"
    if profile.code != CODE:
        older = profile.code > CODE
        return f"the profile expects speech code {profile.code}; this herde speaks {CODE}" + ("; update herde" if older else "")
    if backend not in profile.backends:
        return f"the profile admits {', '.join(sorted(profile.backends))}; this worker runs {backend}"
    want_model, want_clone = profile.backends[backend]
    if model != want_model:
        return f"this worker loads {model or 'no checkpoint'}; the profile expects {want_model}"
    if profile.voices and clone_model != want_clone:
        return f"this worker's Base model is {clone_model or 'none'}; the profile expects {want_clone}"
    # The stub loads no weights: nothing to run in any format.
    if backend != "stub" and dtype not in profile.dtypes:
        return f"this worker runs in {dtype or 'an unknown format'}; the profile admits {', '.join(profile.dtypes)}"
    return ""


def judge(pcm16: bytes, rate: int, text: str, silence_ok: bool = False) -> tuple[bool, float, str]:
    """A check take: whether it is speech as long as its words need, neither silent nor clipped — and its pace.
    `silence_ok`: the stub speaks silence on purpose (its length is all a project's tests can hear)."""
    samples = array.array("h")
    samples.frombytes(pcm16[: len(pcm16) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    seconds = len(samples) / rate if rate > 0 else 0.0
    cps = round(len(text) / seconds, 1) if seconds > 0 else 0.0
    if not CPS_MIN <= cps <= CPS_MAX:
        return False, cps, f"{cps} characters a second ({CPS_MIN:.0f} to {CPS_MAX:.0f} fit its words)"
    if not silence_ok and math.sqrt(sum(s * s for s in samples) / len(samples)) < SILENT_RMS:
        return False, cps, "silent"
    clipped = sum(1 for s in samples if s >= 32767 or s <= -32767) / len(samples)
    if clipped > CLIPPED_SHARE:
        return False, cps, f"clips ({clipped:.1%} of its samples at full scale)"
    return True, cps, ""


class ProfileStore:
    """One project's profile on disk: `profile.json` and `files/<sha256>.wav` under `<state>/projects/<name>/`.
    The owner's state folder is private, and so is this: a recording is the project's."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.files = folder / "files"

    def load(self) -> Profile | None:
        """The profile last applied, if it is still whole; else None (the next poll fetches it again)."""
        try:
            return parse(json.loads((self.folder / "profile.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    def save(self, profile: Profile) -> None:
        _write(self.folder / "profile.json", json.dumps(profile.document, ensure_ascii=False).encode())

    def path(self, sha: str) -> Path:
        return self.files / f"{sha}.wav"

    def has(self, sha: str) -> bool:
        return self.path(sha).is_file()

    def put(self, sha: str, data: bytes) -> None:
        if hashlib.sha256(data).hexdigest() != sha:
            raise ProfileError(f"the file {sha[:12]}… is not what its SHA-256 names")
        _write(self.path(sha), data)

    def voices(self, profile: Profile) -> dict[str, OwnVoice]:
        """The profile's voices, each recording checked like the owner's own (PCM WAV, 3–30 s) and against its hash."""
        out: dict[str, OwnVoice] = {}
        for name, langs in profile.voices.items():
            refs = {}
            for lang, rec in langs.items():
                try:
                    refs[lang] = reference(self.path(rec.file), rec.text, rec.file)
                except BadRecording as e:
                    raise ProfileError(f"{name}.{lang}: the recording {e}") from None
            out[name.lower()] = OwnVoice(name=name, refs=refs)
        return out

    def verify(self, profile: Profile) -> list[str]:
        """The files that are no longer what the profile names (gone, or changed): removed, so they are fetched again."""
        bad = []
        for sha in profile.files():
            path = self.path(sha)
            try:
                same = hashlib.sha256(path.read_bytes()).hexdigest() == sha
            except OSError:
                same = False
            if not same:
                bad.append(sha)
                path.unlink(missing_ok=True)
        return bad

    def prune(self, profile: Profile) -> None:
        """Recordings no longer named: a voice the project removed or took again."""
        keep = {f"{sha}.wav" for sha in profile.files()}
        if self.files.is_dir():
            for p in self.files.iterdir():
                if p.name not in keep:
                    p.unlink(missing_ok=True)

    def clear(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)


def folder_name(project: str) -> str:
    """A project's name as a folder: the config allows dots, and "." or ".." must not leave `projects/`."""
    return project if set(project) - {"."} else "_" * len(project)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(data)
    os.replace(tmp, path)
