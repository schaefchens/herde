import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# The package next to the tests, whichever directory pytest starts from (and whether or not it is installed).
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "protocol" / "fixtures"
SCHEMA = ROOT / "protocol" / "schema.json"
KEY = "s3cret-" + "k" * 57


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def all_fixtures() -> list[tuple[str, dict]]:
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(FIXTURES.glob("*.json"))]


@pytest.fixture
def no_env(monkeypatch):
    """No Docker defaults from the environment of whoever runs the tests."""
    for name in ("HERDE_CONFIG", "HERDE_TTS_BACKEND", "HERDE_OLLAMA", "HERDE_UI_HOST", "HERDE_CONTAINER"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def never_caffeinate(monkeypatch):
    """`herde run` re-executes itself under caffeinate on a Mac: in a test that would replace pytest itself."""
    monkeypatch.setenv("HERDE_AWAKE", "1")


# --- a project's profile (PROTOCOL.md, Profiles) --------------------------------------------------------------

SPEECH = {"chunk_chars": 120, "pause_ms": 300, "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.1, "mp3_kbps": 128}
FAITH = "Das war Mein Gott ist größer. Schön, dass ihr dabei seid."


def wav_bytes(seconds: float = 4.0, rate: int = 8000, amplitude: int = 0) -> bytes:
    """A mono 16-bit PCM WAV: silence, or a square wave of `amplitude` (speech-like enough for a level check)."""
    import io
    import wave

    frames = bytearray()
    for i in range(int(seconds * rate)):
        frames += (amplitude if (i // 20) % 2 else -amplitude).to_bytes(2, "little", signed=True)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return out.getvalue()


def profile_doc(voices=None, backends=None, speech=None, takes=None, every=21600, code=1, dtypes=("bf16",)) -> dict:
    """A project's `profile` answer, with its hash: the stub admitted, unless `backends` says otherwise."""
    import copy

    from herde.protocol import canonical_hash

    tts = copy.deepcopy({
        "model": "qwen3-tts-1.7b", "code": code, "backends": backends or {"stub": {"model": "", "clone_model": ""}},
        "dtypes": list(dtypes), "speech": speech or SPEECH, "voices": voices or [], "checks": {"every": every, "takes": takes or []},
    })
    return {"profile": canonical_hash({"tts": tts}), "tts": tts}


def voice_entry(name: str, recordings: dict[str, bytes], words: str = FAITH) -> dict:
    import hashlib

    return {"id": name, "langs": {lang: {"file": hashlib.sha256(data).hexdigest(), "bytes": len(data), "text": words} for lang, data in recordings.items()}}
