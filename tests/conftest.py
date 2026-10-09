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
