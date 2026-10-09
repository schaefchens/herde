"""The owner's commands: join a project, pause and resume, take over an arche-worker config."""

import stat
import tomllib

import pytest
from conftest import KEY

from herde import cli
from herde.config import load
from herde.protocol import Joined
from herde.state import State

URL = "https://radio.example.org/api/worker/v2"


@pytest.fixture
def joined(monkeypatch):
    calls = []

    async def fake_join(url, code, name, version, platform, transport=None):
        calls.append((url, code, name))
        return Joined(project="ARCHE", worker_id=7, worker_name=name, trust="lender", key=KEY)

    monkeypatch.setattr(cli, "join", fake_join)
    return calls


def test_join_writes_a_private_key_file_and_a_project_block(tmp_path, joined, no_env, capsys):
    path = tmp_path / "config.toml"
    assert cli.main(["--config", str(path), "join", URL, "7KQ3-M2XP", "--name", "Peters-PC"]) == 0
    assert joined == [(URL, "7KQ3-M2XP", "Peters-PC")]
    key = tmp_path / "keys" / "arche.key"
    assert key.read_text().strip() == KEY
    assert stat.S_IMODE(key.stat().st_mode) == 0o600 and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(key.parent.stat().st_mode) == 0o700
    config = load(path)
    assert (config.name, config.projects[0].name, config.projects[0].url, config.projects[0].key) == ("Peters-PC", "arche", URL, KEY)
    assert KEY not in path.read_text(), "the key lives in its own file"
    assert "lender" in capsys.readouterr().out


def test_joining_again_renews_the_key_and_keeps_one_block(tmp_path, joined, no_env):
    path = tmp_path / "config.toml"
    cli.main(["--config", str(path), "join", URL, "CODE-1"])
    text = path.read_text()
    path.write_text(text.replace('# weight = 1', 'weight = 3  # the owner wrote this'))
    assert cli.main(["--config", str(path), "join", URL, "CODE-2"]) == 0
    assert path.read_text().count("[[projects]]") == 1 and "the owner wrote this" in path.read_text(), "comments and edits kept"
    assert cli.main(["--config", str(path), "join", URL.replace("radio", "other"), "CODE-3"]) == 0
    names = [p.name for p in load(path).projects]
    assert names == ["arche", "arche-2"], "a second project of the same name gets its own"


def test_join_refuses_a_plain_http_address_elsewhere(tmp_path, joined, no_env):
    assert cli.main(["--config", str(tmp_path / "c.toml"), "join", "http://radio.example.org/api/worker/v2", "CODE"]) == 2
    assert joined == [] and not (tmp_path / "c.toml").exists()


def test_the_docker_image_sets_up_its_first_config_through_the_environment(tmp_path, joined, monkeypatch):
    monkeypatch.setenv("HERDE_TTS_BACKEND", "torch")
    monkeypatch.setenv("HERDE_OLLAMA", "http://ollama:11434")
    monkeypatch.setenv("HERDE_UI_HOST", "0.0.0.0")
    path = tmp_path / "config.toml"
    cli.main(["--config", str(path), "join", URL, "CODE"])
    data = tomllib.loads(path.read_text())
    assert data["tts"]["backend"] == "torch" and data["text"]["ollama"] == "http://ollama:11434" and data["ui_host"] == "0.0.0.0"


def test_pause_and_resume_from_another_process(tmp_path, capsys):
    path = tmp_path / "config.toml"
    assert cli.main(["--config", str(path), "pause", "2h"]) == 0
    paused, until = State(tmp_path).paused()
    assert paused and until is not None
    assert cli.main(["--config", str(path), "pause"]) == 0
    assert State(tmp_path).paused() == (True, None)
    assert cli.main(["--config", str(path), "resume"]) == 0
    assert State(tmp_path).paused() == (False, None)
    assert cli.main(["--config", str(path), "pause", "someday"]) == 2
    assert cli.main(["--config", str(path), "status"]) == 0
    assert "has not run" in capsys.readouterr().out


def test_an_arche_worker_config_is_taken_over(tmp_path, no_env):
    old_dir = tmp_path / "arche-worker"
    (old_dir / "voices").mkdir(parents=True)
    (old_dir / "voices" / "Faith.de.txt").write_text("Words.")
    old = old_dir / "config.toml"
    old.write_text(
        'name = "Studio-Mac"\nmp3_bitrate = "128k"\n[engine]\nchunk_chars = 1100\nhf_home = "~/hf"\n'
        f'[[stations]]\nname = "production"\nurl = "https://radio.schaefchens.de"\nkey = "{KEY}"\n'
        f'[[stations]]\nname = "dev"\nurl = "http://localhost:8080/"\nkey = "{"d" * 64}"\n'
    )
    path = tmp_path / "herde" / "config.toml"
    assert cli.main(["--config", str(path), "import-arche-worker", str(old)]) == 0
    c = load(path)
    assert [(p.name, p.url) for p in c.projects] == [("production", "https://radio.schaefchens.de/api/worker/v2"), ("dev", "http://localhost:8080/api/worker/v2")]
    assert c.projects[0].key == KEY and c.projects[1].key == "d" * 64
    assert (c.name, c.mp3_bitrate, c.tts.backend, c.tts.chunk_chars) == ("Studio-Mac", "128k", "mlx", 1100)
    assert (tmp_path / "herde" / "voices" / "Faith.de.txt").read_text() == "Words."
    assert cli.main(["--config", str(path), "import-arche-worker", str(old)]) == 0
    assert len(load(path).projects) == 2, "importing twice adds nothing"


def test_run_needs_a_project_and_something_to_offer(tmp_path, no_env):
    path = tmp_path / "config.toml"
    path.write_text("[tts]\nbackend = \"stub\"\n")
    path.chmod(0o600)
    assert cli.main(["--config", str(path), "run"]) == 2
    path.write_text(f'[[projects]]\nname = "p"\nurl = "{URL}"\nkey = "{KEY}"\n')
    assert cli.main(["--config", str(path), "run"]) == 2
