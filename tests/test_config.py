import os
from pathlib import Path

import pytest
from conftest import KEY

from herde.config import TTS_DEFAULTS, ConfigError, config_path, load, parse

URL = "https://radio.example.org/api/worker/v2"


def project(**over):
    return {"name": "arche", "url": URL, "key": KEY} | over


def test_a_config_with_one_project_and_the_defaults(no_env):
    c = parse({"projects": [project(url=URL + "/")]})
    p = c.projects[0]
    assert (p.url, p.weight, p.kinds, p.models, p.windows) == (URL, 1, ("tts", "text"), (), ())
    assert (c.mp3_bitrate, c.keep_awake, c.gpu_slots, c.ui_port, c.ui_host) == ("96k", True, 1, 8737, "127.0.0.1")
    assert c.tts.backend == "" and c.text.backend == "", "nothing runs that the owner did not choose"
    assert KEY not in repr(c), "a key never shows in a repr (and so in no log line)"


def test_plain_http_only_for_this_machine():
    assert parse({"projects": [project(url="http://localhost:8080/api/worker/v2")]}).projects[0].url == "http://localhost:8080/api/worker/v2"
    assert parse({"projects": [project(url="http://127.0.0.1:8080/api/worker/v2")]})
    with pytest.raises(ConfigError, match="https"):
        parse({"projects": [project(url="http://radio.example.org/api/worker/v2")]})


def test_a_service_name_beside_it_is_local_only_in_a_container(monkeypatch):
    with pytest.raises(ConfigError, match="https"):
        parse({"projects": [project(url="http://arche/api/worker/v2")]})
    monkeypatch.setattr("herde.config.IN_CONTAINER", True)
    assert parse({"projects": [project(url="http://arche/api/worker/v2")]})


@pytest.mark.parametrize("url", [URL.replace("https://", "https://user:pw@"), URL + "?x=1", URL + "#f", "radio.example.org", "ftp://radio.example.org"])
def test_odd_addresses_are_refused(url):
    with pytest.raises(ConfigError):
        parse({"projects": [project(url=url)]})


@pytest.mark.parametrize("key", ["", "short", "k" * 31, "k" * 20 + " " + "k" * 20])
def test_a_key_must_be_the_one_the_project_gave(key):
    with pytest.raises(ConfigError, match="key"):
        parse({"projects": [project(key=key)]})


def test_keys_live_in_files_only_their_owner_can_read(tmp_path, monkeypatch):
    monkeypatch.setattr("herde.config.IN_CONTAINER", False)
    (tmp_path / "keys").mkdir()
    key = tmp_path / "keys" / "arche.key"
    key.write_text(KEY + "\n")
    key.chmod(0o644)
    raw = {"name": "arche", "url": URL, "key_file": "keys/arche.key"}
    with pytest.raises(ConfigError, match="chmod 600"):
        parse({"projects": [raw]}, tmp_path / "config.toml")
    key.chmod(0o600)
    assert parse({"projects": [raw]}, tmp_path / "config.toml").projects[0].key == KEY, "relative to the state folder, trimmed"
    with pytest.raises(ConfigError, match="no key file"):
        parse({"projects": [dict(raw, key_file="keys/other.key")]}, tmp_path / "config.toml")
    with pytest.raises(ConfigError, match="key_file"):
        parse({"projects": [{"name": "arche", "url": URL}]})


def test_projects_are_needed_once_each_under_names_of_their_own():
    assert parse({}).projects == (), "a config without projects reads (join adds them)"
    with pytest.raises(ConfigError, match="once"):
        parse({"projects": [project(), project(name="again")]})
    with pytest.raises(ConfigError, match="once"):
        parse({"projects": [project(), project(url=URL.replace("radio", "other"))]})
    with pytest.raises(ConfigError, match="name"):
        parse({"projects": [project(name="two words")]})


def test_what_a_project_may_use():
    p = parse({"projects": [project(weight=3, kinds=["text"], models=["gemma4-31b"], windows=["Mon-Fri 18:00-08:00"])]}).projects[0]
    assert (p.weight, p.kinds, p.models, p.windows[0].text) == (3, ("text",), ("gemma4-31b",), "Mon-Fri 18:00-08:00")
    for bad in ({"weight": 0}, {"kinds": ["video"]}, {"kinds": []}, {"models": ["gemma 4"]}, {"windows": ["Someday"]}):
        with pytest.raises(ConfigError):
            parse({"projects": [project(**bad)]})


def test_speech_settings_and_their_defaults_per_backend(no_env, tmp_path):
    c = parse({"tts": {"backend": "mlx"}}, tmp_path / "config.toml")
    assert (c.tts.model, c.tts.revision) == (TTS_DEFAULTS["mlx"]["model"], TTS_DEFAULTS["mlx"]["revision"])
    assert c.tts.voices_dir == tmp_path / "voices", "own voices next to the config"
    assert c.tts.chunk_chars == 300, "as the demo did, until the lab decides"
    t = parse({"tts": {"backend": "torch", "chunk_chars": 1100, "voices": "~/voices", "device": "cuda:1"}}).tts
    assert (t.model, t.chunk_chars, t.voices_dir, t.device) == ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 1100, Path.home() / "voices", "cuda:1")
    for bad in ({"backend": "coqui"}, {"chunk_chars": 10}, {"chunk_chars": 5000}, {"chunk_chars": True}, {"task_timeout_factor": 0.5}):
        with pytest.raises(ConfigError):
            parse({"tts": bad})


def test_the_docker_image_chooses_defaults_through_the_environment(no_env, monkeypatch):
    monkeypatch.setenv("HERDE_TTS_BACKEND", "torch")
    monkeypatch.setenv("HERDE_OLLAMA", "http://ollama:11434")
    c = parse({"text": {"models": [{"name": "gemma4-31b", "tag": "gemma4:31b"}]}})
    assert (c.tts.backend, c.text.backend, c.text.ollama) == ("torch", "ollama", "http://ollama:11434")
    assert parse({"tts": {"backend": ""}}).tts.backend == "", "the config has the last word"


def test_text_models_map_short_names_to_ollamas_tags():
    t = parse({"text": {"models": [{"name": "gemma4-31b", "tag": "gemma4:31b-mlx", "context": 32768}], "keep_alive": "2h"}}).text
    assert (t.backend, t.models[0].tag, t.models[0].context, t.keep_alive) == ("ollama", "gemma4:31b-mlx", 32768, "2h")
    for bad in ([{"name": "gemma 4", "tag": "x"}], [{"name": "g", "tag": ""}], [{"name": "g", "tag": "a b"}], [{"name": "g", "tag": "x"}, {"name": "g", "tag": "y"}]):
        with pytest.raises(ConfigError):
            parse({"text": {"models": bad}})
    with pytest.raises(ConfigError):
        parse({"text": {"backend": "llamafile"}})


def test_settings_are_checked():
    for bad in ({"mp3_bitrate": "loud"}, {"keep_awake": "yes"}, {"gpu_slots": 0}, {"ui_port": 70000}, {"windows": "Someday 25:00-26:00"},
                {"limits": {"tts_chars": 0}}, {"limits": "big"}):
        with pytest.raises(ConfigError):
            parse(bad)
    assert parse({"limits": {"tts_chars": 3000}}).limits.tts_chars == 3000


def test_a_file_others_can_read_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("herde.config.IN_CONTAINER", False)
    path = tmp_path / "config.toml"
    path.write_text(f'[[projects]]\nname = "p"\nurl = "{URL}"\nkey = "{KEY}"\n')
    path.chmod(0o644)
    with pytest.raises(ConfigError, match="chmod 600"):
        load(path)
    path.chmod(0o600)
    assert load(path).projects[0].name == "p"
    monkeypatch.setattr("herde.config.IN_CONTAINER", True)
    path.chmod(0o777)
    assert load(path), "a mounted folder's modes say nothing in a container (Docker Desktop on Windows shows 0777)"


def test_no_file_says_what_to_do(tmp_path):
    with pytest.raises(ConfigError, match="herde join"):
        load(tmp_path / "missing.toml")


def test_the_path_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("HERDE_CONFIG", str(tmp_path / "c.toml"))
    assert config_path() == tmp_path / "c.toml"
    assert config_path(str(tmp_path / "x.toml")) == tmp_path / "x.toml"
    monkeypatch.delenv("HERDE_CONFIG")
    assert config_path() == Path(os.path.expanduser("~/.config/herde/config.toml"))
