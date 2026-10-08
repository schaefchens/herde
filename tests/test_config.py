import os

import pytest

from arche_worker.config import DEFAULT_MODEL, DEFAULT_REVISION, ConfigError, config_path, load, parse

KEY = "k" * 64


def station(**over):
    return {"name": "production", "url": "https://radio.example.org", "key": KEY} | over


def test_a_config_with_one_station_and_the_defaults():
    c = parse({"stations": [station(url="https://radio.example.org/")]})
    assert c.stations[0].url == "https://radio.example.org"
    assert (c.model, c.revision, c.mp3_bitrate, c.keep_awake, c.task_timeout_factor) == (DEFAULT_MODEL, DEFAULT_REVISION, "96k", True, 4.0)
    assert KEY not in repr(c), "a key never shows in a repr (and so in no log line)"


def test_plain_http_only_for_this_machine():
    assert parse({"stations": [station(url="http://localhost:8080")]}).stations[0].url == "http://localhost:8080"
    assert parse({"stations": [station(url="http://127.0.0.1:8080")]})
    with pytest.raises(ConfigError, match="https"):
        parse({"stations": [station(url="http://radio.example.org")]})


@pytest.mark.parametrize("url", ["https://user:pw@radio.example.org", "https://radio.example.org/?x=1", "https://radio.example.org/#f", "radio.example.org", "ftp://radio.example.org"])
def test_odd_addresses_are_refused(url):
    with pytest.raises(ConfigError):
        parse({"stations": [station(url=url)]})


@pytest.mark.parametrize("key", ["", "short", "k" * 31, "k" * 20 + " " + "k" * 20])
def test_a_key_must_be_the_one_mod_showed(key):
    with pytest.raises(ConfigError, match="key"):
        parse({"stations": [station(key=key)]})


def test_stations_are_needed_once_each():
    with pytest.raises(ConfigError, match="stations"):
        parse({})
    with pytest.raises(ConfigError, match="once"):
        parse({"stations": [station(), station(name="again")]})


def test_how_long_a_take_may_be_is_a_setting():
    assert parse({"stations": [station()]}).chunk_chars == 300, "as the demo did, until the lab decides"
    assert parse({"stations": [station()], "engine": {"chunk_chars": 1100}}).chunk_chars == 1100
    for bad in (10, 5000, "long", True):
        with pytest.raises(ConfigError, match="chunk_chars"):
            parse({"stations": [station()], "engine": {"chunk_chars": bad}})


def test_settings_are_checked():
    with pytest.raises(ConfigError):
        parse({"stations": [station()], "mp3_bitrate": "loud"})
    with pytest.raises(ConfigError):
        parse({"stations": [station()], "task_timeout_factor": 0.5})
    with pytest.raises(ConfigError):
        parse({"stations": [station()], "keep_awake": "yes"})


def test_a_file_others_can_read_is_refused(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(f'[[stations]]\nname = "p"\nurl = "https://radio.example.org"\nkey = "{KEY}"\n')
    path.chmod(0o644)
    with pytest.raises(ConfigError, match="chmod 600"):
        load(path)
    path.chmod(0o600)
    assert load(path).stations[0].name == "p"


def test_no_file_says_what_to_do(tmp_path):
    with pytest.raises(ConfigError, match="config.example.toml"):
        load(tmp_path / "missing.toml")


def test_the_path_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCHE_WORKER_CONFIG", str(tmp_path / "c.toml"))
    assert config_path() == tmp_path / "c.toml"
    assert config_path(str(tmp_path / "x.toml")) == tmp_path / "x.toml"


def test_own_voices_live_next_to_the_config_unless_told_otherwise(tmp_path):
    from arche_worker.config import DEFAULT_CLONE_MODEL, DEFAULT_CLONE_REVISION

    c = parse({"stations": [station()]}, tmp_path / "config.toml")
    assert (c.clone_model, c.clone_revision, c.voices_dir) == (DEFAULT_CLONE_MODEL, DEFAULT_CLONE_REVISION, tmp_path / "voices")
    c = parse({"stations": [station()], "clone": {"voices": "~/voices", "model": "local/base", "revision": "abc"}})
    assert (c.clone_model, c.clone_revision, c.voices_dir) == ("local/base", "abc", __import__("pathlib").Path.home() / "voices")
    with pytest.raises(ConfigError, match=r"\[clone\]"):
        parse({"stations": [station()], "clone": "voices"})
    with pytest.raises(ConfigError, match="folder"):
        parse({"stations": [station()], "clone": {"voices": ""}})
