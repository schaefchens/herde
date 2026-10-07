"""The worker's config: which stations it serves, with which key, and how it speaks.

A TOML file readable by its owner only — a key fetches what listeners sent
to the station, prayer requests included.
"""

from __future__ import annotations

import os
import re
import socket
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .engine import CHUNK_CHARS, CHUNK_CHARS_MAX

DEFAULT_PATH = Path.home() / ".config" / "arche-worker" / "config.toml"
DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
# The revision the qwen3-tts-ui demo pins: its download can be reused as is.
DEFAULT_REVISION = "52f4770fd9726457eae3d3b6aa92047a25a10776"
# Plain http only to this machine: anything else would send the key in the clear.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Station:
    name: str
    url: str
    key: str = field(repr=False)


@dataclass(frozen=True)
class Config:
    name: str
    model: str
    revision: str
    hf_home: Path | None
    mp3_bitrate: str
    keep_awake: bool
    task_timeout_factor: float
    stations: tuple[Station, ...]
    path: Path | None = None
    chunk_chars: int = CHUNK_CHARS


def config_path(arg: str | None = None) -> Path:
    return Path(arg or os.environ.get("ARCHE_WORKER_CONFIG") or DEFAULT_PATH).expanduser()


def load(path: Path) -> Config:
    if not path.is_file():
        raise ConfigError(f"No config at {path}. Copy worker/config.example.toml there and fill in the station keys.")
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigError(f"{path} can be read by others; it holds station keys. Run: chmod 600 {path}")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} is not valid TOML: {e}") from None
    return parse(data, path)


def parse(data: dict, path: Path | None = None) -> Config:
    engine = data.get("engine", {})
    if not isinstance(engine, dict):
        raise ConfigError("[engine] must be a table")
    raw_stations = data.get("stations")
    if not isinstance(raw_stations, list) or not raw_stations:
        raise ConfigError("Add at least one [[stations]] block with name, url and key")
    stations = tuple(_station(s, i) for i, s in enumerate(raw_stations))
    if len({s.url for s in stations}) != len(stations):
        raise ConfigError("Each station may appear only once")

    bitrate = str(data.get("mp3_bitrate", "96k"))
    if not re.fullmatch(r"\d{2,3}k", bitrate):
        raise ConfigError('mp3_bitrate must look like "96k"')
    factor = data.get("task_timeout_factor", 4.0)
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not 1 <= factor <= 20:
        raise ConfigError("task_timeout_factor must be a number between 1 and 20")
    keep_awake = data.get("keep_awake", True)
    if not isinstance(keep_awake, bool):
        raise ConfigError("keep_awake must be true or false")
    hf_home = engine.get("hf_home")
    chunk = engine.get("chunk_chars", CHUNK_CHARS)
    if not isinstance(chunk, int) or isinstance(chunk, bool) or not 50 <= chunk <= CHUNK_CHARS_MAX:
        raise ConfigError(f"[engine] chunk_chars must be a whole number between 50 and {CHUNK_CHARS_MAX}")
    return Config(
        name=str(data.get("name") or socket.gethostname())[:80],
        model=str(engine.get("model", DEFAULT_MODEL)),
        revision=str(engine.get("revision", DEFAULT_REVISION)),
        hf_home=Path(str(hf_home)).expanduser() if hf_home else None,
        mp3_bitrate=bitrate,
        keep_awake=keep_awake,
        task_timeout_factor=float(factor),
        stations=stations,
        path=path,
        chunk_chars=chunk,
    )


def _station(raw: object, index: int) -> Station:
    if not isinstance(raw, dict):
        raise ConfigError(f"Station {index + 1} must be a table")
    url = str(raw.get("url", "")).strip().rstrip("/")
    parsed = urlparse(url)
    label = str(raw.get("name") or parsed.hostname or f"station {index + 1}")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError(f"{label}: url must be like https://radio.example.org")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError(f"{label}: url must have no login, query or fragment")
    if parsed.scheme == "http" and parsed.hostname not in LOCAL_HOSTS:
        raise ConfigError(f"{label}: use https (plain http is only for this machine)")
    key = str(raw.get("key", ""))
    if len(key) < 32 or re.search(r"\s", key):
        raise ConfigError(f"{label}: key must be the key /mod showed when the worker was added")
    return Station(name=label[:80], url=url, key=key)
