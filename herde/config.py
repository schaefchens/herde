"""The worker's config: which projects it serves, what it may run, and when.

A TOML file (`~/.config/herde/config.toml`, in Docker `/config/config.toml`).
Keys live in files of their own (`keys/<project>.key`, written by `herde
join`), readable by the owner only — a key fetches what a project's people
sent, prayer requests included. Never in an environment variable: `docker
inspect` shows those to anyone who can run docker.
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

from .engines.tts.core import CHUNK_CHARS, CHUNK_CHARS_MAX
from .protocol import KINDS, MODEL
from .windows import Window, WindowError, parse_all

DEFAULT_DIR = Path.home() / ".config" / "herde"
DEFAULT_PATH = DEFAULT_DIR / "config.toml"
# Set by the Docker image: file modes on a mounted folder say nothing there (Docker Desktop on Windows shows 0777).
IN_CONTAINER = os.environ.get("HERDE_CONTAINER") == "1"
# Plain http only to this machine (or, in Docker, a service beside it): anything else would send the key in the clear.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
PROJECT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,40}")
TTS_BACKENDS = ("mlx", "torch", "stub")
TEXT_BACKENDS = ("ollama", "stub")
# The checkpoints each backend loads, pinned where a revision is known to work.
TTS_DEFAULTS: dict[str, dict[str, str]] = {
    # The revisions the qwen3-tts-ui demo pins on the M1 Max: its download can be reused as is.
    "mlx": {
        "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16", "revision": "52f4770fd9726457eae3d3b6aa92047a25a10776",
        "clone_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", "clone_revision": "a6eb4f68e4b056f1215157bb696209bc82a6db48",
    },
    "torch": {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "revision": "",
        "clone_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "clone_revision": "",
    },
    "stub": {"model": "", "revision": "", "clone_model": "", "clone_revision": ""},
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    url: str
    key: str = field(repr=False)
    weight: int = 1
    kinds: tuple[str, ...] = KINDS
    # Model names this project may use; empty: every one this worker has.
    models: tuple[str, ...] = ()
    windows: tuple[Window, ...] = ()


@dataclass(frozen=True)
class TextModel:
    name: str
    tag: str
    context: int = 8192


@dataclass(frozen=True)
class TtsConfig:
    backend: str = ""
    model: str = ""
    revision: str = ""
    clone_model: str = ""
    clone_revision: str = ""
    hf_home: Path | None = None
    voices_dir: Path = DEFAULT_DIR / "voices"
    chunk_chars: int = CHUNK_CHARS
    device: str = "cuda:0"
    task_timeout_factor: float = 4.0


@dataclass(frozen=True)
class TextConfig:
    backend: str = ""
    ollama: str = "http://127.0.0.1:11434"
    keep_alive: str = "60m"
    models: tuple[TextModel, ...] = ()


@dataclass(frozen=True)
class Limits:
    tts_chars: int = 2000
    text_input_chars: int = 200_000
    text_max_tokens: int = 4096


@dataclass(frozen=True)
class Config:
    name: str
    state_dir: Path
    path: Path | None = None
    mp3_bitrate: str = "96k"
    keep_awake: bool = True
    gpu_slots: int = 1
    ui_port: int = 8737
    ui_host: str = "127.0.0.1"
    windows: tuple[Window, ...] = ()
    limits: Limits = Limits()
    tts: TtsConfig = TtsConfig()
    text: TextConfig = TextConfig()
    projects: tuple[ProjectConfig, ...] = ()


def config_path(arg: str | None = None) -> Path:
    return Path(arg or os.environ.get("HERDE_CONFIG") or DEFAULT_PATH).expanduser()


def load(path: Path) -> Config:
    if not path.is_file():
        raise ConfigError(f"No config at {path}. Join a project (herde join <url> <code>) or copy config.example.toml there.")
    _private(path, "it may hold keys")
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} is not valid TOML: {e}") from None
    return parse(data, path)


def parse(data: dict, path: Path | None = None) -> Config:
    folder = path.parent if path else DEFAULT_DIR
    state_dir = _path(data.get("state_dir"), "state_dir") or folder
    bitrate = str(data.get("mp3_bitrate", "96k"))
    if not re.fullmatch(r"\d{2,3}k", bitrate):
        raise ConfigError('mp3_bitrate must look like "96k"')
    keep_awake = data.get("keep_awake", True)
    if not isinstance(keep_awake, bool):
        raise ConfigError("keep_awake must be true or false")
    raw_projects = data.get("projects", [])
    if not isinstance(raw_projects, list):
        raise ConfigError("projects must be [[projects]] blocks")
    projects = tuple(_project(p, i, state_dir) for i, p in enumerate(raw_projects))
    if len({p.url for p in projects}) != len(projects) or len({p.name for p in projects}) != len(projects):
        raise ConfigError("Each project may appear only once, under a name of its own")
    return Config(
        name=str(data.get("name") or socket.gethostname())[:80],
        state_dir=state_dir,
        path=path,
        mp3_bitrate=bitrate,
        keep_awake=keep_awake,
        gpu_slots=_int(data, "gpu_slots", 1, 1, 8),
        ui_port=_int(data, "ui_port", 8737, 0, 65535),
        ui_host=str(data.get("ui_host", "127.0.0.1")),
        windows=_windows(data.get("windows", []), "windows"),
        limits=_limits(_table(data, "limits")),
        tts=_tts(_table(data, "tts"), folder),
        text=_text(_table(data, "text")),
        projects=projects,
    )


def _table(data: dict, name: str) -> dict:
    t = data.get(name, {})
    if not isinstance(t, dict):
        raise ConfigError(f"[{name}] must be a table")
    return t


def _int(data: dict, key: str, default: int, low: int, high: int, where: str = "") -> int:
    v = data.get(key, default)
    if not isinstance(v, int) or isinstance(v, bool) or not low <= v <= high:
        raise ConfigError(f"{where}{key} must be a whole number between {low} and {high}")
    return v


def _path(v: object, what: str) -> Path | None:
    if v is None or v == "":
        return None
    if not isinstance(v, str):
        raise ConfigError(f"{what} must be a folder")
    return Path(v).expanduser()


def _windows(v: object, where: str) -> tuple[Window, ...]:
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ConfigError(f'{where} must be a list like ["Mon-Fri 18:00-08:00", "Sat-Sun"]')
    try:
        return parse_all(v)
    except WindowError as e:
        raise ConfigError(f"{where}: {e}") from None


def _limits(t: dict) -> Limits:
    return Limits(
        tts_chars=_int(t, "tts_chars", 2000, 1, 20000, "[limits] "),
        text_input_chars=_int(t, "text_input_chars", 200_000, 1, 2_000_000, "[limits] "),
        text_max_tokens=_int(t, "text_max_tokens", 4096, 1, 65536, "[limits] "),
    )


def _tts(t: dict, folder: Path) -> TtsConfig:
    backend = str(t.get("backend", os.environ.get("HERDE_TTS_BACKEND", "")))
    if backend not in ("", *TTS_BACKENDS):
        raise ConfigError(f"[tts] backend must be one of {', '.join(TTS_BACKENDS)} (or \"\" for none)")
    defaults = TTS_DEFAULTS.get(backend, TTS_DEFAULTS["stub"])
    chunk = _int(t, "chunk_chars", CHUNK_CHARS, 50, CHUNK_CHARS_MAX, "[tts] ")
    factor = t.get("task_timeout_factor", 4.0)
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not 1 <= factor <= 20:
        raise ConfigError("[tts] task_timeout_factor must be a number between 1 and 20")
    return TtsConfig(
        backend=backend,
        model=str(t.get("model", defaults["model"])),
        revision=str(t.get("revision", defaults["revision"])),
        clone_model=str(t.get("clone_model", defaults["clone_model"])),
        clone_revision=str(t.get("clone_revision", defaults["clone_revision"])),
        hf_home=_path(t.get("hf_home"), "[tts] hf_home"),
        # Our own voices' recordings: next to the config unless [tts] voices says otherwise.
        voices_dir=_path(t.get("voices"), "[tts] voices") or folder / "voices",
        chunk_chars=chunk,
        device=str(t.get("device", "cuda:0")),
        task_timeout_factor=float(factor),
    )


def _text(t: dict) -> TextConfig:
    backend = str(t.get("backend", "ollama" if t.get("models") else ""))
    if backend not in ("", *TEXT_BACKENDS):
        raise ConfigError(f"[text] backend must be one of {', '.join(TEXT_BACKENDS)} (or \"\" for none)")
    url = str(t.get("ollama", os.environ.get("HERDE_OLLAMA", "http://127.0.0.1:11434"))).rstrip("/")
    if urlparse(url).scheme not in ("http", "https"):
        raise ConfigError("[text] ollama must be an address like http://127.0.0.1:11434")
    raw = t.get("models", [])
    if not isinstance(raw, list):
        raise ConfigError("[text] models must be [[text.models]] blocks")
    models = []
    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            raise ConfigError(f"[[text.models]] {i + 1} must be a table")
        name, tag = str(m.get("name", "")), str(m.get("tag", ""))
        if not MODEL.fullmatch(name):
            raise ConfigError(f"[[text.models]] {i + 1}: name must be like gemma4-31b (letters, digits, . _ -)")
        if not tag or len(tag) > 200 or re.search(r"\s", tag):
            raise ConfigError(f"[[text.models]] {name}: tag must be Ollama's name for it, like gemma4:31b")
        models.append(TextModel(name=name, tag=tag, context=_int(m, "context", 8192, 512, 1_048_576, f"[[text.models]] {name}: ")))
    if len({m.name for m in models}) != len(models):
        raise ConfigError("[[text.models]]: each name only once")
    return TextConfig(backend=backend, ollama=url, keep_alive=str(t.get("keep_alive", "60m")), models=tuple(models))


def _project(raw: object, index: int, state_dir: Path) -> ProjectConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"Project {index + 1} must be a table")
    url = str(raw.get("url", "")).strip().rstrip("/")
    parsed = urlparse(url)
    label = str(raw.get("name") or parsed.hostname or f"project {index + 1}")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError(f"{label}: url must be the project's worker address, like https://radio.example.org/api/worker/v2")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError(f"{label}: url must have no login, query or fragment")
    if parsed.scheme == "http" and parsed.hostname not in LOCAL_HOSTS and not _service_name(parsed.hostname):
        raise ConfigError(f"{label}: use https (plain http is only for this machine)")
    if not PROJECT_NAME.fullmatch(label):
        raise ConfigError(f"{label}: a project's name may hold letters, digits and . _ - (at most 40)")
    if "key" in raw:
        key = str(raw["key"])
    elif "key_file" in raw:
        key = read_key(_key_path(str(raw["key_file"]), state_dir), label)
    else:
        raise ConfigError(f"{label}: needs key_file (herde join writes one)")
    if len(key) < 32 or re.search(r"\s", key):
        raise ConfigError(f"{label}: the key must be the one the project gave this worker")
    kinds = raw.get("kinds", list(KINDS))
    if not isinstance(kinds, list) or not kinds or any(k not in KINDS for k in kinds):
        raise ConfigError(f"{label}: kinds must be a list of {', '.join(KINDS)}")
    models = raw.get("models", [])
    if not isinstance(models, list) or any(not isinstance(m, str) or not MODEL.fullmatch(m) for m in models):
        raise ConfigError(f"{label}: models must be a list of model names")
    return ProjectConfig(
        name=label, url=url, key=key, weight=_int(raw, "weight", 1, 1, 100, f"{label}: "),
        kinds=tuple(k for k in KINDS if k in kinds), models=tuple(models), windows=_windows(raw.get("windows", []), f"{label}: windows"),
    )


def _service_name(host: str) -> bool:
    """A name without dots, like "arche": a container beside this one on its Docker network."""
    return IN_CONTAINER and "." not in host


def _key_path(value: str, state_dir: Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else state_dir / p


def read_key(path: Path, label: str) -> str:
    if not path.is_file():
        raise ConfigError(f"{label}: no key file at {path}")
    _private(path, "it is a key")
    return path.read_text(encoding="utf-8").strip()


def _private(path: Path, why: str) -> None:
    if IN_CONTAINER:
        return
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigError(f"{path} can be read by others; {why}. Run: chmod 600 {path}")
