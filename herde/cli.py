"""herde's commands: run, check, join, pause, resume, status, bench, lab (and import-arche-worker).

`run` serves every project in the config until stopped (SIGTERM gives running
tasks back first). The rest are for the computer's owner: join a project with
its invite, pause for a while, see what the worker does, measure this
computer's speed, or compare voices by ear.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import socket
import sys
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path

from . import VERSION
from .config import Config, ConfigError, ProjectConfig, config_path, load, parse
from .connection import Connection, ProjectError, join
from .engines.text import build as build_text
from .engines.tts import build as build_tts
from .engines.tts.mp3 import EncodeError, encode, find_ffmpeg
from .estimates import Speeds
from .machine import platform_info
from .protocol import KIND_VERSIONS, poll_body
from .serve import DEFAULT_PORT, default_data, serve_command
from .state import PauseError, State, parse_pause
from .worker import Link, Worker

log = logging.getLogger("herde")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="herde", description="Do AI work — speech, text — for projects that host the herde protocol.")
    parser.add_argument("--config", help="config file (default ~/.config/herde/config.toml, or $HERDE_CONFIG)")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="serve every project in the config (the default)")
    sub.add_parser("check", help="the config, ffmpeg, Ollama and each project, without loading a model")
    j = sub.add_parser("join", help="join a project with the invite its admin made")
    j.add_argument("url", help="the project's worker address, e.g. https://radio.example.org/api/worker/v2")
    j.add_argument("code", help="the one-time invite code")
    j.add_argument("--name", help="how this computer shows up to the project (default: the config's name)")
    j.add_argument("--as", dest="as_", help="the project's name in this config (default: the name it gives)")
    p = sub.add_parser("pause", help="take no new work: 2h, 30m, until 18:00, or until resumed")
    p.add_argument("spec", nargs="*", help="e.g. 2h; nothing: until `herde resume`")
    sub.add_parser("resume", help="take work again")
    s = sub.add_parser("status", help="what the worker does now")
    s.add_argument("--json", action="store_true")
    sub.add_parser("bench", help="measure this computer's speed with the real models (they load first)")
    lab_p = sub.add_parser("lab", help="speak a cases file in several ways, to compare by ear")
    lab_p.add_argument("cases", help="a cases file (ARCHE's bin/replay-show.php writes one)")
    lab_p.add_argument("--out", help="where the clips go (default ~/herde-lab/<date-time>)")
    lab_p.add_argument("--chunks", default="300,whole,sentence", help="how a text is cut into takes: sizes, 'whole', 'sentence'")
    lab_p.add_argument("--temperatures", default="0.7", help="e.g. 0.6,0.7,0.8")
    lab_p.add_argument("--only", default="", help="only the cases whose id contains this")
    imp = sub.add_parser("import-arche-worker", help="take over an arche-worker config: its keys, voices and speech settings")
    imp.add_argument("old", nargs="?", default=str(Path.home() / ".config" / "arche-worker" / "config.toml"))
    sv = sub.add_parser("serve", help="a little project of your own: invite workers, give them jobs, hear and read what they make")
    sv.add_argument("--host", default=os.environ.get("HERDE_SERVE_HOST", "127.0.0.1"), help="where the page listens (default: this computer only)")
    sv.add_argument("--port", type=int, default=DEFAULT_PORT)
    sv.add_argument("--data", help="its database and clips (default ~/.local/share/herde-serve, or $HERDE_SERVE_DATA)")
    sv.add_argument("--name", default="herde serve", help="how the project names itself to its workers")
    sv.add_argument("--public-url", default="", help="the https address workers on other computers reach it at (behind a reverse proxy)")
    args = parser.parse_args(argv)
    _logging(args.verbose)
    command = args.command or "run"
    if command == "serve":
        return serve_command(args.host, args.port, Path(args.data).expanduser() if args.data else default_data(), args.name, args.public_url)
    path = config_path(args.config)

    if command == "join":
        return join_command(path, args)
    if command == "import-arche-worker":
        return import_arche_worker(Path(args.old).expanduser(), path)
    if command in ("pause", "resume", "status"):
        state = State(_state_dir(path))
        if command == "pause":
            return pause_command(state, args.spec)
        if command == "resume":
            state.resume()
            print("Resumed: this computer takes work again.")
            return 0
        return status_command(state, args.json)
    try:
        config = load(path)
    except ConfigError as e:
        log.error("%s", e)
        return 2
    if command == "check":
        return check(config)
    if command == "bench":
        return bench(config)
    if command == "lab":
        return lab_command(config, args)
    if not config.projects:
        log.error("No project in %s: join one first (herde join <url> <code>)", path)
        return 2
    if not config.tts.backend and not config.text.backend:
        # A worker with nothing to offer would only be told to upgrade (it declares no kind a project gives out).
        log.error("Nothing to offer: set [tts] backend, or [[text.models]], in %s", path)
        return 2
    if config.keep_awake:
        _keep_awake(argv)
    return run(config)


def _logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx logs every request URL at INFO; a key never goes into a URL, but the noise helps nobody.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _keep_awake(argv: list[str]) -> None:
    """Run again under `caffeinate -i`: an idle Mac must not sleep while it serves its projects."""
    if sys.platform != "darwin" or os.environ.get("HERDE_AWAKE") == "1":
        return
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        return
    os.environ["HERDE_AWAKE"] = "1"
    os.execv(caffeinate, [caffeinate, "-i", sys.executable, "-m", "herde", *argv])


def _state_dir(path: Path) -> Path:
    try:
        return load(path).state_dir
    except ConfigError:
        return path.parent


# --- run ---------------------------------------------------------------------------------------------------

def build(config: Config) -> Worker:
    state = State(config.state_dir)
    tts = build_tts(config.tts, config.limits.tts_chars)
    text = build_text(config.text, config.limits.text_max_tokens)
    speeds = Speeds(state, config.tts.backend)
    links = [Link(p, Connection(p.name, p.url, p.key)) for p in config.projects]
    return Worker(config, links, tts, text, state, speeds, platform=platform_info())


def run(config: Config) -> int:
    if config.tts.backend:
        try:
            find_ffmpeg()
        except EncodeError as e:
            log.error("%s", e)
            return 1
    worker = build(config)
    log.info("herde %s on %s: serving %s", VERSION, config.name, ", ".join(p.name for p in config.projects))
    try:
        asyncio.run(_serve(config, worker))
    except KeyboardInterrupt:
        return 130
    return 0


async def _serve(config: Config, worker: Worker) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.stop)
    page = None
    if config.ui_port:
        from .ui import StatusPage

        try:
            page = StatusPage(config.ui_host, config.ui_port, worker.state, worker.status, lambda: loop.call_soon_threadsafe(worker.wake.set))
            page.start()
            log.info("Status page: http://127.0.0.1:%d", page.port)
        except OSError as e:
            log.warning("No status page on port %d (%s)", config.ui_port, e)
            page = None
    try:
        await worker.run()
    finally:
        if page is not None:
            page.stop()


# --- check -------------------------------------------------------------------------------------------------

def check(config: Config) -> int:
    """The config, ffmpeg, Ollama and each project's answer — without loading a model or taking a task."""
    ok = True
    print(f"config: {config.path} ({len(config.projects)} project{'s' if len(config.projects) != 1 else ''}), state in {config.state_dir}")
    tts = build_tts(config.tts, config.limits.tts_chars)
    if tts is not None:
        print(f"speech: {config.tts.backend} ({config.tts.model or 'no model'})")
        try:
            print(f"ffmpeg: {find_ffmpeg()}")
        except EncodeError as e:
            print(f"ffmpeg: {e}")
            ok = False
        listed = ", ".join(f"{o.name} ({', '.join(o.languages)})" for o in tts.own.values())
        print(f"own voices: {listed or 'none'} in {config.tts.voices_dir}")
    else:
        print("speech: none ([tts] backend)")
    text = build_text(config.text, config.limits.text_max_tokens)
    if text is not None:
        asyncio.run(text.refresh())
        if not text.reachable:
            print(f"text: Ollama at {config.text.ollama} does not answer")
            ok = False
        for name, m in sorted(text.models.items()):
            print(f"text: {name} ({m.tag}): {'ready' + (' (in memory)' if name in text.loaded else '') if name in text.available else 'missing — ollama pull ' + m.tag}")
        asyncio.run(text.close())
    else:
        print("text: none ([text] backend)")
    platform = platform_info()
    for p in config.projects:
        ok = asyncio.run(_check_project(config, p, platform)) and ok
    return 0 if ok else 1


async def _check_project(config: Config, p: ProjectConfig, platform: dict) -> bool:
    conn = Connection(p.name, p.url, p.key)
    caps = {"name": config.name, "platform": platform, "engines": []}
    kinds = {k: KIND_VERSIONS[k] for k in p.kinds}
    try:
        answer = await conn.poll(poll_body("loading", caps, True, [], {k: 0 for k in kinds}, kinds, f"herde/{VERSION}"), time.monotonic())
        print(f"{p.name} ({p.url}): ok, next poll in {answer.retry_after:.0f} s")
        return True
    except ProjectError as e:
        print(f"{p.name} ({p.url}): {e}")
        return False
    finally:
        await conn.close()


# --- join and import ---------------------------------------------------------------------------------------

def join_command(path: Path, args: argparse.Namespace) -> int:
    url = args.url.strip().rstrip("/")
    existing: Config | None = None
    if path.is_file():
        try:
            existing = load(path)
        except ConfigError as e:
            log.error("%s", e)
            return 2
    name = (args.name or (existing.name if existing else "") or socket.gethostname())[:80]
    try:
        parse({"projects": [{"name": "check", "url": url, "key": "k" * 32}]})
    except ConfigError as e:
        log.error("%s", str(e).replace("check: ", ""))
        return 2
    try:
        joined = asyncio.run(join(url, args.code.strip(), name, f"herde/{VERSION}", platform_info()))
    except ProjectError as e:
        log.error("%s", e)
        return 1
    state_dir = existing.state_dir if existing else path.parent
    same = next((p for p in (existing.projects if existing else ()) if p.url == url), None)
    project = same.name if same else _unique(args.as_ or _slug(joined.project), {p.name for p in (existing.projects if existing else ())})
    key_file = Path("keys") / f"{project}.key"
    write_secret(state_dir / key_file, joined.key)
    if same is None:
        if not path.is_file():
            write_secret(path, starter_config(name))
        with path.open("a", encoding="utf-8") as f:
            f.write(project_block(project, url, str(key_file)))
    elif "key" in _raw_project(path, url):
        log.warning("%s keeps its key in the config itself: replace it with key_file = \"%s\"", project, key_file)
    try:
        load(path)
    except ConfigError as e:
        log.error("The config no longer reads after joining: %s", e)
        return 1
    print(f"Joined {joined.project} as {joined.worker_name or name} ({joined.trust or 'trust unknown'}); its key is in {state_dir / key_file}.")
    print("Start (or restart) the worker to serve it: herde run")
    return 0


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", name.lower()).strip("-.")[:40]
    return slug or "project"


def _unique(name: str, taken: set[str]) -> str:
    candidate, n = name, 2
    while candidate in taken:
        candidate = f"{name[:36]}-{n}"
        n += 1
    return candidate


def _raw_project(path: Path, url: str) -> dict:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return next((p for p in data.get("projects", []) if isinstance(p, dict) and str(p.get("url", "")).rstrip("/") == url), {})


def _toml(s: str) -> str:
    # A TOML basic string: JSON's escaping is a subset of TOML's.
    return json.dumps(s, ensure_ascii=False)


def write_secret(path: Path, text: str) -> None:
    """A file only its owner can read, written whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with __import__("contextlib").suppress(OSError):
        path.parent.chmod(0o700)
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    os.replace(tmp, path)


def starter_config(name: str, tts_backend: str | None = None) -> str:
    """The first config of a computer that joins a project; the Docker image's defaults come from its environment."""
    backend = os.environ.get("HERDE_TTS_BACKEND", "") if tts_backend is None else tts_backend
    lines = [
        "# herde worker — written by `herde join`. Every setting is in config.example.toml (herde's repository).",
        f"name = {_toml(name)}",
    ]
    if os.environ.get("HERDE_UI_HOST"):
        lines.append(f"ui_host = {_toml(os.environ['HERDE_UI_HOST'])}")
    lines += [
        "# When this computer works for others; nothing here means always:",
        '# windows = ["Mon-Fri 18:00-08:00", "Sat-Sun"]',
        "",
        "[tts]",
        f"backend = {_toml(backend)}",
        "",
        "[text]",
    ]
    if os.environ.get("HERDE_OLLAMA"):
        lines.append(f"ollama = {_toml(os.environ['HERDE_OLLAMA'])}")
    lines += [
        "# The models this computer offers: the name projects ask for, and Ollama's tag (ollama pull it first).",
        "# [[text.models]]",
        '# name = "gemma4-31b"',
        '# tag = "gemma4:31b"',
        "# context = 32768",
        "",
    ]
    return "\n".join(lines)


def project_block(name: str, url: str, key_file: str) -> str:
    return (
        f"\n[[projects]]\nname = {_toml(name)}\nurl = {_toml(url)}\nkey_file = {_toml(key_file)}\n"
        "# weight = 1\n"
        '# kinds = ["tts", "text"]\n'
        '# windows = ["Mon-Fri 18:00-08:00"]\n'
    )


def import_arche_worker(old: Path, path: Path) -> int:
    """An arche-worker config taken over: each station becomes a project (its v2 address), its key a key file,
    its voices folder and speech settings come along. The old worker is left as it is."""
    try:
        with old.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        log.error("cannot read %s: %s", old, e)
        return 2
    engine = data.get("engine", {}) if isinstance(data.get("engine"), dict) else {}
    clone = data.get("clone", {}) if isinstance(data.get("clone"), dict) else {}
    existing = load(path) if path.is_file() else None
    state_dir = existing.state_dir if existing else path.parent
    if existing is None:
        name = str(data.get("name") or socket.gethostname())[:80]
        text = starter_config(name, tts_backend="mlx")
        extra = [f"mp3_bitrate = {_toml(str(data.get('mp3_bitrate', '96k')))}"]
        tts_lines = []
        for key, value in (("model", engine.get("model")), ("revision", engine.get("revision")), ("hf_home", engine.get("hf_home")),
                           ("clone_model", clone.get("model")), ("clone_revision", clone.get("revision"))):
            if value:
                tts_lines.append(f"{key} = {_toml(str(value))}")
        if isinstance(engine.get("chunk_chars"), int):
            tts_lines.append(f"chunk_chars = {engine['chunk_chars']}")
        text = text.replace(f"name = {_toml(name)}\n", f"name = {_toml(name)}\n" + "\n".join(extra) + "\n", 1)
        text = text.replace('[tts]\nbackend = "mlx"\n', '[tts]\nbackend = "mlx"\n' + "".join(line + "\n" for line in tts_lines), 1)
        write_secret(path, text)
    known = {p.url for p in (existing.projects if existing else ())}
    names = {p.name for p in (existing.projects if existing else ())}
    added = []
    for st in data.get("stations", []):
        if not isinstance(st, dict) or not st.get("url") or not st.get("key"):
            continue
        url = str(st["url"]).rstrip("/") + "/api/worker/v2"
        if url in known:
            continue
        project = _unique(_slug(str(st.get("name") or "station")), names)
        names.add(project)
        key_file = Path("keys") / f"{project}.key"
        write_secret(state_dir / key_file, str(st["key"]))
        with path.open("a", encoding="utf-8") as f:
            f.write(project_block(project, url, str(key_file)))
        added.append(f"{project} ({url})")
    voices_old = Path(str(clone.get("voices"))).expanduser() if clone.get("voices") else old.parent / "voices"
    voices_new = state_dir / "voices"
    if voices_old.is_dir() and not voices_new.exists():
        shutil.copytree(voices_old, voices_new)
        print(f"voices: copied {voices_old} to {voices_new}")
    try:
        load(path)
    except ConfigError as e:
        log.error("The new config does not read: %s", e)
        return 1
    print(f"config: {path}")
    print("projects added: " + (", ".join(added) if added else "none (already there)"))
    print("Each station must speak protocol 2 (its /api/worker/v2) before this worker can serve it.")
    return 0


# --- pause, resume, status ----------------------------------------------------------------------------------

def pause_command(state: State, spec: list[str]) -> int:
    try:
        until = parse_pause(" ".join(spec), datetime.now())
    except PauseError as e:
        log.error("%s", e)
        return 2
    state.pause(until.timestamp() if until is not None else None)
    print(f"Paused until {until:%H:%M}" + (f" on {until:%a %d %b}" if until.date() != datetime.now().date() else "") if until else "Paused until you resume (herde resume).")
    print("Running work finishes first; nothing new is taken meanwhile.")
    return 0


def status_command(state: State, as_json: bool) -> int:
    s = state.status()
    paused, until = state.paused()
    if as_json:
        print(json.dumps({"status": s, "paused": paused, "paused_until": until}, indent=2))
        return 0
    if paused:
        print("Paused " + (f"until {time.strftime('%H:%M', time.localtime(until))}" if until else "until resumed"))
    if not s:
        print("The worker has not run here yet (no status file).")
        return 0
    age = int(time.time() - s.get("updated", 0))
    print(f"{s.get('name')} (herde {s.get('version')}), status from {age} s ago" + (" — is it running?" if age > 30 else ""))
    for e in s.get("engines", []):
        print(f"  {e.get('kind')}: {'ready' if e.get('ready') else 'loading' if e.get('loading') else 'not ready'}")
    for p in s.get("projects", []):
        running = ", ".join(f"task {r['task']} ({r['kind']}, {int(r['progress'] * 100)} %)" for r in p.get("running", [])) or "idle"
        print(f"  {p['name']}: {'connected' if p.get('ok') else p.get('last_error') or 'connecting'}; {running}; done {p.get('done', 0)}, failed {p.get('failed', 0)}")
    return 0


# --- bench and lab ------------------------------------------------------------------------------------------

BENCH_TEXT = {
    "de": "Schön, dass ihr da seid. Gleich hört ihr ein Lied über Hoffnung, die auch in dunklen Zeiten trägt. Bleibt dran, und wenn ihr mögt, schreibt uns, wofür wir heute beten dürfen.",
    "en": "Good to have you with us. Next is a song about hope that carries us even through dark times. Stay with us, and if you like, tell us what we may pray for today.",
}


def bench(config: Config) -> int:
    """This computer's speed with the real models, kept for the scheduler (the first estimate of a fresh install)."""
    state = State(config.state_dir)
    speeds = Speeds(state, config.tts.backend)
    tts = build_tts(config.tts, config.limits.tts_chars)
    if tts is not None:
        print(f"speech: loading {config.tts.backend} …")
        tts.load()
        rates = []
        for lang, text in BENCH_TEXT.items():
            for _ in range(2):
                t0 = time.monotonic()
                speech = tts.synthesize(text, "Sohee", lang)
                seconds = time.monotonic() - t0
                rates.append(len(text) / seconds)
                print(f"  {lang}: {len(text)} characters, {speech.ms / 1000:.1f} s of speech in {seconds:.1f} s")
        speeds.set_tts(sorted(rates)[len(rates) // 2])
        print(f"speech: {speeds.tts_cps():.1f} characters a second")
    text_engine = build_text(config.text, config.limits.text_max_tokens)
    if text_engine is not None:
        asyncio.run(_bench_text(text_engine, speeds))
    speeds.save(force=True)
    print(f"kept in {state.speeds_path}")
    return 0


async def _bench_text(engine, speeds: Speeds) -> None:
    from .protocol import Message, TextInput

    await engine.refresh()
    filler = " ".join(["The station plays songs, and between them a host speaks a few warm words."] * 120)
    schema = {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string"}}, "additionalProperties": False}
    for name in sorted(engine.available):
        inp = TextInput(model=name, max_tokens=200, temperature=0.7, messages=(Message("user", f"Summarize in one sentence, as JSON: {filler}"),), json_schema=schema)
        out = await engine.generate(inp, cancelled=lambda: False)
        read = out.input_tokens / max(0.001, out.read_ms / 1000)
        write = out.output_tokens / max(0.001, out.write_ms / 1000)
        speeds.set_text(name, read, write, out.load_ms / 1000 if out.load_ms > 1000 else None)
        print(f"text {name}: reads {read:.0f} tokens/s, writes {write:.1f} tokens/s" + (f", loaded in {out.load_ms / 1000:.1f} s" if out.load_ms > 1000 else ""))
    await engine.close()


def lab_command(config: Config, args: argparse.Namespace) -> int:
    """Speak a cases file in several variants for a listening test (herde.lab); no project is asked."""
    from . import lab

    try:
        cases = lab.load_cases(Path(args.cases).expanduser(), args.only)
        chosen = lab.variants(args.chunks, args.temperatures)
        ffmpeg = find_ffmpeg()
    except (lab.LabError, EncodeError) as e:
        log.error("%s", e)
        return 2
    out = Path(args.out).expanduser() if args.out else Path.home() / "herde-lab" / time.strftime("%Y%m%d-%H%M%S")
    tts = build_tts(config.tts, max(config.limits.tts_chars, 20000))
    if tts is None:
        log.error("lab needs speech: set [tts] backend in %s", config.path)
        return 2
    # A running worker shares the GPU: its takes and these wait for each other.
    log.info("Loading %s for %d cases × %d variants (a running worker slows down meanwhile)", config.tts.model or config.tts.backend, len(cases), len(chosen))
    tts.load()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    index = lab.run(tts.synthesize, lambda pcm, rate: encode(pcm, rate, config.mp3_bitrate, ffmpeg), cases, chosen, out, stop.is_set)
    print(f"Listen: open {index}")
    return 0
