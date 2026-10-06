"""The worker loop and its commands: `run`, `check`, `once`.

One take at a time across every station (one GPU), the stations asked in the
config's order. While the model is not loaded nothing is polled, so the
station sees this worker offline and its hosts fall back to other voices.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
import time
from typing import Callable

from . import VERSION
from .config import Config, ConfigError, config_path, load
from .engine import LANGUAGES, MODEL_NAME, VOICES, Engine, InvalidTask, Speech
from .mp3 import EncodeError, encode, find_ffmpeg
from .station import BadKey, Inactive, RateLimited, Station, StationError, Task

log = logging.getLogger("arche_worker")

# How long a station that refused us is left alone (s): a wrong key or a
# worker switched off in /mod will not change by itself within seconds.
PAUSE_REFUSED = 300.0
PAUSE_RATE_LIMITED = 60.0
PAUSE_ERROR_MAX = 60.0
UPLOAD_TRIES = 3


class Stopped(Exception):
    pass


class Worker:
    def __init__(
        self,
        config: Config,
        engine: Engine,
        stations: list[Station],
        encoder: Callable[[bytes, int, str], tuple[bytes, int]] | None = None,
        sleep: Callable[[float], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.engine = engine
        self.stations = stations
        self.encoder = encoder or (lambda pcm, rate, bitrate: encode(pcm, rate, bitrate))
        self.wake = threading.Event()
        self.sleep = sleep or self.wake.wait
        self.clock = clock
        self.stopping = False
        self.retry_after: dict[str, float] = {}
        self.paused_until: dict[str, float] = {}
        self.errors: dict[str, int] = {}
        # A take given up on that still holds the GPU: no new task until it ends.
        self.stuck: threading.Thread | None = None

    def report(self, ready: bool = True) -> dict:
        return {
            "name": self.config.name,
            "version": f"arche-worker {VERSION}",
            "engine": {"model": MODEL_NAME, "checkpoint": self.config.model, "revision": self.config.revision},
            "voices": [{"id": v, "label": v.replace("_", " ")} for v in VOICES],
            "languages": list(LANGUAGES),
            "ready": ready,
        }

    def stop(self, *_: object) -> None:
        self.stopping = True
        self.wake.set()

    def run(self) -> None:
        while not self.stopping:
            if not self.step() and not self.stopping:
                self.sleep(self.idle_seconds())

    def step(self) -> bool:
        """Ask the stations in order; the first task found is done. True when one was."""
        if not self.engine.ready:
            return False
        if self.stuck is not None:
            if self.stuck.is_alive():
                return False
            self.stuck = None
        now = self.clock()
        for station in self.stations:
            if self.paused_until.get(station.name, 0.0) > now:
                continue
            try:
                poll = station.poll(self.report())
            except BadKey as e:
                self._pause(station, PAUSE_REFUSED, str(e), logging.ERROR)
                continue
            except Inactive as e:
                self._pause(station, PAUSE_REFUSED, str(e), logging.WARNING)
                continue
            except RateLimited as e:
                self._pause(station, PAUSE_RATE_LIMITED, str(e), logging.WARNING)
                continue
            except StationError as e:
                errors = self.errors.get(station.name, 0) + 1
                self.errors[station.name] = errors
                self._pause(station, min(PAUSE_ERROR_MAX, 5.0 * 2 ** (errors - 1)), str(e), logging.WARNING)
                continue
            self.errors.pop(station.name, None)
            self.retry_after[station.name] = poll.retry_after
            if poll.task is not None:
                self.handle(station, poll.task)
                return True
        return False

    def idle_seconds(self) -> float:
        now = self.clock()
        open_ = [self.retry_after.get(s.name, 10.0) for s in self.stations if self.paused_until.get(s.name, 0.0) <= now]
        if open_:
            return max(1.0, min(60.0, min(open_)))
        soonest = min(self.paused_until.get(s.name, now + 10.0) for s in self.stations)
        return max(1.0, min(60.0, soonest - now))

    def handle(self, station: Station, task: Task) -> None:
        started = self.clock()
        try:
            speech = self._synthesize(task)
            mp3, ms = self.encoder(speech.pcm16, speech.sample_rate, self.config.mp3_bitrate)
        except Stopped:
            self._fail(station, task, "the worker stopped", retry=True)
            return
        except InvalidTask as e:
            self._fail(station, task, str(e), retry=False)
            return
        except TimeoutError:
            log.warning("%s task %d: no take within its time; given back", station.name, task.id)
            self._fail(station, task, "timed out", retry=True)
            return
        except EncodeError as e:
            self._fail(station, task, str(e), retry=True)
            return
        except Exception as e:  # anything unforeseen: another worker, or a later try, may do better
            # The details only with --verbose: an error message may quote what was spoken.
            log.debug("%s task %d failed", station.name, task.id, exc_info=True)
            self._fail(station, task, type(e).__name__, retry=True)
            return
        outcome = self._upload(station, task, mp3, ms)
        log.info(
            "%s task %d (%s, %s, %s, %d chars): %d ms of speech in %.1f s, %s",
            station.name, task.id, task.purpose, task.lang, task.voice, len(task.text), ms, self.clock() - started, outcome,
        )

    def _synthesize(self, task: Task) -> Speech:
        box: dict[str, object] = {}

        def work() -> None:
            try:
                box["speech"] = self.engine.synthesize(task.text, task.voice, task.lang, task.instruct, task.temperature, task.seed)
            except BaseException as e:  # handed to the waiting thread below
                box["error"] = e

        # A daemon thread: a stop mid-take must not wait for the GPU to finish.
        thread = threading.Thread(target=work, name="qwen", daemon=True)
        thread.start()
        deadline = self.clock() + task.expected_ms / 1000 * self.config.task_timeout_factor + 30
        while thread.is_alive():
            thread.join(timeout=0.2)
            if self.stopping:
                raise Stopped()
            if thread.is_alive() and self.clock() > deadline:
                self.stuck = thread
                raise TimeoutError()
        if "error" in box:
            raise box["error"]  # type: ignore[misc]
        return box["speech"]  # type: ignore[return-value]

    def _upload(self, station: Station, task: Task, mp3: bytes, ms: int) -> str:
        for attempt in range(1, UPLOAD_TRIES + 1):
            try:
                return station.upload(task.id, mp3, ms)
            except (BadKey, Inactive, RateLimited) as e:
                self._pause(station, PAUSE_RATE_LIMITED if isinstance(e, RateLimited) else PAUSE_REFUSED, str(e), logging.WARNING)
                return "refused"
            except StationError as e:
                if attempt == UPLOAD_TRIES:
                    # Its lease runs out at the station, which gives the task to a worker again.
                    log.warning("%s task %d: upload failed (%s); the station will give it out again", station.name, task.id, e)
                    return "lost"
                self.sleep(2.0 * attempt)
        return "lost"

    def _fail(self, station: Station, task: Task, error: str, retry: bool) -> None:
        log.warning("%s task %d: %s", station.name, task.id, error)
        try:
            station.fail(task.id, error, retry)
        except StationError as e:
            log.warning("%s task %d: could not report the failure (%s)", station.name, task.id, e)

    def _pause(self, station: Station, seconds: float, why: str, level: int) -> None:
        self.paused_until[station.name] = self.clock() + seconds
        log.log(level, "%s (next try in %d s)", why, int(seconds))


def _logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx logs every request URL at INFO; a key never goes into a URL, but the noise helps nobody.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _keep_awake(argv: list[str]) -> None:
    """Run again under `caffeinate -i`: an idle Mac must not sleep while it serves the station."""
    if sys.platform != "darwin" or os.environ.get("ARCHE_WORKER_AWAKE") == "1":
        return
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        return
    os.environ["ARCHE_WORKER_AWAKE"] = "1"
    os.execv(caffeinate, [caffeinate, "-i", sys.executable, "-m", "arche_worker", *argv])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="arche-worker", description="Speak an ARCHE station's host lines with Qwen3-TTS on this Mac.")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "check", "once"])
    parser.add_argument("--config", help="config file (default ~/.config/arche-worker/config.toml)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _logging(args.verbose)
    try:
        config = load(config_path(args.config))
    except ConfigError as e:
        log.error("%s", e)
        return 2

    if args.command == "check":
        return check(config)
    if args.command == "run" and config.keep_awake:
        _keep_awake(argv)

    stations = [Station(s) for s in config.stations]
    engine = Engine(config.model, config.revision, config.hf_home)
    worker = Worker(config, engine, stations)
    try:
        find_ffmpeg()
        log.info("Loading %s (first time: a 4 GB download)", config.model)
        started = time.monotonic()
        engine.load()
        log.info("Model ready in %.1f s; serving %s", time.monotonic() - started, ", ".join(s.name for s in stations))
        # From here a stop finishes cleanly: the current task is given back first.
        signal.signal(signal.SIGTERM, worker.stop)
        signal.signal(signal.SIGINT, worker.stop)
        if args.command == "once":
            log.info("one task done" if worker.step() else "no task waiting")
            return 0
        worker.run()
        return 0
    except EncodeError as e:
        log.error("%s", e)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        for s in stations:
            s.close()


def check(config: Config) -> int:
    """The config, ffmpeg and each station's answer — without loading the model or taking a task."""
    ok = True
    print(f"config: {config.path} ({len(config.stations)} station{'s' if len(config.stations) != 1 else ''})")
    try:
        print(f"ffmpeg: {find_ffmpeg()}")
    except EncodeError as e:
        print(f"ffmpeg: {e}")
        ok = False
    worker = Worker(config, Engine(config.model, config.revision, config.hf_home), [])
    for s in config.stations:
        station = Station(s)
        try:
            poll = station.poll(worker.report(ready=False))
            print(f"{s.name} ({s.url}): ok, next poll in {poll.retry_after:.0f} s")
        except StationError as e:
            print(f"{s.name} ({s.url}): {e}")
            ok = False
        finally:
            station.close()
    return 0 if ok else 1
