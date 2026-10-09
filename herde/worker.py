"""The worker: one connection per project, its polls the heartbeat, one scheduler for all of them.

Every project is polled on its own `retry_after`, busy or not: a long take no
longer makes a computer look gone, and each poll keeps the leases of the tasks
it lists alive. Offers from all projects meet in one place, and the scheduler
picks the next by class, due time and the owner's weights (scheduler.py).

The GPU is one resource: `gpu_slots` tasks at once across speech and text
(one on a Mac — Gemma writing and Qwen speaking at once made both two to five
times slower), and one task at a time per engine. A slot is taken before the
claim goes out, so two claims never race for it; a take that runs over its
time keeps its slot until the GPU is really free again.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import VERSION
from .config import Config, ProjectConfig
from .connection import BadKey, Connection, Gone, Inactive, ProjectError, RateLimited, Upgrade
from .engines.text import EngineError
from .engines.tts.core import Cancelled, InvalidTask, TooLarge, TtsEngine
from .engines.tts.mp3 import EncodeError, encode
from .estimates import Speeds
from .protocol import KIND_VERSIONS, MODEL, Offer, Task, TextOutput, TtsInput, caps_hash, poll_body
from .scheduler import charge, order
from .state import State
from .windows import is_open

log = logging.getLogger("herde")

# How long a project that refused us is left alone (s): a wrong key or a worker
# switched off by its admin will not change by itself within seconds.
PAUSE_REFUSED = 300.0
PAUSE_RATE_LIMITED = 60.0
PAUSE_UPGRADE = 3600.0
PAUSE_ERROR_MAX = 60.0
UPLOAD_TRIES = 3
# Without news, the scheduler looks again this often: a pause ends, a claim_by passes.
TICK = 1.0
# A task still running this long after it was due is stopped: nobody can use it any more.
LATE_GRACE = 5.0
TEXT_TIMEOUT_FACTOR = 4.0
TEXT_TIMEOUT_BASE = 60.0
REFRESH_TEXT = 30.0
STATUS_EVERY = 5.0
# On a stop, running tasks get this long to report themselves given back.
SHUTDOWN_GRACE = 10.0


@dataclass
class Running:
    task: Task
    started: float
    progress: float = 0.0
    reason: str = ""
    # A take we stopped waiting for keeps its slot until its thread ends; _run must not free it then.
    slot_handed: bool = False
    # For the GPU thread (it checks between two takes) and for the loop (which stops waiting at once).
    cancel: threading.Event = field(default_factory=threading.Event)
    stopped: asyncio.Event = field(default_factory=asyncio.Event)

    def stop(self, reason: str) -> None:
        if not self.reason:
            self.reason = reason
        self.cancel.set()
        self.stopped.set()

    def set_progress(self, p: float) -> None:
        self.progress = max(self.progress, min(1.0, p))


class Link:
    """One project as the worker sees it."""

    def __init__(self, config: ProjectConfig, conn: Connection):
        self.config = config
        self.name = config.name
        self.conn = conn
        self.offers: list[Offer] = []
        self.backoff_until = 0.0
        self.errors = 0
        self.sent_hash: str | None = None
        self.need_caps = True
        self.running: dict[int, Running] = {}
        self.poke = asyncio.Event()
        self.last_ok: float | None = None
        self.last_error = ""
        self.done = 0
        self.failed = 0


class Worker:
    def __init__(
        self,
        config: Config,
        links: list[Link],
        tts: TtsEngine | None,
        text,
        state: State,
        speeds: Speeds,
        platform: dict | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        local_now: Callable[[], datetime] = datetime.now,
        encoder: Callable[[bytes, int], tuple[bytes, int]] | None = None,
        scale: float = 1.0,
    ):
        self.config = config
        self.links = links
        self.tts = tts
        self.text = text
        self.state = state
        self.speeds = speeds
        self.platform = platform or {"os": "unknown", "arch": "unknown"}
        self.clock = clock
        self.wall = wall
        self.local_now = local_now
        self.encoder = encoder or (lambda pcm, rate: encode(pcm, rate, config.mp3_bitrate))
        # Tests run the worker's waits faster; the clocks themselves are never scaled.
        self.scale = scale
        self.stopping = False
        self.gpu_busy = 0
        self.engine_busy = {"tts": False, "text": False}
        self.spent: dict[str, float] = {}
        self.wake = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.loading: set[str] = set()
        self._tasks: set[asyncio.Task] = set()
        # One thread for speech: a take never shares the GPU with another take.
        self._tts_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._status: dict = {}

    # --- running ---------------------------------------------------------------------------------------

    async def run(self) -> None:
        loops = [asyncio.create_task(self._poll_loop(link), name=f"poll {link.name}") for link in self.links]
        loops.append(asyncio.create_task(self._dispatch_loop(), name="dispatch"))
        loops.append(asyncio.create_task(self._housekeeping_loop(), name="housekeeping"))
        if self.tts is not None and not self.tts.ready:
            loops.append(asyncio.create_task(self._load_tts(), name="load tts"))
        try:
            await self.stop_event.wait()
        finally:
            await self._shut_down(loops)

    def stop(self, *_: object) -> None:
        if self.stopping:
            return
        self.stopping = True
        log.info("Stopping: running tasks are given back")
        for link in self.links:
            for r in link.running.values():
                r.stop("shutdown")
            link.poke.set()
        self.wake.set()
        self.stop_event.set()

    async def _shut_down(self, loops: list[asyncio.Task]) -> None:
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=SHUTDOWN_GRACE)
        for t in loops:
            t.cancel()
        for t in loops:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        for link in self.links:
            await link.conn.close()
        if self.text is not None:
            await self.text.close()
        self.speeds.save(force=True)
        self._write_status()
        self._tts_pool.shutdown(wait=False, cancel_futures=True)

    async def _load_tts(self) -> None:
        self.loading.add("tts")
        started = self.clock()
        log.info("Loading speech (%s%s)", self.tts.backend.name, "" if self.tts.backend.name == "stub" else "; the first time a download of several GB")
        try:
            await asyncio.to_thread(self.tts.load)
        except Exception as e:  # without speech, text may still be served
            log.error("Speech could not be loaded (%s: %s); this worker offers no speech until it restarts", type(e).__name__, e)
            self.tts = None
        else:
            log.info("Speech ready in %.1f s", self.clock() - started)
        finally:
            self.loading.discard("tts")
            for link in self.links:
                link.poke.set()
            self.wake.set()

    # --- what a project is told --------------------------------------------------------------------------

    def paused(self, link: Link | None = None) -> tuple[bool, int | None]:
        """Whether no new work is taken (for a project: its own hours too), and in how many seconds that ends."""
        on, until = self.state.paused(self.wall())
        if on:
            return True, (max(0, int(until - self.wall())) if until is not None else None)
        sets = [ws for ws in (self.config.windows, link.config.windows if link is not None else ()) if ws]
        if sets:
            now = self.local_now()
            if not all(is_open(ws, now) for ws in sets):
                nxt = _next_open_all(sets, now)
                return True, (max(0, int((nxt - now).total_seconds())) if nxt is not None else None)
        return False, None

    def kinds(self, link: Link) -> dict[str, list[int]]:
        have = {"tts": self.tts is not None, "text": self.text is not None}
        return {k: KIND_VERSIONS[k] for k in link.config.kinds if have.get(k)}

    def caps(self, link: Link) -> dict:
        engines: list[dict] = []
        allowed = set(link.config.models)
        if self.tts is not None and self.tts.ready and "tts" in link.config.kinds:
            entry = self.tts.caps(self.speeds.tts_advertised()["chars_per_s"])
            if not allowed or entry["model"] in allowed:
                engines.append(entry)
        if self.text is not None and self.text.ready and "text" in link.config.kinds:
            engines.extend(e for e in self.text.caps(self.speeds.text_advertised) if not allowed or e["model"] in allowed)
        return {"name": self.config.name, "platform": self.platform, "engines": engines}

    def can_start(self, kind: str) -> bool:
        return not self.engine_busy.get(kind, True) and self.gpu_busy < self.config.gpu_slots

    def poll_body(self, link: Link) -> dict:
        caps = self.caps(link)
        paused, resume_in = self.paused(link)
        ready_engines = bool(caps["engines"])
        if not ready_engines and self.loading:
            state = "loading"
        elif paused or self.stopping:
            state = "paused"
        else:
            state = "ready"
        kinds = self.kinds(link)
        # Free only where an engine is ready: Ollama down, a model still loading, is no slot to offer into.
        ready = {"tts": self.tts is not None and self.tts.ready, "text": self.text is not None and self.text.ready}
        free = {k: (1 if state == "ready" and ready[k] and self.can_start(k) else 0) for k in kinds}
        running = [{"task": r.task.id, "lease": r.task.lease, "progress": round(r.progress, 2)} for r in link.running.values()]
        send_caps = link.need_caps or link.sent_hash != caps_hash(caps)
        return poll_body(state, caps, send_caps, running, free, kinds, f"herde/{VERSION}", resume_in if state == "paused" else None)

    # --- polling -----------------------------------------------------------------------------------------

    async def _poll_loop(self, link: Link) -> None:
        while not self.stopping:
            wait = link.backoff_until - self.clock()
            if wait > 0:
                # A task ending meanwhile changes nothing: the project is left alone until then.
                link.poke.clear()
                await self._nap(link, wait)
                continue
            # Cleared before the poll is made, not after: a task that ends while it is on its way is reported at once.
            link.poke.clear()
            body = self.poll_body(link)
            try:
                answer = await link.conn.poll(body, self.clock())
            except ProjectError as e:
                link.offers = []
                self._refused(link, e)
                continue
            link.errors, link.last_ok, link.last_error = 0, self.wall(), ""
            if "caps" in body:
                link.sent_hash, link.need_caps = body["caps_hash"], False
            if answer.need_caps:
                link.need_caps = True
            link.offers = [o for o in answer.offers if o.task not in link.running]
            for task_id, lease in answer.cancel:
                r = link.running.get(task_id)
                if r is not None and r.task.lease == lease:
                    r.stop("cancelled")
            self.wake.set()
            await self._nap(link, answer.retry_after)

    async def _nap(self, link: Link, seconds: float) -> None:
        """Sleep until the next poll — or at once after a task ended (to report it and hear what is new)."""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(link.poke.wait(), timeout=max(0.0, seconds * self.scale))

    def _refused(self, link: Link, e: ProjectError) -> None:
        link.last_error = str(e)
        if isinstance(e, BadKey):
            self._back_off(link, PAUSE_REFUSED, e, logging.ERROR)
        elif isinstance(e, Inactive):
            self._back_off(link, PAUSE_REFUSED, e, logging.WARNING)
        elif isinstance(e, RateLimited):
            self._back_off(link, e.retry_after or PAUSE_RATE_LIMITED, e, logging.WARNING)
        elif isinstance(e, Upgrade):
            self._back_off(link, PAUSE_UPGRADE, e, logging.ERROR)
        else:
            link.errors += 1
            self._back_off(link, min(PAUSE_ERROR_MAX, 5.0 * 2 ** (link.errors - 1)), e, logging.WARNING)

    def _back_off(self, link: Link, seconds: float, why: Exception, level: int) -> None:
        link.backoff_until = self.clock() + seconds
        log.log(level, "%s (next try in %d s)", why, int(seconds))

    # --- choosing and claiming ---------------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        while not self.stopping:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.wake.wait(), timeout=TICK * self.scale)
            self.wake.clear()
            self._stop_late()
            await self.dispatch()

    async def dispatch(self) -> None:
        """Claim the most urgent offers the free slots allow."""
        while not self.stopping and self.gpu_busy < self.config.gpu_slots:
            if self.paused()[0]:
                return
            now = self.clock()
            offers = [o for link in self.links if link.backoff_until <= now and not self.paused(link)[0] for o in link.offers]
            weights = {link.name: link.config.weight for link in self.links}
            started = False
            for offer in order(offers, now, self.fits, self.spent, weights):
                if not self.can_start(offer.kind):
                    continue
                if await self._claim(self._link(offer.project), offer):
                    started = True
                    break
            if not started:
                return

    def fits(self, o: Offer) -> bool:
        """Whether this worker can, may, and in time will do an offer."""
        link = self._link(o.project)
        if o.kind not in link.config.kinds or (link.config.models and o.model not in link.config.models) or o.task in link.running:
            return False
        limits = self.config.limits
        if o.kind == "tts":
            if self.tts is None or not self.tts.ready or not self.tts.can(o.model, o.voice, o.lang) or o.chars > limits.tts_chars:
                return False
            seconds = self.speeds.tts_seconds(o.chars or 200)
        else:
            if self.text is None or not self.text.ready or not self.text.can(o.model):
                return False
            if o.max_tokens > limits.text_max_tokens or o.input_tokens * 4 > limits.text_input_chars:
                return False
            seconds = self.speeds.text_seconds(o.model, o.input_tokens, o.max_tokens, o.model in self.text.loaded)
        return Speeds.fits(self.clock(), seconds, o.due_by)

    async def _claim(self, link: Link, offer: Offer) -> bool:
        # The slot is taken before the claim goes out: no second claim may race for it.
        self._take(offer.kind)
        link.offers = [o for o in link.offers if o.task != offer.task]
        lease = str(uuid.uuid4())
        try:
            task = await link.conn.claim(offer.task, lease, self.clock())
        except Gone:
            self._release(offer.kind)
            return False
        except ProjectError as e:
            self._release(offer.kind)
            self._refused(link, e)
            return False
        if task.kind != offer.kind or not _fits_input(task):
            # Not what was offered, or an input with ids that do not fit the protocol: never near a model.
            self._release(offer.kind)
            await self._report_fail(link, task, "invalid", True, "the task's input does not fit the protocol")
            return False
        if offer.cls == "background":
            charge(self.spent, link.name, link.config.weight)
        running = Running(task=task, started=self.clock())
        link.running[task.id] = running
        t = asyncio.create_task(self._run(link, running), name=f"{link.name} task {task.id}")
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return True

    def _take(self, kind: str) -> None:
        self.engine_busy[kind] = True
        self.gpu_busy += 1

    def _release(self, kind: str) -> None:
        self.engine_busy[kind] = False
        self.gpu_busy = max(0, self.gpu_busy - 1)
        self.wake.set()

    def _stop_late(self) -> None:
        now = self.clock()
        for link in self.links:
            for r in link.running.values():
                if r.task.due_by is not None and now > r.task.due_by + LATE_GRACE:
                    r.stop("late")

    def _link(self, name: str) -> Link:
        for link in self.links:
            if link.name == name:
                return link
        raise KeyError(name)

    # --- doing a task ------------------------------------------------------------------------------------

    async def _run(self, link: Link, r: Running) -> None:
        task = r.task
        kind = task.kind
        started = self.clock()
        try:
            if kind == "tts":
                mp3, ms = await self._speak(r)
                outcome = await self._deliver(link, r, lambda: link.conn.result_audio(task.id, task.lease, mp3, ms))
                what = f"{ms} ms of speech"
            else:
                output = await self._write(r)
                self.speeds.observe_text(task.input.model, output)
                outcome = await self._deliver(link, r, lambda: link.conn.result_text(task.id, task.lease, output))
                what = f"{output.output_tokens} tokens ({output.finish})"
            if outcome == "done":
                link.done += 1
            log.info("%s task %d (%s, %s%s): %s in %.1f s, %s", link.name, task.id, kind, task.cls, _what(task), what, self.clock() - started, outcome)
        except Cancelled:
            if r.reason == "shutdown":
                await self._report_fail(link, task, "shutdown", True, "worker stopping")
            else:
                log.info("%s task %d: %s; stopped", link.name, task.id, "too late" if r.reason == "late" else "cancelled by the project")
        except _Stuck:
            link.failed += 1
            log.warning("%s task %d: no take within its time; given back (the GPU stays busy until the take ends)", link.name, task.id)
            await self._report_fail(link, task, "timeout", True, "timed out")
        except TimeoutError:
            link.failed += 1
            await self._report_fail(link, task, "timeout", True, "timed out")
        except TooLarge as e:
            await self._report_fail(link, task, "too_large", True, str(e))
        except InvalidTask as e:
            await self._report_fail(link, task, "invalid", True, str(e))
        except (EncodeError, EngineError) as e:
            link.failed += 1
            await self._report_fail(link, task, "engine", True, str(e))
        except Exception as e:  # anything unforeseen: another worker, or a later try, may do better
            link.failed += 1
            # The details only with --verbose: an error message may quote the input.
            log.debug("%s task %d failed", link.name, task.id, exc_info=True)
            await self._report_fail(link, task, "engine", True, type(e).__name__)
        finally:
            link.running.pop(task.id, None)
            if not r.slot_handed:
                self._release(kind)
            link.poke.set()

    async def _speak(self, r: Running) -> tuple[bytes, int]:
        inp = r.task.input
        assert isinstance(inp, TtsInput) and self.tts is not None
        tts = self.tts
        loop = asyncio.get_running_loop()
        timeout = self.speeds.tts_seconds(len(inp.text)) * self.config.tts.task_timeout_factor + 30

        def work() -> tuple[bytes, int, float]:
            t0 = time.monotonic()
            speech = tts.synthesize(inp.text, inp.voice, inp.lang, inp.instruct, inp.temperature, inp.seed, cancelled=r.cancel.is_set, progress=r.set_progress)
            seconds = time.monotonic() - t0
            mp3, ms = self.encoder(speech.pcm16, speech.sample_rate)
            return mp3, ms, seconds

        fut = loop.run_in_executor(self._tts_pool, work)
        stopper = asyncio.ensure_future(r.stopped.wait())
        try:
            done, _ = await asyncio.wait({fut, stopper}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopper.cancel()
        if fut in done:
            mp3, ms, seconds = fut.result()
            self.speeds.observe_tts(len(inp.text), seconds)
            return mp3, ms
        # Given up, or stopped mid-take: the thread ends at its next piece, and holds the GPU until then.
        r.cancel.set()
        if fut.done():
            fut.exception()  # ended meanwhile: nothing to hold
        else:
            r.slot_handed = True

            def free(f: asyncio.Future) -> None:
                if not f.cancelled():
                    f.exception()  # read, so asyncio does not log it as never retrieved
                self._release(r.task.kind)

            fut.add_done_callback(free)
        if stopper in done:
            raise Cancelled()
        raise _Stuck()

    async def _write(self, r: Running) -> TextOutput:
        inp = r.task.input
        assert self.text is not None
        est = self.speeds.text_seconds(inp.model, 0, inp.max_tokens, inp.model in self.text.loaded) + inp.chars / 4 / self.speeds.text(inp.model)["read_tps"]
        gen = asyncio.ensure_future(self.text.generate(inp, cancelled=r.cancel.is_set, progress=r.set_progress))
        stopper = asyncio.ensure_future(r.stopped.wait())
        try:
            done, _ = await asyncio.wait({gen, stopper}, timeout=est * TEXT_TIMEOUT_FACTOR + TEXT_TIMEOUT_BASE, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stopper.cancel()
        if gen in done:
            return gen.result()
        # Closing Ollama's stream stops its writing: the GPU is free when the cancel is through.
        gen.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await gen
        if stopper in done:
            raise Cancelled()
        raise TimeoutError()

    async def _deliver(self, link: Link, r: Running, send: Callable[[], Awaitable[str]]) -> str:
        for attempt in range(1, UPLOAD_TRIES + 1):
            if r.reason in ("cancelled", "late"):
                return "cancelled"
            try:
                return await send()
            except (BadKey, Inactive, RateLimited, Upgrade) as e:
                self._refused(link, e)
                return "refused"
            except ProjectError as e:
                if attempt == UPLOAD_TRIES:
                    # Its lease runs out at the project, which gives the task to a worker again.
                    log.warning("%s task %d: could not hand it back (%s); the project will give it out again", link.name, r.task.id, e)
                    return "lost"
                await asyncio.sleep(2.0 * attempt * self.scale)
        return "lost"

    async def _report_fail(self, link: Link, task: Task, reason: str, retry: bool, error: str) -> None:
        log.warning("%s task %d: %s (%s)", link.name, task.id, error, reason)
        try:
            await link.conn.fail(task.id, task.lease, reason, retry, error)
        except ProjectError as e:
            log.warning("%s task %d: could not report it (%s)", link.name, task.id, e)

    # --- housekeeping ------------------------------------------------------------------------------------

    async def _housekeeping_loop(self) -> None:
        last_text = -REFRESH_TEXT
        while not self.stopping:
            now = self.clock()
            if self.text is not None and now - last_text >= REFRESH_TEXT:
                last_text = now
                before = (self.text.ready, frozenset(self.text.available), frozenset(self.text.loaded))
                await self.text.refresh()
                if (self.text.ready, frozenset(self.text.available), frozenset(self.text.loaded)) != before:
                    self.wake.set()
            self.speeds.save()
            self._write_status()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stop_event.wait(), timeout=STATUS_EVERY * self.scale)

    def status(self) -> dict:
        return self._status

    def _write_status(self) -> None:
        paused, resume_in = self.paused()
        now = self.clock()
        self._status = {
            "name": self.config.name,
            "version": VERSION,
            "updated": int(self.wall()),
            "paused": paused,
            "resume_in": resume_in,
            "pause_file": self.state.paused(self.wall())[0],
            "engines": self._engine_status(),
            "projects": [
                {
                    "name": link.name, "url": link.conn.url, "kinds": list(link.config.kinds), "weight": link.config.weight,
                    "ok": link.last_ok is not None and link.backoff_until <= now, "last_error": link.last_error,
                    "offers": len(link.offers), "done": link.done, "failed": link.failed,
                    "running": [
                        {"task": r.task.id, "kind": r.task.kind, "class": r.task.cls, "progress": round(r.progress, 2), "seconds": int(now - r.started)}
                        for r in link.running.values()
                    ],
                }
                for link in self.links
            ],
        }
        with contextlib.suppress(OSError):
            self.state.save_status(self._status)

    def _engine_status(self) -> list[dict]:
        out = []
        if self.tts is not None or "tts" in self.loading:
            out.append({
                "kind": "tts", "backend": self.tts.backend.name if self.tts else "", "ready": bool(self.tts and self.tts.ready),
                "loading": "tts" in self.loading, "voices": len(self.tts.voices()) if self.tts else 0,
                "chars_per_s": round(self.speeds.tts_cps(), 1),
            })
        if self.text is not None:
            out.append({
                "kind": "text", "ready": self.text.ready,
                "models": [{"name": n, "available": n in self.text.available, "loaded": n in self.text.loaded} for n in sorted(self.text.models)],
            })
        return out


class _Stuck(Exception):
    """A take that ran over its time: given back, its GPU thread still running."""


def _what(task: Task) -> str:
    """What a log line may say about a task: never its text."""
    inp = task.input
    if isinstance(inp, TtsInput):
        return f", {inp.lang}, {inp.voice}, {len(inp.text)} chars"
    return f", {inp.model}"


def _fits_input(task: Task) -> bool:
    inp = task.input
    if isinstance(inp, TtsInput):
        return True  # checked in parse_claim
    return bool(MODEL.fullmatch(inp.model))


def _next_open_all(sets: list, now: datetime) -> datetime | None:
    """The first minute in the next eight days when every set of hours is open."""
    t = now.replace(second=0, microsecond=0)
    for _ in range(8 * 24 * 60):
        if all(is_open(ws, t) for ws in sets):
            return max(t, now)
        t += timedelta(minutes=1)
    return None
