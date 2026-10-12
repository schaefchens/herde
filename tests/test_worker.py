"""The worker loop against fake projects that follow the protocol (httpx.MockTransport) and stand-in engines:
no network, no model, no GPU — and the waits run a hundred times faster."""

import asyncio
import json
import logging
import re
import threading
import time
from datetime import datetime

import httpx
import pytest
from conftest import KEY

from herde.config import ProjectConfig, TextModel, parse
from herde.connection import Connection
from herde.engines.text.stub import StubText
from herde.engines.tts.core import TtsEngine
from herde.engines.tts.stub_backend import StubBackend
from herde.estimates import Speeds
from herde.profile import ProfileStore
from herde.state import State
from herde.windows import parse_all
from herde.worker import Link, Worker

TEXT = "Maria aus Hamburg bittet um Gebet für ihre Mutter."
INSTRUCT = "Warm and calm."
SCHEMA = {"type": "object", "required": ["de"], "properties": {"de": {"type": "string"}}}


class FakeProject:
    """A project as PROTOCOL.md describes it, in memory: a queue, leases, offers, results."""

    def __init__(self, name="p", retry_after=1, retakes=0, journal=None, offer_always=False):
        self.name = name
        # Offers even to a paused worker: so a test can let a worker see every project's offers before it may choose.
        self.offer_always = offer_always
        self.journal = journal if journal is not None else []
        self.retry_after = retry_after
        self.retakes = retakes
        self.tasks: dict[int, dict] = {}
        self.requests: list[tuple[str, dict]] = []
        self.forced: list[tuple[int, dict]] = []
        self.cancelled: set[int] = set()
        # A profile (PROTOCOL.md, Profiles): the answer to `profile`, the files it names, and whether speech waits for a worker to run it.
        self.profile: dict | None = None
        self.files: dict[str, bytes] = {}
        self.gate = True

    def add(self, id, kind="tts", cls="live", due_in=None, claim_in=None, voice="Sohee", lang="de", text=TEXT, model=None):
        model = model or ("qwen3-tts-1.7b" if kind == "tts" else "gemma4-31b")
        if kind == "tts":
            inp = {"text": text, "lang": lang, "voice": voice, "model": model, "instruct": INSTRUCT, "temperature": 0.7, "seed": 0}
            size = {"chars": len(text)}
        else:
            inp = {"model": model, "messages": [{"role": "user", "content": text}], "json_schema": SCHEMA, "max_tokens": 256}
            size = {"input_tokens": len(text) // 4, "max_tokens": 256}
        self.tasks[id] = {"id": id, "kind": kind, "class": cls, "input": inp, "size": size, "state": "queued", "lease": None,
                          "due_in": due_in, "claim_in": claim_in, "voice": voice, "lang": lang, "model": model}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.split("/api/worker/v2/")[1]
        body = _body(request)
        self.requests.append((path, body))
        self.journal.append((self.name, path))
        if self.forced:
            status, answer = self.forced.pop(0)
            return httpx.Response(status, json=answer)
        if path == "profile":
            return httpx.Response(200, json=self.profile) if self.profile else httpx.Response(404, json={"error": "not_found"})
        if path.startswith("files/"):
            data = self.files.get(path.removeprefix("files/")) if self.profile else None
            return httpx.Response(200, content=data) if data is not None else httpx.Response(404, json={"error": "not_found"})
        if path == "poll":
            said = body.get("profile") or {}
            in_step = not self.profile or not self.gate or (said.get("hash") == self.profile["profile"] and said.get("state") == "ok")
            offers = [] if not (self.offer_always or any(body["free"].values())) else [
                self._offer(t) for t in self.tasks.values() if t["state"] == "queued" and (t["kind"] != "tts" or in_step)
            ]
            cancel = [{"task": r["task"], "lease": r["lease"]} for r in body["running"] if r["task"] in self.cancelled]
            answer = {"now": 0, "offers": offers[:10], "cancel": cancel, "need_caps": False, "retry_after": self.retry_after}
            if self.profile:
                answer["profile"] = self.profile["profile"]
            return httpx.Response(200, json=answer)
        m = re.fullmatch(r"tasks/(\d+)/(claim|result|fail)", path)
        t = self.tasks.get(int(m.group(1)))
        action = m.group(2)
        if action == "claim":
            if t is None or (t["state"] != "queued" and t["lease"] != body["lease"]):
                return httpx.Response(409, json={"error": "gone"})
            t["state"], t["lease"] = "leased", body["lease"]
            task = {"id": t["id"], "kind": t["kind"], "class": t["class"], "lease_in": 45, "input": t["input"]}
            if t["due_in"] is not None:
                task["due_in"] = t["due_in"]
            return httpx.Response(200, json={"task": task})
        if t is None or t["state"] != "leased" or t["lease"] != body.get("lease"):
            return httpx.Response(409, json={"error": "gone"})
        if action == "result":
            if self.retakes:
                self.retakes -= 1
                t["state"], t["lease"] = "queued", None
                t["input"]["seed"] += 1
                return httpx.Response(200, json={"ok": False, "retake": True})
            t["state"] = "done"
            t["result"] = body
            return httpx.Response(200, json={"ok": True})
        t["state"] = "queued" if body.get("retry") else "failed"
        t["lease"] = None
        return httpx.Response(200, json={"ok": True})

    def _offer(self, t: dict) -> dict:
        o = {"task": t["id"], "kind": t["kind"], "class": t["class"], "model": t["model"], "size": t["size"]}
        if t["kind"] == "tts":
            o |= {"voice": t["voice"], "lang": t["lang"]}
        for k in ("claim_in", "due_in"):
            if t[k] is not None:
                o[k] = t[k]
        return o

    def calls(self, path_end):
        return [b for p, b in self.requests if p.endswith(path_end)]

    def polls(self):
        return self.calls("poll")


def _body(request: httpx.Request) -> dict:
    kind = request.headers.get("content-type", "")
    if kind.startswith("multipart/"):
        content = request.content if hasattr(request, "_content") else request.read()
        fields = dict(re.findall(rb'name="(\w+)"\r\n\r\n([^\r]*)\r\n', content))
        return {k.decode(): v.decode() for k, v in fields.items()} | {"audio": b"name=\"audio\"" in content}
    return json.loads(request.content or b"{}")


class Backend(StubBackend):
    """The silent stub, able to hold a take (block) or fail."""

    def __init__(self, block: threading.Event | None = None, error: Exception | None = None):
        super().__init__()
        self.block = block
        self.error = error
        self.calls = []

    def preset(self, text, speaker, language, instruct, sampling):
        self.calls.append((text, speaker, language, instruct))
        self.sampling = sampling
        if self.block is not None:
            self.block.wait(10)
        if self.error is not None:
            raise self.error
        return super().preset(text, speaker, language, instruct, sampling)

    def clone(self, text, ref, language, sampling):
        self.calls.append((text, ref.path.name, language, ""))
        self.sampling = sampling
        return super().clone(text, ref, language, sampling)


def make(projects, tmp_path, backend=None, text=False, windows=(), local_now=None, gpu_slots=1, weights=None, project_windows=None, load=True, bitrates=None):
    config = parse({"gpu_slots": gpu_slots, "tts": {"backend": "stub"}, "ui_port": 0}, tmp_path / "config.toml")
    if windows:
        config = config.__class__(**{**config.__dict__, "windows": parse_all(windows)})
    links = []
    for p in projects:
        pc = ProjectConfig(name=p.name, url=f"https://{p.name}.example.org/api/worker/v2", key=KEY, weight=(weights or {}).get(p.name, 1),
                           windows=parse_all((project_windows or {}).get(p.name, [])))
        links.append(Link(pc, Connection(pc.name, pc.url, pc.key, transport=httpx.MockTransport(p.handler)), ProfileStore(tmp_path / "projects" / p.name)))
    tts = TtsEngine(backend or Backend())
    if load:
        tts.load()

    def encoder(pcm, rate, bitrate):
        if bitrates is not None:
            bitrates.append(bitrate)
        return b"ID3MP3", len(pcm) // 2 * 1000 // rate

    text_engine = StubText((TextModel(name="gemma4-31b", tag="x"),)) if text else None
    state = State(tmp_path)
    speeds = Speeds(state, "stub")
    return Worker(config, links, tts, text_engine, state, speeds, platform={"os": "test", "arch": "x"},
                  local_now=local_now or datetime.now, encoder=encoder, scale=0.01)


def serve(worker, until, timeout=8.0, after=None):
    """Run the worker until `until()` holds (then, if given, call `after` and wait for it too), then stop it."""

    async def scenario():
        runner = asyncio.create_task(worker.run())
        deadline = time.monotonic() + timeout
        for check in [until] + ([after] if after else []):
            while not check():
                if runner.done():
                    runner.result()
                assert time.monotonic() < deadline, "timed out"
                await asyncio.sleep(0.005)
        worker.stop()
        await asyncio.wait_for(runner, 15)

    asyncio.run(scenario())


def test_a_task_is_claimed_spoken_and_handed_back_with_its_lease(tmp_path):
    p = FakeProject()
    p.add(7, due_in=300)
    backend = Backend()
    w = make([p], tmp_path, backend)
    serve(w, lambda: p.tasks[7]["state"] == "done")
    claim, result = p.calls("claim")[0], p.calls("result")[0]
    assert re.fullmatch(r"[0-9a-f-]{36}", claim["lease"]), "a fresh lease per claim"
    assert result["lease"] == claim["lease"] and result["audio"] is True and int(result["ms"]) > 0
    assert backend.calls == [(TEXT, "Sohee", "German", INSTRUCT)]
    first = p.polls()[0]
    assert first["caps"]["engines"][0]["model"] == "qwen3-tts-1.7b", "the first poll carries the caps"
    assert all("caps" not in b for b in p.polls()[2:]), "afterwards only their hash"


def test_a_long_take_keeps_its_lease_alive_through_polls(tmp_path):
    release = threading.Event()
    p = FakeProject()
    p.add(7, due_in=600)
    w = make([p], tmp_path, Backend(block=release))

    def held():
        return [b for b in p.polls() if any(r["task"] == 7 for r in b["running"])]

    def finished():
        release.set()
        return p.tasks[7]["state"] == "done"

    serve(w, lambda: len(held()) >= 3, after=finished)
    listed = held()[0]
    assert listed["running"][0]["lease"] == p.calls("claim")[0]["lease"]
    assert listed["free"] == {"tts": 0}, "busy: nothing more to offer it"
    assert listed["state"] == "ready", "busy is not offline"


def test_a_cancel_stops_the_take_and_nothing_more_is_sent(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    release = threading.Event()
    p = FakeProject()
    p.add(7, due_in=600)
    w = make([p], tmp_path, Backend(block=release))
    serve(w, lambda: bool(p.calls("claim")), after=lambda: (p.cancelled.add(7) or True) and "cancelled by the project" in caplog.text)
    release.set()
    assert p.calls("result") == [] and p.calls("fail") == []


def test_a_stop_gives_a_running_task_back(tmp_path):
    release = threading.Event()
    p = FakeProject()
    p.add(7, due_in=600)
    w = make([p], tmp_path, Backend(block=release))
    serve(w, lambda: bool(p.calls("claim")))
    release.set()
    fails = p.calls("fail")
    assert len(fails) == 1 and fails[0]["reason"] == "shutdown" and fails[0]["retry"] is True
    assert fails[0]["lease"] == p.calls("claim")[0]["lease"]
    assert p.tasks[7]["state"] == "queued", "back in the queue, nothing counted"


def test_a_retake_is_claimed_again_with_a_new_lease_and_seed(tmp_path):
    p = FakeProject(retakes=1)
    p.add(7, due_in=600)
    w = make([p], tmp_path)
    serve(w, lambda: p.tasks[7]["state"] == "done")
    claims = p.calls("claim")
    assert len(claims) == 2 and claims[0]["lease"] != claims[1]["lease"]
    assert p.tasks[7]["input"]["seed"] == 1


def test_a_live_task_of_one_project_goes_before_background_work_of_another(tmp_path):
    journal = []
    dev = FakeProject("dev", journal=journal, offer_always=True)
    prod = FakeProject("prod", journal=journal, offer_always=True)
    dev.add(1, cls="background")
    prod.add(2, cls="live", due_in=600)
    w = make([dev, prod], tmp_path, weights={"dev": 10})
    # Paused until both projects' offers are in hand: then the worker chooses between them.
    w.state.pause(None)
    serve(w, lambda: len(dev.polls()) >= 2 and len(prod.polls()) >= 2,
          after=lambda: (w.state.resume() or True) and dev.tasks[1]["state"] == "done" and prod.tasks[2]["state"] == "done")
    claims = [name for name, path in journal if path.endswith("/claim")]
    assert claims == ["prod", "dev"], "weights never outrank a moment on air"


def test_paused_it_says_so_and_takes_nothing(tmp_path):
    p = FakeProject()
    p.add(7, due_in=600)
    w = make([p], tmp_path)
    w.state.pause(None)
    serve(w, lambda: len(p.polls()) >= 3)
    assert p.calls("claim") == []
    assert p.polls()[-1]["state"] == "paused" and p.polls()[-1]["free"] == {"tts": 0}
    assert "resume_in" not in p.polls()[-1], "until resumed: no time to say"


def test_outside_its_hours_it_says_when_it_is_back(tmp_path):
    p = FakeProject()
    p.add(7, due_in=600)
    wednesday_noon = datetime(2026, 10, 7, 12, 0)
    w = make([p], tmp_path, windows=["Mon-Fri 18:00-08:00"], local_now=lambda: wednesday_noon)
    serve(w, lambda: len(p.polls()) >= 2)
    assert p.calls("claim") == []
    assert p.polls()[-1]["state"] == "paused" and p.polls()[-1]["resume_in"] == 6 * 3600


def test_a_projects_own_hours_stop_only_its_work(tmp_path):
    a, b = FakeProject("a"), FakeProject("b")
    a.add(1, due_in=600)
    b.add(2, due_in=600)
    wednesday_noon = datetime(2026, 10, 7, 12, 0)
    w = make([a, b], tmp_path, local_now=lambda: wednesday_noon, project_windows={"a": ["Sat-Sun"]})
    serve(w, lambda: b.tasks[2]["state"] == "done")
    assert a.calls("claim") == [] and a.polls()[-1]["state"] == "paused"


@pytest.mark.parametrize(("status", "error", "pause"), [(401, "bad_worker_key", 300), (403, "worker_inactive", 300), (426, "upgrade_required", 3600)])
def test_refusals_leave_a_project_alone_for_a_while(tmp_path, status, error, pause):
    p = FakeProject()
    p.forced = [(status, {"error": error})]
    w = make([p], tmp_path)
    serve(w, lambda: w.links[0].backoff_until > 0)
    assert len(p.polls()) == 1
    assert pause - 5 < w.links[0].backoff_until - time.monotonic() <= pause


def test_rate_limits_and_server_trouble_back_off(tmp_path):
    p = FakeProject()
    p.forced = [(429, {"error": "rate_limited", "retry_after": 120}), (503, {}), (503, {})]
    w = make([p], tmp_path)
    serve(w, lambda: w.links[0].backoff_until > 0)
    assert 110 < w.links[0].backoff_until - time.monotonic() <= 120
    w = make([p], tmp_path)
    serve(w, lambda: w.links[0].errors == 2)
    assert 5 < w.links[0].backoff_until - time.monotonic() <= 10, "5 s, then 10 s: doubling"


def test_an_engine_failure_is_reported_without_the_words(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    p = FakeProject()
    p.add(7, due_in=600)
    w = make([p], tmp_path, Backend(error=RuntimeError(f"metal out of memory while saying {TEXT}")))
    serve(w, lambda: bool(p.calls("fail")))
    fail = p.calls("fail")[0]
    assert (fail["reason"], fail["retry"], fail["error"]) == ("engine", True, "RuntimeError")
    assert TEXT not in caplog.text and KEY not in caplog.text and INSTRUCT not in caplog.text, "the details only with --verbose: a message may quote the words"


def test_what_it_cannot_do_or_finish_in_time_is_not_claimed(tmp_path):
    p = FakeProject()
    p.add(1, voice="Hope", due_in=600)
    p.add(3, due_in=2, text="Ein langer Satz. " * 40)
    p.add(4, kind="text", due_in=600)
    w = make([p], tmp_path)
    serve(w, lambda: len(p.polls()) >= 3)
    assert p.calls("claim") == [], "an unknown voice, too little time, and text without a text engine"


def test_a_text_task_is_answered_in_its_schema(tmp_path):
    p = FakeProject()
    p.add(9, kind="text", cls="background")
    w = make([p], tmp_path, text=True)
    serve(w, lambda: p.tasks[9]["state"] == "done")
    result = p.tasks[9]["result"]
    assert json.loads(result["output"]["content"]) == {"de": "Stub text."}
    assert result["output"]["finish"] == "stop"
    kinds = p.polls()[0]["kinds"]
    assert kinds == {"tts": [1], "text": [1]}


def test_a_cancel_for_another_lease_changes_nothing(tmp_path):
    release = threading.Event()
    p = FakeProject()
    p.add(7, due_in=600)
    w = make([p], tmp_path, Backend(block=release))
    step = {"forced": False}

    def after():
        if not step["forced"]:
            step["forced"] = True
            p.forced = [(200, {"now": 0, "offers": [], "cancel": [{"task": 7, "lease": "someone-elses-lease"}], "need_caps": False, "retry_after": 1})]
            return False
        if p.forced:
            return False
        release.set()
        return p.tasks[7]["state"] == "done"

    serve(w, lambda: bool(p.calls("claim")), after=after)
    assert p.calls("fail") == []
