"""`herde serve`, the little project: its page and API, its side of the protocol against the protocol's own
example answers, the queue's rules — and a real worker doing a speech and a text job through it, end to end."""

import asyncio
import json
import time
from datetime import datetime

import httpx
import pytest
from conftest import fixture

from herde.config import ProjectConfig, TextModel, parse
from herde.connection import Connection
from herde.engines.text.stub import StubText
from herde.engines.tts.core import TtsEngine
from herde.engines.tts.stub_backend import StubBackend
from herde.estimates import Speeds
from herde.protocol import caps_hash
from herde.serve import Project, Server
from herde.state import State
from herde.worker import Link, Worker

CAPS = {
    "name": "studio-mac",
    "platform": {"os": "darwin", "arch": "arm64", "accelerator": "Apple M1 Max"},
    "engines": [
        {"kind": "tts", "model": "qwen3-tts-1.7b", "location": "local", "max_chars": 2000,
         "voices": [{"id": "Sohee", "label": "Sohee", "langs": ["de", "en"], "instruct": True},
                    {"id": "Faith", "label": "Faith (own voice)", "langs": ["de"], "instruct": False, "own": True}]},
        {"kind": "text", "model": "gemma4-31b", "location": "local", "json_schema": True, "context": 32768, "max_tokens": 4096, "loaded": True},
    ],
}
MP3 = b"ID3" + b"\x00" * 200


class Clock:
    def __init__(self):
        self.now = 1_760_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def served(tmp_path):
    clock = Clock()
    project = Project(tmp_path / "data", "test project", clock=clock)
    server = Server(project, "127.0.0.1", 0)
    server.start()
    http = httpx.Client(base_url=f"http://127.0.0.1:{server.port}", timeout=10)
    yield project, server, http, clock
    http.close()
    server.stop()


def admin(http, project, method, path, body=None, **headers):
    return http.request(method, "/api/" + path, json=body, headers={"X-Herde-Token": project.token, **headers})


def joined(http, project, name="Studio Mac"):
    code = admin(http, project, "POST", "invites", {"name": name}).json()["code"]
    r = http.post("/api/worker/v2/join", json={"code": code, "name": "x", "version": "herde/test", "platform": {"os": "linux", "arch": "x86_64"}})
    assert r.status_code == 200, r.text
    return r.json()["key"]


def poll(http, key, caps=CAPS, send_caps=True, **more):
    body = {"protocol": 2, "version": "herde/test", "kinds": {"tts": [1], "text": [1]}, "state": "ready", "caps_hash": caps_hash(caps),
            "running": [], "free": {"tts": 1, "text": 1}} | more
    if send_caps:
        body["caps"] = caps
    return http.post("/api/worker/v2/poll", json=body, headers={"X-Worker-Key": key})


def call(http, key, job, action, body):
    return http.post(f"/api/worker/v2/tasks/{job}/{action}", json=body, headers={"X-Worker-Key": key})


def speech(http, key, job, lease, audio=MP3):
    return http.post(f"/api/worker/v2/tasks/{job}/result", data={"lease": lease, "ms": "2100"},
                     files={"audio": (f"{job}.mp3", audio, "audio/mpeg")}, headers={"X-Worker-Key": key})


def submit(http, project, **job):
    job = {"kind": "tts", "text": "Gleich hört ihr ein Lied über Hoffnung.", "voice": "Sohee", "lang": "de"} | job
    r = admin(http, project, "POST", "jobs", job)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def diff(actual, expected, path="$"):
    """Where an answer differs from a fixture's: the same keys and kinds of values; a list's items like one of the fixture's."""
    kind = lambda v: "number" if isinstance(v, (int, float)) and not isinstance(v, bool) else type(v).__name__  # noqa: E731
    if isinstance(expected, list) and expected:
        if not isinstance(actual, list):
            return [f"{path} is no list"]
        out = []
        for i, item in enumerate(actual):
            tries = [diff(item, candidate, f"{path}[{i}]") for candidate in expected]
            out += min(tries, key=len)
        return out
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path} is no object"]
        out = [f"{path} lacks {k}" for k in expected if k not in actual] + [f"{path} has unexpected {k}" for k in actual if k not in expected]
        return out + [d for k in actual if k in expected for d in diff(actual[k], expected[k], f"{path}.{k}")]
    if expected is None or expected == []:
        return []
    return [] if kind(actual) == kind(expected) else [f"{path} is a {kind(actual)}, not a {kind(expected)}"]


def like(name, response):
    f = fixture(name)["response"]
    assert response.status_code == f["status"], (name, response.text)
    assert diff(response.json(), f["body"]) == [], name


# --- the page and its API ------------------------------------------------------------------------------

def test_the_page_and_its_api_need_the_token_and_this_computer(served):
    project, _, http, _ = served
    page = http.get("/")
    assert page.status_code == 200 and "herde" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"], "no inline script, no outside assets"
    assert http.get("/app.js").headers["content-type"].startswith("text/javascript")
    assert http.get("/api/state").status_code == 401
    assert admin(http, project, "GET", "state", **{"X-Herde-Token": "wrong"}).status_code == 401
    state = admin(http, project, "GET", "state").json()
    assert state["name"] == "test project" and state["workers"] == [] and state["address"].endswith("/api/worker/v2")
    rebound = http.get("/api/state", headers={"X-Herde-Token": project.token, "Host": "attacker.example:8740"})
    assert rebound.status_code == 403, "a name rebound to this computer does not reach the page"
    assert (project.data / "token").stat().st_mode & 0o077 == 0, "the token file is the user's only"


def test_an_invite_works_once_and_wrong_codes_are_counted(served):
    project, _, http, _ = served
    inv = admin(http, project, "POST", "invites", {"name": "Peter's PC"}).json()
    assert inv["join"] == f"herde join http://127.0.0.1:{served[1].port}/api/worker/v2 {inv['code']}"
    body = {"code": inv["code"].lower().replace("-", " "), "name": "x", "version": "v", "platform": {"os": "linux", "arch": "x86_64"}}
    first = http.post("/api/worker/v2/join", json=body)
    assert first.status_code == 200 and first.json()["worker"]["name"] == "Peter's PC"
    assert http.post("/api/worker/v2/join", json=body).json() == {"error": "bad_code"}, "used once"
    wrong = body | {"code": "zzzz" * 6}
    assert {http.post("/api/worker/v2/join", json=wrong).status_code for _ in range(9)} == {401}
    assert http.post("/api/worker/v2/join", json=wrong).status_code == 429, "ten wrong codes an hour"


# --- the protocol, against its example answers ------------------------------------------------------------

def test_the_answers_have_the_protocol_s_shapes(served):
    project, _, http, _ = served
    platform = {"os": "linux", "arch": "x86_64"}
    code = admin(http, project, "POST", "invites", {"name": "Peter's PC"}).json()["code"]
    joined_ = http.post("/api/worker/v2/join", json={"code": code, "name": "x", "version": "v", "platform": platform})
    like("join.ok", joined_)
    like("join.bad-code", http.post("/api/worker/v2/join", json={"code": code, "name": "x", "version": "v", "platform": platform}))
    key = joined_.json()["key"]
    live = submit(http, project, **{"class": "live", "due_in": 170})
    submit(http, project, lang="en")
    submit(http, project, kind="text", model="gemma4-31b", prompt="Say hello.", max_tokens=1024, **{"class": "background"})
    submit(http, project, voice="Faith", **{"class": "background"})
    like("poll.first", poll(http, key))
    like("poll.need-caps", poll(http, key, send_caps=False, caps_hash="0" * 64))
    like("claim.tts", call(http, key, live, "claim", {"lease": "4d8f0b2e-7c1a-4b5e-9f3d-2a6c8e1b0d47"}))
    like("claim.gone", call(http, key, live, "claim", {"lease": "another-lease-1"}))
    running = [{"task": live, "lease": "4d8f0b2e-7c1a-4b5e-9f3d-2a6c8e1b0d47", "progress": 0.5}, {"task": 999, "lease": "a9c05e1f-3b7d"}]
    like("poll.busy", poll(http, key, send_caps=False, running=running, free={"tts": 0, "text": 0}))
    like("poll.paused", poll(http, key, send_caps=False, state="paused", resume_in=3600, free={"tts": 0, "text": 0}))
    like("poll.upgrade", poll(http, key, kinds={"tts": [7]}))
    like("result.tts", speech(http, key, live, "4d8f0b2e-7c1a-4b5e-9f3d-2a6c8e1b0d47"))
    like("result.gone", speech(http, key, live, "never-claimed-01"))
    text = next(j for j in admin(http, project, "GET", "state").json()["jobs"] if j["kind"] == "text")["id"]
    call(http, key, text, "claim", {"lease": "text-lease-00001"})
    like("result.text", call(http, key, text, "result", fixture("result.text")["request"] | {"lease": "text-lease-00001"}))
    line = submit(http, project)
    call(http, key, line, "claim", {"lease": "line-lease-00001"})
    like("fail.shutdown", call(http, key, line, "fail", {"lease": "line-lease-00001", "reason": "shutdown", "retry": True, "error": "stopping"}))
    call(http, key, line, "claim", {"lease": "line-lease-00002"})
    like("fail.invalid", call(http, key, line, "fail", {"lease": "line-lease-00002", "reason": "invalid", "retry": True, "error": "x"}))
    like("error.bad-key", poll(http, "f" * 64))
    worker = admin(http, project, "GET", "state").json()["workers"][0]["id"]
    admin(http, project, "PATCH", f"workers/{worker}", {"active": False})
    like("error.inactive", poll(http, key))


# --- the queue's rules ------------------------------------------------------------------------------

def test_offers_match_voice_language_and_model(served):
    project, _, http, _ = served
    key = joined(http, project)
    sohee = submit(http, project)
    faith_en = submit(http, project, voice="Faith", lang="en")
    other_model = submit(http, project, model="another-tts")
    gemma = submit(http, project, kind="text", model="gemma4-31b", prompt="Hi")
    qwen = submit(http, project, kind="text", model="qwen3.6-35b", prompt="Hi")
    offered = [o["task"] for o in poll(http, key).json()["offers"]]
    assert sohee in offered and gemma in offered
    assert faith_en not in offered, "Faith speaks German only"
    assert other_model not in offered and qwen not in offered, "a model it does not run"
    no_voices = {**CAPS, "engines": [{**CAPS["engines"][0], "voices": []}]}
    assert poll(http, key, caps=no_voices).json()["offers"] == [], "a worker that names no voices gets no speech"


def test_a_lapsed_lease_goes_back_and_its_late_result_counts_until_someone_else_claims(served):
    project, _, http, clock = served
    a, b = joined(http, project, "A"), joined(http, project, "B")
    job = submit(http, project)
    poll(http, a), poll(http, b)
    assert call(http, a, job, "claim", {"lease": "lease-of-a-001"}).status_code == 200
    assert call(http, b, job, "claim", {"lease": "lease-of-b-001"}).json() == {"error": "gone"}
    clock.now += 46
    assert job in [o["task"] for o in poll(http, b).json()["offers"]], "A went quiet: offered again"
    view = admin(http, project, "GET", f"jobs/{job}").json()
    assert view["state"] == "queued" and view["failures"] == 1
    assert speech(http, a, job, "lease-of-a-001").json() == {"ok": True}, "nobody claimed it since: A's late clip is taken"
    job2 = submit(http, project)
    call(http, a, job2, "claim", {"lease": "lease-of-a-002"})
    clock.now += 46
    poll(http, b)
    assert call(http, b, job2, "claim", {"lease": "lease-of-b-002"}).status_code == 200
    assert speech(http, a, job2, "lease-of-a-002").status_code == 409, "taken over: A's late clip is gone"


def test_a_poll_keeps_a_long_job_and_names_a_cancelled_one_back(served):
    project, _, http, clock = served
    key = joined(http, project)
    long_, cancelled = submit(http, project), submit(http, project)
    poll(http, key)
    call(http, key, long_, "claim", {"lease": "long-lease-0001"})
    call(http, key, cancelled, "claim", {"lease": "gone-lease-0001"})
    admin(http, project, "DELETE", f"jobs/{cancelled}")
    for _ in range(10):
        clock.now += 30
        running = [{"task": long_, "lease": "long-lease-0001"}, {"task": cancelled, "lease": "gone-lease-0001"}]
        answer = poll(http, key, send_caps=False, running=running, free={"tts": 0, "text": 0}).json()
        assert answer["cancel"] == [{"task": cancelled, "lease": "gone-lease-0001"}] and answer["retry_after"] <= 15
    assert admin(http, project, "GET", f"jobs/{long_}").json()["state"] == "leased", "five minutes, kept alive by its polls"
    assert speech(http, key, cancelled, "gone-lease-0001").status_code == 409, "a cancelled job's clip is turned away"
    assert speech(http, key, long_, "long-lease-0001").json() == {"ok": True}


def test_a_result_sent_twice_gets_the_same_answer_and_a_broken_one_is_refused(served):
    project, _, http, _ = served
    key = joined(http, project)
    job = submit(http, project)
    poll(http, key)
    call(http, key, job, "claim", {"lease": "twice-lease-001"})
    assert speech(http, key, job, "twice-lease-001", audio=b"<html>not a clip</html>").status_code == 422
    view = admin(http, project, "GET", f"jobs/{job}").json()
    assert view["state"] == "queued" and view["failures"] == 1, "back in the queue as a mishap"
    call(http, key, job, "claim", {"lease": "twice-lease-002"})
    assert speech(http, key, job, "twice-lease-002").json() == {"ok": True}
    assert speech(http, key, job, "twice-lease-002").json() == {"ok": True}, "sent again: the same answer"
    assert len(list((project.data / "results").glob(f"{job}-*.mp3"))) == 1, "and one clip kept"
    clip = admin(http, project, "GET", f"jobs/{job}/audio")
    assert clip.headers["content-type"] == "audio/mpeg" and clip.content == MP3


def test_failures_shutdown_counts_nothing_invalid_skips_that_worker_three_give_up(served):
    project, _, http, _ = served
    a, b = joined(http, project, "A"), joined(http, project, "B")
    job = submit(http, project)
    poll(http, a), poll(http, b)
    call(http, a, job, "claim", {"lease": "fail-lease-0001"})
    call(http, a, job, "fail", {"lease": "fail-lease-0001", "reason": "shutdown", "retry": True})
    assert admin(http, project, "GET", f"jobs/{job}").json()["failures"] == 0
    call(http, a, job, "claim", {"lease": "fail-lease-0002"})
    call(http, a, job, "fail", {"lease": "fail-lease-0002", "reason": "invalid", "retry": True, "error": "unknown voice"})
    assert job not in [o["task"] for o in poll(http, a).json()["offers"]], "not offered to the worker that cannot"
    assert job in [o["task"] for o in poll(http, b).json()["offers"]], "another may"
    for n in range(2):
        call(http, b, job, "claim", {"lease": f"fail-lease-b{n:03d}"})
        call(http, b, job, "fail", {"lease": f"fail-lease-b{n:03d}", "reason": "engine", "retry": True, "error": "boom"})
    view = admin(http, project, "GET", f"jobs/{job}").json()
    assert view["state"] == "failed" and view["error"] == "boom", "three mishaps: given up"


def test_a_job_is_checked_before_it_is_queued(served):
    project, _, http, _ = served
    bad = [({"kind": "tts", "text": "", "voice": "Sohee", "lang": "de"}, "bad_text"),
           ({"kind": "tts", "text": "x", "voice": "So hee", "lang": "de"}, "bad_voice"),
           ({"kind": "tts", "text": "x", "voice": "Sohee", "lang": "de", "temperature": 3}, "bad_temperature"),
           ({"kind": "text", "model": "gemma4-31b"}, "bad_messages"),
           ({"kind": "text", "model": "gemma4-31b", "prompt": "Hi", "json_schema": "nope"}, "bad_schema"),
           ({"kind": "image"}, "bad_kind")]
    for job, error in bad:
        r = admin(http, project, "POST", "jobs", job)
        assert (r.status_code, r.json()["error"]) == (422, error), job


# --- end to end: a real worker ---------------------------------------------------------------------------

def test_a_real_worker_speaks_and_answers_jobs_from_the_page_end_to_end(tmp_path):
    project = Project(tmp_path / "data", "end to end")
    server = Server(project, "127.0.0.1", 0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    http = httpx.Client(base_url=base, timeout=10)
    try:
        key = joined(http, project)
        spoken = submit(http, project, text="Guten Abend aus dem Test.")
        asked = submit(http, project, kind="text", model="gemma4-31b", prompt="Say hello as JSON.",
                       json_schema={"type": "object", "required": ["de"], "properties": {"de": {"type": "string"}}})
        config = parse({"tts": {"backend": "stub"}, "ui_port": 0}, tmp_path / "config.toml")
        pc = ProjectConfig(name="serve", url=base + "/api/worker/v2", key=key)
        tts = TtsEngine(StubBackend())
        tts.load()
        state = State(tmp_path)
        worker = Worker(config, [Link(pc, Connection(pc.name, pc.url, pc.key))], tts, StubText((TextModel(name="gemma4-31b", tag="x"),)),
                        state, Speeds(state, "stub"), platform={"os": "test", "arch": "x"}, local_now=datetime.now,
                        encoder=lambda pcm, rate: (b"ID3" + b"\x00" * 64, len(pcm) // 2 * 1000 // rate), scale=0.05)

        def done():
            jobs = {j["id"]: j for j in admin(http, project, "GET", "state").json()["jobs"]}
            return all(jobs[i]["state"] == "done" for i in (spoken, asked))

        async def scenario():
            runner = asyncio.create_task(worker.run())
            deadline = time.monotonic() + 20
            while not done():
                assert time.monotonic() < deadline, "timed out"
                if runner.done():
                    runner.result()
                await asyncio.sleep(0.05)
            worker.stop()
            await asyncio.wait_for(runner, 15)

        asyncio.run(scenario())
        answer = admin(http, project, "GET", f"jobs/{asked}").json()
        assert json.loads(answer["output"]["content"]) == {"de": "Stub text."}
        assert admin(http, project, "GET", f"jobs/{spoken}/audio").content.startswith(b"ID3")
        listed = admin(http, project, "GET", "state").json()["workers"][0]
        assert listed["version"].startswith("herde/") and {e["kind"] for e in listed["engines"]} == {"tts", "text"}
    finally:
        http.close()
        server.stop()
