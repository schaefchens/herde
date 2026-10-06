"""The worker against a fake station (httpx.MockTransport) and a fake engine: no network, no model."""

import json
import logging
import threading

import httpx
import pytest

from arche_worker.config import parse
from arche_worker.engine import InvalidTask, Speech
from arche_worker.main import Worker
from arche_worker.station import Station

KEY = "s3cret-" + "k" * 57
TEXT = "Maria aus Hamburg bittet um Gebet für ihre Mutter."
INSTRUCT = "Warm and calm."


def task(id=7, **over):
    return {
        "id": id, "purpose": "break", "text": TEXT, "lang": "de", "voice": "Sohee", "model": "qwen3-tts-1.7b-customvoice",
        "instruct": INSTRUCT, "temperature": 0.7, "seed": 3, "expected_ms": 4000, "lease_until": 2_000_000_000,
    } | over


class FakeStation:
    def __init__(self, polls=None, upload=(200, {"ok": True})):
        self.polls = list(polls or [])
        self.upload = upload
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/worker/poll":
            status, body = self.polls.pop(0) if self.polls else (200, {"task": None, "retry_after": 10})
            if status in (301, 302):
                return httpx.Response(status, headers={"location": "https://elsewhere.example.org/api/worker/poll"})
            return httpx.Response(status, json=body)
        if path.endswith("/audio"):
            return httpx.Response(self.upload[0], json=self.upload[1])
        if path.endswith("/fail"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"error": "not_found"})

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]

    def fails(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/fail")]


class FakeEngine:
    def __init__(self, ready=True, error=None, block=None):
        self.ready = ready
        self.error = error
        self.block = block
        self.calls = []

    def synthesize(self, *args):
        self.calls.append(args)
        if self.block is not None:
            self.block.wait(5)
        if self.error is not None:
            raise self.error
        return Speech(pcm16=b"\x00\x00" * 24000, sample_rate=24000)


class Clock:
    def __init__(self, step=0.0):
        self.now = 1000.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


def worker(stations, engine=None, clock=None, sleeps=None):
    names = [f"s{i}" for i in range(len(stations))]
    config = parse({
        "task_timeout_factor": 1.0,
        "stations": [{"name": n, "url": f"https://{n}.example.org", "key": KEY} for n in names],
    })
    st = [Station(c, transport=httpx.MockTransport(f.handler)) for c, f in zip(config.stations, stations)]
    sleeps = sleeps if sleeps is not None else []
    return Worker(config, engine or FakeEngine(), st, encoder=lambda pcm, rate, bitrate: (b"MP3DATA", 1000), sleep=sleeps.append, clock=clock or Clock())


def test_a_task_is_spoken_and_uploaded():
    station = FakeStation(polls=[(200, {"task": task(), "retry_after": 2})])
    engine = FakeEngine()
    w = worker([station], engine)
    assert w.step() is True
    poll, upload = station.requests
    assert poll.headers["x-arche-worker-key"] == KEY
    assert "authorization" not in poll.headers, "Authorization never reaches PHP on the station's host"
    report = json.loads(poll.content)
    assert report["ready"] is True
    assert report["engine"]["model"] == "qwen3-tts-1.7b-customvoice"
    assert len(report["voices"]) == 9 and {"id": "Sohee", "label": "Sohee"} in report["voices"]
    assert {"en", "de"} <= set(report["languages"])
    assert engine.calls == [(TEXT, "Sohee", "de", INSTRUCT, 0.7, 3)]
    assert upload.url.path == "/api/worker/tasks/7/audio"
    assert upload.headers["x-arche-worker-key"] == KEY
    body = upload.content
    assert b'name="audio"; filename="7.mp3"' in body and b"Content-Type: audio/mpeg" in body and b"MP3DATA" in body
    assert b'name="ms"\r\n\r\n1000' in body


def test_a_retake_or_a_task_gone_reports_no_failure(caplog):
    caplog.set_level(logging.INFO)
    retake = FakeStation(polls=[(200, {"task": task(), "retry_after": 2})], upload=(200, {"ok": False, "retake": True}))
    worker([retake]).step()
    assert retake.fails() == []
    assert "retake" in caplog.text
    gone = FakeStation(polls=[(200, {"task": task(), "retry_after": 2})], upload=(409, {"error": "task_gone"}))
    worker([gone]).step()
    assert gone.fails() == [] and "gone" in caplog.text


def test_idle_it_sleeps_as_long_as_the_station_says():
    station = FakeStation(polls=[(200, {"task": None, "retry_after": 7})])
    sleeps = []
    w = worker([station], sleeps=sleeps)

    def sleep(seconds):
        sleeps.append(seconds)
        w.stop()

    w.sleep = sleep
    w.run()
    assert sleeps == [7.0]


def test_nothing_is_polled_while_the_model_is_not_loaded():
    station = FakeStation()
    assert worker([station], FakeEngine(ready=False)).step() is False
    assert station.requests == []


def test_a_refused_key_pauses_that_station_and_never_shows_the_key(caplog):
    caplog.set_level(logging.DEBUG)
    station = FakeStation(polls=[(401, {"error": "bad_worker_key"})])
    w = worker([station])
    assert w.step() is False
    assert w.step() is False
    assert station.paths() == ["/api/worker/poll"], "left alone after the refusal"
    assert "bad_worker_key" in caplog.text and KEY not in caplog.text


def test_server_errors_back_off_and_redirects_are_not_followed():
    clock = Clock()
    broken = FakeStation(polls=[(503, {}), (503, {})])
    w = worker([broken], clock=clock)
    w.step()
    first = w.paused_until["s0"] - clock.now
    clock.now += first + 1
    w.step()
    assert w.paused_until["s0"] - clock.now > first, "longer each time"
    moved = FakeStation(polls=[(302, {})])
    w = worker([moved])
    w.step()
    assert moved.paths() == ["/api/worker/poll"] and "s0" in w.paused_until


def test_stations_are_asked_in_order_and_one_task_at_a_time():
    first = FakeStation(polls=[(200, {"task": task(id=1), "retry_after": 2})])
    second = FakeStation(polls=[(200, {"task": task(id=2), "retry_after": 2})])
    worker([first, second]).step()
    assert first.paths() == ["/api/worker/poll", "/api/worker/tasks/1/audio"]
    assert second.requests == [], "one take at a time: the second waits for the next pass"
    idle = FakeStation(polls=[(200, {"task": None, "retry_after": 5})])
    busy = FakeStation(polls=[(200, {"task": task(id=3), "retry_after": 2})])
    worker([idle, busy]).step()
    assert busy.paths() == ["/api/worker/poll", "/api/worker/tasks/3/audio"]


def test_a_task_it_cannot_speak_is_given_back_with_the_reason_never_the_text(caplog):
    caplog.set_level(logging.DEBUG)
    invalid = FakeStation(polls=[(200, {"task": task(voice="Hope"), "retry_after": 2})])
    worker([invalid], FakeEngine(error=InvalidTask("unknown voice 'Hope'"))).step()
    assert invalid.fails() == [{"error": "unknown voice 'Hope'", "retry": False}]
    broken = FakeStation(polls=[(200, {"task": task(), "retry_after": 2})])
    worker([broken], FakeEngine(error=RuntimeError("metal out of memory"))).step()
    assert broken.fails() == [{"error": "RuntimeError", "retry": True}]
    assert TEXT not in caplog.text and INSTRUCT not in caplog.text and KEY not in caplog.text


def test_a_take_that_runs_too_long_is_given_back_and_holds_the_worker_until_it_ends():
    release = threading.Event()
    station = FakeStation(polls=[(200, {"task": task(), "retry_after": 2})])
    # Every look at the clock is a minute later: the take's time runs out at once.
    w = worker([station], FakeEngine(block=release), clock=Clock(step=60.0))
    w.step()
    assert station.fails() == [{"error": "timed out", "retry": True}]
    assert w.step() is False and station.paths().count("/api/worker/poll") == 1, "no new task while the GPU is still busy"
    release.set()
    w.stuck.join(5)
    w.step()
    assert station.paths().count("/api/worker/poll") == 2


def test_a_stop_mid_take_gives_the_task_back():
    release = threading.Event()
    station = FakeStation(polls=[(200, {"task": task(), "retry_after": 2})])
    w = worker([station], FakeEngine(block=release))
    threading.Timer(0.3, w.stop).start()
    w.step()
    release.set()
    assert station.fails() == [{"error": "the worker stopped", "retry": True}]


@pytest.mark.parametrize("answer", [{"task": {"id": 0, "text": "x"}}, {"task": {"id": 5, "text": " "}}, ["not", "an", "object"]])
def test_a_garbled_poll_answer_is_an_error_not_a_crash(answer):
    station = FakeStation(polls=[(200, answer)])
    w = worker([station])
    assert w.step() is False
    assert "s0" in w.paused_until
