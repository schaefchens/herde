"""One project's calls against the protocol's own example answers (httpx.MockTransport): no network."""

import asyncio
import json
import re

import httpx
import pytest
from conftest import KEY, fixture

from herde.connection import BadKey, Connection, Gone, Inactive, ProjectError, RateLimited, Upgrade, join
from herde.protocol import TextOutput

URL = "https://radio.example.org/api/worker/v2"


class Replay:
    """Answers each call with a fixture's response and keeps what was sent."""

    def __init__(self, *names, status=None):
        self.answers = [fixture(n)["response"] for n in names]
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status is not None:
            return httpx.Response(self.status[0], headers=self.status[1] if len(self.status) > 1 else None)
        answer = self.answers.pop(0)
        return httpx.Response(answer["status"], json=answer["body"])


def conn(replay) -> Connection:
    return Connection("arche", URL, KEY, transport=httpx.MockTransport(replay))


def run(coro):
    return asyncio.run(coro)


def test_a_poll_goes_to_the_base_address_with_the_key_header_only():
    replay = Replay("poll.first")
    c = conn(replay)
    body = fixture("poll.first")["request"]
    answer = run(c.poll(body, 0.0))
    request = replay.requests[0]
    assert str(request.url) == URL + "/poll"
    assert request.headers["x-worker-key"] == KEY
    assert "authorization" not in request.headers, "an Authorization header never reaches PHP on some hosts"
    assert request.headers["user-agent"].startswith("herde/")
    assert json.loads(request.content) == body
    assert len(answer.offers) == 4


@pytest.mark.parametrize(("name", "error"), [("error.bad-key", BadKey), ("error.inactive", Inactive), ("poll.upgrade", Upgrade)])
def test_refusals_become_errors_the_worker_can_wait_out(name, error):
    with pytest.raises(error):
        run(conn(Replay(name)).poll(fixture("poll.idle")["request"], 0.0))


def test_server_trouble_and_redirects_are_project_errors():
    with pytest.raises(ProjectError):
        run(conn(Replay(status=(503,))).poll({}, 0.0))
    with pytest.raises(ProjectError, match="redirected"):
        run(conn(Replay(status=(302, {"location": "https://elsewhere.example.org/poll"}))).poll({}, 0.0))

    def broken(request):
        raise httpx.ConnectError("down")

    with pytest.raises(ProjectError, match="ConnectError"):
        run(Connection("arche", URL, KEY, transport=httpx.MockTransport(broken)).poll({}, 0.0))


def test_a_claim_gets_the_task_or_learns_it_is_gone():
    lease = fixture("claim.tts")["request"]["lease"]
    replay = Replay("claim.tts", "claim.gone")
    c = conn(replay)
    task = run(c.claim(52, lease, 10.0))
    assert task.id == 52 and task.lease == lease
    assert json.loads(replay.requests[0].content) == {"lease": lease}
    assert replay.requests[0].url.path.endswith("/tasks/52/claim")
    with pytest.raises(Gone):
        run(c.claim(52, lease, 10.0))


def test_speech_goes_back_as_multipart_and_the_answers_are_outcomes():
    lease = fixture("result.tts")["request"]["multipart"]["lease"]
    replay = Replay("result.tts", "result.tts-retake", "result.gone")
    c = conn(replay)
    assert run(c.result_audio(52, lease, b"ID3MP3DATA", 21400)) == "done"
    body = replay.requests[0].content
    assert b'name="audio"; filename="52.mp3"' in body and b"Content-Type: audio/mpeg" in body and b"ID3MP3DATA" in body
    assert re.search(rb'name="lease"\r\n\r\n' + lease.encode(), body) and b'name="ms"\r\n\r\n21400' in body
    assert run(c.result_audio(52, lease, b"x", 61400)) == "retake"
    assert run(c.result_audio(52, lease, b"x", 1)) == "gone"


def test_text_goes_back_as_json():
    f = fixture("result.text")
    replay = Replay("result.text")
    out = f["request"]["output"]
    output = TextOutput(content=out["content"], finish=out["finish"], input_tokens=out["usage"]["input_tokens"], output_tokens=out["usage"]["output_tokens"],
                        load_ms=out["timings"]["load_ms"], read_ms=out["timings"]["read_ms"], write_ms=out["timings"]["write_ms"])
    assert run(conn(replay).result_text(60, f["request"]["lease"], output)) == "done"
    assert json.loads(replay.requests[0].content) == f["request"]


def test_a_failure_is_reported_and_a_gone_task_needs_no_report():
    f = fixture("fail.shutdown")
    replay = Replay("fail.shutdown", "claim.gone")
    c = conn(replay)
    run(c.fail(52, f["request"]["lease"], "shutdown", True, "worker stopping"))
    assert json.loads(replay.requests[0].content) == f["request"]
    run(c.fail(52, f["request"]["lease"], "engine", True, "x" * 500))
    assert len(json.loads(replay.requests[1].content)["error"]) == 200


def test_join_sends_no_key_and_returns_one():
    f = fixture("join.ok")
    replay = Replay("join.ok")
    joined = run(join(URL, f["request"]["code"], "Peter's PC", "herde/0.2.0", f["request"]["platform"], transport=httpx.MockTransport(replay)))
    assert joined.trust == "lender" and len(joined.key) == 64
    request = replay.requests[0]
    assert "x-worker-key" not in request.headers
    assert str(request.url) == URL + "/join"
    assert json.loads(request.content) == {**f["request"], "version": "herde/0.2.0"}
    with pytest.raises(BadKey, match="invite"):
        run(join(URL, "AAAA-BBBB", "x", "herde/0.2.0", {}, transport=httpx.MockTransport(Replay("join.bad-code"))))
    with pytest.raises(RateLimited) as limited:
        run(join(URL, "AAAA-BBBB", "x", "herde/0.2.0", {}, transport=httpx.MockTransport(Replay("error.rate-limited"))))
    assert limited.value.retry_after == 600
