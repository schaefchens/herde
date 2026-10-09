"""The local page: status for everyone on this computer, changes only with the page's own token."""

import http.client
import json
import urllib.parse

import pytest

from herde.state import State
from herde.ui import StatusPage


@pytest.fixture
def page(tmp_path):
    woken = []
    status = {"name": "studio-mac", "version": "0.2.0", "paused": False, "engines": [{"kind": "tts", "ready": True, "backend": "mlx", "voices": 10}],
              "projects": [{"name": "arche", "ok": True, "running": [], "done": 3, "failed": 0}]}
    p = StatusPage("127.0.0.1", 0, State(tmp_path), lambda: status, lambda: woken.append(1))
    p.start()
    p.woken = woken
    yield p
    p.stop()


def request(page, method, path, body=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", page.port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{page.port}"}
    if body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        body = urllib.parse.urlencode(body)
    conn.request(method, path, body=body, headers=headers)
    r = conn.getresponse()
    return r.status, r.read().decode(), dict(r.getheaders())


def test_the_page_shows_what_the_worker_does(page):
    status, body, headers = request(page, "GET", "/")
    assert status == 200 and "studio-mac" in body and "arche" in body and "Pause 1 hour" in body
    assert "default-src 'none'" in headers["Content-Security-Policy"] and headers["X-Frame-Options"] == "DENY"
    assert "<script" not in body
    status, body, _ = request(page, "GET", "/status.json")
    assert json.loads(body)["projects"][0]["done"] == 3


def test_pause_and_resume_with_the_pages_token(page):
    status, _, headers = request(page, "POST", "/pause", {"token": page.token, "hours": "3"})
    assert status == 303 and headers["Location"] == "/"
    paused, until = page.state.paused()
    assert paused and until is not None and page.woken
    request(page, "POST", "/pause", {"token": page.token, "hours": "0"})
    assert page.state.paused() == (True, None)
    _, body, _ = request(page, "GET", "/")
    assert "Resume" in body
    request(page, "POST", "/resume", {"token": page.token})
    assert page.state.paused() == (False, None)


def test_another_site_cannot_pause_it(page):
    status, _, _ = request(page, "POST", "/pause", {"hours": "0"})
    assert status == 403
    status, _, _ = request(page, "POST", "/pause", {"token": "guessed", "hours": "0"})
    assert status == 403
    assert page.state.paused() == (False, None)


def test_a_rebound_name_is_refused(page):
    status, _, _ = request(page, "GET", "/", host="evil.example.org")
    assert status == 403
    status, _, _ = request(page, "POST", "/pause", {"token": page.token}, host="evil.example.org:8737")
    assert status == 403 and page.state.paused() == (False, None)
    assert request(page, "GET", "/", host="localhost:8737")[0] == 200
