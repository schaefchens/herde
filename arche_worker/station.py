"""The station's worker API: poll for a task, upload its MP3, or say why it failed.

The key travels in `X-Arche-Worker-Key`: an Authorization header never reaches
PHP on the station's host. Redirects are not followed, so a moved address
can never take the key to another host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import httpx

from . import VERSION
from .config import Station as StationConfig


class StationError(RuntimeError):
    """The station did not answer as expected; try again later."""


class BadKey(StationError):
    pass


class Inactive(StationError):
    pass


class RateLimited(StationError):
    pass


@dataclass(frozen=True)
class Task:
    id: int
    purpose: str
    lang: str
    voice: str
    model: str
    temperature: float
    seed: int
    expected_ms: int
    lease_until: int
    # Listeners' words, and the host's style: never in a log line.
    text: str = field(repr=False)
    instruct: str = field(repr=False, default="")


@dataclass(frozen=True)
class Poll:
    task: Task | None
    retry_after: float


Outcome = Literal["done", "retake", "gone", "invalid"]


class Station:
    def __init__(self, config: StationConfig, transport: httpx.BaseTransport | None = None):
        self.name = config.name
        self.url = config.url
        self._client = httpx.Client(
            base_url=config.url,
            headers={"X-Arche-Worker-Key": config.key, "User-Agent": f"arche-worker/{VERSION}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def poll(self, report: dict) -> Poll:
        data = self._json(self._post("/api/worker/poll", json=report))
        raw = data.get("task")
        retry = data.get("retry_after", 10)
        retry_after = float(retry) if isinstance(retry, (int, float)) and not isinstance(retry, bool) else 10.0
        return Poll(task=_task(raw) if raw is not None else None, retry_after=max(1.0, min(60.0, retry_after)))

    def upload(self, task_id: int, mp3: bytes, ms: int) -> Outcome:
        response = self._post(
            f"/api/worker/tasks/{task_id}/audio",
            data={"ms": str(int(ms))},
            files={"audio": (f"{task_id}.mp3", mp3, "audio/mpeg")},
        )
        if response.status_code == 409:
            return "gone"
        if response.status_code == 422:
            return "invalid"
        data = self._json(response)
        if data.get("ok") is True:
            return "done"
        if data.get("retake") is True:
            return "retake"
        raise StationError(f"{self.name}: unexpected upload answer")

    def fail(self, task_id: int, error: str, retry: bool) -> None:
        response = self._post(f"/api/worker/tasks/{task_id}/fail", json={"error": error[:200], "retry": retry})
        # Gone meanwhile (cancelled, or the lease ran out): nothing left to report.
        if response.status_code != 409:
            self._json(response)

    def _post(self, path: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.post(path, **kwargs)
        except httpx.HTTPError as e:
            raise StationError(f"{self.name}: {type(e).__name__}") from None
        if response.status_code == 401:
            raise BadKey(f"{self.name} refused the key (bad_worker_key): check the config")
        if response.status_code == 403:
            raise Inactive(f"{self.name}: this worker is switched off in /mod")
        if response.status_code == 429:
            raise RateLimited(f"{self.name}: too many requests")
        if response.is_redirect:
            raise StationError(f"{self.name}: redirected to {response.headers.get('location', '?')}; put the final address in the config")
        if response.status_code >= 500:
            raise StationError(f"{self.name}: HTTP {response.status_code}")
        return response

    def _json(self, response: httpx.Response) -> dict:
        if response.status_code != 200:
            raise StationError(f"{self.name}: HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError:
            raise StationError(f"{self.name}: answer is not JSON") from None
        if not isinstance(data, dict):
            raise StationError(f"{self.name}: answer is not an object")
        return data


def _task(raw: object) -> Task:
    if not isinstance(raw, dict) or not isinstance(raw.get("id"), int) or raw["id"] <= 0:
        raise StationError("the station sent a task without an id")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise StationError(f"task {raw['id']} has no text")
    return Task(
        id=raw["id"],
        purpose=str(raw.get("purpose", "")),
        lang=str(raw.get("lang", "")),
        voice=str(raw.get("voice", "")),
        model=str(raw.get("model", "")),
        temperature=float(raw.get("temperature", 0.7)),
        seed=int(raw.get("seed", 0)),
        expected_ms=int(raw.get("expected_ms") or len(text) * 1000 // 13),
        lease_until=int(raw.get("lease_until", 0)),
        text=text,
        instruct=str(raw.get("instruct") or ""),
    )
