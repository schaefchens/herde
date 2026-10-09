"""One project's calls (protocol/PROTOCOL.md): join, poll, claim, result, fail.

The key travels in `X-Worker-Key`: an Authorization header never reaches PHP
on some shared hosts. Redirects are not followed, so a moved address can never
take the key to another host. A project's error answers become exceptions the
worker knows how to wait out; a task that is gone is no error at all.
"""

from __future__ import annotations

from typing import Literal

import httpx

from . import USER_AGENT
from .protocol import Joined, PollAnswer, ProtocolError, Task, TextOutput, error_code, parse_claim, parse_join, parse_poll


class ProjectError(RuntimeError):
    """The project did not answer as expected; try again later (5 s, doubling up to 60 s)."""


class BadKey(ProjectError):
    pass


class Inactive(ProjectError):
    pass


class RateLimited(ProjectError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class Upgrade(ProjectError):
    """The project speaks another protocol version: ask again in an hour."""


class Gone(Exception):
    """The task is taken, cancelled, too late, or none of this worker's business."""


Outcome = Literal["done", "retake", "gone", "invalid"]


def _client(url: str, headers: dict[str, str], transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        # A trailing slash: "poll" then lands under the base address, not beside it.
        base_url=url.rstrip("/") + "/",
        headers={"User-Agent": USER_AGENT, **headers},
        timeout=httpx.Timeout(60.0, connect=10.0),
        follow_redirects=False,
        transport=transport,
    )


class Connection:
    def __init__(self, name: str, url: str, key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.name = name
        self.url = url
        self._client = _client(url, {"X-Worker-Key": key}, transport)

    async def close(self) -> None:
        await self._client.aclose()

    async def poll(self, body: dict, received: float) -> PollAnswer:
        response = await self._post("poll", json=body)
        try:
            return parse_poll(self.name, self._json(response), received)
        except ProtocolError as e:
            raise ProjectError(f"{self.name}: {e}") from None

    async def claim(self, task_id: int, lease: str, received: float) -> Task:
        response = await self._post(f"tasks/{task_id}/claim", json={"lease": lease})
        if response.status_code in (404, 409):
            raise Gone()
        try:
            return parse_claim(self.name, task_id, lease, self._json(response), received)
        except ProtocolError as e:
            raise ProjectError(f"{self.name}: {e}") from None

    async def result_audio(self, task_id: int, lease: str, mp3: bytes, ms: int) -> Outcome:
        return self._outcome(await self._post(
            f"tasks/{task_id}/result",
            data={"lease": lease, "ms": str(int(ms))},
            files={"audio": (f"{task_id}.mp3", mp3, "audio/mpeg")},
        ))

    async def result_text(self, task_id: int, lease: str, output: TextOutput) -> Outcome:
        return self._outcome(await self._post(f"tasks/{task_id}/result", json=output.body(lease)))

    async def fail(self, task_id: int, lease: str, reason: str, retry: bool, error: str = "") -> None:
        response = await self._post(f"tasks/{task_id}/fail", json={"lease": lease, "reason": reason, "retry": retry, "error": error[:200]})
        # Gone meanwhile (cancelled, or the lease ran out): nothing left to report.
        if response.status_code not in (404, 409):
            self._json(response)

    def _outcome(self, response: httpx.Response) -> Outcome:
        if response.status_code in (404, 409):
            return "gone"
        if response.status_code in (413, 422):
            return "invalid"
        data = self._json(response)
        if data.get("ok") is True:
            return "done"
        if data.get("retake") is True:
            return "retake"
        raise ProjectError(f"{self.name}: unexpected answer to a result")

    async def _post(self, path: str, **kwargs) -> httpx.Response:
        return _checked(self.name, await _send(self.name, self._client, path, **kwargs))

    def _json(self, response: httpx.Response) -> dict:
        return _object(self.name, response)


async def join(url: str, code: str, name: str, version: str, platform: dict, transport: httpx.AsyncBaseTransport | None = None) -> Joined:
    """A one-time invite turned into a key (no key travels: this is how one is made)."""
    async with _client(url, {}, transport) as client:
        response = await _send(url, client, "join", json={"code": code, "name": name, "version": version, "platform": platform})
        if response.status_code == 401:
            raise BadKey(f"{url} did not take the invite code ({error_code(_maybe_json(response))}): it may be mistyped, used or expired")
        response = _checked(url, response)
        try:
            return parse_join(_object(url, response))
        except ProtocolError as e:
            raise ProjectError(f"{url}: {e}") from None


async def _send(label: str, client: httpx.AsyncClient, path: str, **kwargs) -> httpx.Response:
    try:
        return await client.post(path, **kwargs)
    except httpx.HTTPError as e:
        raise ProjectError(f"{label}: {type(e).__name__}") from None


def _checked(label: str, response: httpx.Response) -> httpx.Response:
    code = response.status_code
    if code == 401:
        raise BadKey(f"{label} refused the key ({error_code(_maybe_json(response))}): check the key file, or join again")
    if code == 403:
        raise Inactive(f"{label}: this worker is switched off by the project's admin")
    if code == 426:
        raise Upgrade(f"{label} speaks another protocol version ({_supported(response)}): update herde")
    if code == 429:
        data = _maybe_json(response)
        retry = data.get("retry_after") if isinstance(data, dict) else None
        raise RateLimited(f"{label}: too many requests", float(retry) if isinstance(retry, (int, float)) and not isinstance(retry, bool) else None)
    if response.is_redirect:
        raise ProjectError(f"{label}: redirected to {response.headers.get('location', '?')}; put the final address in the config")
    if code >= 500:
        raise ProjectError(f"{label}: HTTP {code}")
    if code == 400:
        raise ProjectError(f"{label}: the project could not read a request (bad_request)")
    return response


def _maybe_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return None


def _supported(response: httpx.Response) -> str:
    data = _maybe_json(response)
    supported = data.get("supported") if isinstance(data, dict) else None
    return f"it speaks {supported}" if supported else "unknown"


def _object(label: str, response: httpx.Response) -> dict:
    if response.status_code != 200:
        raise ProjectError(f"{label}: HTTP {response.status_code}")
    data = _maybe_json(response)
    if not isinstance(data, dict):
        raise ProjectError(f"{label}: answer is not a JSON object")
    return data
