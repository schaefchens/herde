"""The queue of `herde serve`: the project side of the protocol (protocol/PROTOCOL.md) in one SQLite file.

A little project of its own, for trying herde on one computer and as a template
for a project's side: workers join with an invite, poll, claim a job with a lease
of their own, hand back what they made or say why they could not. ARCHE's PHP
(its server/app/Workers/) does the same on shared hosting, with its own rules on
top (who may see what, a clip's length against its words).

Standard library only. One lock around the database: a handful of workers and
one page polling every few seconds are nothing for SQLite, and every rule below
then holds without a race.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROTOCOL = 2
KINDS = {"tts": [1], "text": [1]}
CLASSES = ("live", "interactive", "background")
CLASS_ORDER = "CASE class WHEN 'live' THEN 0 WHEN 'interactive' THEN 1 ELSE 2 END"
LEASE_SECONDS = 45
RENEW_BELOW = 20
ONLINE_SECONDS = 90
# A worker holds this many jobs at most: one that breaks cannot sit on the whole queue.
HELD_MAX = 4
FAILURES_MAX = 3
OFFERS_MAX = 10
INVITE_SECONDS = 72 * 3600
WRONG_CODES_PER_HOUR = 10
TTS_CHARS_MAX = 2000
TEXT_CHARS_MAX = 200_000
AUDIO_BYTES_MAX = 25 * 1024 * 1024
# Unambiguous letters and digits: a code is read out and typed by hand.
CODE_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"
LEASE = re.compile(r"[A-Za-z0-9-]{8,64}")
VOICE = re.compile(r"[A-Za-z0-9_-]{1,40}")
MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
LANG = re.compile(r"[a-z]{2}")
HASH = re.compile(r"[0-9a-f]{64}")
FAIL_REASONS = ("shutdown", "cancelled", "invalid", "too_large", "engine", "timeout")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS workers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  key_mac TEXT NOT NULL UNIQUE,
  active INTEGER NOT NULL DEFAULT 1,
  caps TEXT NOT NULL DEFAULT '{}',
  caps_hash TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT '',
  resume_at INTEGER NOT NULL DEFAULT 0,
  version TEXT NOT NULL DEFAULT '',
  last_seen INTEGER NOT NULL DEFAULT 0,
  created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS invites (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  code_mac TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL DEFAULT '',
  expires INTEGER NOT NULL,
  used INTEGER NOT NULL DEFAULT 0,
  worker_id INTEGER,
  created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  class TEXT NOT NULL DEFAULT 'interactive',
  model TEXT NOT NULL DEFAULT '',
  voice TEXT NOT NULL DEFAULT '',
  lang TEXT NOT NULL DEFAULT '',
  size INTEGER NOT NULL DEFAULT 0,
  input TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'queued',
  worker_id INTEGER,
  lease TEXT NOT NULL DEFAULT '',
  lease_until INTEGER NOT NULL DEFAULT 0,
  outcome TEXT NOT NULL DEFAULT '',
  failures INTEGER NOT NULL DEFAULT 0,
  failed_by INTEGER NOT NULL DEFAULT 0,
  deadline INTEGER NOT NULL DEFAULT 0,
  due INTEGER NOT NULL DEFAULT 0,
  result TEXT NOT NULL DEFAULT '',
  ms INTEGER NOT NULL DEFAULT 0,
  error TEXT NOT NULL DEFAULT '',
  created INTEGER NOT NULL,
  started INTEGER NOT NULL DEFAULT 0,
  finished INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, id);
"""


class Answer(Exception):
    """An answer other than 200: its status and its body ({"error": code, ...})."""

    def __init__(self, status: int, error: str, **extra: Any):
        super().__init__(error)
        self.status = status
        self.body = {"error": error, **extra}


def family(model: str) -> str:
    """A speech model by its family: which checkpoint speaks a voice is the worker's business."""
    return re.sub(r"-(customvoice|base|voicedesign)$", "", model.strip().lower())


def grouped(code: str) -> str:
    return "-".join(code[i : i + 4] for i in range(0, len(code), 4)).upper()


class Project:
    def __init__(self, data: Path, name: str = "herde serve", clock: Callable[[], float] = time.time):
        self.data = data
        self.name = name[:80] or "herde serve"
        self.clock = clock
        data.mkdir(parents=True, exist_ok=True)
        os.chmod(data, 0o700)
        (data / "results").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(data / "serve.sqlite", check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.secret = self._meta("secret") or self._set_meta("secret", secrets.token_hex(32))
        self.token = self._token()
        self.wrong_codes: dict[str, deque[float]] = {}

    def now(self) -> int:
        return int(self.clock())

    # --- the page and the API (the admin) -------------------------------------------------------------

    def invite(self, name: str) -> dict:
        """A one-time code for a worker to join with: 130 random bits, kept as an HMAC, valid three days."""
        name = " ".join(str(name or "").split())[:60]
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(26))
        now = self.now()
        with self.lock:
            cur = self.db.execute("INSERT INTO invites (code_mac, name, expires, created) VALUES (?, ?, ?, ?)",
                                  (self._mac("invite:" + code), name, now + INVITE_SECONDS, now))
        return {"id": cur.lastrowid, "name": name, "code": grouped(code), "expires": now + INVITE_SECONDS}

    def invites(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id, name, expires FROM invites WHERE used = 0 AND expires > ? ORDER BY id", (self.now(),)).fetchall()
        return [dict(r) for r in rows]

    def withdraw(self, invite_id: int) -> None:
        with self.lock:
            if self.db.execute("DELETE FROM invites WHERE id = ? AND used = 0", (invite_id,)).rowcount != 1:
                raise Answer(404, "not_found")

    def set_active(self, worker_id: int, active: bool) -> None:
        with self.lock:
            if self.db.execute("UPDATE workers SET active = ? WHERE id = ?", (int(active), worker_id)).rowcount != 1:
                raise Answer(404, "not_found")
            if not active:
                self._release(worker_id)

    def remove_worker(self, worker_id: int) -> None:
        with self.lock:
            self._release(worker_id)
            if self.db.execute("DELETE FROM workers WHERE id = ?", (worker_id,)).rowcount != 1:
                raise Answer(404, "not_found")

    def submit(self, raw: Any) -> int:
        """A job, checked as a worker will read it: speech (text, voice, language) or text (a model, messages)."""
        if not isinstance(raw, dict):
            raise Answer(400, "bad_request")
        kind = raw.get("kind")
        cls = raw.get("class", "interactive")
        if cls not in CLASSES:
            raise Answer(422, "bad_class")
        if kind == "tts":
            inp, model, voice, lang, size = self._speech(raw)
        elif kind == "text":
            inp, model, voice, lang, size = self._text(raw)
        else:
            raise Answer(422, "bad_kind")
        now = self.now()
        due_in = raw.get("due_in")
        due = now + int(due_in) if isinstance(due_in, int) and not isinstance(due_in, bool) and due_in > 0 else 0
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO jobs (kind, class, model, voice, lang, size, input, deadline, due, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (kind, cls, model, voice, lang, size, json.dumps(inp, ensure_ascii=False), max(0, due - 30) if due else 0, due, now),
            )
        return int(cur.lastrowid)

    def cancel(self, job_id: int) -> None:
        """Not wanted any more: a worker holding it hears so with its next poll, and its result is turned away."""
        with self.lock:
            if self.db.execute("UPDATE jobs SET state = 'cancelled', error = 'cancelled', finished = ? WHERE id = ? AND state IN ('queued', 'leased')",
                               (self.now(), job_id)).rowcount != 1:
                raise Answer(409, "not_open")

    def job(self, job_id: int) -> dict:
        with self.lock:
            row = self.db.execute("SELECT j.*, w.name AS worker FROM jobs j LEFT JOIN workers w ON w.id = j.worker_id WHERE j.id = ?", (job_id,)).fetchone()
        if row is None:
            raise Answer(404, "not_found")
        return self._job_view(row)

    def audio(self, job_id: int) -> Path:
        with self.lock:
            row = self.db.execute("SELECT kind, state, result FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["kind"] != "tts" or row["state"] != "done" or not row["result"]:
            raise Answer(404, "not_found")
        return self.data / "results" / row["result"]

    def state(self) -> dict:
        """What the page shows: the workers, open invites, the latest jobs."""
        self.maintain()
        with self.lock:
            workers = self.db.execute("SELECT * FROM workers ORDER BY name, id").fetchall()
            jobs = self.db.execute("SELECT j.*, w.name AS worker FROM jobs j LEFT JOIN workers w ON w.id = j.worker_id ORDER BY j.id DESC LIMIT 50").fetchall()
        return {"name": self.name, "workers": [self._worker_view(w) for w in workers], "invites": self.invites(), "jobs": [self._job_view(j) for j in jobs]}

    # --- what workers call (PROTOCOL.md) ---------------------------------------------------------------

    def join(self, raw: Any, ip: str) -> dict:
        """An invite's code becomes a key, once. A wrong, used or expired code all get the same answer, and are counted per address."""
        raw = raw if isinstance(raw, dict) else {}
        code = re.sub(r"[^a-z0-9]", "", str(raw.get("code", "")).lower())
        now = self.now()
        with self.lock:
            row = self.db.execute("SELECT * FROM invites WHERE code_mac = ? AND used = 0 AND expires > ?", (self._mac("invite:" + code), now)).fetchone() if len(code) >= 16 else None
            if row is None:
                tries = self.wrong_codes.setdefault(ip, deque())
                while tries and tries[0] < now - 3600:
                    tries.popleft()
                tries.append(now)
                if len(tries) > WRONG_CODES_PER_HOUR:
                    raise Answer(429, "rate_limited", retry_after=600)
                raise Answer(401, "bad_code")
            if self.db.execute("UPDATE invites SET used = ? WHERE id = ? AND used = 0", (now, row["id"])).rowcount != 1:
                raise Answer(401, "bad_code")
            name = row["name"] or " ".join(str(raw.get("name", "")).split())[:60] or "worker"
            key = secrets.token_hex(32)
            cur = self.db.execute("INSERT INTO workers (name, key_mac, created) VALUES (?, ?, ?)", (name, self._mac("key:" + key), now))
            self.db.execute("UPDATE invites SET worker_id = ? WHERE id = ?", (cur.lastrowid, row["id"]))
        return {"key": key, "worker": {"id": cur.lastrowid, "name": name, "trust": "own"}, "project": {"name": self.name, "protocol": PROTOCOL}}

    def authenticate(self, key: str) -> sqlite3.Row:
        with self.lock:
            row = self.db.execute("SELECT * FROM workers WHERE key_mac = ?", (self._mac("key:" + key),)).fetchone() if len(key) >= 32 else None
        if row is None:
            raise Answer(401, "bad_worker_key")
        if not row["active"]:
            raise Answer(403, "worker_inactive")
        return row

    def poll(self, w: sqlite3.Row, raw: Any) -> dict:
        """The heartbeat and the market: what the worker says is noted, what it holds is kept or named back, and it is offered what it can do."""
        if not isinstance(raw, dict):
            raise Answer(400, "bad_request")
        theirs = raw.get("kinds") if isinstance(raw.get("kinds"), dict) else {}
        kinds = [k for k, versions in KINDS.items() if isinstance(theirs.get(k), list) and set(versions) & {v for v in theirs[k] if isinstance(v, int)}]
        if raw.get("protocol") != PROTOCOL or not kinds:
            raise Answer(426, "upgrade_required", supported={"protocol": [PROTOCOL], "kinds": KINDS})
        caps_hash = str(raw.get("caps_hash", ""))
        if not HASH.fullmatch(caps_hash):
            raise Answer(400, "bad_request")
        now = self.now()
        state = raw.get("state") if raw.get("state") in ("ready", "paused", "loading") else "ready"
        resume_in = raw.get("resume_in")
        resume_at = now + min(resume_in, 30 * 86400) if state == "paused" and isinstance(resume_in, int) and resume_in > 0 else 0
        self.maintain()
        with self.lock:
            sets = {"state": state, "resume_at": resume_at, "version": str(raw.get("version", ""))[:60], "last_seen": now}
            if isinstance(raw.get("caps"), dict):
                sets |= {"caps": json.dumps(caps(raw["caps"]), ensure_ascii=False), "caps_hash": caps_hash}
            self.db.execute(f"UPDATE workers SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?", (*sets.values(), w["id"]))
            w = self.db.execute("SELECT * FROM workers WHERE id = ?", (w["id"],)).fetchone()
            need_caps = "caps" not in raw and caps_hash != w["caps_hash"]
            cancel = self._heartbeat(w, raw.get("running"), now)
            free_raw = raw.get("free") if isinstance(raw.get("free"), dict) else {}
            free = {k: v if isinstance(v := free_raw.get(k), int) and not isinstance(v, bool) else 0 for k in kinds}
            offers = self._offers(w, free, now) if state == "ready" and not need_caps else []
        holding = len(raw.get("running") or []) > len(cancel)
        retry = 2 if offers else 5 if holding else 30 if state == "paused" else 10 if state == "loading" else 3
        return {"now": now, "offers": offers, "cancel": cancel, "need_caps": need_caps, "retry_after": retry}

    def claim(self, w: sqlite3.Row, job_id: int, raw: Any) -> dict:
        lease = str((raw or {}).get("lease", "")) if isinstance(raw, dict) else ""
        if not LEASE.fullmatch(lease):
            raise Answer(400, "bad_request")
        now = self.now()
        with self.lock:
            j = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if j is None:
                raise Answer(409, "gone")
            # The same claim again (its answer lost on the way): the same job.
            if j["state"] == "leased" and j["lease"] == lease and j["worker_id"] == w["id"]:
                return {"task": self._task_view(j, now)}
            open_ = j["state"] == "queued" or (j["state"] == "leased" and j["lease_until"] < now)
            in_time = (j["deadline"] == 0 or j["deadline"] > now) and (j["due"] == 0 or j["due"] > now)
            held = self.db.execute("SELECT COUNT(*) FROM jobs WHERE worker_id = ? AND state = 'leased' AND lease_until >= ?", (w["id"], now)).fetchone()[0]
            if not open_ or not in_time or j["failed_by"] == w["id"] or held >= HELD_MAX or not can_do(_caps(w), j):
                raise Answer(409, "gone")
            # A lease that lapsed without a word from its worker counts as a mishap of the job.
            self.db.execute(
                "UPDATE jobs SET state = 'leased', worker_id = ?, lease = ?, lease_until = ?, outcome = '', started = ?, failures = failures + ? WHERE id = ?",
                (w["id"], lease, now + LEASE_SECONDS, now, int(j["state"] == "leased"), job_id),
            )
            return {"task": self._task_view(self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone(), now)}

    def result(self, w: sqlite3.Row, job_id: int, lease: str, audio: bytes | None, ms: int, output: Any) -> dict:
        """What a worker made. The same result sent again gets the answer the first one got."""
        now = self.now()
        with self.lock:
            j = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if j is None:
                raise Answer(409, "gone")
            if j["lease"] == lease and j["worker_id"] == w["id"] and j["outcome"]:
                return _replay(j["outcome"])
            # A late result whose lease lapsed is still taken while nobody claimed the job since.
            if j["state"] not in ("leased", "queued") or j["lease"] != lease or j["worker_id"] != w["id"]:
                raise Answer(409, "gone")
            if j["kind"] == "tts":
                if not audio or len(audio) > AUDIO_BYTES_MAX or not _is_mp3(audio):
                    self._requeue(j, failures=j["failures"] + 1, error="no MP3", outcome=json.dumps({"status": 422, "body": {"error": "invalid_result"}}))
                    raise Answer(422, "invalid_result")
                name = f"{job_id}-{secrets.token_hex(4)}.mp3"
                tmp = self.data / "results" / (name + ".tmp")
                tmp.write_bytes(audio)
                tmp.replace(self.data / "results" / name)
                result, kept_ms = name, max(0, min(int(ms), 3_600_000))
            else:
                content = output.get("content") if isinstance(output, dict) else None
                if not isinstance(content, str) or len(content) > TEXT_CHARS_MAX:
                    self._requeue(j, failures=j["failures"] + 1, error="no answer", outcome=json.dumps({"status": 422, "body": {"error": "invalid_result"}}))
                    raise Answer(422, "invalid_result")
                usage = output.get("usage") if isinstance(output.get("usage"), dict) else {}
                result = json.dumps({"content": content, "finish": "length" if output.get("finish") == "length" else "stop",
                                     "usage": {"input_tokens": int(usage.get("input_tokens") or 0), "output_tokens": int(usage.get("output_tokens") or 0)}},
                                    ensure_ascii=False)
                kept_ms = 0
            self.db.execute("UPDATE jobs SET state = 'done', result = ?, ms = ?, error = '', outcome = ?, finished = ? WHERE id = ?",
                            (result, kept_ms, json.dumps({"status": 200, "body": {"ok": True}}), now, job_id))
        return {"ok": True}

    def fail(self, w: sqlite3.Row, job_id: int, raw: Any) -> dict:
        raw = raw if isinstance(raw, dict) else {}
        lease = str(raw.get("lease", ""))
        reason = raw.get("reason") if raw.get("reason") in FAIL_REASONS else "engine"
        with self.lock:
            j = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if j is not None and j["lease"] == lease and j["worker_id"] == w["id"] and j["outcome"]:
                return _replay(j["outcome"])
            # Gone meanwhile (cancelled, taken over): nothing left to report.
            if j is None or j["state"] not in ("leased", "queued") or j["lease"] != lease or j["worker_id"] != w["id"]:
                return {"ok": True}
            ok = json.dumps({"status": 200, "body": {"ok": True}})
            if reason in ("shutdown", "cancelled"):
                self._requeue(j, outcome=ok)
                return {"ok": True}
            error = " ".join(str(raw.get("error", "")).split())[:200] or reason
            failures = j["failures"] + 1
            give_up = raw.get("retry") is False or failures >= FAILURES_MAX
            self._requeue(j, failures=failures, error=error, outcome=ok, give_up=give_up,
                          **({"failed_by": w["id"]} if reason in ("invalid", "too_large") else {}))
        return {"ok": True}

    def maintain(self) -> None:
        """Leases that lapsed go back to the queue (a computer asleep mid-job); jobs past their time are cancelled."""
        now = self.now()
        with self.lock:
            for j in self.db.execute("SELECT * FROM jobs WHERE state = 'leased' AND lease_until < ?", (now,)).fetchall():
                # The lease stays with the row: its late result is still taken while nobody claimed the job since.
                self._requeue(j, failures=j["failures"] + 1, error="lease ran out", give_up=j["failures"] + 1 >= FAILURES_MAX, keep_lease=True)
            self.db.execute(
                "UPDATE jobs SET state = 'cancelled', error = 'too late', finished = ? WHERE (state = 'queued' AND deadline > 0 AND deadline <= ?) "
                "OR (state IN ('queued', 'leased') AND due > 0 AND due <= ?)",
                (now, now, now),
            )

    # --- inside ------------------------------------------------------------------------------------------

    def _heartbeat(self, w: sqlite3.Row, running: Any, now: int) -> list[dict]:
        cancel = []
        for r in (running if isinstance(running, list) else [])[:20]:
            job_id = r.get("task") if isinstance(r, dict) else None
            lease = str(r.get("lease", "")) if isinstance(r, dict) else ""
            if not isinstance(job_id, int) or not LEASE.fullmatch(lease):
                continue
            j = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if j is None or j["state"] != "leased" or j["lease"] != lease or j["worker_id"] != w["id"] or (j["due"] and j["due"] <= now):
                cancel.append({"task": job_id, "lease": lease})
            elif j["lease_until"] - now < RENEW_BELOW:
                self.db.execute("UPDATE jobs SET lease_until = ? WHERE id = ?", (now + LEASE_SECONDS, job_id))
        return cancel

    def _offers(self, w: sqlite3.Row, free: dict[str, int], now: int) -> list[dict]:
        rows = self.db.execute(
            f"SELECT * FROM jobs WHERE (state = 'queued' OR (state = 'leased' AND lease_until < ?)) AND (deadline = 0 OR deadline > ?) "
            f"AND (due = 0 OR due > ?) AND failed_by != ? ORDER BY {CLASS_ORDER}, CASE WHEN deadline = 0 THEN 1 ELSE 0 END, deadline, id LIMIT 500",
            (now, now, now, w["id"]),
        ).fetchall()
        offers, have = [], _caps(w)
        for j in rows:
            if free.get(j["kind"], 0) > 0 and can_do(have, j):
                offers.append(self._offer_view(j, now))
                if len(offers) >= OFFERS_MAX:
                    break
        return offers

    def _requeue(self, j: sqlite3.Row, failures: int | None = None, error: str | None = None, outcome: str = "", give_up: bool = False,
                 failed_by: int | None = None, keep_lease: bool = False) -> None:
        inp = json.loads(j["input"])
        # A new seed for another try, unless the worker only went quiet.
        if j["kind"] == "tts" and not keep_lease and not give_up:
            inp["seed"] = int(inp.get("seed", 0)) + 1
        self.db.execute(
            "UPDATE jobs SET state = ?, input = ?, lease_until = 0, outcome = ?, failures = ?, error = ?, failed_by = ?, finished = ? WHERE id = ?",
            ("failed" if give_up else "queued", json.dumps(inp, ensure_ascii=False), outcome if not keep_lease else j["outcome"],
             j["failures"] if failures is None else failures, j["error"] if error is None else error,
             j["failed_by"] if failed_by is None else failed_by, self.now() if give_up else 0, j["id"]),
        )

    def _release(self, worker_id: int) -> None:
        self.db.execute("UPDATE jobs SET state = 'queued', worker_id = NULL, lease = '', lease_until = 0, outcome = '' WHERE worker_id = ? AND state = 'leased'", (worker_id,))

    def _speech(self, raw: dict) -> tuple[dict, str, str, str, int]:
        text = str(raw.get("text", "")).strip()
        voice, lang = str(raw.get("voice", "")), str(raw.get("lang", ""))
        model = str(raw.get("model") or "qwen3-tts-1.7b")
        if not text or len(text) > TTS_CHARS_MAX:
            raise Answer(422, "bad_text")
        if not VOICE.fullmatch(voice) or not LANG.fullmatch(lang) or not MODEL.fullmatch(model):
            raise Answer(422, "bad_voice")
        temperature = raw.get("temperature", 0.7)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0.1 <= temperature <= 1.2:
            raise Answer(422, "bad_temperature")
        inp = {"text": text, "lang": lang, "voice": voice, "model": model, "instruct": str(raw.get("instruct", ""))[:1000],
               "temperature": float(temperature), "seed": int(raw.get("seed", 0)) if isinstance(raw.get("seed"), int) else 0}
        return inp, family(model), voice, lang, len(text)

    def _text(self, raw: dict) -> tuple[dict, str, str, str, int]:
        model = str(raw.get("model", ""))
        if not MODEL.fullmatch(model):
            raise Answer(422, "bad_model")
        messages = raw.get("messages")
        if messages is None and str(raw.get("prompt", "")).strip():
            messages = [{"role": "user", "content": str(raw["prompt"]).strip()}]
        if not isinstance(messages, list) or not messages or not all(
                isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) for m in messages):
            raise Answer(422, "bad_messages")
        system = str(raw.get("system", ""))
        size = len(system) + sum(len(m["content"]) for m in messages)
        if size > TEXT_CHARS_MAX:
            raise Answer(422, "too_large")
        max_tokens = raw.get("max_tokens", 512)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
            raise Answer(422, "bad_max_tokens")
        inp: dict[str, Any] = {"model": model, "messages": [{"role": m["role"], "content": m["content"]} for m in messages], "max_tokens": max_tokens}
        if system:
            inp["system"] = system
        if raw.get("json_schema") is not None:
            if not isinstance(raw["json_schema"], dict):
                raise Answer(422, "bad_schema")
            inp["json_schema"] = raw["json_schema"]
        for k, lo, hi in (("temperature", 0, 2), ("top_p", 0, 1)):
            v = raw.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi:
                inp[k] = float(v)
        if isinstance(raw.get("top_k"), int) and not isinstance(raw["top_k"], bool) and raw["top_k"] >= 1:
            inp["top_k"] = raw["top_k"]
        return inp, model, "", "", size // 4

    def _offer_view(self, j: sqlite3.Row, now: int) -> dict:
        o: dict[str, Any] = {"task": j["id"], "kind": j["kind"], "class": j["class"], "model": j["model"]}
        if j["kind"] == "tts":
            o |= {"voice": j["voice"], "lang": j["lang"], "size": {"chars": j["size"]}}
        else:
            o |= {"size": {"input_tokens": j["size"], "max_tokens": int(json.loads(j["input"]).get("max_tokens", 0))}}
        return o | _times(j, now)

    def _task_view(self, j: sqlite3.Row, now: int) -> dict:
        return {"id": j["id"], "kind": j["kind"], "class": j["class"], "lease_in": max(0, j["lease_until"] - now)} | _times(j, now) | {"input": json.loads(j["input"])}

    def _job_view(self, j: sqlite3.Row) -> dict:
        inp = json.loads(j["input"])
        view = {"id": j["id"], "kind": j["kind"], "class": j["class"], "state": j["state"], "model": j["model"], "voice": j["voice"], "lang": j["lang"],
                # On a sqlite3.Row, `in` looks at the values, not the column names.
                "worker": j["worker"] if "worker" in j.keys() else None, "failures": j["failures"], "error": j["error"],  # noqa: SIM118
                "created": j["created"], "started": j["started"], "finished": j["finished"], "ms": j["ms"]}
        view["input"] = inp.get("text") if j["kind"] == "tts" else (inp.get("messages") or [{}])[-1].get("content", "")
        if j["kind"] == "text" and j["state"] == "done":
            view["output"] = json.loads(j["result"])
        return view

    def _worker_view(self, w: sqlite3.Row) -> dict:
        now = self.now()
        have = _caps(w)
        online = bool(w["active"]) and w["state"] == "ready" and w["last_seen"] >= now - ONLINE_SECONDS
        return {"id": w["id"], "name": w["name"], "active": bool(w["active"]), "online": online,
                "state": w["state"] if w["last_seen"] >= now - ONLINE_SECONDS else "", "resume_at": w["resume_at"], "version": w["version"],
                "last_seen": w["last_seen"], "platform": have.get("platform", {}), "engines": have.get("engines", [])}

    def _token(self) -> str:
        """The page's key: in a file only this user may read (the page asks for it once)."""
        path = self.data / "token"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        token = secrets.token_urlsafe(24)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token)
        return token

    def _mac(self, value: str) -> str:
        return hmac.new(bytes.fromhex(self.secret), value.encode(), hashlib.sha256).hexdigest()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> str:
        self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        return value


def caps(raw: dict) -> dict:
    """A worker's capabilities as kept: what does not fit the protocol's schema is left out, never trusted."""
    out: dict[str, Any] = {"name": str(raw.get("name", ""))[:80], "engines": []}
    if isinstance(raw.get("platform"), dict):
        out["platform"] = {k: v for k, v in raw["platform"].items() if k in ("os", "arch", "accelerator", "memory_gb") and isinstance(v, (str, int, float))}
    for e in (raw.get("engines") if isinstance(raw.get("engines"), list) else [])[:50]:
        if not isinstance(e, dict) or e.get("kind") not in ("tts", "text", "stt") or not MODEL.fullmatch(str(e.get("model", ""))):
            continue
        engine: dict[str, Any] = {"kind": e["kind"], "model": e["model"], "location": "cloud" if e.get("location") == "cloud" else "local"}
        if e["kind"] == "tts":
            engine["voices"] = [
                {"id": v["id"], "label": str(v.get("label", v["id"]))[:60], "langs": [x for x in v.get("langs", []) if isinstance(x, str) and LANG.fullmatch(x)][:40],
                 "instruct": v.get("instruct") is not False, "own": v.get("own") is True}
                for v in (e.get("voices") if isinstance(e.get("voices"), list) else [])[:100]
                if isinstance(v, dict) and VOICE.fullmatch(str(v.get("id", "")))
            ]
        out["engines"].append(engine)
    return out


def can_do(have: dict, j: sqlite3.Row) -> bool:
    """Whether a worker's caps cover a job: a speech model's family with the voice in the job's language, or the text model by name."""
    for e in have.get("engines", []):
        if e.get("kind") != j["kind"]:
            continue
        if j["kind"] == "text" and e.get("model") == j["model"]:
            return True
        if j["kind"] == "tts" and family(e.get("model", "")) == family(j["model"]) and any(
                v["id"].lower() == j["voice"].lower() and j["lang"] in v.get("langs", []) for v in e.get("voices", [])):
            return True
    return False


def _caps(w: sqlite3.Row) -> dict:
    try:
        return json.loads(w["caps"])
    except ValueError:
        return {}


def _times(j: sqlite3.Row, now: int) -> dict:
    out = {}
    if j["deadline"]:
        out["claim_in"] = max(0, j["deadline"] - now)
    if j["due"]:
        out["due_in"] = max(0, j["due"] - now)
    return out


def _replay(outcome: str) -> dict:
    o = json.loads(outcome)
    if o.get("status") != 200:
        raise Answer(int(o.get("status", 409)), str(o.get("body", {}).get("error", "gone")))
    return o["body"]


def _is_mp3(data: bytes) -> bool:
    """An MP3 starts with an ID3 tag or a frame's sync bits."""
    return data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)
