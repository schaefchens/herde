# The herde protocol, version 2

How a worker (a computer with AI models) gets work from a project and hands
back the results. Version 1 was ARCHE's first voice-worker API; this is its
successor for any project and any kind of work.

**Roles:**
- **The project** hosts the calls below, holds the queue, and decides who may see what.
- **The worker** always starts the conversation. It needs no public address and no open port, so it works behind a firewall or NAT.

One worker may serve many projects, and one project many workers.

`schema.json` describes every message, and `fixtures/` holds an example of
every exchange. Both sides test against them.

## Basics

- **Transport.** JSON over HTTPS, under one base address the project names, for example `https://radio.schaefchens.de/api/worker/v2`. Plain HTTP only to the worker's own machine (`localhost`).
- **Key.** It travels in the header `X-Worker-Key`: an `Authorization` header does not reach PHP on some shared hosts.
  - Workers never put a key in a URL.
  - They never follow a redirect, so a moved address cannot carry the key elsewhere.
- **Fields.** Both sides ignore fields they do not know. New optional fields need no new version.
- **Times.**
  - Every limit is given as **seconds from now** (`claim_in`, `due_in`, `lease_in`), never as a clock time, because a lent computer's clock may be off.
  - `now` in a poll answer is the project's own clock, in Unix seconds, for logs.
- **Errors.** Any status other than 200 carries `{"error": "<code>"}`:

  | Status | `error` | The worker then … |
  |---|---|---|
  | 400 | `bad_request` | logs it; a bug on one side |
  | 401 | `bad_worker_key` / `bad_code` | leaves the project alone for 5 min (a key was revoked or mistyped) |
  | 403 | `worker_inactive` | leaves it alone for 5 min (switched off by the project's admin) |
  | 404 | `not_found` | treats it like `gone` for a task |
  | 409 | `gone` | drops that task: it was taken, cancelled, given up, or is none of this worker's business |
  | 413 | `too_large` | fails the task as `too_large` |
  | 422 | `invalid_result` | fails the task as `engine` |
  | 426 | `upgrade_required` | stops asking this project for an hour and says so in its log; `supported` lists the versions the project speaks |
  | 429 | `rate_limited` | waits `retry_after` seconds, else 60 |
  | 5xx, no answer | — | waits 5 s, doubling up to 60 s |

## The calls

| Call | What it is for |
|---|---|
| `POST join` | Turns a one-time invite into a key |
| `POST poll` | Heartbeat, capabilities, and the list of open work |
| `POST tasks/{id}/claim` | Takes one task and gets its input |
| `POST tasks/{id}/result` | Hands back what was made |
| `POST tasks/{id}/fail` | Says why it could not be made |
| `GET tasks/{id}/input` | Reserved: a large input such as audio to transcribe |

### join

The project's admin creates an invite; a worker's owner exchanges it once.

```
POST join   {"code": "7KQ3-M2XP-…", "name": "Peter's PC", "version": "herde/0.2.0",
             "platform": {"os": "linux", "arch": "x86_64", "accelerator": "NVIDIA GeForce RTX 4070"}}
→ 200       {"key": "<secret>", "worker": {"id": 7, "name": "Peter's PC", "trust": "lender"},
             "project": {"name": "ARCHE", "protocol": 2}}
```

- **The code.** At least 128 random bits. The project keeps only an HMAC of it, accepts it once, and lets it expire (ARCHE: 72 hours).
  - A wrong, used or expired code all get the same 401 `bad_code`.
  - Wrong codes are rate-limited per address.
- **Trust** comes from the invite, never from the worker:
  - `own`: our own computer, which may see private work;
  - `lender`: someone else's, which gets public work only.
- **The key** is stored by the worker in a file only its owner can read.

### poll

The heartbeat and the market in one call. A worker polls every `retry_after` seconds, and once more at once after a task ends.

```
POST poll
{
  "protocol": 2,
  "version": "herde/0.2.0",
  "kinds": {"tts": [1], "text": [1]},
  "state": "ready",
  "caps_hash": "3f1a…",
  "caps": { … },
  "running": [{"task": 41, "lease": "b5e1…", "progress": 0.5}],
  "free": {"tts": 0, "text": 0}
}
→ 200
{
  "now": 1760011234,
  "offers": [
    {"task": 52, "kind": "tts", "class": "live", "model": "qwen3-tts-1.7b", "voice": "Sohee", "lang": "de",
     "size": {"chars": 312}, "claim_in": 141, "due_in": 171}
  ],
  "cancel": [{"task": 40, "lease": "a9c0…"}],
  "need_caps": false,
  "retry_after": 3
}
```

**The worker's side:**
- **`state`:**
  - `ready`: takes work;
  - `paused`: its owner paused it, or it is outside its hours. It is online but takes nothing; `resume_in` may say when it is back.
  - `loading`: its models are still loading. The project counts it as not there yet.
- **`free`:** how many tasks of each kind it could start now. One GPU shared by speech and text shows 0 for both while it works.
- **`running`:** the tasks it holds, each with its lease. Listing a task keeps the lease alive (see Leases).
- **`caps`:** what it can do (below). It is sent on the first poll after a start, whenever it changes, and when the project asked with `need_caps`. Otherwise only `caps_hash` goes: the SHA-256 of the caps as canonical JSON (keys sorted, no spaces, UTF-8).
- **`kinds`:** the kinds and kind versions it speaks.

**The project's side:**
- **`offers`:** at most 10 tasks this worker could and may do, most urgent first, as **metadata only**. No text and no input, so a worker learns nothing it does not take.
  - An offer is no promise; another worker may claim it first.
  - A project may send no offers while every `free` is 0.
- **`cancel`:** held tasks the project no longer wants, because they were cancelled, are too late, or their lease lapsed. The worker stops them and sends nothing more about them.
- **`need_caps`:** the project lost or doubts the caps; the next poll should carry them.
- **`retry_after`:** 1–60 seconds. **At most 15 while the worker holds tasks**, so leases stay alive.

**Capabilities:**
```json
{
  "name": "css-mbp",
  "platform": {"os": "darwin", "arch": "arm64", "accelerator": "Apple M1 Max", "memory_gb": 64},
  "engines": [
    {"kind": "tts", "model": "qwen3-tts-1.7b", "location": "local",
     "voices": [
       {"id": "Sohee", "label": "Sohee", "langs": ["de", "en", "fr", "…"], "instruct": true},
       {"id": "Faith", "label": "Faith (own voice)", "langs": ["de", "en"], "instruct": false, "own": true}
     ],
     "max_chars": 2000, "speed": {"chars_per_s": 7}},
    {"kind": "text", "model": "gemma4-31b", "location": "local",
     "json_schema": true, "context": 32768, "max_tokens": 4096,
     "speed": {"read_tps": 67, "write_tps": 11, "load_s": 15}, "loaded": true}
  ]
}
```
- **`model`** is the name projects ask for, the same on every worker. The worker maps it to its local files, for example `gemma4-31b` → Ollama's `gemma4:31b-mlx` on a Mac.
- **`location`:** `local`, or `cloud` for an engine that sends the input to a provider (coming later). A project that must keep data on its own machines offers such work only to local engines.
- **`speed`** is measured by the worker and rounded. Projects may use it to judge whether anyone can finish in time.
- **`instruct`:** whether a voice takes a spoken direction. Own voices are cloned from a recording and take none.

### claim

```
POST tasks/52/claim   {"lease": "4d8f0b2e-…"}
→ 200 {"task": {"id": 52, "kind": "tts", "class": "live", "lease_in": 45, "due_in": 170,
                "input": {"text": "…", "lang": "de", "voice": "Sohee", "model": "qwen3-tts-1.7b",
                          "instruct": "Warm and calm.", "temperature": 0.7, "seed": 0}}}
→ 409 {"error": "gone"}
```

- **The lease** is a token the worker makes for this claim: a random UUID, 8–64 characters of `[A-Za-z0-9-]`. Claim, result and fail all carry it.
  - The same claim sent again gets the same 200.
  - A claim with another token gets `gone` while the first lease holds.
- **`gone`** covers everything a worker may not do: taken, cancelled, too late, too many held at once, or private work for a lender. A worker cannot probe what exists.
- **Before claiming,** a worker should be able to finish by `due_in`. It decides that from its own measured speed.

### result

The answer is idempotent: a result sent again with the same lease gets the answer the first one got, so a lost connection never counts twice.

**Speech** is multipart, because projects may cut JSON bodies (ARCHE at 1 MiB):
```
POST tasks/52/result   multipart/form-data: lease=4d8f…, ms=21400, audio=<52.mp3, audio/mpeg>
```
**Text** is JSON:
```
POST tasks/60/result   {"lease": "…", "output": {"content": "{\"de\": …}", "finish": "stop",
                        "usage": {"input_tokens": 3912, "output_tokens": 288},
                        "timings": {"load_ms": 0, "read_ms": 5840, "write_ms": 26100}}}
```

| Answer | Meaning |
|---|---|
| `{"ok": true}` | Taken; the task is done. |
| `{"ok": false, "retake": true}` | Made but not right, e.g. speech whose length does not fit its words. The task is back in the queue with another seed; this worker may claim it again. |
| 409 `gone` | Nobody wants it any more. |
| 422 `invalid_result` | Not usable at all (not an MP3, not JSON). |

### fail

```
POST tasks/52/fail   {"lease": "4d8f…", "reason": "shutdown", "retry": true, "error": "worker stopping"}
→ 200 {"ok": true}
```

| `reason` | Meaning | What a project should do |
|---|---|---|
| `shutdown` | The worker is stopping | Back to the queue; nothing counted |
| `cancelled` | Stopped because of `cancel` | Nothing; it is cancelled already |
| `invalid` | This worker cannot do it (a voice or model it lacks) | Back to the queue, not offered to this worker again |
| `too_large` | Over this worker's limits | Same as `invalid` |
| `engine` | The model failed | Count a failure; back to the queue until the project's limit |
| `timeout` | It took far longer than it should | Same as `engine` |

- **`retry: false`** says the task cannot succeed anywhere (empty text, a broken schema), and the project may give it up.
- **`error`** is a short reason for logs (at most 200 characters). It never quotes the input.
- A fail sent again with the same lease gets the same answer.

### input (reserved)

`GET tasks/{id}/input` with `X-Worker-Lease: <lease>` will return a large input as binary, such as audio to transcribe (kind `stt`, coming later). The input is given only while the lease holds. Workers never fetch any other URL a task might contain.

## Leases

- **A claimed task belongs to the worker while its polls list it.** Each poll that lists a task moves its lease to 45 s ahead; a project may do so only when fewer than 20 s are left, to save writes.
- **Silence:**
  - A worker that stops polling loses its tasks after 45 s, and they go back to the queue.
  - A lapse counts as a failure, not as a bad result: a sleeping computer says nothing about the work.
- **A task past `due_in`** is cancelled even while held, and turns up in `cancel`.
- **A late result** whose lease lapsed is still taken if nobody claimed the task since. Otherwise it gets `gone`.

## Classes and times

- **`class`:**
  - `live`: tied to a fixed moment (a host break on air);
  - `interactive`: a person waits for it (a test in an admin page);
  - `background`: everything else (recorded lines, look-ups).

  Workers take `live` before `interactive` before `background`.
- **`claim_in`:** after this, nobody should start the task. It is left out when there is no such time.
- **`due_in`:** after this, the result is useless and the task is cancelled. It is left out when there is no such time.

## Kinds

### tts, version 1
| Input | |
|---|---|
| `text` | What to say, already prepared for a voice (≤ the worker's `max_chars`) |
| `lang` | Two-letter code |
| `voice` | A voice id from the worker's caps |
| `model` | E.g. `qwen3-tts-1.7b` |
| `instruct` | Optional spoken direction; ignored by voices with `instruct: false` |
| `temperature` | Optional, 0.1–1.2 |
| `seed` | Integer; a retake gets another |

The output is an MP3, mono; the worker reports its length in `ms`.

### text, version 1
| Input | |
|---|---|
| `model` | A model name from the worker's caps |
| `system` | Optional system prompt |
| `messages` | `[{"role": "user" \| "assistant", "content": "…"}]` |
| `json_schema` | Optional: the answer must be JSON of this schema |
| `max_tokens` | Upper limit of the answer |
| `temperature`, `top_p`, `top_k` | Optional sampling |

- **No tools, no web access, thinking off.**
- **Output:**
  - `content`: the text, or the JSON as text;
  - `finish`: `stop` or `length`;
  - `usage`: `input_tokens`, `output_tokens`;
  - `timings`: `load_ms`, `read_ms`, `write_ms`.

### stt (reserved)
Speech to text, after `input` is defined.

## Duties

**A project:**
- **Offers private work only to `own` workers,** and only to `local` engines when the data must stay on its own machines.
- **Writes nothing for an idle poll.** No rows are touched when nothing changed: a shared host serves listeners from the same few processes. A project may update its "last seen" mark every 30 s.
- **Rests a worker** that keeps failing (ARCHE: three failures in a row, then 10 minutes without offers).
- **Holds lenders to one lease at a time** at most.

**A worker:**
- **Downloads nothing a task names:** models only from its owner's list.
- **Follows no URL found in a task,** and checks every id against the patterns in `schema.json`.
- **Uses no tools in text tasks.**
- **Keeps inputs and outputs only for the task's length,** and never writes them to a log.
- **Holds each kind to its owner's limits:** characters, tokens, seconds.
