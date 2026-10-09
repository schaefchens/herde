"""A project that speaks protocol 2, standard library only: the herde image works against it end to end.

It holds one live speech task and one text task, takes a worker's results the
way PROTOCOL.md says (leases, multipart speech, JSON text), and exits 0 once
both are done — or 1 after two minutes.
"""

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = "integration-" + "k" * 52
SCHEMA = {"type": "object", "required": ["de"], "properties": {"de": {"type": "string"}}}
TASKS = {
    1: {"kind": "tts", "class": "live", "state": "queued", "lease": None,
        "input": {"text": "Guten Abend und willkommen zurück bei uns.", "lang": "de", "voice": "Sohee", "model": "qwen3-tts-1.7b", "instruct": "Warm.", "temperature": 0.7, "seed": 0}},
    2: {"kind": "text", "class": "background", "state": "queued", "lease": None,
        "input": {"model": "stub", "messages": [{"role": "user", "content": "Say hello as JSON."}], "json_schema": SCHEMA, "max_tokens": 64}},
}
LOCK = threading.Lock()


def offer(tid, t):
    o = {"task": tid, "kind": t["kind"], "class": t["class"]}
    if t["kind"] == "tts":
        o |= {"model": "qwen3-tts-1.7b", "voice": "Sohee", "lang": "de", "size": {"chars": len(t["input"]["text"])}, "claim_in": 120, "due_in": 150}
    else:
        o |= {"model": "stub", "size": {"input_tokens": 10, "max_tokens": 64}}
    return o


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("project:", fmt % args, flush=True)

    def do_POST(self):
        if self.headers.get("X-Worker-Key") != KEY:
            return self._json(401, {"error": "bad_worker_key"})
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        path = self.path.split("/api/worker/v2/", 1)[-1]
        with LOCK:
            if path == "poll":
                body = json.loads(raw)
                assert body["protocol"] == 2, body
                offers = [offer(i, t) for i, t in TASKS.items() if t["state"] == "queued"] if any(body["free"].values()) else []
                return self._json(200, {"now": int(time.time()), "offers": offers, "cancel": [], "need_caps": False, "retry_after": 1})
            m = re.fullmatch(r"tasks/(\d+)/(claim|result|fail)", path)
            t = TASKS.get(int(m.group(1))) if m else None
            if t is None:
                return self._json(404, {"error": "not_found"})
            if m.group(2) == "claim":
                lease = json.loads(raw)["lease"]
                if t["state"] != "queued":
                    return self._json(409, {"error": "gone"})
                t["state"], t["lease"] = "leased", lease
                return self._json(200, {"task": {"id": int(m.group(1)), "kind": t["kind"], "class": t["class"], "lease_in": 45, "input": t["input"]}})
            if m.group(2) == "result":
                if t["kind"] == "tts":
                    lease = re.search(rb'name="lease"\r\n\r\n([^\r]+)', raw).group(1).decode()
                    ms = int(re.search(rb'name="ms"\r\n\r\n(\d+)', raw).group(1))
                    ok = lease == t["lease"] and b"audio/mpeg" in raw and ms > 1000
                else:
                    body = json.loads(raw)
                    ok = body["lease"] == t["lease"] and json.loads(body["output"]["content"]) == {"de": "Stub text."}
                if not ok:
                    return self._json(422, {"error": "invalid_result"})
                t["state"] = "done"
                print(f"project: task {m.group(1)} done", flush=True)
                return self._json(200, {"ok": True})
            t["state"], t["lease"] = "queued", None
            return self._json(200, {"ok": True})

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    deadline = time.time() + 120
    while time.time() < deadline:
        with LOCK:
            if all(t["state"] == "done" for t in TASKS.values()):
                print("project: every task done", flush=True)
                sys.exit(0)
        time.sleep(0.5)
    print("project: timed out", {i: t["state"] for i, t in TASKS.items()}, flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
