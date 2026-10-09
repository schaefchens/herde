"""The contract: every example exchange fits the schema, and the worker reads and writes them as they are."""

import json

import pytest
from conftest import SCHEMA, all_fixtures, fixture
from jsonschema import Draft202012Validator

from herde import PROTOCOL
from herde.protocol import KIND_VERSIONS, ProtocolError, caps_hash, parse_claim, parse_join, parse_poll, poll_body

schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
REQUEST = {"join": "JoinRequest", "poll": "PollRequest", "claim": "ClaimRequest", "fail": "FailRequest"}
RESPONSE = {"join": "JoinResponse", "poll": "PollResponse", "claim": "ClaimResponse", "result": "ResultResponse", "fail": "FailResponse"}


def valid(defname: str, instance) -> list[str]:
    sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{defname}"}
    return [f"{list(e.path)}: {e.message}" for e in Draft202012Validator(sub).iter_errors(instance)]


def test_the_schema_is_a_schema():
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(("name", "f"), all_fixtures())
def test_every_fixture_fits_the_schema(name, f):
    call, req, res = f["call"], f["request"], f["response"]
    if call == "result":
        assert not (valid("TtsResultFields", req["multipart"]) if "multipart" in req else valid("TextResult", req)), name
    else:
        assert not valid(REQUEST[call], req), name
    assert not valid(RESPONSE[call] if res["status"] == 200 else "Error", res["body"]), name


def test_the_caps_hash_is_canonical_json():
    first = fixture("poll.first")["request"]
    assert caps_hash(first["caps"]) == first["caps_hash"]
    reordered = json.loads(json.dumps(first["caps"]), object_pairs_hook=lambda pairs: dict(reversed(pairs)))
    assert caps_hash(reordered) == first["caps_hash"], "key order does not change the hash"


def test_the_worker_writes_the_first_poll_as_the_fixture_has_it():
    first = fixture("poll.first")["request"]
    body = poll_body("ready", first["caps"], True, [], {"tts": 1, "text": 1}, KIND_VERSIONS, "herde/0.2.0")
    assert body == first
    assert body["protocol"] == PROTOCOL
    idle = poll_body("ready", first["caps"], False, [], {"tts": 1, "text": 1}, KIND_VERSIONS, "herde/0.2.0")
    assert idle == fixture("poll.idle")["request"], "only the hash when nothing changed"
    paused = poll_body("paused", first["caps"], False, [], {"tts": 0, "text": 0}, KIND_VERSIONS, "herde/0.2.0", resume_in=3600)
    assert paused == fixture("poll.paused")["request"]


def test_offers_become_this_machines_times():
    answer = parse_poll("arche", fixture("poll.first")["response"]["body"], received=500.0)
    assert [o.task for o in answer.offers] == [52, 53, 60, 61]
    live = answer.offers[0]
    assert (live.project, live.kind, live.cls, live.voice, live.lang, live.chars) == ("arche", "tts", "live", "Sohee", "de", 312)
    assert (live.claim_by, live.due_by) == (641.0, 671.0), "seconds from now, on the monotonic clock"
    text = answer.offers[2]
    assert (text.kind, text.model, text.input_tokens, text.max_tokens, text.due_by) == ("text", "gemma4-31b", 3912, 1024, None)
    assert answer.retry_after == 2.0 and answer.cancel == () and answer.need_caps is False


def test_an_offer_that_does_not_fit_is_left_out_and_a_garbled_answer_is_an_error():
    body = fixture("poll.first")["response"]["body"]
    bad = dict(body, offers=[
        {**body["offers"][0], "voice": "../etc/passwd"},
        {**body["offers"][0], "task": 0},
        {**body["offers"][0], "kind": "video"},
        {**body["offers"][2], "model": "gemma 4"},
        body["offers"][1],
    ])
    assert [o.task for o in parse_poll("p", bad, 0.0).offers] == [53]
    for garbled in (["not", "an", "object"], {"offers": []}, {"retry_after": "soon"}):
        with pytest.raises(ProtocolError):
            parse_poll("p", garbled, 0.0)
    assert parse_poll("p", {"retry_after": 900, "offers": []}, 0.0).retry_after == 60.0, "at most a minute"
    assert parse_poll("p", {"retry_after": 0, "offers": []}, 0.0).retry_after == 1.0


def test_a_claim_answer_becomes_a_task_whose_words_stay_out_of_logs():
    lease = fixture("claim.tts")["request"]["lease"]
    task = parse_claim("arche", 52, lease, fixture("claim.tts")["response"]["body"], received=100.0)
    assert (task.kind, task.cls, task.lease, task.due_by) == ("tts", "live", lease, 270.0)
    assert (task.input.voice, task.input.lang, task.input.seed, task.input.temperature) == ("Sohee", "de", 0, 0.7)
    assert task.input.text.startswith("Guten Abend")
    assert "Guten Abend" not in repr(task) and "Warm" not in repr(task)
    text = parse_claim("arche", 60, lease, fixture("claim.text")["response"]["body"], received=0.0)
    assert text.input.model == "gemma4-31b" and text.input.json_schema["required"] == ["de", "en", "delivery"]
    assert "radio host" not in repr(text), "neither the prompt nor the schema in a repr"
    with pytest.raises(ProtocolError):
        parse_claim("arche", 99, lease, fixture("claim.tts")["response"]["body"], 0.0)
    broken = json.loads(json.dumps(fixture("claim.tts")["response"]["body"]))
    broken["task"]["input"]["voice"] = "Sohee; rm -rf"
    with pytest.raises(ProtocolError):
        parse_claim("arche", 52, lease, broken, 0.0)


def test_a_join_answer_gives_the_key_and_the_trust():
    joined = parse_join(fixture("join.ok")["response"]["body"])
    assert (joined.project, joined.worker_id, joined.trust) == ("ARCHE", 7, "lender")
    assert joined.key not in repr(joined)
    with pytest.raises(ProtocolError):
        parse_join({"key": "short", "worker": {}, "project": {"protocol": 2}})
    with pytest.raises(ProtocolError):
        parse_join({**fixture("join.ok")["response"]["body"], "project": {"name": "x", "protocol": 3}})
