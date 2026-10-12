"""A project's profile at work (PROTOCOL.md, Profiles): fetched and kept; its voices for that project only; its
settings in every take; checked before any of its speech and verified again and again; dropped when it is dropped.
Against the fake project of test_worker: no network, no model."""

import hashlib

from conftest import SPEECH, profile_doc, voice_entry, wav_bytes
from test_protocol import valid
from test_worker import Backend, FakeProject, make, serve

from herde.engines.tts.core import CODE
from herde.profile import ProfileStore

DE, EN = wav_bytes(4.0), wav_bytes(5.0)
TAKES = [{"voice": "Faith", "lang": "de", "text": "Guten Abend, hier ist ARCHE."}, {"voice": "Sohee", "lang": "en", "text": "Good evening, this is ARCHE."}]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def station(**kw) -> FakeProject:
    p = FakeProject(name="arche")
    p.profile = profile_doc(voices=[voice_entry("Faith", {"de": DE, "en": EN})], takes=TAKES, **kw)
    p.files = {sha(DE): DE, sha(EN): EN}
    return p


def report(p: FakeProject) -> dict:
    """How the worker stood with the profile in its last poll."""
    return (p.polls() or [{}])[-1].get("profile") or {}


def last_caps(p: FakeProject) -> dict:
    return next(b["caps"] for b in reversed(p.polls()) if "caps" in b and b["caps"]["engines"])["engines"][0]


def link(w, name):
    return next(x for x in w.links if x.name == name)


def fetched(p: FakeProject) -> list[str]:
    return [path for path, _ in p.requests if path == "profile" or path.startswith("files/")]


class Clipped(Backend):
    """Speaks a tenth of a second whatever it is asked: words left out, until it is mended."""

    broken = True

    def preset(self, text, speaker, language, instruct, sampling):
        pcm, rate = super().preset(text, speaker, language, instruct, sampling)
        return (pcm[:4800], rate) if self.broken else (pcm, rate)

    def clone(self, text, ref, language, sampling):
        pcm, rate = super().clone(text, ref, language, sampling)
        return (pcm[:4800], rate) if self.broken else (pcm, rate)


def test_a_new_computer_takes_the_stations_profile_and_checks_it_before_it_speaks(tmp_path):
    p = station()
    p.gate = False  # offered at once: the worker itself must wait for its checks
    text = "Ein Satz. " * 30
    p.add(7, voice="Faith", lang="de", due_in=300)
    p.add(8, voice="Sohee", lang="de", text=text, due_in=300)
    backend, bitrates = Backend(), []
    w = make([p], tmp_path, backend, bitrates=bitrates)
    serve(w, lambda: p.tasks[7]["state"] == "done" and p.tasks[8]["state"] == "done" and report(p).get("state") == "ok")
    assert sorted(fetched(p)) == sorted(["profile", f"files/{sha(DE)}", f"files/{sha(EN)}"]), "the profile, and each recording once"
    assert [c[0] for c in backend.calls[:2]] == [t["text"] for t in TAKES], "its checks first, then the station's speech"
    assert backend.calls[0][1] == f"{sha(DE)}.wav", "Faith from the station's recording"
    pieces = [c[0] for c in backend.calls if c[0] in text]
    assert len(pieces) > 1 and all(len(c) <= 120 for c in pieces), "the profile's piece size, not the owner's 300"
    assert (backend.sampling.top_k, backend.sampling.top_p, backend.sampling.repetition_penalty) == (40, 0.9, 1.1)
    assert set(bitrates) == {"128k"}, "the profile's bitrate"
    said = report(p)
    assert said["hash"] == p.profile["profile"] and said["verified_ago"] < 60 and "detail" not in said
    assert next(b["profile"] for b in p.polls() if "profile" in b)["state"] == "syncing", "its first report: still fetching"
    tts = last_caps(p)
    assert [v["id"] for v in tts["voices"] if v.get("own")] == ["Faith"]
    setup = tts["setup"]
    assert (setup["code"], setup["backend"], setup["speech"]) == (CODE, "stub", SPEECH)
    assert setup["voices"] == {"Faith": {"de": sha(DE), "en": sha(EN)}}, "the recordings as read from disk"
    assert [(c["voice"], c["lang"], c["ok"]) for c in setup["checks"]] == [("Faith", "de", True), ("Sohee", "en", True)]
    assert all(not valid("PollRequest", b) for b in p.polls()), "every poll fits the schema"
    assert (tmp_path / "projects" / "arche" / "profile.json").is_file()


def test_a_stations_voices_are_offered_to_that_station_only(tmp_path):
    arche, other = station(), FakeProject(name="other")
    w = make([arche, other], tmp_path)
    serve(w, lambda: report(arche).get("state") == "ok" and len(other.polls()) > 3)
    assert [v["id"] for v in last_caps(arche)["voices"] if v.get("own")] == ["Faith"]
    assert not [v for v in last_caps(other)["voices"] if v.get("own")], "the other project never hears of Faith"
    assert all("profile" not in b for b in other.polls()), "and nothing of a profile it does not hold"
    assert not fetched(other)


def test_a_computer_that_cannot_run_the_profile_says_why_and_takes_none_of_its_speech(tmp_path):
    pinned = {"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16@" + "5" * 40, "clone_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16@" + "a" * 40}
    p = station(backends={"mlx": pinned})
    p.gate = False
    p.add(7, voice="Sohee", due_in=300)
    w = make([p], tmp_path)
    serve(w, lambda: report(p).get("state") == "unmet" and len(p.polls()) > 5)
    assert "admits mlx" in report(p)["detail"] and "stub" in report(p)["detail"]
    assert p.tasks[7]["state"] == "queued", "offered, and left"
    assert not [v for v in last_caps(p)["voices"] if v.get("own")], "neither the station's voices nor the owner's"


def test_a_failed_check_keeps_the_station_silent_until_a_check_passes(tmp_path):
    p = station()
    p.gate = False
    p.add(7, voice="Sohee", due_in=600)
    backend = Clipped()
    w = make([p], tmp_path, backend)

    def mended():
        backend.broken = False
        link(w, "arche").check_due = 0.0  # its ten minutes are up
        return p.tasks[7]["state"] == "done" and report(p).get("state") == "ok"

    serve(w, lambda: report(p).get("state") == "failing", after=mended)
    states = [(i, b["profile"]) for i, (path, b) in enumerate(p.requests) if path == "poll" and "profile" in b]
    failing = next((i, s) for i, s in states if s["state"] == "failing")
    assert "2 of 2 checks failed" in failing[1]["detail"] and "characters a second" in failing[1]["detail"]
    claims = [i for i, (path, _) in enumerate(p.requests) if path.endswith("claim")]
    assert claims and claims[0] > failing[0], "offered all along, and taken only once a check passed"


def test_a_recording_changed_on_disk_is_noticed_and_fetched_again(tmp_path):
    p = station()
    w = make([p], tmp_path)
    recording = ProfileStore(tmp_path / "projects" / "arche").path(sha(DE))
    changed = []

    def tampered_then_mended():
        if not changed:
            recording.write_bytes(wav_bytes(6.0))
            link(w, "arche").verified_at -= 601  # its ten minutes are up
            changed.append(True)
        return fetched(p).count(f"files/{sha(DE)}") == 2 and report(p).get("state") == "ok"

    serve(w, lambda: report(p).get("state") == "ok", after=tampered_then_mended)
    assert recording.read_bytes() == DE
    assert any(b.get("profile", {}).get("state") == "syncing" and "changed on disk" in b["profile"].get("detail", "") for b in p.polls())


def test_a_profile_the_station_drops_is_forgotten(tmp_path):
    p = station()
    w = make([p], tmp_path)

    def dropped():
        p.profile = None
        return "profile" not in p.polls()[-1] and not (tmp_path / "projects" / "arche").exists() and "caps" in p.polls()[-1]

    serve(w, lambda: report(p).get("state") == "ok", after=dropped)
    assert not [v for v in last_caps(p)["voices"] if v.get("own")], "the owner's voices again (none here)"
    assert "checks" not in last_caps(p)["setup"]


def test_after_a_restart_the_kept_profile_is_checked_again_without_fetching_it(tmp_path):
    p = station()
    serve(make([p], tmp_path), lambda: report(p).get("state") == "ok")
    p.requests.clear()
    backend = Backend()
    w = make([p], tmp_path, backend, load=False)
    serve(w, lambda: report(p).get("state") == "ok")
    assert fetched(p) == [], "nothing fetched again"
    assert sorted(r.path.name for r in backend.learnt) == sorted([f"{sha(DE)}.wav", f"{sha(EN)}.wav"]), "learnt with the models"
    first = p.polls()[0]["profile"]
    assert first["hash"] == p.profile["profile"] and first["state"] == "syncing"
