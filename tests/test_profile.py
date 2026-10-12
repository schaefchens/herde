"""A project's profile on its own: read strictly, met only by the worker's own pins, its takes judged, kept on disk."""

import hashlib
import json
import math
import struct

import pytest
from conftest import FAITH, SPEECH, profile_doc, voice_entry, wav_bytes

from herde.engines.tts.core import CODE
from herde.profile import ProfileError, ProfileStore, folder_name, judge, parse, unmet
from herde.protocol import canonical_hash

MLX = {"model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16@" + "5" * 40, "clone_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16@" + "a" * 40}


def faith():
    return voice_entry("Faith", {"de": wav_bytes(), "en": wav_bytes(5.0)})


def test_a_profile_is_read_with_its_settings_voices_and_checks():
    doc = profile_doc(voices=[faith()], backends={"mlx": MLX}, takes=[{"voice": "Faith", "lang": "de", "text": " Guten Abend. "}], every=3600)
    p = parse(doc)
    assert p.hash == doc["profile"] and p.code == CODE and p.backends == {"mlx": (MLX["model"], MLX["clone_model"])}
    assert (p.settings.chunk_chars, p.settings.pause_ms, p.settings.top_k, p.settings.top_p, p.settings.repetition_penalty, p.mp3_kbps) == (120, 300, 40, 0.9, 1.1, 128)
    assert list(p.voices) == ["Faith"] and sorted(p.voices["Faith"]) == ["de", "en"] and p.voices["Faith"]["de"].text == FAITH
    assert len(p.files()) == 2 and (p.every, p.takes[0].text) == (3600, "Guten Abend.")
    assert FAITH not in repr(p), "the words stay out of log lines"


def test_a_profile_whose_hash_does_not_fit_its_content_is_refused():
    doc = profile_doc()
    doc["tts"]["speech"] = {**SPEECH, "chunk_chars": 300}
    with pytest.raises(ProfileError, match="hash"):
        parse(doc)
    reordered = json.loads(json.dumps(profile_doc()), object_pairs_hook=lambda pairs: dict(reversed(pairs)))
    assert parse(reordered).hash == profile_doc()["profile"], "key order does not change the hash"


@pytest.mark.parametrize(("change", "why"), [
    (lambda t: t["voices"].append(voice_entry("Sohee", {"de": wav_bytes()})), "named like another"),
    (lambda t: t["voices"].extend([voice_entry("Faith", {"de": wav_bytes()}), voice_entry("faith", {"en": wav_bytes()})]), "named like another"),
    (lambda t: t["voices"].append(voice_entry("Faith", {"tlh": wav_bytes()})), "language"),
    (lambda t: t["voices"].append(voice_entry("Faith", {"de": wav_bytes()}, words=" ")), "words"),
    (lambda t: t["checks"].update(takes=[{"voice": "Grace", "lang": "de", "text": "Hallo"}]), "neither a preset"),
    (lambda t: t["checks"].update(every=60), "at least 600"),
    (lambda t: t["speech"].update(top_p=0), "top_p"),
    (lambda t: t["speech"].update(chunk_chars=True), "chunk_chars"),
    (lambda t: t.update(backends={"cuda": {"model": "", "clone_model": ""}}), "no backend"),
    (lambda t: t.update(backends={"mlx": {"model": "x@not-a-commit", "clone_model": ""}}), "no backend"),
    (lambda t: t.update(dtypes=["int4"]), "dtypes"),
])
def test_what_does_not_fit_the_protocol_is_refused_and_says_what(change, why):
    tts = profile_doc()["tts"]
    change(tts)
    with pytest.raises(ProfileError, match=why):
        parse({"profile": canonical_hash({"tts": tts}), "tts": tts})


def test_only_the_workers_own_pins_meet_a_profile():
    p = parse(profile_doc(voices=[faith()], backends={"mlx": MLX, "stub": {"model": "", "clone_model": ""}}))
    assert unmet(p, "mlx", MLX["model"], MLX["clone_model"], "bf16") == ""
    assert "loads" in unmet(p, "mlx", MLX["model"].replace("5", "6"), MLX["clone_model"], "bf16"), "another CustomVoice revision"
    assert "Base model" in unmet(p, "mlx", MLX["model"], "", "bf16"), "own voices need the profile's Base model"
    assert "fp16" in unmet(p, "mlx", MLX["model"], MLX["clone_model"], "fp16")
    assert "torch" in unmet(p, "torch", "Qwen/x@" + "1" * 40, "", "bf16"), "a backend the profile does not admit"
    assert unmet(p, "stub", "", "", "") == "", "the stub loads no weights: no format to hold it to"
    without_voices = parse(profile_doc(backends={"mlx": MLX}))
    assert unmet(without_voices, "mlx", MLX["model"], "", "bf16") == "", "no own voice, no Base model needed"
    newer = parse(profile_doc(code=CODE + 1))
    assert "update herde" in unmet(newer, "stub", "", "", "")


def pcm(seconds: float, amplitude: int, rate: int = 24000) -> bytes:
    """A tone, clipped at full scale as the backends clip a model's samples (to_pcm16)."""
    return b"".join(struct.pack("<h", max(-32767, min(32767, int(amplitude * math.sin(i / 7))))) for i in range(int(seconds * rate)))


def test_a_check_take_must_be_speech_as_long_as_its_words_need():
    words = "x" * 52  # 13 characters a second over 4 s
    assert judge(pcm(4.0, 8000), 24000, words) == (True, 13.0, "")
    ok, cps, why = judge(pcm(0.5, 8000), 24000, words)
    assert not ok and cps == 104.0 and "characters a second" in why, "far too fast: words were left out"
    assert not judge(pcm(20.0, 8000), 24000, words)[0], "far too slow: a take that ran away"
    assert judge(pcm(4.0, 20), 24000, words)[2] == "silent"
    assert judge(pcm(4.0, 0), 24000, words, silence_ok=True)[0], "the stub's silence is on purpose"
    assert "clips" in judge(pcm(4.0, 60000), 24000, words)[2]


def test_the_store_keeps_checked_recordings_and_notices_a_change(tmp_path):
    store = ProfileStore(tmp_path / "projects" / "arche")
    p = parse(profile_doc(voices=[faith()]))
    de, en = (hashlib.sha256(d).hexdigest() for d in (wav_bytes(), wav_bytes(5.0)))
    with pytest.raises(ProfileError, match="SHA-256"):
        store.put(de, b"something else")
    store.put(de, wav_bytes())
    store.put(en, wav_bytes(5.0))
    voices = store.voices(p)
    assert voices["faith"].name == "Faith" and voices["faith"].refs["en"].sha256 == en and voices["faith"].refs["de"].seconds == 4.0
    store.save(p)
    assert store.load() == p and (store.folder / "profile.json").stat().st_mode & 0o077 == 0, "the project's, and private"
    assert store.verify(p) == []
    store.path(de).write_bytes(wav_bytes(6.0))
    assert store.verify(p) == [de] and not store.has(de), "a changed recording goes, to be fetched again"
    (store.files / "old.wav").write_bytes(b"x")
    store.prune(p)
    assert sorted(f.name for f in store.files.iterdir()) == [f"{en}.wav"]
    store.clear()
    assert not store.folder.exists() and store.load() is None


def test_a_recording_that_cannot_be_cloned_from_is_refused(tmp_path):
    store = ProfileStore(tmp_path)
    short = wav_bytes(2.0)
    p = parse(profile_doc(voices=[voice_entry("Faith", {"de": short})]))
    store.put(hashlib.sha256(short).hexdigest(), short)
    with pytest.raises(ProfileError, match=r"Faith\.de: the recording runs 2\.0 s"):
        store.voices(p)


def test_a_projects_name_never_leaves_the_projects_folder():
    assert folder_name("arche") == "arche" and folder_name("radio.example") == "radio.example"
    assert folder_name("..") == "__" and folder_name(".") == "_"
