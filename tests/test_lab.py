"""The listening lab without the model: cases in, one MP3 per case and variant, an index to listen to them."""

import json

import pytest

from arche_worker.engine import InvalidTask, Speech
from arche_worker.lab import Case, LabError, load_cases, run, variants


def test_variants_are_sizes_whole_or_sentence_at_each_temperature():
    v = variants("300,whole,sentence", "0.6,0.8")
    assert [x.name for x in v] == ["300-t0.6", "300-t0.8", "whole-t0.6", "whole-t0.8", "sentence-t0.6", "sentence-t0.8"]
    assert (v[0].chunk_chars, v[0].every_sentence, v[2].chunk_chars, v[4].every_sentence) == (300, False, 2000, True)
    for chunks, temps in (("tiny", "0.7"), ("10", "0.7"), ("300", "hot"), ("300", "3"), ("", "0.7")):
        with pytest.raises(LabError):
            variants(chunks, temps)


def test_cases_come_from_the_replay_file(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [
        {"id": "1643-break-de-new", "lang": "de", "voice": "Sohee", "text": "Hallo.", "instruct": "Warm."},
        {"id": "1643-break-de-aired", "lang": "de", "voice": "Sohee", "text": "Hallo!", "instruct": "Warm."},
        {"id": "empty", "text": "  "},
    ]}))
    assert [c.id for c in load_cases(path)] == ["1643-break-de-new", "1643-break-de-aired"]
    assert [c.id for c in load_cases(path, only="new")] == ["1643-break-de-new"]
    with pytest.raises(LabError):
        load_cases(path, only="nothing")
    path.write_text("not json")
    with pytest.raises(LabError):
        load_cases(path)


def test_every_case_is_spoken_in_every_variant_and_listed(tmp_path):
    calls = []

    def synthesize(text, voice, lang, instruct, temperature, seed, chunk_chars, every_sentence):
        calls.append((text, voice, lang, instruct, temperature, seed, chunk_chars, every_sentence))
        if text == "kaputt":
            raise InvalidTask("unknown voice")
        return Speech(pcm16=b"\x00\x00" * 24000, sample_rate=24000)

    cases = [Case("a<1>", "de", "Sohee", "Gott ist treu & gut.", "Warm <and> calm."), Case("b", "en", "Ryan", "kaputt", "")]
    index = run(synthesize, lambda pcm, rate: (b"MP3", 1000), cases, variants("300,whole", "0.7"), tmp_path)
    assert calls[0] == ("Gott ist treu & gut.", "Sohee", "de", "Warm <and> calm.", 0.7, 0, 300, False)
    assert sorted(p.name for p in tmp_path.glob("*.mp3")) == ["a_1___300-t0.7.mp3", "a_1___whole-t0.7.mp3"]
    page = index.read_text()
    assert "Gott ist treu &amp; gut." in page and "Warm &lt;and&gt; calm." in page, "texts escaped"
    assert page.count("<audio") == 2 and "—" in page, "a case that could not be spoken shows a dash"


def test_a_stop_ends_the_lab_with_what_is_done(tmp_path):
    stop = {"now": False}

    def synthesize(*args, **kw):
        stop["now"] = True
        return Speech(pcm16=b"\x00\x00" * 240, sample_rate=24000)

    run(synthesize, lambda pcm, rate: (b"MP3", 10), [Case("a", "de", "Sohee", "Hallo.", "")], variants("300,whole", "0.7"), tmp_path, lambda: stop["now"])
    assert len(list(tmp_path.glob("*.mp3"))) == 1 and (tmp_path / "index.html").is_file()
