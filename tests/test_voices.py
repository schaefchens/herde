"""Our own voices: a recording and its words per language, found in the voices folder."""

import logging
import wave

from arche_worker.engine import LANGUAGES, VOICES
from arche_worker.voices import discover

WORDS = "Das war Mein Gott ist größer. Schön, dass ihr dabei seid."


def recording(folder, name, seconds=8.0, words=WORDS, rate=8000):
    """A silent PCM WAV of `seconds`, with its words beside it (None: no words file)."""
    path = folder / f"{name}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    if words is not None:
        (folder / f"{name}.txt").write_text(words, encoding="utf-8")
    return path


def found(folder):
    return discover(folder, LANGUAGES, VOICES)


def test_a_voice_is_its_recordings_by_language(tmp_path):
    recording(tmp_path, "Faith.de")
    recording(tmp_path, "Faith.en", words="That was My God Is Greater.\nGreat to have you with us.")
    voices = found(tmp_path)
    assert list(voices) == ["faith"]
    faith = voices["faith"]
    assert (faith.name, faith.languages) == ("Faith", ["de", "en"])
    assert faith.refs["de"].text == WORDS
    assert faith.refs["en"].text == "That was My God Is Greater. Great to have you with us.", "the words on one line"
    assert faith.refs["de"].seconds == 8.0
    assert WORDS not in repr(faith), "the words stay out of log lines"


def test_a_language_without_a_recording_borrows_another(tmp_path):
    recording(tmp_path, "Faith.de")
    faith = found(tmp_path)["faith"]
    assert faith.reference("en").path.name == "Faith.de.wav", "Qwen carries a voice across languages"
    assert faith.reference(" DE ").path.name == "Faith.de.wav"


def test_what_cannot_be_spoken_is_never_offered(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    recording(tmp_path, "NoWords.de", words=None)
    recording(tmp_path, "Empty.de", words="  \n ")
    recording(tmp_path, "Long.de", words="x" * 1001)
    recording(tmp_path, "Short.de", seconds=2.0)
    recording(tmp_path, "Endless.de", seconds=31.0)
    recording(tmp_path, "Unnamed")
    recording(tmp_path, "Klingon.tlh")
    recording(tmp_path, "Two words.de")
    recording(tmp_path, "sohee.de")
    (tmp_path / "Broken.de.wav").write_bytes(b"not a wav")
    (tmp_path / "Broken.de.txt").write_text(WORDS)
    assert found(tmp_path) == {}
    for name in ["NoWords.de.txt", "Empty.de.txt", "Long.de.txt", "Short.de.wav", "Endless.de.wav", "Unnamed.wav", "Klingon.tlh.wav",
                 "Two words.de.wav", "sohee.de.wav", "Broken.de.wav"]:
        assert name in caplog.text, f"{name} is reported"


def test_names_differing_only_in_case_are_one_voice_at_most(tmp_path):
    recording(tmp_path, "Faith.de")
    recording(tmp_path, "faith.en")
    voices = found(tmp_path)
    assert list(voices) == ["faith"] and voices["faith"].name == "Faith" and voices["faith"].languages == ["de"]


def test_no_folder_no_voices(tmp_path):
    assert found(tmp_path / "missing") == {}
