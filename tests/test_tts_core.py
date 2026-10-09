import pytest

from herde.engines.tts.core import CHUNK_CHARS, Cancelled, InvalidTask, Speech, TooLarge, TtsEngine, language_name, max_tokens, split_text, voice_name
from herde.engines.tts.stub_backend import StubBackend


def test_languages_and_voices_by_their_codes_and_names():
    assert language_name("de") == "German"
    assert language_name(" EN ") == "English"
    assert voice_name("sohee") == "Sohee"
    assert voice_name("Ono_Anna") == "Ono_Anna"
    with pytest.raises(InvalidTask):
        language_name("xx")
    with pytest.raises(InvalidTask):
        voice_name("Hope")


def test_max_tokens_leaves_room_for_a_slow_take_and_no_more():
    # 13 characters ≈ one second ≈ 12.5 tokens; room for 2.5 times that, plus 64.
    assert max_tokens("x" * 13) == 96
    assert max_tokens("x" * 300) == 786
    assert max_tokens("") == 64
    assert max_tokens("x" * 100_000) == 8192


def test_long_text_is_cut_at_sentence_ends():
    sentence = "Gott ist treu und begleitet uns durch diesen Tag. "
    chunks = split_text(sentence * 20)
    assert all(len(c) <= CHUNK_CHARS for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert " ".join(chunks) == (sentence * 20).strip()


@pytest.mark.parametrize("abbr", ["z. B.", "Dr.", "bzw.", "ca.", "am 3."])
def test_never_cut_after_an_abbreviation(abbr):
    head = "Wir beten heute gemeinsam für viele Menschen in unserer Stadt und weit darüber hinaus, " * 2
    text = f"{head}{abbr} Oktober und alle, die krank sind."
    for chunk in split_text(text):
        assert not chunk.endswith(abbr.split()[-1]), f"cut after {abbr!r}"


def test_a_word_longer_than_a_chunk_is_cut_hard():
    chunks = split_text("a" * 700)
    assert [len(c) for c in chunks] == [300, 300, 100]


def test_paragraphs_stay_apart():
    assert split_text("Hallo.\nWillkommen.") == ["Hallo.", "Willkommen."]


def test_speech_knows_its_length():
    assert Speech(pcm16=b"\x00\x00" * 24000, sample_rate=24000).ms == 1000


def test_no_take_before_the_model_is_loaded():
    engine = TtsEngine(StubBackend())
    assert not engine.ready
    with pytest.raises(RuntimeError):
        engine.synthesize("Hallo", "Sohee", "de")


def test_a_whole_moment_can_be_one_take_or_one_take_per_sentence():
    moment = "Willkommen zur Gebetsstunde. " * 20
    assert split_text(moment.strip(), 2000) == [moment.strip()], "the whole moment in one take"
    assert split_text("Eins ist gut. Zwei auch! Drei?", 2000, every_sentence=True) == ["Eins ist gut.", "Zwei auch!", "Drei?"]
    assert split_text("Am 3. Oktober, z. B. hier. Und dort.", 2000, every_sentence=True) == ["Am 3. Oktober, z. B. hier.", "Und dort."], "never after an ordinal or an abbreviation"


def test_an_own_voice_is_spoken_from_its_recording_and_a_preset_by_its_name(tmp_path):
    from test_voices import recording

    from herde.engines.tts.voices import discover

    recording(tmp_path, "Faith.de")
    recording(tmp_path, "Faith.en")
    engine = TtsEngine(StubBackend(), own=discover(tmp_path, ("de", "en"), ("Sohee",)))
    assert engine.route("faith", "de").path.name == "Faith.de.wav", "by name, in any case, as the station sends it"
    assert engine.route("Faith", "en").path.name == "Faith.en.wav"
    assert engine.route("sohee", "de") == "Sohee"
    with pytest.raises(InvalidTask):
        engine.route("Hope", "de")


class Recorder(StubBackend):
    """The stub, noting each piece it is given (and able to stop the worker's wait after the first)."""

    def __init__(self, after_first=None):
        super().__init__()
        self.pieces = []
        self.after_first = after_first

    def preset(self, text, speaker, language, instruct, sampling):
        self.pieces.append((text, speaker, language, instruct, sampling.temperature))
        if self.after_first is not None and len(self.pieces) == 1:
            self.after_first()
        return super().preset(text, speaker, language, instruct, sampling)


def loaded(backend, **kw):
    engine = TtsEngine(backend, **kw)
    engine.load()
    return engine


def test_a_text_is_spoken_piece_by_piece_with_a_short_breath_between():
    backend = Recorder()
    sentence = "Gott ist treu und begleitet uns durch diesen Tag. "
    progress = []
    speech = loaded(backend).synthesize(sentence * 10, "sohee", "de", "Warm.", 0.8, 3, progress=progress.append)
    assert len(backend.pieces) > 1 and all(p[1:] == ("Sohee", "German", "Warm.", 0.8) for p in backend.pieces)
    silence = sum(len(p[0]) for p in backend.pieces) / 13 + (len(backend.pieces) - 1) * 0.15
    assert abs(speech.ms / 1000 - silence) < 0.2, "the stub speaks 13 characters a second, plus 150 ms between pieces"
    assert progress[-1] == 1.0 and progress == sorted(progress)


def test_a_cancel_stops_between_two_pieces():
    cancelled = {"now": False}
    backend = Recorder(after_first=lambda: cancelled.update(now=True))
    with pytest.raises(Cancelled):
        loaded(backend).synthesize("Ein Satz hier. " * 60, "Sohee", "de", cancelled=lambda: cancelled["now"])
    assert len(backend.pieces) == 1


def test_more_than_this_worker_speaks_is_too_large_and_unknowns_are_invalid():
    engine = loaded(StubBackend(), max_chars=100)
    with pytest.raises(TooLarge):
        engine.synthesize("x " * 60, "Sohee", "de")
    with pytest.raises(InvalidTask):
        engine.synthesize("Hallo", "Hope", "de")
    with pytest.raises(InvalidTask):
        engine.synthesize("Hallo", "Sohee", "tlh")
    with pytest.raises(InvalidTask):
        engine.synthesize("   ", "Sohee", "de")


def test_the_caps_say_exactly_what_it_speaks(tmp_path):
    from test_voices import recording

    from herde.engines.tts.voices import discover

    recording(tmp_path, "Faith.de")
    recording(tmp_path, "Faith.en")
    engine = loaded(StubBackend(), own=discover(tmp_path, ("de", "en"), ("Sohee",)), max_chars=2000)
    caps = engine.caps(19.6)
    assert (caps["kind"], caps["model"], caps["location"], caps["max_chars"], caps["speed"]) == ("tts", "qwen3-tts-1.7b", "local", 2000, {"chars_per_s": 20})
    voices = {v["id"]: v for v in caps["voices"]}
    assert len(voices) == 10 and voices["Sohee"]["instruct"] is True and len(voices["Sohee"]["langs"]) == 10
    assert voices["Faith"] == {"id": "Faith", "label": "Faith (own voice)", "langs": ["de", "en"], "instruct": False, "own": True}
    assert engine.can("qwen3-tts-1.7b", "Faith", "de") and not engine.can("qwen3-tts-1.7b", "Faith", "fr"), "an own voice in its recorded languages"
    assert engine.can("qwen3-tts-1.7b", "sohee", "fr") and not engine.can("qwen3-tts-1.7b", "Hope", "de")
    assert not engine.can("qwen3-tts-1.7b-customvoice", "Sohee", "de"), "the name projects ask for, exactly"
