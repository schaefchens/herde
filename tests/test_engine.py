import pytest

from arche_worker.engine import CHUNK_CHARS, Engine, InvalidTask, Speech, language_name, max_tokens, split_text, voice_name


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
    engine = Engine("model", "rev")
    assert not engine.ready
    with pytest.raises(RuntimeError):
        engine.synthesize("Hallo", "Sohee", "de")


def test_a_whole_moment_can_be_one_take_or_one_take_per_sentence():
    moment = "Willkommen zur Gebetsstunde. " * 20
    assert split_text(moment.strip(), 2000) == [moment.strip()], "the whole moment in one take"
    assert split_text("Eins ist gut. Zwei auch! Drei?", 2000, every_sentence=True) == ["Eins ist gut.", "Zwei auch!", "Drei?"]
    assert split_text("Am 3. Oktober, z. B. hier. Und dort.", 2000, every_sentence=True) == ["Am 3. Oktober, z. B. hier.", "Und dort."], "never after an ordinal or an abbreviation"
