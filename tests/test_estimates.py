"""Measured speeds: what lets a worker say no to work it cannot finish in time."""

from herde.estimates import DEFAULT_TEXT, DEFAULT_TTS, Speeds
from herde.protocol import TextOutput
from herde.state import State


def test_speech_speed_starts_at_a_default_and_follows_what_is_measured(tmp_path):
    s = Speeds(State(tmp_path), "mlx")
    assert s.tts_cps() == DEFAULT_TTS["mlx"]
    s.observe_tts(300, 10.0)
    assert s.tts_cps() == 30.0, "the first measurement replaces the default"
    s.observe_tts(100, 10.0)
    assert round(s.tts_cps(), 1) == 24.0, "then a moving average"
    assert s.tts_seconds(240) == 240 / s.tts_cps() + 1.0


def test_the_caps_move_only_on_a_clear_change(tmp_path):
    s = Speeds(State(tmp_path), "mlx")
    s.observe_tts(200, 10.0)
    assert s.tts_advertised() == {"chars_per_s": 20.0}
    s.observe_tts(220, 10.0)
    assert s.tts_advertised() == {"chars_per_s": 20.0}, "within a quarter: unchanged (each caps change is a write at the project)"
    for _ in range(10):
        s.observe_tts(400, 10.0)
    assert s.tts_advertised()["chars_per_s"] > 30


def test_text_speeds_come_from_ollamas_counts_and_a_cold_model_costs_its_load(tmp_path):
    s = Speeds(State(tmp_path), "")
    assert s.text("gemma4-31b") == DEFAULT_TEXT
    s.observe_text("gemma4-31b", TextOutput(finish="stop", input_tokens=3900, output_tokens=300, load_ms=15000, read_ms=58000, write_ms=27000))
    t = s.text("gemma4-31b")
    assert round(t["read_tps"]) == 67 and round(t["write_tps"]) == 11 and t["load_s"] == 15.0 and t["out_tokens"] == 300
    warm = s.text_seconds("gemma4-31b", 3900, 1024, loaded=True)
    assert s.text_seconds("gemma4-31b", 3900, 1024, loaded=False) == warm + 15.0


def test_fits_leaves_room_to_spare():
    assert Speeds.fits(0, 10, None)
    assert Speeds.fits(0, 10, 15)
    assert not Speeds.fits(0, 10, 14)


def test_speeds_are_kept_between_runs(tmp_path):
    s = Speeds(State(tmp_path), "torch")
    s.observe_tts(300, 10.0)
    s.save(force=True)
    assert Speeds(State(tmp_path), "torch").tts_cps() == 30.0
    assert Speeds(State(tmp_path), "mlx").tts_cps() == DEFAULT_TTS["mlx"], "per backend"
