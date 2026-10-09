"""A pause written by one process (herde pause, the local page) and read by the running worker."""

import stat
import time
from datetime import datetime, timedelta

import pytest

from herde.state import PauseError, State, parse_pause

NOW = datetime(2026, 10, 9, 14, 30)


def test_pause_specs():
    assert parse_pause("2h", NOW) == NOW + timedelta(hours=2)
    assert parse_pause("30m", NOW) == NOW + timedelta(minutes=30)
    assert parse_pause("45", NOW) == NOW + timedelta(minutes=45), "minutes without a unit"
    assert parse_pause("until 18:00", NOW) == datetime(2026, 10, 9, 18, 0)
    assert parse_pause("08:00", NOW) == datetime(2026, 10, 10, 8, 0), "a time already past today is tomorrow's"
    assert parse_pause("", NOW) is None and parse_pause("forever", NOW) is None
    for bad in ("soon", "25:00", "0m", "40d"):
        with pytest.raises(PauseError):
            parse_pause(bad, NOW)


def test_a_pause_until_a_time_ends_by_itself(tmp_path):
    state = State(tmp_path)
    assert state.paused() == (False, None)
    state.pause(time.time() + 60)
    paused, until = state.paused()
    assert paused and until > time.time()
    assert state.paused(now=until + 1)[0] is False
    assert stat.S_IMODE(state.pause_path.stat().st_mode) == 0o600


def test_a_pause_until_resumed(tmp_path):
    state = State(tmp_path)
    state.pause(None)
    assert State(tmp_path).paused() == (True, None), "another process sees it at once"
    state.resume()
    assert State(tmp_path).paused() == (False, None)


def test_a_broken_file_is_no_pause(tmp_path):
    state = State(tmp_path)
    state.pause_path.write_text("{not json")
    assert state.paused() == (False, None)
    state.save_status({"name": "x"})
    assert state.status() == {"name": "x"}
