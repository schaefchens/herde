"""A lender's hours: when the computer works for others."""

from datetime import datetime

import pytest

from herde.windows import WindowError, is_open, next_open, parse, parse_all

# 2026-10-09 is a Friday.
FRI_21 = datetime(2026, 10, 9, 21, 0)


def test_a_span_across_midnight_belongs_to_the_day_it_starts():
    w = parse("Mon-Fri 18:00-08:00")
    assert w.contains(FRI_21)
    assert w.contains(datetime(2026, 10, 10, 7, 59)), "Friday night runs into Saturday morning"
    assert not w.contains(datetime(2026, 10, 10, 8, 0))
    assert not w.contains(datetime(2026, 10, 10, 21, 0)), "Saturday evening is not in Mon-Fri"
    assert w.contains(datetime(2026, 10, 12, 3, 0)) is False, "Monday 03:00 belongs to Sunday night, not in Mon-Fri"


def test_whole_days_every_day_and_lists():
    assert parse("Sat-Sun").contains(datetime(2026, 10, 11, 12, 0))
    assert not parse("Sat-Sun").contains(FRI_21)
    assert parse("09:00-17:00").contains(datetime(2026, 10, 7, 9, 0)) and not parse("09:00-17:00").contains(datetime(2026, 10, 7, 17, 0))
    assert parse("Mon,Wed,Fri").contains(FRI_21) and not parse("Mon,Wed,Fri").contains(datetime(2026, 10, 8, 12, 0))
    assert parse("Fri-Mon").contains(datetime(2026, 10, 12, 12, 0)), "a range may wrap around the week"
    assert parse("always").contains(FRI_21)
    assert parse("Mon-Fri 00:00-24:00").contains(datetime(2026, 10, 9, 23, 59))


@pytest.mark.parametrize("bad", ["Someday", "Mon-Fri 25:00-26:00", "Mon 18:00-24:30", "", "Mon-Fri 18-08"])
def test_what_is_no_window(bad):
    with pytest.raises(WindowError):
        parse(bad)


def test_open_and_when_next():
    nights = parse_all(["Mon-Fri 18:00-08:00", "Sat-Sun"])
    assert is_open((), FRI_21), "no windows: always"
    assert is_open(nights, FRI_21)
    wednesday_noon = datetime(2026, 10, 7, 12, 0)
    assert not is_open(nights, wednesday_noon)
    assert next_open(nights, wednesday_noon) == datetime(2026, 10, 7, 18, 0)
    assert next_open(nights, FRI_21) == FRI_21
