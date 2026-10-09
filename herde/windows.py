"""When a computer works for others: hours like "Mon-Fri 18:00-08:00", in the computer's own time.

A lender's PC is theirs first: outside its hours the worker takes nothing new
(it still tells its projects it is there, and when it is back). A span that
crosses midnight belongs to the day it starts on: "Fri 22:00-02:00" is Friday
night into Saturday.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
FULL_DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_SPAN = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")


class WindowError(ValueError):
    pass


@dataclass(frozen=True)
class Window:
    text: str
    days: frozenset[int]
    # Minutes after midnight; end <= start crosses midnight, end == start is the whole day.
    start: int = 0
    end: int = 0

    def contains(self, now: datetime) -> bool:
        minute, day = now.hour * 60 + now.minute, now.weekday()
        if self.start < self.end:
            return day in self.days and self.start <= minute < self.end
        if self.start == self.end:
            return day in self.days
        return (day in self.days and minute >= self.start) or ((day - 1) % 7 in self.days and minute < self.end)

    def starts_after(self, now: datetime) -> datetime | None:
        """The first start of this window after `now` (within the next eight days)."""
        base = now.replace(hour=0, minute=0, second=0, microsecond=0)
        for d in range(0, 9):
            day = base + timedelta(days=d)
            if day.weekday() not in self.days:
                continue
            start = day + timedelta(minutes=self.start)
            if start > now:
                return start
        return None


def parse(text: str) -> Window:
    raw = " ".join(str(text).split())
    low = raw.lower()
    if low in ("always", "daily"):
        return Window(text=raw, days=frozenset(range(7)))
    span = _SPAN.search(low)
    start = end = 0
    if span:
        h1, m1, h2, m2 = (int(x) for x in span.groups())
        if h1 > 23 or m1 > 59 or h2 > 24 or m2 > 59 or (h2 == 24 and m2 != 0):
            raise WindowError(f"{raw!r}: hours run from 00:00 to 24:00")
        start, end = h1 * 60 + m1, (h2 * 60 + m2) % (24 * 60)
        low = (low[: span.start()] + low[span.end():]).strip()
    days = _days(low, raw) if low else frozenset(range(7))
    if not span and not low:
        raise WindowError(f"{raw!r}: say which days or hours, like \"Mon-Fri 18:00-08:00\"")
    return Window(text=raw, days=days, start=start, end=end)


def _days(spec: str, raw: str) -> frozenset[int]:
    out: set[int] = set()
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            a, _, b = part.partition("-")
            first, last = _day(a, raw), _day(b, raw)
            d = first
            while True:
                out.add(d)
                if d == last:
                    break
                d = (d + 1) % 7
        else:
            out.add(_day(part, raw))
    return frozenset(out)


def _day(name: str, raw: str) -> int:
    key = name.strip().lower()
    if key in DAYS:
        return DAYS.index(key)
    if key in FULL_DAYS:
        return FULL_DAYS.index(key)
    raise WindowError(f"{raw!r}: {name.strip()!r} is no day (Mon, Tue, Wed, Thu, Fri, Sat, Sun)")


def parse_all(texts: Iterable[str]) -> tuple[Window, ...]:
    return tuple(parse(t) for t in texts)


def is_open(windows: Iterable[Window], now: datetime) -> bool:
    """No windows: always."""
    windows = tuple(windows)
    return not windows or any(w.contains(now) for w in windows)


def next_open(windows: Iterable[Window], now: datetime) -> datetime | None:
    """When the windows open next (now, if they are open)."""
    windows = tuple(windows)
    if is_open(windows, now):
        return now
    starts = [s for s in (w.starts_after(now) for w in windows) if s is not None]
    return min(starts) if starts else None
