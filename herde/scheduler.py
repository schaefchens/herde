"""Which offered task to claim next — pure decisions, no I/O, so every rule has a test.

1. `live` work first (a moment on air), the earliest due first: it cannot wait.
2. Then `interactive` (a person waits), the earliest due first.
3. Then `background`, taking turns across projects by the owner's weights:
   a project of weight 3 gets three tasks for another's one, and none waits
   forever behind a busy one (the old worker asked stations in order, and the
   first one starved the rest).

Left out before any of that: what nobody should start any more, what this
worker cannot or may not do, and what it could not finish before it is due.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable

from .protocol import Offer

CLASS_ORDER = {"live": 0, "interactive": 1, "background": 2}


def order(
    offers: Iterable[Offer],
    now: float,
    fits: Callable[[Offer], bool],
    spent: dict[str, float],
    weights: dict[str, int],
) -> list[Offer]:
    """The offers worth claiming, most urgent first."""
    candidates = [o for o in offers if (o.claim_by is None or o.claim_by > now) and (o.due_by is None or o.due_by > now) and fits(o)]
    inf = math.inf
    urgent = sorted(
        (o for o in candidates if o.cls != "background"),
        key=lambda o: (CLASS_ORDER[o.cls], o.due_by if o.due_by is not None else inf, o.claim_by if o.claim_by is not None else inf, o.received, o.project, o.task),
    )
    return urgent + _turns([o for o in candidates if o.cls == "background"], spent, weights)


def _turns(offers: list[Offer], spent: dict[str, float], weights: dict[str, int]) -> list[Offer]:
    """Background work, interleaved across projects by weight; each project's own order kept."""
    queues: dict[str, deque[Offer]] = {}
    for o in offers:
        queues.setdefault(o.project, deque()).append(o)
    credit = {p: spent.get(p, 0.0) for p in queues}
    out: list[Offer] = []
    while queues:
        p = min(queues, key=lambda q: (credit[q], q))
        out.append(queues[p].popleft())
        credit[p] += 1.0 / max(1, weights.get(p, 1))
        if not queues[p]:
            del queues[p]
    return out


def charge(spent: dict[str, float], project: str, weight: int) -> None:
    """A background task claimed: its project's turn is used up by 1/weight."""
    spent[project] = spent.get(project, 0.0) + 1.0 / max(1, weight)
    # Keep the numbers small: only the differences between projects matter.
    low = min(spent.values())
    if low > 1000:
        for k in spent:
            spent[k] -= low
