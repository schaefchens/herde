"""Which offer next: live by due time, then interactive, then background in turns by weight."""

from herde.protocol import Offer
from herde.scheduler import charge, order


def offer(task, cls="background", project="a", due=None, claim=None, received=0.0, kind="tts"):
    return Offer(project=project, task=task, kind=kind, cls=cls, model="qwen3-tts-1.7b", voice="Sohee", lang="de", chars=200,
                 claim_by=claim, due_by=due, received=received)


def ids(offers):
    return [(o.project, o.task) for o in offers]


def test_live_first_by_the_earliest_due_then_interactive_then_background():
    offers = [
        offer(1, "background"), offer(2, "interactive", due=300), offer(3, "live", due=200), offer(4, "live", due=100), offer(5, "interactive", due=100),
    ]
    assert ids(order(offers, 0, lambda o: True, {}, {})) == [("a", 4), ("a", 3), ("a", 5), ("a", 2), ("a", 1)]


def test_a_live_task_of_another_project_beats_background_work():
    offers = [offer(1, "background", project="dev"), offer(2, "live", project="prod", due=500)]
    assert ids(order(offers, 0, lambda o: True, {}, {"dev": 10, "prod": 1}))[0] == ("prod", 2), "weights never outrank a moment on air"


def test_background_work_takes_turns_by_weight():
    offers = [offer(i, project="a") for i in range(1, 7)] + [offer(i, project="b") for i in range(11, 17)]
    first = ids(order(offers, 0, lambda o: True, {}, {"a": 2, "b": 1}))[:6]
    assert [p for p, _ in first] == ["a", "b", "a", "a", "b", "a"], "two for a, one for b, neither waits for ever"
    assert [t for p, t in first if p == "a"] == [1, 2, 3, 4], "each project's own order kept"


def test_turns_remember_what_was_already_taken():
    spent = {}
    for _ in range(3):
        charge(spent, "a", 1)
    offers = [offer(1, project="a"), offer(2, project="b")]
    assert ids(order(offers, 0, lambda o: True, spent, {"a": 1, "b": 1}))[0] == ("b", 2)


def test_what_nobody_should_start_or_this_worker_cannot_do_is_left_out():
    offers = [offer(1, "live", claim=50, due=100), offer(2, "live", due=40), offer(3, "live", due=200), offer(4)]
    assert ids(order(offers, 60, lambda o: o.task != 4, {}, {})) == [("a", 3)]


def test_spent_turns_stay_small():
    spent = {"a": 2000.0, "b": 1500.0}
    charge(spent, "a", 1)
    assert spent == {"a": 501.0, "b": 0.0}
