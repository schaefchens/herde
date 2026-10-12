"""herde: a worker that does AI work — speech, text — for projects that host the herde protocol.

A project holds the queue and decides who may see what (protocol/PROTOCOL.md);
the worker connects out to every project it serves, chooses the most urgent
task it can finish in time, runs it on its own models and hands back the result.
"""

VERSION = "0.5.1"
# The protocol this worker speaks (version 1 was ARCHE's first voice-worker API).
PROTOCOL = 2
USER_AGENT = f"herde/{VERSION}"
