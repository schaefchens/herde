"""How fast this computer is, measured as it works: what lets it say no to work it cannot finish in time.

Each finished task moves an exponential moving average. The numbers a
project sees in the caps move only on a clear change (a quarter), and
rounded: every change of the caps is a write at the project, and a shared host
serves its listeners from the same few processes.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import TextOutput
    from .state import State

ALPHA = 0.3
# Room for what estimates cannot know (a busy disk, a hotter GPU): a task is claimed only with this much spare.
SAFETY = 1.25
SPARE_SECONDS = 2.0
# Until measured. The M1 Max's live log: 200–250 characters in 7–13 s.
DEFAULT_TTS = {"mlx": 20.0, "torch": 15.0, "stub": 1000.0}
DEFAULT_TEXT = {"read_tps": 50.0, "write_tps": 10.0, "load_s": 30.0, "out_tokens": 300.0}
SAVE_EVERY = 60.0


def _sig(x: float, digits: int = 2) -> float:
    if x <= 0:
        return 0.0
    return round(x, -math.floor(math.log10(abs(x))) + digits - 1)


class Speeds:
    def __init__(self, state: State | None = None, tts_backend: str = "", clock=time.monotonic):
        self.state = state
        self.tts_backend = tts_backend
        self.clock = clock
        saved = state.speeds() if state is not None else {}
        self.data: dict = saved if isinstance(saved, dict) else {}
        self.data.setdefault("tts", {})
        self.data.setdefault("text", {})
        self.data.setdefault("advertised", {})
        self._saved_at = clock()
        self._dirty = False

    # --- speech ----------------------------------------------------------------------------------------

    def tts_cps(self) -> float:
        v = self.data["tts"].get(self.tts_backend)
        return float(v) if isinstance(v, (int, float)) and v > 0 else DEFAULT_TTS.get(self.tts_backend, 10.0)

    def observe_tts(self, chars: int, seconds: float) -> None:
        if chars <= 0 or seconds <= 0:
            return
        self._ema(self.data["tts"], self.tts_backend, chars / seconds)

    def tts_seconds(self, chars: int) -> float:
        return chars / self.tts_cps() + 1.0

    def tts_advertised(self) -> dict:
        return {"chars_per_s": self._advertise(f"tts:{self.tts_backend}", self.tts_cps(), lambda v: float(max(1, round(v))))}

    # --- text ------------------------------------------------------------------------------------------

    def text(self, model: str) -> dict[str, float]:
        m = self.data["text"].get(model)
        out = dict(DEFAULT_TEXT)
        if isinstance(m, dict):
            out.update({k: float(v) for k, v in m.items() if k in DEFAULT_TEXT and isinstance(v, (int, float)) and v > 0})
        return out

    def observe_text(self, model: str, out: TextOutput) -> None:
        m = self.data["text"].setdefault(model, {})
        if out.input_tokens > 0 and out.read_ms > 0:
            self._ema(m, "read_tps", out.input_tokens / (out.read_ms / 1000))
        if out.output_tokens > 0 and out.write_ms > 0:
            self._ema(m, "write_tps", out.output_tokens / (out.write_ms / 1000))
            self._ema(m, "out_tokens", float(out.output_tokens))
        # A load of more than a second was a cold start: that is what a cold model costs.
        if out.load_ms > 1000:
            self._ema(m, "load_s", out.load_ms / 1000)

    def text_seconds(self, model: str, input_tokens: int, max_tokens: int, loaded: bool) -> float:
        s = self.text(model)
        out_tokens = min(max_tokens, s["out_tokens"]) if max_tokens > 0 else s["out_tokens"]
        return (0.0 if loaded else s["load_s"]) + input_tokens / s["read_tps"] + out_tokens / s["write_tps"] + 1.0

    def text_advertised(self, model: str) -> dict:
        s = self.text(model)
        return {
            "read_tps": self._advertise(f"text:{model}:read_tps", s["read_tps"], _sig),
            "write_tps": self._advertise(f"text:{model}:write_tps", s["write_tps"], _sig),
            "load_s": self._advertise(f"text:{model}:load_s", s["load_s"], lambda v: float(max(5, 5 * round(v / 5)))),
        }

    def set_tts(self, chars_per_s: float) -> None:
        """A measurement taken on purpose (`herde bench`): it replaces the average."""
        self.data["tts"][self.tts_backend] = chars_per_s
        self._dirty = True

    def set_text(self, model: str, read_tps: float, write_tps: float, load_s: float | None = None) -> None:
        m = self.data["text"].setdefault(model, {})
        m["read_tps"], m["write_tps"] = read_tps, write_tps
        if load_s is not None:
            m["load_s"] = load_s
        self._dirty = True

    # --- feasibility -------------------------------------------------------------------------------------

    @staticmethod
    def fits(now: float, seconds: float, due_by: float | None) -> bool:
        """Whether work estimated at `seconds` ends before `due_by`, with room to spare."""
        return due_by is None or now + seconds * SAFETY + SPARE_SECONDS <= due_by

    # --- keeping ---------------------------------------------------------------------------------------

    def _ema(self, table: dict, key: str, value: float) -> None:
        old = table.get(key)
        table[key] = value if not isinstance(old, (int, float)) or old <= 0 else old + ALPHA * (value - old)
        self._dirty = True
        self.save()

    def _advertise(self, key: str, value: float, rounding) -> float:
        ads = self.data["advertised"]
        old = ads.get(key)
        if not isinstance(old, (int, float)) or old <= 0 or abs(value - old) / old > 0.25:
            ads[key] = rounding(value)
            self._dirty = True
        return float(ads[key])

    def save(self, force: bool = False) -> None:
        if self.state is None or not self._dirty:
            return
        if force or self.clock() - self._saved_at >= SAVE_EVERY:
            self.state.save_speeds(self.data)
            self._saved_at = self.clock()
            self._dirty = False
