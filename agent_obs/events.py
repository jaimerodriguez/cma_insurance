"""The semantic event log: "what happened, in order".

One JSON object per line: ``{ts, run_id, seq, level, event, **fields}``. Fields
pass through the configured redactor before they reach the sink.

``seq`` is a monotonic counter. Timestamps are second-resolution strings (they
match the rest of this project's date handling), which is too coarse to order
events inside a fast turn — the sequence number is what gives a total order.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Any

from .redact import Redactor
from .sinks import Sink

# Levels, coarse on purpose. `debug` is where high-volume per-block events go.
LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}

# Row keys the envelope owns. A field of the same name would otherwise overwrite
# the row's own metadata — silently, which is worse than the TypeError that
# shadowing the *parameter* used to raise: losing the `event` name makes the row
# unfindable. Colliding fields are kept, prefixed with `field.`.
_RESERVED_KEYS = frozenset({"ts", "run_id", "seq", "level", "event", "session_id"})


class EventLog:
    def __init__(self, sink: Sink, redactor: Redactor, *, run_id: str,
                 level: str = "debug"):
        self.sink = sink
        self.redactor = redactor
        self.run_id = run_id
        self.threshold = LEVELS.get(level, 10)
        self._seq = itertools.count(1)
        self._lock = threading.Lock()
        self.session_id: str | None = None
        self.counts: dict[str, int] = {}

    def set_session_id(self, session_id: str | None) -> None:
        """Record the provider session id on every later event.

        Unlike the project this was adapted from, learning the session id does not
        rename anything — files are keyed on the run id. It only enriches rows.
        """
        self.session_id = session_id

    def event(self, name: str, /, *, level: str = "info", **fields: Any) -> None:
        """Append one event. ``name`` is **positional-only** and ``level`` is the
        only reserved keyword.

        The ``/`` matters: arbitrary field names arrive via ``**fields``, and a
        caller logging a field that happens to be called ``name`` — a tool name, a
        function name, a persona name — would otherwise collide with this
        parameter and raise ``got multiple values for argument 'name'``. Making it
        positional-only sends ``name=…`` to ``fields`` where it belongs.
        """
        if LEVELS.get(level, 20) < self.threshold:
            return
        with self._lock:
            seq = next(self._seq)
            self.counts[name] = self.counts.get(name, 0) + 1
        payload = self.redactor.event(name, fields)
        row: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self.run_id,
            "seq": seq,
            "level": level,
            "event": name,
            "session_id": self.session_id,
        }
        for key, value in payload.items():
            row[f"field.{key}" if key in _RESERVED_KEYS else key] = value
        self.sink.write(row)

    # Convenience wrappers so callers read as logging calls. `name` is
    # positional-only here for the same reason as in event().
    def debug(self, name: str, /, **fields: Any) -> None:
        self.event(name, level="debug", **fields)

    def info(self, name: str, /, **fields: Any) -> None:
        self.event(name, level="info", **fields)

    def warn(self, name: str, /, **fields: Any) -> None:
        self.event(name, level="warn", **fields)

    def error(self, name: str, /, **fields: Any) -> None:
        self.event(name, level="error", **fields)

    def stderr_sink(self, line: str) -> None:
        """Passed to ``ClaudeAgentOptions.stderr``.

        Everything is kept, at a level that reflects severity — the source project
        dropped non-matching lines entirely, which meant CLI diagnostics vanished.
        Volume is handled by the level threshold and sink rotation, not by
        discarding at the source.
        """
        low = line.lower()
        level = "error" if ("error" in low or "fatal" in low) else (
            "warn" if "warn" in low else "debug")
        self.event("cli.stderr", level=level, line=line)

    def tail(self, n: int = 20) -> list[str]:
        return self.sink.tail(n)
