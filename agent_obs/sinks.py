"""JSONL sinks — the single implementation of "append a dict as one line".

The source project this was distilled from had three near-copies of the
open-append-rename dance (one per layer), only one of which took a lock. This is
that logic once, with the pieces the copies were missing: a mutex, a size cap,
and rotation.

Files are named for the **run id**, not the provider session id. That was a
deliberate change: this project has three backends (Agent SDK, direct API,
Managed Agents) whose session identifiers arrive at different times and in
different shapes, and one of them has no per-turn session id at all. A run id we
mint ourselves is available before the first write, is the same across all three,
and removes the pending-file rename — along with the orphaned ``pending-*.jsonl``
files it left behind on crashes. Provider session ids are recorded *inside* the
files and in ``var/runs.jsonl``, so you can still find a run from a session id.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class Sink:
    """Interface a sink must satisfy (also the no-op implementation)."""

    path: Path | None = None

    def write(self, row: dict[str, Any]) -> None:  # pragma: no cover - no-op
        pass

    def tail(self, n: int = 20) -> list[str]:
        return []

    def close(self) -> None:
        pass


class NullSink(Sink):
    """Used when a layer is switched off, so call sites need no guards."""


class JsonlSink(Sink):
    """Append-only JSONL file with a lock, a size cap, and rotation.

    Thread-safe: the wire proxy writes from its own handler threads while the
    event log writes from the main/async thread, and both may target the same
    directory. One lock per sink is enough because each sink owns its file.
    """

    def __init__(self, path: Path, *, max_bytes: int = 32 * 1024 * 1024,
                 max_files: int = 5):
        self.path = path
        self.max_bytes = max_bytes
        self.max_files = max_files
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dropped = 0        # rows lost to write errors; surfaced by /obs

    def write(self, row: dict[str, Any]) -> None:
        try:
            line = json.dumps(row, default=str) + "\n"
        except (TypeError, ValueError):
            # Never let an unserialisable field take down the caller's turn.
            self.dropped += 1
            return
        with self._lock:
            try:
                self._rotate_if_needed(len(line))
                with self.path.open("a") as f:
                    f.write(line)
            except OSError:
                self.dropped += 1

    def write_text_line(self, text: str) -> None:
        """For pre-serialised lines (OTel spans arrive as JSON already)."""
        with self._lock:
            try:
                self._rotate_if_needed(len(text) + 1)
                with self.path.open("a") as f:
                    f.write(text.rstrip("\n") + "\n")
            except OSError:
                self.dropped += 1

    def _rotate_if_needed(self, incoming: int) -> None:
        """Rotate to <name>.1.jsonl … keeping `max_files` generations.

        Called with the lock held. Rotation is what keeps wire capture bounded:
        every turn resends the whole conversation, so an uncapped capture file
        grows with the square of the turn count.
        """
        if self.max_bytes <= 0 or not self.path.exists():
            return
        if self.path.stat().st_size + incoming <= self.max_bytes:
            return
        for gen in range(self.max_files - 1, 0, -1):
            src = self._gen_path(gen)
            if src.exists():
                dst = self._gen_path(gen + 1)
                if gen + 1 >= self.max_files:
                    src.unlink(missing_ok=True)      # oldest generation falls off
                else:
                    src.replace(dst)
        self.path.replace(self._gen_path(1))

    def _gen_path(self, gen: int) -> Path:
        # foo.jsonl -> foo.1.jsonl (keeps the .jsonl suffix so tooling still works)
        return self.path.with_suffix(f".{gen}{self.path.suffix}")

    def tail(self, n: int = 20) -> list[str]:
        if not self.path.exists():
            return []
        try:
            return self.path.read_text().splitlines()[-n:]
        except OSError:
            return []

    def read_all(self) -> list[dict[str, Any]]:
        """Parsed rows, skipping anything malformed (a torn last line on crash)."""
        rows: list[dict[str, Any]] = []
        if not self.path.exists():
            return rows
        try:
            text = self.path.read_text()
        except OSError:
            return rows
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows
