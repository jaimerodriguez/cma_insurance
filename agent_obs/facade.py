"""``Observability`` — the single object a front-end constructs.

Replaces the god-object this was distilled from (a class called ``TraceLogger``
that also owned an OTel provider and an HTTP proxy, reached as
``trace.otel.span(...)`` and ``trace.capture_base_url``). Here the layers are
named for what they are:

    obs.events   semantic event log
    obs.spans    OTel tracer (session → turn → tool)
    obs.usage    per-turn token/cost ledger
    obs.wire     capture proxy (or None)

A run is a context manager::

    with Observability.start(ObsConfig.from_env(), front_end="repl") as obs:
        ...

Files are keyed on ``obs.run_id``. Provider session ids are recorded via
``note_session()`` into ``var/runs.jsonl``, so a session id still finds its run
without the pending-file renames the original used.

``current()`` returns the active run, or a disabled instance if none is running,
so instrumentation (notably the tool-dispatch decorator) needs no null checks.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from .config import ObsConfig
from .events import EventLog
from .redact import build_redactor
from .sinks import JsonlSink, NullSink, Sink
from .spans import NullTracer, build_tracer, record_turn_usage
from .usage import UsageLedger


class Observability:
    """One run's worth of instrumentation across every layer."""

    def __init__(self, config: ObsConfig, *, front_end: str = "unknown",
                 run_id: str | None = None):
        self.config = config
        self.front_end = front_end
        self.run_id = run_id or _mint_run_id(front_end)
        self.started_at = time.time()
        self.enabled = config.enabled
        self.session_ids: dict[str, str] = {}      # backend -> latest session id
        self._tool_calls = 0
        self._closed = False
        self._session_cm = None

        redactor = build_redactor(config.redact, config.max_field_chars,
                                  config.preview_chars, config.min_collapse_chars,
                                  collapse_structures=config.collapse_structures,
                                  min_node_chars=config.min_node_chars)

        def sink(name: str, on: bool) -> Sink:
            if not (self.enabled and on):
                return NullSink()
            return JsonlSink(config.traces_dir / f"{self.run_id}.{name}.jsonl",
                             max_bytes=config.max_file_bytes,
                             max_files=config.max_files)

        self._event_sink = sink("events", config.events)
        self._span_sink = sink("otel", config.spans)
        self._wire_sink = sink("wire", config.wire)
        self._ledger_sink = (
            JsonlSink(config.ledger_path, max_bytes=config.max_file_bytes,
                      max_files=config.max_files)
            if self.enabled and config.usage else NullSink())

        self.events = EventLog(self._event_sink, redactor, run_id=self.run_id)
        self.spans = build_tracer(self._span_sink, run_id=self.run_id,
                                  service_name=config.service_name,
                                  otlp_endpoint=config.otlp_endpoint,
                                  enabled=self.enabled and config.spans)
        self.usage = UsageLedger(self._ledger_sink, run_id=self.run_id)

        self.wire = None
        self.wire_error: str | None = None
        if self.enabled and config.wire:
            self._start_wire(redactor)

    # --- lifecycle -------------------------------------------------------

    def _start_wire(self, redactor) -> None:
        from .shape import build_shapers
        from .wire import CaptureProxy, ProxyTargetError
        try:
            self.wire = CaptureProxy(
                self._wire_sink, redactor,
                upstream=self.config.upstream,
                paths=self.config.wire_paths,
                response_mode=self.config.wire_responses,
                events=self.events,
                shapers=build_shapers(self.config.wire_tools,
                                      self.config.wire_tools_keep),
            )
        except (ProxyTargetError, OSError) as exc:
            # A failed proxy must not stop the app — it is instrumentation.
            self.wire_error = f"{type(exc).__name__}: {exc}"

    @classmethod
    def start(cls, config: ObsConfig | None = None, *, front_end: str = "unknown",
              install_current: bool = True, **overrides: Any) -> Observability:
        """Build a run, make it the process-wide current one, and log its header."""
        cfg = (config or ObsConfig.from_env())
        if overrides:
            cfg = cfg.with_overrides(**overrides)
        obs = cls(cfg, front_end=front_end)
        if install_current:
            set_current(obs)
        obs._open_session_span()
        obs.events.info("run.start", front_end=front_end, pid=os.getpid(),
                        **{f"cfg.{k}": v for k, v in cfg.describe().items()})
        if obs.wire_error:
            obs.events.warn("wire.unavailable", error=obs.wire_error)
        obs._append_run_index({"event": "start"})
        return obs

    def _open_session_span(self) -> None:
        if isinstance(self.spans, NullTracer):
            return
        # Held open for the whole run so turn spans have a parent. Entered manually
        # rather than with `with`, because the run's lifetime is the REPL's.
        self._session_cm = self.spans.session(
            "session", **{"claude.front_end": self.front_end,
                          "claude.run_id": self.run_id})
        self._session_cm.__enter__()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        t = self.usage.run_totals
        self.events.info(
            "run.end",
            duration_s=round(time.time() - self.started_at, 1),
            turns=t.turns, tool_calls=self._tool_calls,
            input_tokens=t.input_tokens, output_tokens=t.output_tokens,
            cache_read_input_tokens=t.cache_read_input_tokens,
            cache_creation_input_tokens=t.cache_creation_input_tokens,
            cost_usd=t.total_cost_usd or None,
            wire_captured=self.wire.captured if self.wire else 0,
            dropped_rows=sum(getattr(s, "dropped", 0) for s in
                             (self._event_sink, self._span_sink,
                              self._wire_sink, self._ledger_sink)),
        )
        self._append_run_index({"event": "end", "turns": t.turns,
                                "tool_calls": self._tool_calls})
        if self._session_cm is not None:
            try:
                self._session_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
        self.spans.shutdown()
        if self.wire:
            self.wire.shutdown()
        for s in (self._event_sink, self._span_sink, self._wire_sink,
                  self._ledger_sink):
            s.close()
        if current() is self:
            set_current(None)

    def __enter__(self) -> Observability:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- correlation -----------------------------------------------------

    def note_session(self, session_id: str | None, *, backend: str = "sdk") -> None:
        """Record a provider session id for this run (idempotent)."""
        if not session_id or self.session_ids.get(backend) == session_id:
            return
        self.session_ids[backend] = session_id
        self.events.set_session_id(session_id)
        self.events.info("session.id", backend=backend, session_id=session_id)
        self._append_run_index({"event": "session", "backend": backend,
                                "session_id": session_id})

    def _append_run_index(self, payload: dict[str, Any]) -> None:
        """``var/runs.jsonl`` maps run ids to session ids and front-ends.

        This is what replaces renaming files when a session id shows up: one small
        index, greppable, and it survives a run that never learns a session id.
        """
        if not (self.enabled and self.config.events):
            return
        try:
            path = self.config.runs_index_path
            path.parent.mkdir(parents=True, exist_ok=True)
            import json
            row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "run_id": self.run_id, "front_end": self.front_end,
                   "trace_id": self.spans.trace_id, **payload}
            with path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass

    # --- helpers used by instrumentation ---------------------------------

    @property
    def tool_tracing(self) -> bool:
        return self.enabled and self.config.tool_tracing

    def count_tool_call(self) -> None:
        self._tool_calls += 1

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def base_url(self) -> str | None:
        """Base URL to point a client at, or None when wire capture is off."""
        return self.wire.base_url if self.wire else None

    def client_env(self) -> dict[str, str]:
        """Env additions for a spawned CLI (``ClaudeAgentOptions.env``)."""
        return {"ANTHROPIC_BASE_URL": self.base_url} if self.base_url else {}

    @contextmanager
    def turn(self, kind: str, /, **attributes: Any) -> Iterator[Any]:
        """Wrap one model round-trip: a span, start/end events, and timing.

        Yields the span so the caller can attach the ledger record with
        ``record_turn_usage``.

        ``kind`` is positional-only so an attribute of the same name lands in
        ``attributes`` instead of colliding with this parameter.
        """
        started = time.monotonic()
        tools_before = self._tool_calls
        # Merged into a dict rather than passed as `kind=kind, **attributes`: an
        # attribute called "kind" would make that form raise "got multiple values".
        # Merging lets the caller's value simply win.
        fields = {"kind": kind, **attributes}
        self.events.info("turn.start", **fields)
        with self.spans.turn(f"turn.{kind}", **{
                "claude.turn.kind": kind,
                **{f"claude.{k}": v for k, v in attributes.items()}}) as span:
            try:
                yield span
            except BaseException as exc:
                self.events.error("turn.failed", **{
                    **fields,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "error": f"{type(exc).__name__}: {exc}"})
                raise
            else:
                self.events.info("turn.end", **{
                    **fields,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "tool_calls": self._tool_calls - tools_before})

    def record_turn(self, span: Any, record: Any) -> Any:
        """Attach a ``TurnRecord``'s measurements to the turn span."""
        record_turn_usage(span, record)
        return record

    # --- reporting -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """What ``/obs`` prints."""
        t = self.usage.run_totals
        return {
            "run_id": self.run_id,
            "front_end": self.front_end,
            "trace_id": self.spans.trace_id,
            "enabled": self.enabled,
            "layers": {
                "events": bool(self.config.events and self.enabled),
                "spans": bool(self.config.spans and self.enabled)
                         and not isinstance(self.spans, NullTracer),
                "usage": bool(self.config.usage and self.enabled),
                "tools": self.tool_tracing,
                "wire": self.wire is not None,
            },
            "redact": self.config.redact,
            "wire_tools": self.config.wire_tools,
            "wire_base_url": self.base_url,
            "wire_captured": self.wire.captured if self.wire else 0,
            "wire_error": self.wire_error,
            "session_ids": dict(self.session_ids),
            "turns": t.turns,
            "tool_calls": self._tool_calls,
            "tokens": t.all_tokens,
            "cost_usd": t.total_cost_usd or None,
            "files": {k: str(v) for k, v in self.paths().items()},
        }

    def paths(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for label, sink in (("events", self._event_sink), ("otel", self._span_sink),
                            ("wire", self._wire_sink), ("ledger", self._ledger_sink)):
            if getattr(sink, "path", None) is not None:
                out[label] = sink.path
        return out

    def tail(self, n: int = 20) -> list[str]:
        return self.events.tail(n)

    def ledger_rows(self) -> list[dict[str, Any]]:
        read = getattr(self._ledger_sink, "read_all", None)
        return read() if read else []


class _Disabled(Observability):
    """What ``current()`` returns with no active run: every layer a no-op.

    Exists so instrumentation can call ``current().events.info(...)``
    unconditionally — the alternative is a null check at every call site, which is
    how call sites start disagreeing about whether tracing is on.
    """

    def __init__(self) -> None:
        super().__init__(ObsConfig(enabled=False), front_end="none",
                         run_id="disabled")


_DISABLED: Observability | None = None
_CURRENT: Observability | None = None


def _mint_run_id(front_end: str) -> str:
    """Sortable and unique: ``<front_end>-YYYYmmdd-HHMMSS-<rand>``."""
    return f"{front_end}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def set_current(obs: Observability | None) -> None:
    global _CURRENT
    _CURRENT = obs


def current() -> Observability:
    """The active run, or a disabled stand-in. Never returns None."""
    global _DISABLED
    if _CURRENT is not None:
        return _CURRENT
    if _DISABLED is None:
        _DISABLED = _Disabled()
    return _DISABLED
