"""OpenTelemetry spans, file-backed, with a real hierarchy.

Two differences from the project this was adapted from:

1. **Nesting.** There, every turn was a root span and tool calls were span
   *events*, so there was no per-tool latency and no way to roll a session up.
   Here it is ``session`` → ``turn`` → ``tool``, which is what makes a trace
   viewer useful and gives tool duration for free.
2. **Optional dependency.** ``opentelemetry-sdk`` may not be installed; if the
   import fails the tracer degrades to a no-op with the same API, so nothing
   guards its call sites.

Attribute names follow the OTel GenAI semantic conventions where they exist
(``gen_ai.*``) so these traces mean something in any backend without a mapping
layer. Cost and cache counters have no semconv equivalent yet and are namespaced
under ``claude.*`` rather than invented at the top level.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from .sinks import Sink

try:  # opentelemetry-sdk is optional
    from opentelemetry.context import Context
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
        SpanExporter,
        SpanExportResult,
    )
    from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context
    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    OTEL_AVAILABLE = False
    SpanExporter = object  # type: ignore[assignment,misc]


class _NullSpan:
    """Enough of the Span API for our call sites, doing nothing."""

    def set_attribute(self, *_a, **_k) -> None: pass
    def set_attributes(self, *_a, **_k) -> None: pass
    def add_event(self, *_a, **_k) -> None: pass
    def set_status(self, *_a, **_k) -> None: pass
    def record_exception(self, *_a, **_k) -> None: pass
    def end(self) -> None: pass
    def is_recording(self) -> bool: return False
    def get_span_context(self) -> None: return None


class NullTracer:
    """No-op tracer used when spans are disabled or OTel is missing."""

    available = False
    trace_id = None

    @contextmanager
    def session(self, *_a, **_k) -> Iterator[Any]:
        yield _NullSpan()

    @contextmanager
    def turn(self, *_a, **_k) -> Iterator[Any]:
        yield _NullSpan()

    @contextmanager
    def tool(self, *_a, **_k) -> Iterator[Any]:
        yield _NullSpan()

    def shutdown(self) -> None:
        pass


if OTEL_AVAILABLE:

    class JsonlSpanExporter(SpanExporter):  # type: ignore[misc]
        """Writes each finished span as one JSON line into a Sink."""

        def __init__(self, sink: Sink):
            self.sink = sink

        def export(self, spans: Sequence["ReadableSpan"]) -> "SpanExportResult":
            try:
                for span in spans:
                    line = span.to_json(indent=None)
                    writer = getattr(self.sink, "write_text_line", None)
                    if writer is not None:
                        writer(line)
                return SpanExportResult.SUCCESS
            except Exception:
                return SpanExportResult.FAILURE

        def shutdown(self) -> None:
            pass

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            return True


class SessionTracer:
    """Owns a **private** TracerProvider, never the global one.

    That is what lets two front-ends (or two sessions in one process) keep their
    spans separate, and it means importing this module has no global side effects.
    """

    available = True

    def __init__(self, sink: Sink, *, run_id: str, service_name: str,
                 otlp_endpoint: str | None = None):
        self.run_id = run_id
        self._root: Any = None
        self._root_ctx: Any = None
        self._provider = TracerProvider(
            resource=Resource.create({
                "service.name": service_name,
                "service.instance.id": run_id,
            })
        )
        # Immediate export: a crash mid-session must still leave the finished
        # spans on disk, which a batching processor would not guarantee.
        self._provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(sink)))
        if otlp_endpoint:
            self._add_otlp(otlp_endpoint)
        self._tracer = self._provider.get_tracer("agent_obs")

    def _add_otlp(self, endpoint: str) -> None:
        """Best-effort OTLP export alongside the file. Missing exporter package
        is not an error — the file sink is the source of truth."""
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError:
            return
        self._provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

    @property
    def trace_id(self) -> str | None:
        if self._root is None:
            return None
        ctx = self._root.get_span_context()
        return f"0x{ctx.trace_id:032x}" if ctx else None

    # --- span levels -----------------------------------------------------

    @contextmanager
    def session(self, name: str = "session", /, **attributes: Any) -> Iterator[Any]:
        """Root span for a whole run. Turn spans attach underneath it."""
        with self._tracer.start_as_current_span(name) as span:
            _set(span, attributes)
            self._root = span
            self._root_ctx = set_span_in_context(span)
            try:
                yield span
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                raise
            finally:
                self._root = None
                self._root_ctx = None

    @contextmanager
    def turn(self, name: str, /, **attributes: Any) -> Iterator[Any]:
        """One request/response cycle. Parented to the session span when open."""
        with self._tracer.start_as_current_span(name, context=self._root_ctx) as span:
            _set(span, attributes)
            try:
                yield span
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                raise

    @contextmanager
    def tool(self, tool_name: str, /, **attributes: Any) -> Iterator[Any]:
        """One tool execution. Parented to whatever span is current (the turn)."""
        with self._tracer.start_as_current_span(f"tool.{tool_name}") as span:
            _set(span, {"gen_ai.tool.name": tool_name, **attributes})
            try:
                yield span
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                raise

    def shutdown(self) -> None:
        self._provider.shutdown()


def _set(span: Any, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            try:
                span.set_attribute(key, value)
            except Exception:
                span.set_attribute(key, str(value))


def build_tracer(sink: Sink, *, run_id: str, service_name: str,
                 otlp_endpoint: str | None, enabled: bool) -> Any:
    if not enabled or not OTEL_AVAILABLE:
        return NullTracer()
    return SessionTracer(sink, run_id=run_id, service_name=service_name,
                         otlp_endpoint=otlp_endpoint)


def record_turn_usage(span: Any, record: Any) -> None:
    """Copy a ``TurnRecord``'s measurements onto a turn span."""
    if record is None:
        return
    _set(span, {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": record.model,
        "gen_ai.usage.input_tokens": record.input_tokens,
        "gen_ai.usage.output_tokens": record.output_tokens,
        "gen_ai.response.finish_reasons": record.stop_reason,
        "claude.cache.read_input_tokens": record.cache_read_input_tokens,
        "claude.cache.creation_input_tokens": record.cache_creation_input_tokens,
        "claude.cost.usd": record.total_cost_usd,
        "claude.backend": record.backend,
        "claude.session.id": record.session_id,
        "claude.tool_calls": record.tool_calls,
        "claude.num_turns": record.num_turns,
        "claude.duration.api_ms": record.duration_api_ms,
        "claude.duration.wall_ms": record.duration_ms,
    })
    if record.is_error and OTEL_AVAILABLE:
        span.set_status(Status(StatusCode.ERROR, "result.is_error"))
