"""Per-turn usage ledger, with one adapter per backend.

``TurnRecord`` is backend-neutral: the three ways this project reaches a model
(Claude Agent SDK, the Anthropic Messages API directly, Managed Agents) expose
their accounting in three different shapes, and each gets an adapter that
produces the same record. That keeps ``var/ledger.jsonl`` comparable across
backends — which is the whole point, since choosing between them is a cost
question.

Every row carries ``schema_version``. The source project's stats code
reconstructed records from raw dicts by reading dataclass field defaults, so a
field rename silently corrupted all-time history; a version field makes that a
detectable condition instead.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .sinks import Sink

SCHEMA_VERSION = 1


@dataclass
class TurnRecord:
    """One model round-trip, whichever backend served it."""

    ts: str
    run_id: str
    backend: str                       # sdk | api | cma
    turn: int
    schema_version: int = SCHEMA_VERSION
    session_id: str | None = None
    role: str | None = None            # adjuster | insurer | agent
    identity: str | None = None
    model: str | None = None
    kind: str = "chat"                 # chat | maintenance | classify | …
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_cost_usd: float | None = None
    duration_ms: int = 0
    duration_api_ms: int = 0
    num_turns: int = 0
    tool_calls: int = 0
    stop_reason: str | None = None
    is_error: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def all_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_read_input_tokens + self.cache_creation_input_tokens)


@dataclass
class Totals:
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_cost_usd: float = 0.0
    duration_ms: int = 0
    duration_api_ms: int = 0
    tool_calls: int = 0
    errors: int = 0

    def add(self, r: TurnRecord) -> None:
        self.turns += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.cache_read_input_tokens += r.cache_read_input_tokens
        self.cache_creation_input_tokens += r.cache_creation_input_tokens
        self.total_cost_usd += r.total_cost_usd or 0.0
        self.duration_ms += r.duration_ms
        self.duration_api_ms += r.duration_api_ms
        self.tool_calls += r.tool_calls
        self.errors += 1 if r.is_error else 0

    @property
    def all_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens
                + self.cache_read_input_tokens + self.cache_creation_input_tokens)

    @property
    def cache_hit_ratio(self) -> float:
        denom = (self.cache_read_input_tokens + self.cache_creation_input_tokens
                 + self.input_tokens)
        return self.cache_read_input_tokens / denom if denom else 0.0


class UsageLedger:
    """Append-only ledger plus in-process totals for the current run."""

    def __init__(self, sink: Sink, *, run_id: str):
        self.sink = sink
        self.run_id = run_id
        self.run_totals = Totals()
        self._turn = 0

    def record(self, rec: TurnRecord) -> TurnRecord:
        self.run_totals.add(rec)
        self.sink.write(asdict(rec))
        return rec

    def next_turn(self) -> int:
        self._turn += 1
        return self._turn

    # --- adapters --------------------------------------------------------

    def from_sdk_result(self, result: Any, **ctx: Any) -> TurnRecord:
        """Claude Agent SDK ``ResultMessage``.

        The richest source: the CLI has already totalled the whole agent loop, so
        cost, wall-vs-API duration, and the internal turn count all arrive here.
        ``total_cost_usd`` is ``None`` under subscription auth — tokens are still
        recorded, which is why cost is nullable throughout.
        """
        usage = getattr(result, "usage", None) or {}
        if not isinstance(usage, dict):        # some versions expose an object
            usage = {k: getattr(usage, k, None)
                     for k in ("input_tokens", "output_tokens",
                               "cache_read_input_tokens", "cache_creation_input_tokens")}
        return self._build(
            backend="sdk",
            session_id=getattr(result, "session_id", None),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            total_cost_usd=getattr(result, "total_cost_usd", None),
            duration_ms=getattr(result, "duration_ms", 0),
            duration_api_ms=getattr(result, "duration_api_ms", 0),
            num_turns=getattr(result, "num_turns", 0),
            stop_reason=getattr(result, "stop_reason", None)
                        or getattr(result, "subtype", None),
            is_error=bool(getattr(result, "is_error", False)),
            **ctx,
        )

    def from_api_response(self, response: Any, *, wall_ms: int = 0,
                          **ctx: Any) -> TurnRecord:
        """Anthropic Messages API response.

        One record per ``messages.create`` call, so a turn that loops through
        several tool round-trips produces several rows — accumulate by run or by
        ``extra.loop_turn`` when comparing against the SDK backend, which reports
        the whole loop as one row.
        """
        usage = getattr(response, "usage", None)
        # Pop unconditionally: the caller usually passes the model it asked for, and
        # the response echoes the model that actually served the request. Prefer the
        # response (a fallback would otherwise go unrecorded), but the key has to
        # leave `ctx` either way or it arrives twice.
        requested_model = ctx.pop("model", None)
        return self._build(
            backend="api",
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None),
            total_cost_usd=None,          # the API does not price the call for us
            duration_ms=wall_ms,
            model=getattr(response, "model", None) or requested_model,
            stop_reason=getattr(response, "stop_reason", None),
            **ctx,
        )

    def from_cma_usage(self, usage: Any, *, session_id: str | None = None,
                       wall_ms: int = 0, **ctx: Any) -> TurnRecord:
        """Managed Agents. Usage arrives on session events and its shape is not
        guaranteed, so every field is read defensively and a missing one is 0."""
        def get(name: str) -> Any:
            if usage is None:
                return None
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None)

        return self._build(
            backend="cma",
            session_id=session_id,
            input_tokens=get("input_tokens"),
            output_tokens=get("output_tokens"),
            cache_read_input_tokens=get("cache_read_input_tokens"),
            cache_creation_input_tokens=get("cache_creation_input_tokens"),
            duration_ms=wall_ms,
            **ctx,
        )

    def _build(self, *, backend: str, **fields: Any) -> TurnRecord:
        extra = fields.pop("extra", None) or {}
        rec = TurnRecord(
            ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            run_id=self.run_id,
            backend=backend,
            turn=self.next_turn(),
            input_tokens=int(fields.pop("input_tokens", 0) or 0),
            output_tokens=int(fields.pop("output_tokens", 0) or 0),
            cache_read_input_tokens=int(fields.pop("cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(fields.pop("cache_creation_input_tokens", 0) or 0),
            duration_ms=int(fields.pop("duration_ms", 0) or 0),
            duration_api_ms=int(fields.pop("duration_api_ms", 0) or 0),
            num_turns=int(fields.pop("num_turns", 0) or 0),
            tool_calls=int(fields.pop("tool_calls", 0) or 0),
            extra=extra,
            **fields,
        )
        return self.record(rec)


def summarize(rows: list[dict[str, Any]], by: str = "backend") -> dict[str, Totals]:
    """Bucket ledger rows for reporting.

    Rows are read as plain dicts and only known keys are copied onto a fresh
    record, so an older or newer ``schema_version`` degrades to partial totals
    instead of raising.
    """
    buckets: dict[str, Totals] = {}
    known = set(TurnRecord.__dataclass_fields__)
    for row in rows:
        name = str(row.get(by) or "-")
        payload = {k: v for k, v in row.items() if k in known}
        payload.setdefault("ts", "")
        payload.setdefault("run_id", "")
        payload.setdefault("backend", "-")
        payload.setdefault("turn", 0)
        try:
            rec = TurnRecord(**payload)
        except TypeError:
            continue
        buckets.setdefault(name, Totals()).add(rec)
    return buckets


def format_totals(t: Totals) -> str:
    cost = f"${t.total_cost_usd:.4f}" if t.total_cost_usd else "n/a"
    return (f"turns={t.turns} in={t.input_tokens} out={t.output_tokens} "
            f"cache_read={t.cache_read_input_tokens} cache_write={t.cache_creation_input_tokens} "
            f"hit={t.cache_hit_ratio:.0%} tools={t.tool_calls} cost={cost} "
            f"wall={t.duration_ms}ms")
