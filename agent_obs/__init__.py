"""``agent_obs`` — tracing, logging, and usage accounting for LLM agent front-ends.

Four layers, each independently switchable, each writing append-only JSONL:

    events   what happened, in order      var/traces/<run_id>.events.jsonl
    spans    OTel session → turn → tool   var/traces/<run_id>.otel.jsonl
    wire     the real HTTP payloads       var/traces/<run_id>.wire.jsonl
    usage    per-turn tokens and cost     var/ledger.jsonl

Typical use::

    import agent_obs

    with agent_obs.Observability.start(front_end="repl", wire=True) as obs:
        with obs.turn("chat", role="adjuster") as span:
            result = ...                       # call the model
            obs.record_turn(span, obs.usage.from_sdk_result(result, role="adjuster"))

Backend integration:

* **Claude Agent SDK** — pass ``hooks=build_hooks(obs)`` and
  ``env=obs.client_env()`` into ``ClaudeAgentOptions``, and feed every streamed
  message to ``observe_sdk_message(obs, message)``.
* **Anthropic API / Managed Agents** — construct the client with
  ``base_url=obs.base_url`` when it is not None.
* **Tools** — decorate the dispatcher with ``@traced_dispatch``; that covers all
  backends at once, because they all execute tools in this process.

Nothing here changes agent behaviour. The hooks are tracing-only and return
``{}``; ``QuotaGuard`` is the single opt-in exception and is not installed unless
you ask for it.
"""

from __future__ import annotations

from .config import PROJECT_ROOT, ObsConfig
from .events import EventLog
from .facade import Observability, current, set_current
from .logbridge import install_logging, uninstall_logging
from .redact import (
    DevRedactor,
    FlowRedactor,
    NullRedactor,
    StrictRedactor,
    build_redactor,
)
from .shape import TOOL_SHAPE_MODES, Shaper, ToolSchemaShaper, build_shapers
from .sinks import JsonlSink, NullSink
from .spans import OTEL_AVAILABLE, NullTracer, SessionTracer, record_turn_usage
from .tooltrace import trace_callable, traced_dispatch
from .usage import SCHEMA_VERSION, Totals, TurnRecord, UsageLedger, format_totals, summarize

__all__ = [
    "ObsConfig",
    "PROJECT_ROOT",
    "Observability",
    "current",
    "set_current",
    "EventLog",
    "JsonlSink",
    "NullSink",
    "build_redactor",
    "FlowRedactor",
    "StrictRedactor",
    "DevRedactor",
    "NullRedactor",
    "Shaper",
    "ToolSchemaShaper",
    "build_shapers",
    "TOOL_SHAPE_MODES",
    "SessionTracer",
    "NullTracer",
    "OTEL_AVAILABLE",
    "record_turn_usage",
    "TurnRecord",
    "Totals",
    "UsageLedger",
    "summarize",
    "format_totals",
    "SCHEMA_VERSION",
    "traced_dispatch",
    "trace_callable",
    "install_logging",
    "uninstall_logging",
    "build_hooks",
    "observe_sdk_message",
    "QuotaGuard",
    "HOOK_EVENTS",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Defer the SDK-dependent helpers so importing this package never requires
    ``claude_agent_sdk`` — the ``--use-key`` and Managed Agents paths do not use it."""
    if name in ("build_hooks", "observe_sdk_message", "QuotaGuard", "HOOK_EVENTS"):
        from . import sdk
        return getattr(sdk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
