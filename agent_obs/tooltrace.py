"""Tool-call tracing at the function boundary — the backend-independent seam.

This project executes every tool in its own process: the Agent SDK path routes
through in-process MCP handlers, the direct-API path runs the loop locally, and
Managed Agents dispatches ``agent.custom_tool_use`` events back to us. All three
converge on one function, ``repl._dispatch(table, name, args)``.

So tool tracing does not need hooks, does not need the SDK, and does not need to
be duplicated per backend: decorate that one function and every tool call in the
project is traced, with arguments, result size, duration, and exceptions.

Hooks (see ``sdk.py``) remain useful for what only they can see — permission
decisions, the model-side ``tool_use_id``, compaction, subagents — but they are
enrichment, not the source of truth for "a tool ran".
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from .facade import current


def traced_dispatch(fn: Callable[..., str]) -> Callable[..., str]:
    """Decorator for a ``_dispatch(table, name, args) -> str`` tool dispatcher.

    Resolves the active ``Observability`` at call time via ``agent_obs.current()``
    rather than taking it as an argument. That is deliberate: it means the three
    existing call sites need no changes, and a module that imported ``_dispatch``
    before observability was started (``cma.py`` does exactly this) still gets the
    traced version. With no active run, ``current()`` returns a no-op and the
    wrapper costs one attribute lookup.
    """

    @functools.wraps(fn)
    def wrapper(table: dict[str, Any], name: str, args: dict[str, Any], *rest, **kw) -> str:
        obs = current()
        if not obs.tool_tracing:
            return fn(table, name, args, *rest, **kw)

        permitted = name in (table or {})
        obs.events.info("tool.start", tool=name, permitted=permitted,
                        arg_keys=sorted((args or {}).keys()), args=args)
        started = time.monotonic()
        with obs.spans.tool(name, **{"claude.tool.permitted": permitted}) as span:
            try:
                result = fn(table, name, args, *rest, **kw)
            except BaseException as exc:
                # _dispatch catches tool exceptions itself, so reaching here means
                # the dispatcher failed — worth recording loudly.
                elapsed = int((time.monotonic() - started) * 1000)
                obs.events.error("tool.dispatch_failed", tool=name,
                                 duration_ms=elapsed,
                                 error=f"{type(exc).__name__}: {exc}")
                span.set_attribute("claude.tool.dispatch_failed", True)
                raise
            elapsed = int((time.monotonic() - started) * 1000)
            # _dispatch reports tool failures as a JSON {"error": ...} string.
            failed = isinstance(result, str) and result.lstrip().startswith('{"error"')
            span.set_attribute("claude.tool.result_bytes", len(result or ""))
            span.set_attribute("claude.tool.duration_ms", elapsed)
            span.set_attribute("claude.tool.failed", failed)
            obs.count_tool_call()
            obs.events.event(
                "tool.error" if failed else "tool.end",
                level="warn" if failed else "info",
                tool=name, duration_ms=elapsed,
                result_bytes=len(result or ""), result=result,
            )
            return result

    wrapper.__wrapped_by_agent_obs__ = True   # type: ignore[attr-defined]
    return wrapper


def trace_callable(name: str | None = None, *, kind: str = "call"):
    """General-purpose decorator for any function worth timing.

    Used for the deterministic (no-LLM) paths — ``process_incident`` sweeps,
    maintenance passes — so they show up on the same timeline as model turns.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        label = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            obs = current()
            if not obs.enabled:
                return fn(*args, **kwargs)
            started = time.monotonic()
            # Field is `func`, not `name`: `name` reads as the event name next to
            # the event name itself, and it used to collide with EventLog.event's
            # own parameter (that is now positional-only, so it would work — but
            # the clearer field is still the right one).
            obs.events.debug(f"{kind}.start", func=label)
            with obs.spans.tool(label, **{"claude.call.kind": kind}):
                try:
                    return fn(*args, **kwargs)
                finally:
                    obs.events.debug(f"{kind}.end", func=label,
                                     duration_ms=int((time.monotonic() - started) * 1000))
        return wrapper
    return decorate
