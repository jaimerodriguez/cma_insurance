"""Configuration for the observability package.

One dataclass, one env-var prefix (``OBS_``). Every layer has an independent
on/off switch so a run can trace events but skip the wire proxy, or capture the
wire but skip spans, without touching call sites.

Defaults: everything cheap is on, wire capture is off (it is the expensive,
privacy-sensitive layer), and redaction is ``flow`` — new content readable, text
already written collapsed to a short preview. ``strict`` is the no-content-at-all
mode for traces that might leave this machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Response-capture modes for the wire proxy.
#   none    — request bodies only
#   summary — + response status/headers and, for SSE, event-type counts and byte totals
#   full    — + the whole response body (SSE included). Large; use for debugging only.
WIRE_RESPONSE_MODES = ("none", "summary", "full")

# Redaction modes, least to most revealing. See redact.py for the full rules.
#   flow    — new content in full, anything already written collapses to a preview
#   strict  — no content at all, only sizes and hashes
#   dev     — everything kept, each string truncated
#   none    — passthrough (credentials are still dropped)
REDACT_MODES = ("flow", "strict", "dev", "none")

# How much of each tool *definition* the wire layer keeps. See shape.py.
#   full     — verbatim (default; the capture stays faithful)
#   skeleton — name, parameter names, required list, description length, hash
#   names    — name and hash only
TOOL_SHAPE_MODES = ("full", "skeleton", "names")


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = (os.environ.get(name) or default).strip().lower()
    return raw if raw in allowed else default


@dataclass(frozen=True)
class ObsConfig:
    """What to record, where, and how aggressively to redact it."""

    # Master switch. False makes every layer a no-op without removing call sites.
    enabled: bool = True

    # Root for all output. Per-run files land in <var_dir>/traces/.
    var_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "var")

    # --- layers ----------------------------------------------------------
    events: bool = True          # semantic event log (JSONL)
    spans: bool = True           # OpenTelemetry spans (JSONL, optional OTLP)
    usage: bool = True           # per-turn token/cost ledger
    tool_tracing: bool = True    # wrap tool dispatch (all three backends)
    wire: bool = False           # local capture proxy — off unless asked for

    # --- wire proxy ------------------------------------------------------
    # Request paths to record. Substring match, so "/v1/messages" covers the
    # Messages API and "/v1/beta/sessions" covers Managed Agents session traffic.
    wire_paths: tuple[str, ...] = ("/v1/messages", "/v1/beta")
    wire_responses: str = "summary"
    # Tool-definition shaping. `full` by default: shaping is lossy in a way
    # redaction is not, so it is opt-in and a shaped row is stamped `_shaped`.
    # Worth turning on because the `claude` CLI ships every native tool and every
    # MCP server on the machine — measured here at 110 of 118 definitions and 95%
    # of the request — and third-party descriptions carry account state.
    wire_tools: str = "full"
    # fnmatch patterns kept verbatim. Normally one pattern for the MCP server this
    # app registers, so a tool added later keeps its schema automatically.
    wire_tools_keep: tuple[str, ...] = ()
    # Upstream the proxy forwards to. Read at construction (never at import) so a
    # mutated ANTHROPIC_BASE_URL cannot make the proxy forward to itself.
    upstream: str = "https://api.anthropic.com"

    # --- redaction / size ------------------------------------------------
    redact: str = "flow"
    # First-sight cap per string. 4000 fits a typical tool-schema description or
    # system prompt whole, which is the point — those are read once per run.
    max_field_chars: int = 4000
    preview_chars: int = 20         # flow mode: preview length for repeated text
    # flow mode: strings shorter than this are never collapsed, because the
    # replacement object would be bigger than the text and short repeated strings
    # are labels (tool/function/model/session ids), not content.
    min_collapse_chars: int = 200
    # flow mode: also collapse a repeated dict/list, not just a repeated string.
    # Without this a resent JSON Schema is rewritten in full every request — every
    # string inside one is a field name below `min_collapse_chars`. On by default
    # (it is lossless and greppable by `sha`); set OBS_COLLAPSE_STRUCTURES=0 to
    # get the previous string-only behaviour back.
    collapse_structures: bool = True
    # Floor for the above. Higher than the string floor because collapsing a
    # container costs its shape as well as its text.
    min_node_chars: int = 500
    max_file_bytes: int = 32 * 1024 * 1024   # rotate a sink past this
    max_files: int = 5              # rotated generations to keep per sink

    # --- spans -----------------------------------------------------------
    service_name: str = "claude-agent-transcriber2"
    otlp_endpoint: str | None = None    # e.g. http://localhost:4318/v1/traces

    @property
    def traces_dir(self) -> Path:
        return self.var_dir / "traces"

    @property
    def ledger_path(self) -> Path:
        return self.var_dir / "ledger.jsonl"

    @property
    def runs_index_path(self) -> Path:
        return self.var_dir / "runs.jsonl"

    def with_overrides(self, **kwargs: Any) -> ObsConfig:
        return replace(self, **kwargs)

    @classmethod
    def from_env(cls, **overrides: Any) -> ObsConfig:
        """Build from ``OBS_*`` env vars; explicit kwargs win over the environment.

        Recognised:
          OBS_ENABLED, OBS_VAR_DIR, OBS_EVENTS, OBS_SPANS, OBS_USAGE,
          OBS_TOOL_TRACING, OBS_WIRE, OBS_WIRE_PATHS (comma-separated),
          OBS_WIRE_RESPONSES (none|summary|full), OBS_UPSTREAM,
          OBS_WIRE_TOOLS (full|skeleton|names), OBS_WIRE_TOOLS_KEEP (comma-separated
          fnmatch patterns),
          OBS_REDACT (flow|strict|dev|none), OBS_MAX_FIELD_CHARS, OBS_PREVIEW_CHARS,
          OBS_MIN_COLLAPSE_CHARS, OBS_COLLAPSE_STRUCTURES, OBS_MIN_NODE_CHARS,
          OBS_MAX_FILE_BYTES,
          OBS_MAX_FILES, OBS_SERVICE_NAME, OBS_OTLP_ENDPOINT
        """
        base = cls()
        var_dir = os.environ.get("OBS_VAR_DIR")
        paths = os.environ.get("OBS_WIRE_PATHS")
        keep = os.environ.get("OBS_WIRE_TOOLS_KEEP")
        cfg = cls(
            enabled=_flag("OBS_ENABLED", base.enabled),
            var_dir=Path(var_dir).expanduser() if var_dir else base.var_dir,
            events=_flag("OBS_EVENTS", base.events),
            spans=_flag("OBS_SPANS", base.spans),
            usage=_flag("OBS_USAGE", base.usage),
            tool_tracing=_flag("OBS_TOOL_TRACING", base.tool_tracing),
            wire=_flag("OBS_WIRE", base.wire),
            wire_paths=(tuple(p.strip() for p in paths.split(",") if p.strip())
                        if paths else base.wire_paths),
            wire_responses=_choice("OBS_WIRE_RESPONSES", base.wire_responses,
                                   WIRE_RESPONSE_MODES),
            wire_tools=_choice("OBS_WIRE_TOOLS", base.wire_tools, TOOL_SHAPE_MODES),
            wire_tools_keep=(tuple(p.strip() for p in keep.split(",") if p.strip())
                             if keep else base.wire_tools_keep),
            upstream=os.environ.get("ANTHROPIC_BASE_URL") or base.upstream,
            redact=_choice("OBS_REDACT", base.redact, REDACT_MODES),
            max_field_chars=_int("OBS_MAX_FIELD_CHARS", base.max_field_chars),
            preview_chars=_int("OBS_PREVIEW_CHARS", base.preview_chars),
            min_collapse_chars=_int("OBS_MIN_COLLAPSE_CHARS", base.min_collapse_chars),
            collapse_structures=_flag("OBS_COLLAPSE_STRUCTURES",
                                      base.collapse_structures),
            min_node_chars=_int("OBS_MIN_NODE_CHARS", base.min_node_chars),
            max_file_bytes=_int("OBS_MAX_FILE_BYTES", base.max_file_bytes),
            max_files=_int("OBS_MAX_FILES", base.max_files),
            service_name=os.environ.get("OBS_SERVICE_NAME") or base.service_name,
            otlp_endpoint=os.environ.get("OBS_OTLP_ENDPOINT") or base.otlp_endpoint,
        )
        return replace(cfg, **overrides) if overrides else cfg

    def describe(self) -> dict[str, object]:
        """Flat view for /obs displays and the run header event."""
        return {
            "enabled": self.enabled,
            "var_dir": str(self.var_dir),
            "events": self.events,
            "spans": self.spans,
            "usage": self.usage,
            "tool_tracing": self.tool_tracing,
            "wire": self.wire,
            "wire_paths": list(self.wire_paths),
            "wire_responses": self.wire_responses,
            "wire_tools": self.wire_tools,
            "wire_tools_keep": list(self.wire_tools_keep),
            "redact": self.redact,
            "preview_chars": self.preview_chars,
            "min_collapse_chars": self.min_collapse_chars,
            "collapse_structures": self.collapse_structures,
            "otlp_endpoint": self.otlp_endpoint,
        }
