"""Body shaping for the wire path: dropping payload we know we never want.

Distinct from redaction, and deliberately a separate layer.

A ``Redactor`` decides *how much of a value* to write and knows nothing about the
Anthropic request schema — that is what lets one redaction mode apply to the event
log and the wire log alike. A ``Shaper`` is the opposite: it knows exactly what a
``/v1/messages`` body looks like and decides *which parts are worth keeping at
all*. Mixing the two would put tool-name matching inside a generic walker and give
``StrictRedactor`` a special case it has no use for.

Shapers run **before** the redactor (``CaptureProxy._record``), for three reasons:

* hashes stay meaningful — a skeleton's ``sha`` must cover the original tool, and
  in ``flow`` mode the second request's description has already become a
  ``{"_t": "seen"}`` marker by the time the redactor is done with it;
* ``strict`` mode still works — it summarises ``body`` wholesale, so it must see a
  real structure, not one already reduced to ``{"_t": "str", "chars": …}``;
* the redactor stays the last thing to touch every row, so credential stripping
  remains unconditional. A shaper is never a second path to disk.

Shaping is applied to a *parsed copy* of the request body. The bytes forwarded
upstream are the original ones, read before any of this runs, so nothing here can
alter the request the model actually receives.

**Shaping is lossy in a way redaction is not**, so a shaped row is stamped
``"_shaped": ["tools:skeleton"]``. A reader must be able to tell a filtered
capture from a faithful one; when you need ground truth, re-run with
``wire_tools="full"``.

Why tool definitions specifically: on the Agent SDK path the ``claude`` CLI
assembles the tool list, and it ships every native tool plus every MCP server
configured on the machine — not just the ones this app registered. On a measured
run here that was 118 tool definitions totalling 268 KB per request, of which 8
tools (4.4 KB) belonged to this project. The other 110 are pure boilerplate on
every turn, and third-party servers interpolate account state into their
descriptions — real Slack user ids and workspace URLs were present in that
capture. So this is a privacy control at least as much as a size one.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from typing import Any, Protocol

# How much of each tool definition to keep:
#   full     — everything (no shaper installed; the default)
#   skeleton — name, parameter names, required list, description length, hash
#   names    — name and hash only
TOOL_SHAPE_MODES = ("full", "skeleton", "names")


class Shaper(Protocol):
    """Applied to a wire row before the redactor sees it."""

    def shape(self, row: dict[str, Any]) -> dict[str, Any]: ...


def _sha(value: Any) -> str:
    """Fingerprint of a tool definition, stable across key ordering.

    This is the part that makes elision auditable: two turns whose tool lists
    carry the same ``sha`` per entry provably shipped identical definitions, so
    "did the tool list change mid-run?" is still answerable from a shaped trace.
    """
    try:
        canon = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError, RecursionError):
        canon = repr(value)
    return hashlib.sha256(canon.encode("utf-8", "replace")).hexdigest()[:12]


class ToolSchemaShaper:
    """Reduce ``body.tools`` entries to a skeleton, except those matching ``keep``.

    ``keep`` holds :mod:`fnmatch` patterns tested against the tool name, so the
    keep-set is normally one pattern for the MCP server this app registers
    (``"mcp__insurance__*"``) rather than a hand-maintained list of tool names. A
    list would silently drop the schema of any tool added later — which is exactly
    the schema you would want. A pattern keeps it.

    The skeleton still answers what a trace gets asked of a tool list: was the tool
    offered, did its definition change between turns, and did the model pass an
    argument the schema never declared.
    """

    def __init__(self, mode: str = "skeleton", keep: tuple[str, ...] = ()):
        if mode not in TOOL_SHAPE_MODES:
            raise ValueError(f"mode must be one of {TOOL_SHAPE_MODES}, got {mode!r}")
        self.mode = mode
        self.keep = tuple(keep)

    def _kept(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.keep)

    def _one(self, tool: dict[str, Any]) -> dict[str, Any]:
        name = tool.get("name")
        if not isinstance(name, str) or self._kept(name):
            return tool
        digest = _sha(tool)
        if self.mode == "names":
            return {"name": name, "_t": "tool_elided", "sha": digest}
        schema = tool.get("input_schema")
        schema = schema if isinstance(schema, dict) else {}
        props = schema.get("properties")
        required = schema.get("required")
        description = tool.get("description")
        return {
            "name": name,
            "_t": "tool_skeleton",
            "params": sorted(props) if isinstance(props, dict) else [],
            "required": list(required) if isinstance(required, list) else [],
            "desc_chars": len(description) if isinstance(description, str) else 0,
            "sha": digest,
        }

    def shape(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.mode == "full":
            return row
        body = row.get("body")
        if not isinstance(body, dict):
            return row
        tools = body.get("tools")
        if not isinstance(tools, list) or not tools:
            return row
        shaped = [self._one(t) if isinstance(t, dict) else t for t in tools]
        if shaped == tools:
            return row
        out = dict(row)
        # `tools` already exists in `body`, so this replaces it in place and the
        # key order a reader is used to is preserved.
        out["body"] = {**body, "tools": shaped}
        marks = row.get("_shaped")
        out["_shaped"] = [*marks, f"tools:{self.mode}"] if isinstance(marks, list) \
            else [f"tools:{self.mode}"]
        return out


def build_shapers(wire_tools: str = "full",
                  wire_tools_keep: tuple[str, ...] = ()) -> tuple[Shaper, ...]:
    """Shapers implied by the config. Empty tuple means capture verbatim."""
    if wire_tools == "full":
        return ()
    return (ToolSchemaShaper(wire_tools, wire_tools_keep),)
