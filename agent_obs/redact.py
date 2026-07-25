"""Redaction, applied at the sink boundary rather than per call site.

Everything written to disk passes through a ``Redactor``, so a call site can
never forget to redact — the only way to log raw content is to configure a
permissive redactor deliberately.

Four modes:

* ``flow`` (default) — **first sight in full, repeats as previews.** Each string
  is written whole the first time it appears (capped at ``max_field_chars``);
  every later appearance of the same text becomes
  ``{"_t": "seen", "preview": <first 20 chars>, "chars": N, "sha": …}`` — but only
  for strings above ``min_collapse_chars``, so labels (tool names, function names,
  model ids, session ids) stay verbatim however often they repeat.

  The same rule applies one level up, to **whole subtrees**: a dict or list whose
  canonical JSON exceeds ``min_node_chars`` and has been written before collapses
  to ``{"_t": "seen_node", "kind": …, "chars": N, "sha": …}``. Without this, a
  resent tool schema is only *partly* collapsed — its description is one long
  string and does collapse, but its ``input_schema`` is a nest of short strings,
  every one of them below the floor, so the bulk of it was rewritten verbatim on
  every request. Structural dedup is what actually makes the "resent payload
  collapses" claim below true. Switch it off with ``collapse_structures=False``.

  Because every request resends the whole conversation, this is what makes a wire
  log readable: turn 1 shows everything, turn 8 shows only the genuinely new
  message with the history collapsed to a skeleton you can still follow. The
  ``sha`` points back to the full text earlier in the same file. It also bounds
  growth — an uncapped capture grows with the square of the turn count.
* ``strict`` — structural only. Every string becomes ``{chars, sha}``. Keeps
  measurements (prompt sizes, tool counts, cache-breakpoint placement) with no
  content whatsoever. Use when traces might leave this machine.
* ``dev`` — every string kept, truncated to ``max_field_chars``. No repeat
  collapsing, so a long conversation gets very large.
* ``none`` — passthrough. Local debugging on synthetic data only.

Credentials are dropped in **every** mode, including ``none``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, Protocol

# Keys whose values are text the model wrote, read, or was told — the content we
# summarise in strict mode rather than store.
_CONTENT_KEYS = frozenset({
    "content", "text", "system", "prompt", "input", "arguments", "result",
    "thinking", "signature", "message", "messages", "preview", "reply",
    "tool_input", "tool_response", "args", "output",
})

# Diagnostic strings kept readable even in strict mode, truncated rather than
# hashed. A deliberate, documented hole in the "no content on disk" guarantee: an
# exception type and message is the single most useful field in a trace, and a
# hashed error makes the whole log worthless for debugging. The truncation bounds
# how much incidental data an error string can carry.
_DIAGNOSTIC_KEYS = frozenset({"error", "proxy_error", "line", "message_error"})
_DIAGNOSTIC_MAX_CHARS = 300

# Keys that must never be written in any mode, even in `none`. Auth material has
# no diagnostic value and every one of these is a credential.
_NEVER = frozenset({
    "authorization", "x-api-key", "api_key", "apikey", "cookie", "set-cookie",
    "anthropic_api_key", "password", "secret", "token", "access_token",
    "refresh_token", "bearer",
})

_REDACTED = "<redacted>"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


def _canonical(value: Any) -> str | None:
    """Order-independent JSON for a container, or None if it will not serialise.

    ``sort_keys`` is what makes two dicts that differ only in key order hash the
    same. ``default=str`` keeps a stray non-JSON value from making the whole
    subtree unhashable — the sinks stringify the same way, so the canonical form
    matches what would have been written.
    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError, RecursionError):
        return None


def summarize(value: Any) -> Any:
    """Shape summary of a value: what it was, how big, and a stable fingerprint."""
    if isinstance(value, str):
        return {"_t": "str", "chars": len(value), "sha": _hash(value)}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        # Keep structural keys (type, role, name, ids, cache_control) — they are the
        # cache/tool-layout signal — and summarise the rest.
        out: dict[str, Any] = {}
        for key, sub in value.items():
            lk = str(key).lower()
            if lk in _NEVER:
                out[key] = _REDACTED
            elif lk in ("type", "role", "name", "id", "tool_use_id",
                        "cache_control", "ttl", "stop_reason", "model",
                        # "_t" marks a value this module already summarised (the
                        # wire proxy pre-summarises SSE streams); re-hashing it
                        # would only obscure our own marker.
                        "_t"):
                out[key] = sub
            else:
                out[key] = summarize(sub)
        return out
    if isinstance(value, (list, tuple)):
        return {"_t": "list", "len": len(value),
                "items": [summarize(v) for v in value[:50]]}
    return {"_t": type(value).__name__}


class Redactor(Protocol):
    """Applied to every row on its way to a sink."""

    def event(self, name: str, fields: dict[str, Any]) -> dict[str, Any]: ...

    def wire(self, row: dict[str, Any]) -> dict[str, Any]: ...


class NullRedactor:
    """Passthrough, except credentials — those are dropped in every mode."""

    mode = "none"

    def event(self, name: str, fields: dict[str, Any]) -> dict[str, Any]:
        return _strip_secrets(fields)

    def wire(self, row: dict[str, Any]) -> dict[str, Any]:
        return _strip_secrets(row)


class DevRedactor:
    """Keeps content but caps every string, so one huge tool result cannot
    balloon a trace file."""

    mode = "dev"

    def __init__(self, max_field_chars: int = 2000):
        self.max_field_chars = max_field_chars

    def _walk(self, value: Any, depth: int = 0) -> Any:
        if depth > 12:
            return "<depth-limit>"
        if isinstance(value, str):
            if len(value) <= self.max_field_chars:
                return value
            return value[:self.max_field_chars] + f"…(+{len(value) - self.max_field_chars} chars)"
        if isinstance(value, dict):
            return {k: (_REDACTED if str(k).lower() in _NEVER else self._walk(v, depth + 1))
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            head = [self._walk(v, depth + 1) for v in value[:200]]
            if len(value) > 200:
                head.append(f"…(+{len(value) - 200} items)")
            return head
        return value

    def event(self, name: str, fields: dict[str, Any]) -> dict[str, Any]:
        return self._walk(fields)

    def wire(self, row: dict[str, Any]) -> dict[str, Any]:
        return self._walk(row)


class StrictRedactor:
    """Structure and measurements, never content.

    Event fields: scalars pass through (they are counters, ids, durations, flags);
    anything under a content-ish key is summarised.
    Wire rows: the body is summarised wholesale, which preserves the message/tool
    structure and `cache_control` placement while dropping the text.
    """

    mode = "strict"

    def event(self, name: str, fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in fields.items():
            lk = str(key).lower()
            if lk in _NEVER:
                out[key] = _REDACTED
            elif isinstance(value, (int, float, bool)) or value is None:
                out[key] = value
            elif lk in _DIAGNOSTIC_KEYS and isinstance(value, str):
                out[key] = (value if len(value) <= _DIAGNOSTIC_MAX_CHARS
                            else value[:_DIAGNOSTIC_MAX_CHARS] + "…")
            elif lk not in _CONTENT_KEYS and _is_identifier_list(value):
                # A list of short strings under a non-content key is schema, not
                # data: tool argument *names*, configured paths, MCP server names.
                # Hashing these would delete the signal and protect nothing.
                out[key] = list(value)
            elif lk in _CONTENT_KEYS or not isinstance(value, str):
                out[key] = summarize(value)
            else:
                # A short non-content string (a tool name, a role, an id) is safe
                # and is the whole point of the event log.
                out[key] = value if len(value) <= 200 else summarize(value)
        return out

    def wire(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for key in ("body", "response_body"):
            if key in out:
                out[key] = summarize(out[key])
        if isinstance(out.get("headers"), dict):
            out["headers"] = {k: (_REDACTED if k.lower() in _NEVER else v)
                              for k, v in out["headers"].items()}
        return out


class FlowRedactor:
    """Full text on first sight; a short preview on every repeat.

    Deduplication is by content hash across the whole run, not by position in the
    ``messages`` array. That matters because the same text moves position between
    requests (a user message at index 3 this turn was index 1 last turn) and
    because it catches repeats anywhere — resent tool schemas and an unchanged
    system prompt collapse for exactly the same reason the message history does.

    Repeats are caught at two granularities: individual strings, and whole
    dict/list subtrees (see ``collapse_structures``). The subtree pass is the one
    that handles payloads built from many short strings — a JSON Schema is the
    common case, and no per-string rule can collapse one, because every string in
    it is a field name well under the floor.

    The ``sha`` on a collapsed entry is the same ``sha`` the full text was written
    under, so the first occurrence is findable by grep in the same file.
    """

    mode = "flow"

    def __init__(self, max_field_chars: int = 4000, preview_chars: int = 20,
                 min_collapse_chars: int = 200, *,
                 collapse_structures: bool = True, min_node_chars: int = 500):
        self.max_field_chars = max_field_chars
        self.preview_chars = preview_chars
        self.collapse_structures = collapse_structures
        # Higher than the string floor on purpose. Collapsing a string costs you
        # its text; collapsing a container costs you its *shape* too — which keys
        # were present, how the message blocks nested — and shape is most of what
        # a wire trace is read for. A floor here keeps the small containers
        # (content blocks, cache_control markers) visible and only collapses the
        # payload-sized ones that actually dominate a file.
        self.min_node_chars = min_node_chars
        # Never collapse a string smaller than the object that would replace it.
        # A `{_t, preview, chars, sha}` entry serialises to ~70 characters, so
        # collapsing short strings makes the file both larger and harder to read —
        # and short repeated strings are almost always labels rather than content:
        # function names, tool names, model ids, session ids, roles. Content worth
        # deduplicating (message bodies, system prompts, tool schemas) is far above
        # this floor.
        self.min_collapse_chars = min_collapse_chars
        # The wire proxy redacts on its own handler threads while the event log
        # redacts on the main thread, and both share this set.
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def _string(self, value: str) -> Any:
        # Short strings always pass through verbatim, however often they repeat.
        if len(value) < self.min_collapse_chars:
            return value
        digest = _hash(value)
        with self._lock:
            first_time = digest not in self._seen
            if first_time:
                self._seen.add(digest)
        if not first_time:
            return {"_t": "seen", "preview": value[:self.preview_chars],
                    "chars": len(value), "sha": digest}
        if len(value) <= self.max_field_chars:
            return value
        return {"_t": "truncated", "text": value[:self.max_field_chars],
                "chars": len(value), "sha": digest}

    def _node(self, value: Any) -> dict[str, Any] | None:
        """A ``seen_node`` marker if this container was written before, else None.

        Registering happens on the *first* sight, before the children are walked,
        so a nested repeat still collapses against the outer occurrence.
        """
        canon = _canonical(value)
        if canon is None or len(canon) < self.min_node_chars:
            return None
        digest = _hash(canon)
        with self._lock:
            if digest not in self._seen:
                self._seen.add(digest)
                return None
        return {"_t": "seen_node",
                "kind": "list" if isinstance(value, (list, tuple)) else "dict",
                "chars": len(canon), "sha": digest}

    def _walk(self, value: Any, depth: int = 0) -> Any:
        if depth > 12:
            return "<depth-limit>"
        if isinstance(value, str):
            return self._string(value)
        if isinstance(value, (dict, list, tuple)):
            # depth 0 is the row itself: collapsing that would replace a whole
            # record with a marker and lose its envelope.
            if self.collapse_structures and depth > 0:
                marker = self._node(value)
                if marker is not None:
                    return marker
            if isinstance(value, dict):
                return {k: (_REDACTED if str(k).lower() in _NEVER
                            else self._walk(v, depth + 1))
                        for k, v in value.items()}
            return [self._walk(v, depth + 1) for v in value]
        return value

    def event(self, name: str, fields: dict[str, Any]) -> dict[str, Any]:
        return self._walk(fields)

    def wire(self, row: dict[str, Any]) -> dict[str, Any]:
        return self._walk(row)


def _is_identifier_list(value: Any) -> bool:
    """True for a short list of short strings — names, keys, paths.

    Bounded on both length and element size so a list of message bodies can never
    slip through as "identifiers".
    """
    if not isinstance(value, (list, tuple)) or len(value) > 64:
        return False
    return all(isinstance(v, str) and len(v) <= 80 for v in value)


def _strip_secrets(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return "<depth-limit>"
    if isinstance(value, dict):
        return {k: (_REDACTED if str(k).lower() in _NEVER else _strip_secrets(v, depth + 1))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_secrets(v, depth + 1) for v in value]
    return value


def build_redactor(mode: str, max_field_chars: int = 4000,
                   preview_chars: int = 20,
                   min_collapse_chars: int = 200, *,
                   collapse_structures: bool = True,
                   min_node_chars: int = 500) -> Redactor:
    if mode == "none":
        return NullRedactor()
    if mode == "dev":
        return DevRedactor(max_field_chars)
    if mode == "strict":
        return StrictRedactor()
    return FlowRedactor(max_field_chars, preview_chars, min_collapse_chars,
                        collapse_structures=collapse_structures,
                        min_node_chars=min_node_chars)
