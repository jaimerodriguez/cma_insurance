"""Generate Anthropic tool schemas from the ``AGENT_TOOLS`` registry.

Introspects each function in ``tools.AGENT_TOOLS`` — its signature, type hints,
and Google-style docstring — and emits a JSON schema in the shape the Anthropic
Messages API expects::

    {"name": ..., "description": ..., "input_schema": {"type": "object", ...}}

Because it reads the live functions, the schemas stay in sync with the code.
Nested dataclass parameters (e.g. ``DynamicPolicies``) become object schemas,
``StrEnum`` params become string enums, and ``IntFlag`` params (``ReportType``)
become integer bitmasks.

Run as a script to print the full schema list, which is also written to
``agent_tools_schema.json``:

    python3 agent_schemas.py
"""

import inspect
import json
import re
import types
import typing
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum, IntFlag
from pathlib import Path

from tools import AGENT_TOOLS

SCHEMA_FILE = Path(__file__).with_name("agent_tools_schema.json")

# A JSON-schema fragment, and a whole Anthropic tool schema. Both are plain
# dicts at runtime; the aliases exist so the annotations say which of the two a
# function deals in, and so a bare ``dict`` does not leak "unknown value type"
# into every caller — ``roles.schemas_for_role`` and ``mcp_server.tool_specs``
# both re-export this shape.
JsonSchema = dict[str, typing.Any]
ToolSchema = dict[str, typing.Any]

_PRIMITIVES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_SECTION_HEADERS = ("Args:", "Arguments:", "Attributes:", "Returns:", "Raises:", "Yields:")
_PARAM_HEADERS = ("Args:", "Arguments:", "Attributes:")
_ENTRY_RE = re.compile(r"([A-Za-z_]\w*)\s*(?:\([^)]*\))?:\s*(.*)")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return ``(summary, {param_name: description})`` from a Google-style docstring."""
    if not doc:
        return "", {}
    lines = doc.splitlines()

    # Summary: everything up to the first recognized section header.
    summary_parts: list[str] = []
    for line in lines:
        if line.strip() in _SECTION_HEADERS:
            break
        summary_parts.append(line.strip())
    summary = " ".join(p for p in summary_parts).strip()

    # Param descriptions from the Args/Attributes section(s).
    params: dict[str, str] = {}
    in_params = False
    base_indent: int | None = None
    key: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped in _PARAM_HEADERS:
            in_params, base_indent, key = True, None, None
            continue
        if stripped in _SECTION_HEADERS:  # a different section ends param parsing
            in_params, key = False, None
            continue
        if not in_params or not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if base_indent is None:
            base_indent = indent
        match = _ENTRY_RE.match(stripped)
        if indent <= base_indent and match:
            key = match.group(1)
            params[key] = match.group(2).strip()
        elif key:  # continuation line for the current param
            params[key] += " " + stripped
    return summary, params


def _is_special(tp: object) -> bool:
    """True if ``tp`` is an Enum subclass or a dataclass (needs custom mapping)."""
    return (isinstance(tp, type) and issubclass(tp, Enum)) or is_dataclass(tp)


def _as_nullable(schema: JsonSchema) -> JsonSchema:
    """Widen a schema so it also accepts ``null``.

    A parameter annotated ``X | None`` genuinely accepts ``None``, and the model
    is routinely handed defaults that contain one — the ADJUSTER prompt embeds
    the policies object verbatim, ``agent_override_autoapprove_reason: null``
    included, and tells the model to pass it through. Emitting the non-null
    schema made the MCP server reject those calls with
    ``Input validation error: None is not of type 'string'`` *before* the tool
    ran, so the failure never reached ``_dispatch`` and could not be caught
    there. The schema has to match the signature.
    """
    out = dict(schema)
    kind = out.get("type")
    if isinstance(kind, str):
        if kind != "null":
            out["type"] = [kind, "null"]
    elif isinstance(kind, list) and "null" not in kind:
        out["type"] = [*kind, "null"]
    # An enum constrains values independently of `type`, so a nullable enum has
    # to list None too or null stays rejected.
    if isinstance(out.get("enum"), list) and None not in out["enum"]:
        out["enum"] = [*out["enum"], None]
    return out


def _type_to_schema(tp: object, description: str = "") -> JsonSchema:
    """Map a Python type annotation to a JSON-schema fragment."""
    origin = typing.get_origin(tp)
    if origin is types.UnionType or origin is typing.Union:
        # Optional[...] / A | B — prefer an enum/dataclass member, and keep the
        # nullability rather than dropping it (see ``_as_nullable``).
        all_args = typing.get_args(tp)
        args = [a for a in all_args if a is not type(None)]
        chosen = next((a for a in args if _is_special(a)), args[0] if args else str)
        schema = _type_to_schema(chosen, description)
        return _as_nullable(schema) if len(args) != len(all_args) else schema

    schema: JsonSchema = {}
    if isinstance(tp, type) and issubclass(tp, IntFlag):
        members = ", ".join(f"{m.name}={m.value}" for m in tp)
        schema = {"type": "integer"}
        description = (
            f"{description} Integer bitmask; combine values by adding. "
            f"Members: {members}."
        ).strip()
    elif isinstance(tp, type) and issubclass(tp, Enum):
        schema = {"type": "string", "enum": [m.value for m in tp]}
    elif is_dataclass(tp):
        schema = _dataclass_to_schema(tp)
    elif tp in _PRIMITIVES:
        schema = {"type": _PRIMITIVES[tp]}
    else:
        schema = {"type": "string"}  # safe fallback

    if description:
        schema["description"] = description
    return schema


def _dataclass_to_schema(dc: type) -> JsonSchema:
    """Build an object schema for a dataclass parameter (e.g. ``DynamicPolicies``)."""
    _, attr_desc = _parse_docstring(dc.__doc__)
    hints = typing.get_type_hints(dc)
    properties: dict[str, JsonSchema] = {}
    required: list[str] = []
    for f in fields(dc):
        properties[f.name] = _type_to_schema(hints.get(f.name, str), attr_desc.get(f.name, ""))
        if f.default is MISSING and f.default_factory is MISSING:
            required.append(f.name)
    schema: JsonSchema = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def build_tool_schema(func: typing.Callable[..., typing.Any]) -> ToolSchema:
    """Build one Anthropic tool schema from a function in ``AGENT_TOOLS``."""
    summary, arg_desc = _parse_docstring(func.__doc__)
    hints = typing.get_type_hints(func)
    properties: dict[str, JsonSchema] = {}
    required: list[str] = []
    for name, param in inspect.signature(func).parameters.items():
        if name == "self":
            continue
        properties[name] = _type_to_schema(hints.get(name, str), arg_desc.get(name, ""))
        if param.default is inspect.Parameter.empty:
            required.append(name)

    input_schema: JsonSchema = {"type": "object", "properties": properties,
                                "additionalProperties": False}
    if required:
        input_schema["required"] = required
    return {"name": func.__name__, "description": summary, "input_schema": input_schema}


def build_tool_schemas() -> list[ToolSchema]:
    """Build the Anthropic tool schema list for every function in ``AGENT_TOOLS``."""
    return [build_tool_schema(func) for func in AGENT_TOOLS]


if __name__ == "__main__":
    schemas = build_tool_schemas()
    SCHEMA_FILE.write_text(json.dumps(schemas, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(schemas, indent=2))
    print(f"\n{len(schemas)} tool schemas written to {SCHEMA_FILE.name}")
