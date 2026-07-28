"""Tests for the generated tool schemas — the contract between us and the model.

These validate with ``jsonschema``, the same library the MCP server runs against
``inputSchema`` before it dispatches (``mcp/server/lowlevel/server.py``). That
placement is the whole point: a schema mismatch is rejected *before* the tool
function is entered, so it cannot be caught, logged, or recovered from anywhere
in ``tools.py`` or ``repl._dispatch``. The only place to catch it is here.

The bug that prompted the file: every optional parameter (``X | None``) was
emitted as a non-null schema, so a payload carrying an explicit ``null`` — which
the ADJUSTER prompt hands the model and tells it to pass through verbatim —
failed with ``Input validation error: None is not of type 'string'``.

Run with:  .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import inspect
import json
import os
import types
import typing
from dataclasses import MISSING, fields, is_dataclass

import jsonschema
import pytest

import agent_schemas
import roles
import tools

SCHEMAS = {s["name"]: s for s in agent_schemas.build_tool_schemas()}
TOOL_NAMES = sorted(SCHEMAS)


def _is_optional(tp: object) -> bool:
    return (typing.get_origin(tp) in (typing.Union, types.UnionType)
            and type(None) in typing.get_args(tp))


def _minimal_args(fn) -> dict:
    """A payload with every *required* parameter filled with a schema-valid value."""
    props = SCHEMAS[fn.__name__]["input_schema"]["properties"]
    out = {}
    for name, param in inspect.signature(fn).parameters.items():
        if param.default is not inspect.Parameter.empty:
            continue
        kind = props[name].get("type")
        kind = kind[0] if isinstance(kind, list) else kind
        enum = props[name].get("enum")
        out[name] = (enum[0] if enum else
                     {"string": "x", "integer": 1, "number": 1.0,
                      "boolean": True, "object": {}}.get(kind, "x"))
    return out


# --- the regression that prompted these ------------------------------------

def _policies_from_prompt(adjuster_id: str = "jaime") -> dict:
    """The policies object as it literally appears in the built ADJUSTER prompt.

    Parsed out of the rendered text rather than rebuilt from the dataclass on
    purpose. Re-deriving it with ``asdict`` would compare the schema against
    itself — both sides come from ``DynamicPolicies``, so the assertion passes no
    matter what the prompt actually says. The prompt is the artifact the model
    obeys, so the prompt is what has to be checked.
    """
    prompt = roles.build_system_prompt(roles.Role.ADJUSTER, adjuster_id)
    # Anchored on the phrase that introduces the object, not on a full
    # sentence: the wording around it is prose and will be edited.
    marker = "auto-approval policies"
    assert marker in prompt, "prompt no longer introduces the policies object"
    start = prompt.index("{", prompt.index(marker))
    obj, _ = json.JSONDecoder().raw_decode(prompt[start:])
    return obj


def test_the_policies_object_the_prompt_embeds_is_a_valid_argument():
    """`build_system_prompt` embeds this object and says "pass this exact object".

    If it does not validate, we are instructing the model to make a call the MCP
    server will reject before it reaches us — which is exactly what happened.
    """
    policies = _policies_from_prompt()
    assert None in policies.values(), "fixture is meaningless if no field is null"

    for tool_name in ("process_incident", "escalate_incident"):
        jsonschema.validate(
            {"incident_id": "case-3001", "policies": policies},
            SCHEMAS[tool_name]["input_schema"],
        )


def test_prompt_and_schema_agree_on_the_policies_field_set():
    """`additionalProperties: false` makes an extra key a hard rejection, so the
    object in the prompt and the schema have to name exactly the same fields."""
    embedded = set(_policies_from_prompt())
    declared = set(SCHEMAS["process_incident"]["input_schema"]["properties"]["policies"]["properties"])
    assert embedded == declared


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_optional_parameters_accept_null(name):
    """`X | None` in the signature must mean null-accepting in the schema.

    A model that passes `reason=None` explicitly — rather than omitting it — is
    doing something the Python signature allows, so the schema has to allow it.
    """
    fn = next(f for f in tools.AGENT_TOOLS if f.__name__ == name)
    schema = SCHEMAS[name]["input_schema"]
    hints = typing.get_type_hints(fn)
    base = _minimal_args(fn)

    for pname, param in inspect.signature(fn).parameters.items():
        if param.default is not inspect.Parameter.empty and _is_optional(hints.get(pname)):
            jsonschema.validate({**base, pname: None}, schema)


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_optional_dataclass_fields_accept_null(name):
    """Same rule one level down — a nested object's optional fields."""
    fn = next(f for f in tools.AGENT_TOOLS if f.__name__ == name)
    schema = SCHEMAS[name]["input_schema"]
    props = schema["properties"]
    base = _minimal_args(fn)

    for pname, tp in typing.get_type_hints(fn).items():
        if pname == "return":
            continue
        dc = next((a for a in typing.get_args(tp) if is_dataclass(a)), None)
        if dc is None:
            continue
        for f in fields(dc):
            if not _is_optional(typing.get_type_hints(dc).get(f.name)):
                continue
            filled = {g.name: (None if _is_optional(typing.get_type_hints(dc).get(g.name))
                               else g.default)
                      for g in fields(dc) if g.default is not MISSING}
            jsonschema.validate({**base, pname: {**filled, f.name: None}}, schema)
            assert props[pname]["properties"][f.name].get("type") is not None


# --- general schema hygiene -------------------------------------------------

@pytest.mark.parametrize("name", TOOL_NAMES)
def test_every_default_bearing_parameter_is_optional_in_the_schema(name):
    """A parameter with a Python default must not be listed as `required`."""
    fn = next(f for f in tools.AGENT_TOOLS if f.__name__ == name)
    required = set(SCHEMAS[name]["input_schema"].get("required", []))
    for pname, param in inspect.signature(fn).parameters.items():
        if param.default is not inspect.Parameter.empty:
            assert pname not in required, f"{name}.{pname} has a default but is required"


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_required_parameters_match_the_signature(name):
    fn = next(f for f in tools.AGENT_TOOLS if f.__name__ == name)
    expected = {p for p, v in inspect.signature(fn).parameters.items()
                if v.default is inspect.Parameter.empty}
    assert set(SCHEMAS[name]["input_schema"].get("required", [])) == expected


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_schemas_are_themselves_valid_json_schema(name):
    jsonschema.Draft202012Validator.check_schema(SCHEMAS[name]["input_schema"])


@pytest.mark.parametrize("name", TOOL_NAMES)
def test_minimal_call_validates(name):
    """Required params alone — the smallest legal call — must pass."""
    fn = next(f for f in tools.AGENT_TOOLS if f.__name__ == name)
    jsonschema.validate(_minimal_args(fn), SCHEMAS[name]["input_schema"])


def test_checked_in_schema_file_matches_the_code():
    """`agent_tools_schema.json` is generated; a stale copy silently ships the
    old contract to the Managed Agents control plane."""
    import json
    on_disk = json.loads(agent_schemas.SCHEMA_FILE.read_text(encoding="utf-8"))
    assert on_disk == agent_schemas.build_tool_schemas(), (
        "agent_tools_schema.json is stale — run `python3 agent_schemas.py`"
    )


def test_every_role_only_exposes_tools_that_exist():
    for role in roles.Role:
        for tool_name in roles.ROLE_TOOLS[role]:
            assert tool_name in SCHEMAS, f"{role.value} allows unknown tool {tool_name!r}"


# --- credential loading (cma.py) --------------------------------------------

def test_cma_rejects_a_placeholder_api_key(monkeypatch):
    """An API key outranks an `ant auth login` profile in the SDK's resolution
    order, so a placeholder does not fall through to a working profile — it goes
    out on the wire and comes back as a 401. Fail up front instead."""
    import cma
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ENTER_HERE")
    with pytest.raises(SystemExit, match="does not look like an API key"):
        cma._load_api_key()


def test_cma_accepts_a_real_looking_key(monkeypatch):
    import cma
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 60)
    assert "ANTHROPIC_API_KEY" in cma._load_api_key()


def test_cma_does_not_let_dotenv_override_an_exported_key(monkeypatch):
    """`load_dotenv` defaults to `override=False`, so a key already in the shell
    beats `.env`. Asserted because flipping that would silently swap which
    account a run bills to."""
    import cma
    exported = "sk-ant-api03-EXPORTED" + "y" * 40
    monkeypatch.setenv("ANTHROPIC_API_KEY", exported)
    cma._load_api_key()
    assert os.environ["ANTHROPIC_API_KEY"] == exported


def test_cma_falls_through_to_ambient_credentials(monkeypatch):
    """No key must not be fatal — `anthropic.Anthropic()` also resolves an
    `ant auth login` profile, and exiting would break that path.

    `load_dotenv()` searches upward from *cma.py's own directory*, not the cwd,
    so the project `.env` is found however the script is launched — which is the
    behaviour we want, and why `chdir` cannot be used to simulate its absence
    here. Stubbing the loader isolates the branch under test.
    """
    import dotenv

    import cma
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ambient credentials" in cma._load_api_key()
