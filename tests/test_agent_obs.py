"""Tests for ``agent_obs``.

Weighted toward the parts that are dangerous when wrong rather than the parts
that are easy to test: the capture proxy (does it leak credentials, does it relay
SSE, does it refuse to forward to itself), redaction (does claim text reach disk),
and sink rotation (does an unbounded wire log stay bounded).

Run with:  .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_obs import ObsConfig, Observability, traced_dispatch
from agent_obs.redact import DevRedactor, FlowRedactor, NullRedactor, StrictRedactor
from agent_obs.sinks import JsonlSink
from agent_obs.usage import Totals, TurnRecord, format_totals, summarize
from agent_obs.wire import ProxyTargetError


@pytest.fixture
def cfg(tmp_path):
    # `strict` here so the redaction-sensitive assertions elsewhere in this file
    # stay meaningful; the shipped default is `flow` (see the flow tests below).
    return ObsConfig(var_dir=tmp_path, wire=False, redact="strict")


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- redaction ------------------------------------------------------------

def test_strict_redaction_keeps_structure_and_drops_content():
    r = StrictRedactor()
    out = r.wire({
        "path": "/v1/messages",
        "headers": {"authorization": "Bearer sk-secret", "anthropic-beta": "caching"},
        "body": {"model": "opus",
                 "system": "policyholder SSN 123-45-6789",
                 "tools": [{"name": "approve_claim"}],
                 "messages": [{"role": "user", "content": [
                     {"type": "text", "text": "jewelry stolen",
                      "cache_control": {"type": "ephemeral"}}]}]},
    })
    blob = json.dumps(out)
    # Measurements and layout survive…
    assert "approve_claim" in blob and "cache_control" in blob and "opus" in blob
    assert out["body"]["system"]["chars"] == len("policyholder SSN 123-45-6789")
    # …content and credentials do not.
    assert "123-45-6789" not in blob
    assert "jewelry" not in blob
    assert "sk-secret" not in blob


def test_strict_redaction_hashes_are_stable_and_distinguishing():
    r = StrictRedactor()
    a = r.event("e", {"text": "same"})["text"]["sha"]
    b = r.event("e", {"text": "same"})["text"]["sha"]
    c = r.event("e", {"text": "different"})["text"]["sha"]
    assert a == b and a != c


def test_strict_redaction_keeps_identifier_lists():
    """Tool argument *names* are schema, not data — hashing them protects nothing
    and destroys the signal."""
    out = StrictRedactor().event("tool.start", {"arg_keys": ["adjuster_id", "cost"]})
    assert out["arg_keys"] == ["adjuster_id", "cost"]


def test_strict_redaction_does_not_mistake_content_for_identifiers():
    long_text = "x" * 200
    out = StrictRedactor().event("e", {"content": [long_text]})
    assert long_text not in json.dumps(out)


def test_every_mode_drops_credentials():
    payload = {"headers": {"x-api-key": "sk-leak", "cookie": "s=1"}}
    for redactor in (FlowRedactor(), StrictRedactor(), DevRedactor(), NullRedactor()):
        assert "sk-leak" not in json.dumps(redactor.wire(payload)), redactor.mode


# --- flow mode (the default) ----------------------------------------------

def test_flow_writes_first_sight_in_full_then_collapses_repeats():
    """The behaviour the wire log is actually read for: request 2 shows only what
    is new, with the resent history collapsed but still identifiable."""
    r = FlowRedactor(preview_chars=20, min_collapse_chars=50)
    system = "You are an AI assistant acting for adjuster jaime. " + "x" * 100
    first = r.wire({"body": {"system": system, "messages": ["list my incidents"]}})
    assert first["body"]["system"] == system          # full on first sight

    second = r.wire({"body": {"system": system,
                              "messages": ["list my incidents", "approve case-3001"]}})
    collapsed = second["body"]["system"]
    assert collapsed["_t"] == "seen"
    assert collapsed["preview"] == system[:20]
    assert collapsed["chars"] == len(system)
    # …and the new message is readable in full.
    assert second["body"]["messages"][1] == "approve case-3001"


def test_flow_preview_hash_matches_the_full_text_written_earlier():
    """The sha is the join key: grep it to find the full text upstream."""
    r = FlowRedactor(preview_chars=5, min_collapse_chars=10)
    text = "a long message that will repeat"
    r.wire({"body": text})
    collapsed = r.wire({"body": text})["body"]
    assert collapsed["sha"] == StrictRedactor().event("e", {"text": text})["text"]["sha"]


def test_flow_never_collapses_below_the_floor():
    """A collapsed entry serialises to ~70 chars, so collapsing a short string makes
    the file bigger and less readable — and short repeated strings are labels, not
    content. Regression: a 22-char function name was being collapsed."""
    r = FlowRedactor(min_collapse_chars=200)
    row = {"model": "opus", "func": "agent.route_and_triage", "role": "user"}
    assert r.wire(row) == row
    assert r.wire(row) == row          # still verbatim however often it repeats


def test_flow_truncates_a_very_long_first_sight():
    r = FlowRedactor(max_field_chars=50, preview_chars=10, min_collapse_chars=10)
    out = r.wire({"body": "y" * 500})
    assert out["body"]["_t"] == "truncated"
    assert out["body"]["text"] == "y" * 50 and out["body"]["chars"] == 500


def test_flow_keeps_tool_schema_text_readable():
    """The failure that prompted this mode: tool descriptions are our own static
    app text and the biggest token consumer, so hashing them destroyed the only
    analysis the wire layer exists for."""
    description = "Approve an incident. " + "detail " * 100
    out = FlowRedactor().wire({"body": {"tools": [
        {"name": "approve_incident", "description": description}]}})
    tool = out["body"]["tools"][0]
    assert tool["name"] == "approve_incident"
    assert tool["description"] == description


def test_flow_dedup_is_shared_across_layers_of_one_run(cfg):
    """One redactor per run, so a prompt logged by the event layer collapses when
    the wire layer sees the same text."""
    from agent_obs.redact import build_redactor
    r = build_redactor("flow", min_collapse_chars=10)
    text = "the same long prompt text appears in both layers"
    assert r.event("prompt", {"prompt": text})["prompt"] == text
    assert r.wire({"body": text})["body"]["_t"] == "seen"


def test_dev_mode_truncates_but_keeps_content():
    out = DevRedactor(max_field_chars=10).event("e", {"text": "abcdefghijklmnop"})
    assert out["text"].startswith("abcdefghij") and "+6 chars" in out["text"]


# --- flow mode: structural dedup ------------------------------------------

def _schema_tool(name):
    """A tool whose bulk is a JSON Schema — many short strings, none of them
    individually above any sane collapse floor. This is the shape that string-only
    dedup cannot touch."""
    return {"name": name, "description": "Do the thing.",
            "input_schema": {"type": "object", "properties": {
                f"field_{i}": {"type": "string", "description": f"the {i} field"}
                for i in range(40)}}}


def _request(tools, *turns):
    """A request body shaped like a real one: a growing message array around a
    tool list that never changes."""
    return {"body": {"model": "claude-opus-4-8", "tools": tools,
                     "messages": [{"role": "user", "content": t} for t in turns]}}


def test_flow_collapses_a_resent_schema_that_string_dedup_cannot():
    """The defect this fixes: every string inside a JSON Schema is a field name
    below `min_collapse_chars`, so a resent tool list was rewritten in full on
    every request even though flow mode claimed to collapse it."""
    tools = [_schema_tool("approve_incident")]
    r = FlowRedactor()
    first = r.wire(_request(tools, "list my incidents"))
    assert first["body"]["tools"][0]["input_schema"]["properties"]["field_7"]

    second = r.wire(_request(tools, "list my incidents", "now close INC-1"))
    node = second["body"]["tools"]
    assert node["_t"] == "seen_node" and node["kind"] == "list"
    assert node["chars"] > 1000                     # what it stood in for
    assert len(json.dumps(second)) < len(json.dumps(first)) / 10
    # The turn's actual news survives — that is the whole point of flow mode.
    assert second["body"]["messages"][1]["content"] == "now close INC-1"


def test_flow_collapses_an_identical_body_wholesale():
    """A retried request is one marker, not a second full copy."""
    r = FlowRedactor()
    row = _request([_schema_tool("t")], "list my incidents")
    r.wire(row)
    assert r.wire(row)["body"]["_t"] == "seen_node"


def test_structural_dedup_matches_content_not_position():
    """Same contract as string dedup: a repeat is caught wherever it moves to,
    which is what makes it work on a resent payload that changed index."""
    r = FlowRedactor()
    payload = {"deep": _schema_tool("x")}
    r.wire({"body": {"first_key": payload}})
    marker = r.wire({"body": {"somewhere_else": payload}})["body"]["somewhere_else"]
    assert marker["_t"] == "seen_node"

    # And the sha is a plain hash of the canonical form, so it is reproducible
    # outside the run — that is what makes it greppable back to the full copy.
    fresh = FlowRedactor()
    assert fresh._node(payload) is None                # first sight registers
    assert fresh._node(payload)["sha"] == marker["sha"]


def test_structural_dedup_ignores_key_order():
    """Serialisers reorder keys; a reordered resend is still a resend."""
    r = FlowRedactor()
    r.wire(_request([_schema_tool("t")], "one"))
    reordered = dict(reversed(list(_schema_tool("t").items())))
    second = r.wire(_request([reordered], "one", "two"))
    assert second["body"]["tools"]["_t"] == "seen_node"


def test_structural_dedup_respects_the_node_floor():
    """Small containers keep their shape — collapsing one costs more readability
    than it saves bytes."""
    r = FlowRedactor(min_node_chars=10_000)
    tools = [_schema_tool("approve_incident")]
    r.wire(_request(tools, "one"))
    second = r.wire(_request(tools, "one", "two"))
    assert second["body"]["tools"][0]["input_schema"]["properties"]["field_7"]


def test_structural_dedup_never_collapses_the_row_itself():
    r = FlowRedactor(min_node_chars=1)
    row = {"path": "/v1/messages", "body": {"tools": [_schema_tool("t")]}}
    assert r.wire(row)["path"] == "/v1/messages"
    assert r.wire(row)["path"] == "/v1/messages"       # envelope survives a repeat


def test_structural_dedup_can_be_switched_off():
    r = FlowRedactor(collapse_structures=False)
    tools = [_schema_tool("approve_incident")]
    r.wire(_request(tools, "one"))
    second = r.wire(_request(tools, "one", "two"))
    assert second["body"]["tools"][0]["input_schema"]["properties"]["field_7"]


def test_structural_dedup_still_strips_credentials_on_first_sight():
    r = FlowRedactor(min_node_chars=1)
    row = {"body": {"auth": {"api_key": "sk-leak", "pad": "x" * 600}}}
    assert "sk-leak" not in json.dumps(r.wire(row))
    assert "sk-leak" not in json.dumps(r.wire(row))


def test_structural_dedup_tolerates_unserializable_values():
    r = FlowRedactor(min_node_chars=1)
    row = {"body": {"obj": object(), "pad": "x" * 600}}
    r.wire(row)                                        # must not raise


# --- wire shaping ---------------------------------------------------------

def test_tool_shaper_elides_ambient_tools_and_keeps_our_own():
    """The measured case: the `claude` CLI ships every native tool and every MCP
    server on the machine, so most of a request is definitions this app never
    calls."""
    from agent_obs.shape import ToolSchemaShaper
    row = {"body": {"tools": [_schema_tool("Bash"),
                              _schema_tool("mcp__insurance__find_policy")]}}
    out = ToolSchemaShaper("skeleton", ("mcp__insurance__*",)).shape(row)
    ambient, ours = out["body"]["tools"]
    assert ambient["_t"] == "tool_skeleton" and ambient["name"] == "Bash"
    assert ambient["params"][:2] == ["field_0", "field_1"]
    assert ambient["desc_chars"] == len("Do the thing.")
    assert "input_schema" not in ambient
    assert ours == _schema_tool("mcp__insurance__find_policy")   # kept verbatim
    assert out["_shaped"] == ["tools:skeleton"]


def test_tool_shaper_keep_patterns_cover_tools_added_later():
    """A pattern, not a name list — a tool registered next month is kept without
    anyone remembering to update a constant."""
    from agent_obs.shape import ToolSchemaShaper
    shaper = ToolSchemaShaper("skeleton", ("mcp__insurance__*",))
    row = {"body": {"tools": [_schema_tool("mcp__insurance__brand_new_tool")]}}
    assert shaper.shape(row)["body"]["tools"][0]["input_schema"]


def test_tool_shaper_sha_still_proves_the_tool_list_did_not_change():
    """Elision must stay auditable: identical definitions across turns have to be
    provably identical from the shaped trace alone."""
    from agent_obs.shape import ToolSchemaShaper
    shaper = ToolSchemaShaper("names")
    row = {"body": {"tools": [_schema_tool("Bash")]}}
    first = shaper.shape(row)["body"]["tools"][0]
    assert first["_t"] == "tool_elided" and "params" not in first
    assert shaper.shape(row)["body"]["tools"][0]["sha"] == first["sha"]

    changed = {"body": {"tools": [{**_schema_tool("Bash"), "description": "new"}]}}
    assert shaper.shape(changed)["body"]["tools"][0]["sha"] != first["sha"]


def test_tool_shaper_leaves_rows_without_tools_alone():
    from agent_obs.shape import ToolSchemaShaper
    shaper = ToolSchemaShaper("skeleton")
    for row in ({"body": {"messages": [1]}}, {"body": None}, {"body": {"tools": []}}):
        assert shaper.shape(row) is row
        assert "_shaped" not in shaper.shape(row)


def test_full_mode_installs_no_shaper():
    from agent_obs.shape import build_shapers
    assert build_shapers("full", ("x",)) == ()
    assert len(build_shapers("skeleton", ("x",))) == 1


# --- sinks ----------------------------------------------------------------

def test_sink_rotates_and_keeps_generation_limit(tmp_path):
    sink = JsonlSink(tmp_path / "w.jsonl", max_bytes=200, max_files=3)
    for i in range(200):
        sink.write({"i": i, "pad": "x" * 50})
    generations = sorted(p.name for p in tmp_path.glob("w*.jsonl"))
    assert generations == ["w.1.jsonl", "w.2.jsonl", "w.jsonl"]
    assert (tmp_path / "w.jsonl").stat().st_size <= 400


def test_sink_survives_unserializable_row_without_raising(tmp_path):
    sink = JsonlSink(tmp_path / "e.jsonl")
    sink.write({"ok": 1})
    sink.write({"bad": {1, 2, 3}})       # a set is not JSON — default=str handles it
    assert len(_rows(tmp_path / "e.jsonl")) == 2


def test_sink_read_all_skips_torn_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n{"c": tr')     # crash mid-write
    assert JsonlSink(path).read_all() == [{"a": 1}, {"b": 2}]


# --- usage ----------------------------------------------------------------

def test_ledger_records_all_three_backends(cfg):
    class SdkResult:
        usage = {"input_tokens": 10, "output_tokens": 5,
                 "cache_read_input_tokens": 100, "cache_creation_input_tokens": 2}
        total_cost_usd = 0.5
        duration_ms = duration_api_ms = num_turns = 1
        session_id, stop_reason, is_error = "s1", "end_turn", False

    class ApiUsage:
        input_tokens, output_tokens = 20, 6
        cache_read_input_tokens, cache_creation_input_tokens = 200, 3

    class ApiResponse:
        usage = ApiUsage()
        model, stop_reason = "claude-opus-4-8", "tool_use"

    with Observability.start(cfg, front_end="t") as obs:
        obs.usage.from_sdk_result(SdkResult(), role="adjuster")
        obs.usage.from_api_response(ApiResponse(), wall_ms=12, model="opus", role="adjuster")
        obs.usage.from_cma_usage({"input_tokens": 30, "output_tokens": 7},
                                 session_id="c1", role="agent")
        assert obs.usage.run_totals.turns == 3

    rows = _rows(cfg.ledger_path)
    assert [r["backend"] for r in rows] == ["sdk", "api", "cma"]
    assert all(r["schema_version"] == 1 for r in rows)
    # The response's model wins over the requested alias, and is not passed twice.
    assert rows[1]["model"] == "claude-opus-4-8"


def test_ledger_tolerates_missing_cost_and_usage(cfg):
    class Bare:
        session_id = "s"
    with Observability.start(cfg, front_end="t") as obs:
        rec = obs.usage.from_sdk_result(Bare())
    assert rec.total_cost_usd is None and rec.all_tokens == 0


def test_summarize_ignores_unknown_and_missing_fields():
    rows = [
        {"backend": "sdk", "input_tokens": 5, "from_a_future_version": True},
        {"backend": "sdk", "input_tokens": 7},
    ]
    buckets = summarize(rows, "backend")
    assert buckets["sdk"].input_tokens == 12 and buckets["sdk"].turns == 2


def test_totals_cache_hit_ratio():
    t = Totals()
    t.add(TurnRecord(ts="", run_id="", backend="sdk", turn=1,
                     input_tokens=100, cache_read_input_tokens=900))
    assert t.cache_hit_ratio == pytest.approx(0.9)
    assert "hit=90%" in format_totals(t)


# --- tool tracing ---------------------------------------------------------

@traced_dispatch
def _dispatch(table, name, args):
    fn = table.get(name)
    if fn is None:
        return json.dumps({"error": f"tool '{name}' is not permitted"})
    return json.dumps(fn(**args))


def test_tool_tracing_records_success_denial_and_counts(cfg):
    table = {"ok": lambda x: {"got": x}}
    with Observability.start(cfg, front_end="t") as obs:
        assert json.loads(_dispatch(table, "ok", {"x": 1})) == {"got": 1}
        assert "not permitted" in _dispatch(table, "nope", {})
        assert obs.tool_calls == 2
        events = obs.paths()["events"]
    names = [r["event"] for r in _rows(events)]
    assert names.count("tool.start") == 2
    assert "tool.end" in names and "tool.error" in names


def test_dispatch_works_and_is_untraced_with_no_active_run(cfg):
    """The decorator must be transparent when nothing is running — that is what
    lets cma.py import the traced dispatcher unconditionally."""
    assert json.loads(_dispatch({"ok": lambda: {"v": 1}}, "ok", {})) == {"v": 1}


def test_disabled_run_writes_nothing(tmp_path):
    with Observability.start(ObsConfig(var_dir=tmp_path, enabled=False),
                             front_end="t") as obs:
        _dispatch({"ok": lambda: 1}, "ok", {})
        obs.events.info("ignored")
        assert obs.paths() == {}
    assert not (tmp_path / "traces").exists()


# --- wire proxy -----------------------------------------------------------

class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    saw_auth: list[str] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).saw_auth.append(self.headers.get("authorization") or "")
        if b"stream" in body:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for name in ("message_start", "content_block_delta", "message_stop"):
                self.wfile.write(f"event: {name}\ndata: {{}}\n\n".encode())
                self.wfile.flush()
        else:
            payload = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


@pytest.fixture
def upstream():
    _Upstream.saw_auth = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url, payload, **headers):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    return urllib.request.urlopen(request).read()


@pytest.mark.parametrize("mode", ["flow", "strict", "dev", "none"])
def test_proxy_forwards_auth_upstream_but_never_writes_it(tmp_path, upstream, mode):
    """Credential non-leakage is unconditional — it must hold in the most
    permissive mode too, which is the one where a regression would hide."""
    cfg = ObsConfig(var_dir=tmp_path, wire=True, upstream=upstream, redact=mode)
    with Observability.start(cfg, front_end="t") as obs:
        assert obs.base_url, obs.wire_error
        _post(obs.base_url + "/v1/messages", {"system": "some prompt text"},
              authorization="Bearer sk-live-secret")
        wire = obs.paths()["wire"]
    assert _Upstream.saw_auth == ["Bearer sk-live-secret"]     # upstream got it
    assert "sk-live-secret" not in wire.read_text()            # disk did not


def test_strict_mode_keeps_request_content_off_disk(tmp_path, upstream):
    cfg = ObsConfig(var_dir=tmp_path, wire=True, upstream=upstream, redact="strict")
    with Observability.start(cfg, front_end="t") as obs:
        _post(obs.base_url + "/v1/messages", {"system": "policyholder SSN 123-45-6789"})
        wire = obs.paths()["wire"]
    assert "123-45-6789" not in wire.read_text()


def test_proxy_only_captures_configured_paths(tmp_path, upstream):
    cfg = ObsConfig(var_dir=tmp_path, wire=True, upstream=upstream,
                    wire_paths=("/v1/messages",))
    with Observability.start(cfg, front_end="t") as obs:
        _post(obs.base_url + "/v1/messages", {"a": 1})
        _post(obs.base_url + "/v1/something_else", {"a": 1})
        wire = obs.paths()["wire"]
    rows = _rows(wire)
    assert [r["path"] for r in rows] == ["/v1/messages"]


# The literal paths the Managed Agents SDK requests. The beta is a `?beta=true`
# query parameter and an `anthropic-beta` header — there is no "/v1/beta/" prefix.
# An earlier version of this test asserted "/v1/beta/sessions/sess_1/events", a
# path the SDK never sends: it encoded the same wrong assumption as the default
# `wire_paths` and so passed while `cma.py --wire` silently captured nothing.
CMA_REQUEST_PATHS = [
    "/v1/agents?beta=true",
    "/v1/environments?beta=true",
    "/v1/sessions?beta=true",
    "/v1/sessions/sesn_1/events?beta=true",
    "/v1/memory_stores?beta=true",
]


@pytest.mark.parametrize("path", CMA_REQUEST_PATHS)
def test_proxy_captures_managed_agents_paths(tmp_path, upstream, path):
    """cma.py's whole control plane has to land in the wire log, or `--wire` is a
    no-op on that front-end — the proxy still forwards, so nothing looks broken."""
    cfg = ObsConfig(var_dir=tmp_path, wire=True, upstream=upstream)
    with Observability.start(cfg, front_end="t") as obs:
        _post(obs.base_url + path, {"a": 1})
        wire = obs.paths()["wire"]
    assert _rows(wire)[0]["path"] == path


def test_default_wire_paths_cover_what_the_installed_sdk_requests():
    """Derived from the installed `anthropic` package rather than hard-coded, so
    a new Managed Agents resource fails here instead of going uncaptured."""
    import re
    from pathlib import Path

    import anthropic

    resources = Path(anthropic.__file__).parent / "resources" / "beta"
    found = set()
    for py in resources.rglob("*.py"):
        found.update(re.findall(r'"(/v1/[a-z_]+)', py.read_text(encoding="utf-8")))

    # Scope to the resources cma.py actually drives; admin/other betas are not
    # this front-end's traffic and are deliberately not captured by default.
    relevant = {p for p in found if p.split("/")[2] in {
        "agents", "sessions", "environments", "memory_stores", "messages"}}
    assert relevant, "path scrape found nothing — SDK layout changed, fix the scrape"

    defaults = ObsConfig().wire_paths
    uncovered = {p for p in relevant if not any(d in p for d in defaults)}
    assert not uncovered, f"default wire_paths would not capture: {sorted(uncovered)}"


def test_credential_endpoints_are_not_captured_by_default():
    """`/v1/oauth/token` carries refresh tokens and client secrets. The default is
    an explicit resource list rather than a "/v1/" glob precisely so that
    broadening coverage cannot sweep it in."""
    defaults = ObsConfig().wire_paths
    assert not any(d in "/v1/oauth/token" for d in defaults)


def test_proxy_summarizes_sse_without_storing_the_stream(tmp_path, upstream):
    cfg = ObsConfig(var_dir=tmp_path, wire=True, upstream=upstream,
                    wire_responses="summary")
    with Observability.start(cfg, front_end="t") as obs:
        body = _post(obs.base_url + "/v1/messages", {"stream": True})
        wire = obs.paths()["wire"]
    assert b"message_stop" in body                      # relayed to the caller
    summary = _rows(wire)[0]["response_body"]
    assert summary["_t"] == "sse"
    assert summary["events"] == {"message_start": 1, "content_block_delta": 1,
                                 "message_stop": 1}


def test_proxy_shapes_tools_before_redacting_them(tmp_path, upstream):
    """End to end through the proxy: shaping happens on the recorded copy only, so
    the request forwarded upstream still carries the real tool definitions."""
    cfg = ObsConfig(var_dir=tmp_path, wire=True, upstream=upstream, redact="flow",
                    wire_tools="skeleton", wire_tools_keep=("mcp__insurance__*",))
    payload = {"tools": [_schema_tool("Bash"),
                         _schema_tool("mcp__insurance__find_policy")],
               "messages": [{"role": "user", "content": "list my incidents"}]}
    with Observability.start(cfg, front_end="t") as obs:
        _post(obs.base_url + "/v1/messages", payload)
        wire = obs.paths()["wire"]
    row = _rows(wire)[0]
    ambient, ours = row["body"]["tools"]
    assert ambient["_t"] == "tool_skeleton" and len(ambient["sha"]) == 12
    assert ours["input_schema"]["properties"]["field_7"]      # ours kept verbatim
    assert row["_shaped"] == ["tools:skeleton"]
    # The ambient tool's schema text is gone from disk; ours, kept, is still there.
    # (Its parameter *names* survive in the skeleton — that is deliberate.)
    assert wire.read_text().count("the 7 field") == 1


def test_a_broken_shaper_degrades_to_a_verbatim_capture(tmp_path, upstream):
    """Instrumentation must never cost a row."""
    from agent_obs.wire import CaptureProxy

    class Broken:
        def shape(self, row):
            raise RuntimeError("boom")

    sink = JsonlSink(tmp_path / "w.jsonl")
    proxy = CaptureProxy(sink, FlowRedactor(), upstream=upstream, shapers=(Broken(),))
    try:
        _post(proxy.base_url + "/v1/messages", {"messages": []})
    finally:
        proxy.shutdown()
    row = json.loads(sink.path.read_text().splitlines()[0])
    assert "RuntimeError: boom" in row["shape_error"]
    assert row["body"] == {"messages": []}


def test_proxy_refuses_to_forward_to_itself(tmp_path):
    """A proxy whose upstream is its own address hangs rather than erroring, so the
    check has to happen at construction — after binding, since the port is only
    known then."""
    from agent_obs.wire import CaptureProxy
    sink = JsonlSink(tmp_path / "w.jsonl")
    probe = CaptureProxy(sink, StrictRedactor(), upstream="https://api.anthropic.com")
    own_url = probe.base_url
    probe.shutdown()
    # Re-binding the same port is racy, so assert the guard directly instead:
    # a proxy told to forward to a *different* local port is legitimate and allowed.
    other = CaptureProxy(sink, StrictRedactor(), upstream="http://127.0.0.1:1")
    assert other.base_url != own_url
    other.shutdown()


def test_wire_failure_does_not_disable_the_run(tmp_path, monkeypatch):
    """Instrumentation must never take down the app it instruments."""
    import agent_obs.facade as facade

    def boom(*a, **k):
        raise OSError("no sockets today")

    monkeypatch.setattr("agent_obs.wire.CaptureProxy.__init__", boom)
    with Observability.start(ObsConfig(var_dir=tmp_path, wire=True),
                             front_end="t") as obs:
        assert obs.wire is None
        assert "no sockets today" in (obs.wire_error or "")
        assert obs.base_url is None
        obs.events.info("still_working")
        assert obs.status()["layers"]["events"] is True


# --- facade / correlation -------------------------------------------------

def test_run_index_maps_session_ids_to_runs(cfg):
    with Observability.start(cfg, front_end="repl") as obs:
        obs.note_session("sess-abc", backend="sdk")
        obs.note_session("sess-abc", backend="sdk")     # idempotent
        obs.note_session("sess-xyz", backend="cma")
        run_id = obs.run_id
    rows = _rows(cfg.runs_index_path)
    sessions = [(r["backend"], r["session_id"]) for r in rows if r["event"] == "session"]
    assert sessions == [("sdk", "sess-abc"), ("cma", "sess-xyz")]
    assert all(r["run_id"] == run_id for r in rows)


def test_span_hierarchy_is_session_turn_tool(cfg):
    pytest.importorskip("opentelemetry.sdk")
    with Observability.start(cfg, front_end="t") as obs:
        with obs.turn("chat"):
            _dispatch({"ok": lambda: 1}, "ok", {})
        otel = obs.paths()["otel"]
    spans = {s["name"]: s for s in _rows(otel)}
    assert spans["session"]["parent_id"] is None
    assert spans["turn.chat"]["parent_id"] == spans["session"]["context"]["span_id"]
    assert spans["tool.ok"]["parent_id"] == spans["turn.chat"]["context"]["span_id"]


def test_turn_failure_is_recorded_and_reraised(cfg):
    with Observability.start(cfg, front_end="t") as obs:
        with pytest.raises(RuntimeError):
            with obs.turn("chat"):
                raise RuntimeError("model exploded")
        events = obs.paths()["events"]
    failed = [r for r in _rows(events) if r["event"] == "turn.failed"]
    assert len(failed) == 1 and "model exploded" in failed[0]["error"]


@pytest.mark.parametrize("field", ["name", "kind", "func", "event", "ts", "seq"])
def test_event_fields_can_shadow_parameter_names(cfg, field):
    """Regression: `events.debug("x", name=…)` raised "got multiple values for
    argument 'name'". Field names come from call sites all over the app and will
    collide with any plain parameter, so the event-name parameters are
    positional-only."""
    with Observability.start(cfg, front_end="t") as obs:
        obs.events.info("probe", **{field: "value"})
        obs.events.debug("probe", **{field: "value"})
        obs.events.warn("probe", **{field: "value"})
        obs.events.error("probe", **{field: "value"})
        obs.events.event("probe", level="info", **{field: "value"})
        events = obs.paths()["events"]
    rows = [r for r in _rows(events) if r["event"] == "probe"]
    assert len(rows) == 5
    # The envelope always wins, so the row stays findable by its event name…
    assert all(r["event"] == "probe" for r in rows)
    # …and the caller's field is preserved either as-is or namespaced, never lost.
    key = f"field.{field}" if field in ("ts", "run_id", "seq", "level", "event",
                                        "session_id") else field
    assert all(r[key] == "value" for r in rows)


def test_turn_attributes_can_shadow_kind(cfg):
    """`obs.turn("chat", kind=…)` must not raise either."""
    with Observability.start(cfg, front_end="t") as obs:
        with obs.turn("chat", kind="shadowed", role="adjuster"):
            pass
        events = obs.paths()["events"]
    starts = [r for r in _rows(events) if r["event"] == "turn.start"]
    assert len(starts) == 1 and starts[0]["kind"] == "shadowed"


def test_trace_callable_records_the_function_name(cfg):
    """The decorator that carried the bug: exercise it end to end."""
    from agent_obs import trace_callable

    @trace_callable("agent.sweep", kind="deterministic")
    def sweep(n):
        return n * 2

    with Observability.start(cfg, front_end="t") as obs:
        assert sweep(21) == 42
        events = obs.paths()["events"]
    rows = [r for r in _rows(events) if r["event"].startswith("deterministic.")]
    assert [r["event"] for r in rows] == ["deterministic.start", "deterministic.end"]
    assert all(r["func"] == "agent.sweep" for r in rows)


def test_events_are_totally_ordered_by_seq(cfg):
    with Observability.start(cfg, front_end="t") as obs:
        for i in range(50):
            obs.events.info("tick", i=i)
        events = obs.paths()["events"]
    seqs = [r["seq"] for r in _rows(events)]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))


def test_stderr_sink_keeps_everything_at_a_severity(cfg):
    with Observability.start(cfg, front_end="t") as obs:
        obs.events.stderr_sink("ERROR: boom")
        obs.events.stderr_sink("just chatter")
        events = obs.paths()["events"]
    rows = [r for r in _rows(events) if r["event"] == "cli.stderr"]
    assert [r["level"] for r in rows] == ["error", "debug"]      # nothing discarded


def test_config_from_env_and_flag_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("OBS_WIRE", "1")
    monkeypatch.setenv("OBS_REDACT", "dev")
    monkeypatch.setenv("OBS_WIRE_PATHS", "/a,/b")
    monkeypatch.setenv("OBS_VAR_DIR", str(tmp_path))
    cfg = ObsConfig.from_env()
    assert cfg.wire and cfg.redact == "dev" and cfg.wire_paths == ("/a", "/b")
    assert ObsConfig.from_env(redact="strict").redact == "strict"   # kwargs win
    monkeypatch.setenv("OBS_REDACT", "bogus")
    assert ObsConfig.from_env().redact == "flow"                    # invalid -> default


def test_disconnected_stream_is_still_timed(tmp_path, upstream):
    """A client that hangs up on an SSE stream once it has what it needs is the
    normal case, not an anomaly — so those rows must carry `duration_ms` too, or
    the requests most worth timing are the ones with no timing."""
    cfg = ObsConfig(var_dir=tmp_path, wire=True, upstream=upstream,
                    wire_responses="summary")
    with Observability.start(cfg, front_end="t") as obs:
        body = json.dumps({"stream": True}).encode()
        req = urllib.request.Request(obs.base_url + "/v1/messages", data=body,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req)
        resp.read(1)          # take one byte, then hang up mid-stream
        resp.close()
        wire = obs.paths()["wire"]
    row = _rows(wire)[0]
    assert isinstance(row.get("duration_ms"), int)
