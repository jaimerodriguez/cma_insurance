"""Tests for the MCP facade.

Two things are worth pinning here.

**That it stays a facade.** The model must see byte-identical tool schemas
whether the tools arrive as Managed Agents custom tools or over MCP, and results
must match what `repl._dispatch` returns directly. If those drift, the facade has
started owning domain behaviour.

**That concurrency is actually serialized.** `storage.py` is whole-file
read-modify-write with no locking — safe today only because every other backend
dispatches serially. An HTTP server does not, and that is the one hazard the
facade has to solve with new code rather than delegation.

Run with:  .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import httpx
import pytest

import cma
import mcp_server
import repl
import roles
from roles import Role

TOKEN = "test-token-not-a-real-secret"


# --- config -----------------------------------------------------------------

def test_off_by_default_so_the_custom_tool_path_is_untouched(monkeypatch):
    for var in ("MCP_TOOLS", "MCP_PUBLIC_URL", "MCP_BEARER_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert mcp_server.McpConfig.from_env().active is False


def test_active_auto_derives_from_having_a_url_and_a_token():
    cfg = mcp_server.McpConfig(public_url="https://x.example.com", token="t")
    assert cfg.active is True
    assert cfg.with_overrides(token="").active is False
    assert cfg.with_overrides(public_url="").active is False


@pytest.mark.parametrize("raw,expected", [("1", True), ("on", True), ("0", False)])
def test_mcp_tools_env_var_overrides_the_derivation(monkeypatch, raw, expected):
    monkeypatch.setenv("MCP_TOOLS", raw)
    monkeypatch.delenv("MCP_PUBLIC_URL", raising=False)
    monkeypatch.delenv("MCP_BEARER_TOKEN", raising=False)
    assert mcp_server.McpConfig.from_env().active is expected


def test_describe_never_leaks_the_token():
    cfg = mcp_server.McpConfig(public_url="https://x.example.com", token="hunter2")
    assert "hunter2" not in json.dumps(cfg.describe())


def test_url_joining_survives_a_trailing_slash():
    cfg = mcp_server.McpConfig(public_url="https://x.example.com/")
    assert cfg.url_for(Role.AGENT) == "https://x.example.com/mcp/agent"


# --- it is a facade, not a reimplementation ---------------------------------

@pytest.mark.parametrize("role", list(Role))
def test_schemas_are_identical_to_the_custom_tool_declarations(role):
    """The model must see the same tool contract in either mode. `custom_tools_for_role`
    is the Anthropic-shaped declaration; strip its `type` discriminator and the
    rest must match ours exactly."""
    ours = list(mcp_server.tool_specs(role))
    theirs = [{k: v for k, v in t.items() if k != "type"}
              for t in cma.custom_tools_for_role(role)]
    assert sorted(ours, key=lambda s: s["name"]) == sorted(theirs, key=lambda s: s["name"])


@pytest.mark.parametrize("role", list(Role))
def test_mcp_tools_publish_our_schemas_verbatim(role):
    """`inputSchema` must be our generated schema object, not something re-derived
    from the Python signature — the reason FastMCP is not used here."""
    by_name = {t.name: t for t in mcp_server.mcp_tools(role)}
    for spec in mcp_server.tool_specs(role):
        assert by_name[spec["name"]].inputSchema == spec["input_schema"]
        assert by_name[spec["name"]].description == spec["description"]


@pytest.mark.parametrize("role,count", [(Role.ADJUSTER, 21), (Role.INSURER, 8),
                                        (Role.AGENT, 11)])
def test_each_endpoint_serves_exactly_its_role(role, count):
    names = {s["name"] for s in mcp_server.tool_specs(role)}
    assert names == roles.ROLE_TOOLS[role]
    assert len(names) == count


def test_endpoints_do_not_widen_to_the_delegation_union():
    """Per-role endpoints recover what `roles.DELEGATES_TO` gave up: an AGENT
    session no longer has to serve adjuster tools, because an MCP subagent calls
    its own endpoint."""
    agent = {s["name"] for s in mcp_server.tool_specs(Role.AGENT)}
    assert "update_resolution" not in agent          # adjuster-only
    assert agent < roles.allowed_tool_names(roles.session_roles(Role.AGENT))


# --- serialization ----------------------------------------------------------

def test_dispatch_is_serialized_end_to_end(monkeypatch):
    """The lock has to span the whole call, not just the write: two callers that
    each read before either writes lose one another's changes."""
    inflight, peak = 0, 0
    lock = threading.Lock()

    def slow_tool(**kw):
        nonlocal inflight, peak
        with lock:
            inflight += 1
            peak = max(peak, inflight)
        time.sleep(0.02)
        with lock:
            inflight -= 1
        return "ok"

    table = {"slow_tool": slow_tool}
    threads = [threading.Thread(target=mcp_server.dispatch_guarded,
                                args=(table, "slow_tool", {})) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert peak == 1, f"{peak} concurrent dispatches — storage.py cannot take that"


def test_concurrent_writes_do_not_lose_each_other(world):
    """The prize. Without the lock this loses notes and can raise JSONDecodeError
    on a half-written file."""
    import storage
    import tools as domain
    from conftest import AUTO_POLICY, MED_ADJUSTER, PLAIN_INSURER
    from data_entities import ReportType

    ids = [domain.create_incident(MED_ADJUSTER, ReportType.CAR_ACCIDENT, "x",
                                  PLAIN_INSURER, AUTO_POLICY, 100).id
           for _ in range(5)]
    table = roles.dispatch_table(Role.ADJUSTER)
    notes = [(i, f"note-{n}") for n in range(4) for i in ids]

    def work(item):
        incident_id, note = item
        mcp_server.dispatch_guarded(table, "append_incident_history",
                                    {"incident_id": incident_id, "note": note})

    threads = [threading.Thread(target=work, args=(item,)) for item in notes]
    for t in threads: t.start()
    for t in threads: t.join()

    saved = storage.load_incidents()          # must not raise on a torn file
    for incident_id in ids:
        history = saved[incident_id].history or ""
        missing = [n for n in range(4) if f"note-{n}" not in history]
        assert not missing, f"{incident_id} lost notes {missing}"


# --- live HTTP --------------------------------------------------------------

@pytest.fixture
def live_server():
    """A real uvicorn server on a free port, torn down after the test."""
    import uvicorn

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    cfg = mcp_server.McpConfig(host="127.0.0.1", port=port, token=TOKEN)
    server = uvicorn.Server(uvicorn.Config(mcp_server.build_app(cfg),
                                           host=cfg.host, port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("server did not come up")
    yield cfg
    server.should_exit = True


def _get(url: str, token: str | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_healthz_is_open_and_reports_the_tool_counts(live_server):
    status, body = _get(f"http://{live_server.host}:{live_server.port}/healthz")
    assert status == 200
    assert json.loads(body)["tools"] == {r.value: len(mcp_server.tool_specs(r))
                                         for r in Role}


@pytest.mark.parametrize("token", [None, "wrong-token"])
def test_endpoints_reject_a_missing_or_wrong_token(live_server, token):
    status, _ = _get(live_server.local_url_for(Role.AGENT), token)
    assert status == 401


@contextlib.asynccontextmanager
async def _client(cfg, role):
    """An authenticated MCP session against one role's endpoint.

    Auth rides on the httpx client rather than a `headers=` kwarg: that is how
    `streamable_http_client` takes it (the older `streamablehttp_client` spelling
    is deprecated).
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {TOKEN}"}) as http_client:
        async with streamable_http_client(cfg.local_url_for(role),
                                          http_client=http_client) as (r, w, _):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                yield sess


def _call(cfg, role, tool, args):
    """Run one MCP tool call against the live server and return the text payload."""
    async def go():
        async with _client(cfg, role) as sess:
            return (await sess.call_tool(tool, args)).content[0].text
    return asyncio.run(go())


def _listed(cfg, role):
    async def go():
        async with _client(cfg, role) as sess:
            return sorted(t.name for t in (await sess.list_tools()).tools)
    return asyncio.run(go())


@pytest.mark.parametrize("role", list(Role))
def test_tools_list_over_the_wire_matches_the_role(live_server, role):
    assert set(_listed(live_server, role)) == roles.ROLE_TOOLS[role]


def test_result_is_identical_to_calling_dispatch_directly(live_server, world):
    """The facade must not transform results — same bytes as the other backends."""
    args = {"adjuster_id": "jaime"}
    over_mcp = _call(live_server, Role.AGENT, "list_incidents", args)
    direct = repl._dispatch(roles.dispatch_table(Role.AGENT), "list_incidents", args)
    assert over_mcp == direct


def test_a_tool_the_role_lacks_is_refused_not_raised(live_server):
    """`update_resolution` is adjuster-only. The AGENT endpoint must refuse it in
    `_dispatch`'s error shape, not blow up."""
    text = _call(live_server, Role.AGENT, "update_resolution",
                 {"incident_id": "x", "adjuster": "y", "resolution": "approved"})
    assert json.loads(text) == {
        "error": "tool 'update_resolution' is not permitted for this role"}


def test_bad_argument_type_is_reported_in_the_dispatch_error_shape(live_server):
    text = _call(live_server, Role.AGENT, "list_incidents", {"adjuster_id": 12345})
    assert "Input validation error" in json.loads(text)["error"]


def test_the_policies_object_the_prompt_embeds_is_accepted(live_server, world):
    """The regression that bit us before: the ADJUSTER prompt hands the model a
    policies object containing a null and says to pass it verbatim."""
    import agent_memory
    from dataclasses import asdict

    policies = asdict(agent_memory.effective_policies("jaime"))
    assert None in policies.values()
    text = _call(live_server, Role.ADJUSTER, "process_incident",
                 {"incident_id": "no-such-case", "policies": policies})
    assert "Input validation error" not in text     # not found is fine; rejected is not


def test_the_bare_endpoint_url_does_not_redirect(live_server):
    """A `Mount` per role 307s `/mcp/agent` to `/mcp/agent/`, and an MCP client
    that does not follow redirects — the current SDK's does not — just fails. The
    URL we record on the agent has to work exactly as written."""
    url = live_server.local_url_for(Role.AGENT)
    req = urllib.request.Request(url, data=b"{}", method="POST")
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Content-Type", "application/json")
    try:
        status = urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        status = e.code
    assert status not in (301, 302, 307, 308), f"bare URL redirected ({status})"


def test_an_unknown_role_endpoint_is_a_clean_404(live_server):
    status, _ = _get(f"http://{live_server.host}:{live_server.port}"
                     f"{live_server.path_prefix}/nope", TOKEN)
    assert status == 404
