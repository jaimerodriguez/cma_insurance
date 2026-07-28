"""Tests for locating and updating the Managed Agents per-role agents.

Network-free: the Anthropic client is stubbed. What is worth pinning here is the
*decision* logic — which agent a role resolves to, and whether the live config
counts as drifted — because those are what make `update_agents` a safe repeat
operation rather than something that mints a version on every run.

The tool comparison in particular encodes a fact about the API discovered by
inspecting real agents: tools come back with server-side defaults filled in, so
a prebuilt toolset sent as `{"type": "agent_toolset_20260401"}` is echoed with
`default_config` and `configs` attached.

Run with:  .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import cma
import mcp_server
from roles import Role


_CONFIG = {"environment_id": "env_1",
           "agents": {r.value: f"agent_{r.value}" for r in Role},
           "memory_stores": {"jaime": "memstore_j", "sam": "memstore_s"}}


class _Tool(SimpleNamespace):
    """Stands in for a pydantic tool model — only `model_dump` is used."""

    def model_dump(self, exclude_none: bool = False) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if not exclude_none or v is not None}


def _live_agent(role: Role, *, agent_id: str = "agent_live", version: int = 1,
                system: str | None = None, tools: list[dict] | None = None,
                model: str | None = None, roster: list[str] | None = None):
    """An agent object shaped like the API's, in sync with the code by default.

    ``multiagent`` comes back as an object whose roster entries are resolved
    agent references, not the bare id strings that were sent — which is why
    ``_roster_ids`` compares ids rather than entry shapes.
    """
    desired = cma._desired_agent(role, _CONFIG)
    raw = tools if tools is not None else desired["tools"]
    ids = roster if roster is not None else cma._roster_ids(desired["multiagent"])
    return SimpleNamespace(
        id=agent_id, name=desired["name"], version=version,
        system=desired["system"] if system is None else system,
        tools=[_Tool(**t) for t in raw],
        model=SimpleNamespace(id=model or desired["model"], effort=None, speed="standard"),
        multiagent=(SimpleNamespace(
            type="coordinator",
            agents=[SimpleNamespace(type="agent", id=i, version=1) for i in ids])
            if ids else None),
        # Echoed back as models, not the dicts we sent — and empty in the default
        # custom-tool mode, which is what `_desired_agent` will also produce.
        mcp_servers=[SimpleNamespace(type="url", name=s["name"], url=s["url"])
                     for s in desired["mcp_servers"]],
    )


class _FakeAgents:
    def __init__(self, by_id: dict):
        self.by_id, self.updated = by_id, []

    def retrieve(self, agent_id, **kw):
        import anthropic
        if not agent_id.startswith("agent_"):
            raise anthropic.BadRequestError(
                "Invalid agent ID.", response=SimpleNamespace(status_code=400,
                                                              headers={}, request=None),
                body=None)
        if agent_id not in self.by_id:
            raise anthropic.NotFoundError(
                "not found", response=SimpleNamespace(status_code=404,
                                                      headers={}, request=None),
                body=None)
        return self.by_id[agent_id]

    def list(self, **kw):
        return list(self.by_id.values())

    def update(self, agent_id, **fields):
        self.updated.append((agent_id, fields))
        live = self.by_id[agent_id]
        return SimpleNamespace(id=agent_id, version=live.version + 1)


def _client(by_id: dict):
    return SimpleNamespace(beta=SimpleNamespace(agents=_FakeAgents(by_id)))


@pytest.fixture
def in_sync(monkeypatch, tmp_path):
    """A client whose three agents exactly match the code, plus a scratch config."""
    monkeypatch.setattr(cma, "CONFIG_FILE", tmp_path / "cma_config.json")
    agents = {f"agent_{r.value}": _live_agent(r, agent_id=f"agent_{r.value}")
              for r in Role}
    config = {"environment_id": "env_1",
              "agents": {r.value: f"agent_{r.value}" for r in Role},
              "memory_stores": {}}
    return _client(agents), config


# --- locating ---------------------------------------------------------------

def test_finds_each_role_by_recorded_id(in_sync):
    client, config = in_sync
    for role in Role:
        assert cma.find_agent(client, config, role).id == f"agent_{role.value}"


@pytest.mark.parametrize("bad_id,reason", [
    ("agent_01ABSENT", "well-formed but absent -> 404"),
    ("not-an-agent-id", "malformed -> 400, which is *not* NotFoundError"),
])
def test_falls_back_to_name_when_the_recorded_id_is_unusable(in_sync, bad_id, reason):
    """The config file is local and disposable; the agents are not. Without the
    fallback a bad id would create a duplicate `insurance-<role>` agent."""
    client, config = in_sync
    config = copy.deepcopy(config)
    config["agents"]["insurer"] = bad_id
    assert cma.find_agent(client, config, Role.INSURER).id == "agent_insurer", reason


def test_finds_by_name_with_no_config_at_all(in_sync):
    client, _ = in_sync
    assert cma.find_agent(client, {}, Role.AGENT).id == "agent_agent"


def test_returns_none_when_the_role_has_no_agent(in_sync):
    client, config = in_sync
    del client.beta.agents.by_id["agent_insurer"]
    config = copy.deepcopy(config)
    config["agents"].pop("insurer")
    assert cma.find_agent(client, config, Role.INSURER) is None


# --- drift ------------------------------------------------------------------

def test_no_drift_when_the_agent_matches_the_code(in_sync):
    client, config = in_sync
    for role in Role:
        assert cma.agent_drift(cma.find_agent(client, config, role), role, config) == []


def test_server_filled_tool_defaults_are_not_drift():
    """The API echoes a prebuilt toolset back with `default_config` and `configs`
    populated. Treating that as drift would update on every single run."""
    desired = cma._desired_agent(Role.ADJUSTER, _CONFIG)["tools"]
    echoed = [dict(t) for t in desired]
    for t in echoed:
        if t.get("type") == "agent_toolset_20260401":
            t.update(configs=[], default_config={"enabled": True,
                                                 "permission_policy": {"type": "always_allow"}})
            break
    else:
        pytest.fail("adjuster no longer carries a prebuilt toolset — update this test")
    assert cma._tools_match(desired, [_Tool(**t) for t in echoed])


@pytest.mark.parametrize("mutate,expected", [
    (lambda ts: ts[:-1], "tools"),
    (lambda ts: [{**ts[0], "description": "CHANGED"}, *ts[1:]], "tools"),
    (lambda ts: [*ts, {"type": "custom", "name": "zz", "description": "d",
                       "input_schema": {"type": "object", "properties": {}}}], "tools"),
])
def test_tool_changes_are_detected(mutate, expected):
    desired = cma._desired_agent(Role.INSURER, _CONFIG)["tools"]
    live = _live_agent(Role.INSURER, tools=mutate(copy.deepcopy(desired)))
    assert expected in cma.agent_drift(live, Role.INSURER, _CONFIG)


def test_system_and_model_changes_are_detected():
    assert "system" in cma.agent_drift(_live_agent(Role.AGENT, system="old"), Role.AGENT, _CONFIG)
    assert "model" in cma.agent_drift(
        _live_agent(Role.AGENT, model="claude-haiku-4-5"), Role.AGENT, _CONFIG)


# --- updating ---------------------------------------------------------------

def test_update_is_a_no_op_when_nothing_drifted(in_sync):
    client, config = in_sync
    rows = cma.update_agents(client, config)
    assert {r["action"] for r in rows} == {"unchanged"}
    assert client.beta.agents.updated == []


def test_update_pushes_the_code_config_when_drifted(in_sync):
    client, config = in_sync
    client.beta.agents.by_id["agent_agent"].system = "stale prompt"

    rows = {r["role"]: r for r in cma.update_agents(client, config)}
    assert rows["agent"]["action"] == "updated"
    assert rows["agent"]["changed"] == ["system"]
    assert rows["insurer"]["action"] == "unchanged"

    (agent_id, sent), = client.beta.agents.updated
    assert agent_id == "agent_agent"
    assert sent["system"] == cma._agent_system(Role.AGENT)
    assert sent["tools"] == cma.agent_tools_for_role(Role.AGENT)
    assert sent["model"] == cma.MODEL_BY_ROLE[Role.AGENT]
    assert "version" not in sent, (
        "sending `version` asks for optimistic concurrency and 409s on any "
        "mismatch — wrong for a declarative apply that owns these agents"
    )


def test_force_updates_even_without_drift(in_sync):
    client, config = in_sync
    rows = cma.update_agents(client, config, only=[Role.INSURER], force=True)
    assert rows[0]["action"] == "updated" and rows[0]["changed"] == ["(forced)"]
    assert len(client.beta.agents.updated) == 1


def test_only_restricts_the_roles_touched(in_sync):
    client, config = in_sync
    for a in client.beta.agents.by_id.values():
        a.system = "stale"
    rows = cma.update_agents(client, config, only=[Role.ADJUSTER])
    assert [r["role"] for r in rows] == ["adjuster"]
    assert [aid for aid, _ in client.beta.agents.updated] == ["agent_adjuster"]


def test_missing_agent_is_reported_not_created(in_sync):
    """`update_agents` updates; provisioning stays `setup`'s job."""
    client, config = in_sync
    del client.beta.agents.by_id["agent_insurer"]
    config = copy.deepcopy(config)
    config["agents"].pop("insurer")
    rows = {r["role"]: r for r in cma.update_agents(client, config)}
    assert rows["insurer"]["action"] == "missing"
    assert client.beta.agents.updated == []


def test_a_name_match_repairs_the_recorded_id(in_sync, tmp_path):
    client, config = in_sync
    config = copy.deepcopy(config)
    config["agents"]["adjuster"] = "agent_01STALE"
    cma.update_agents(client, config, only=[Role.ADJUSTER])
    assert config["agents"]["adjuster"] == "agent_adjuster"
    assert cma.CONFIG_FILE.exists(), "the repaired id should be persisted"


# --- prompts ----------------------------------------------------------------

@pytest.mark.parametrize("role", list(Role))
def test_every_role_system_prompt_builds(role):
    """`str.format` raises KeyError for a placeholder it was not given, so a
    renamed field breaks prompt construction — and every session with it."""
    assert cma._agent_system(role).strip()


# --- delegation (multiagent roster) -----------------------------------------

def test_only_the_agent_role_gets_a_coordinator_roster():
    """Delegation is driven by `roles.DELEGATES_TO`, shared with the Agent SDK
    front-end so the two backends cannot disagree on who may delegate."""
    assert cma._multiagent_config(Role.AGENT, _CONFIG) == {
        "type": "coordinator", "agents": ["agent_adjuster"]}
    for role in (Role.ADJUSTER, Role.INSURER):
        assert cma._multiagent_config(role, _CONFIG) is None


def test_no_roster_when_the_target_agent_does_not_exist_yet():
    """An empty roster is rejected by the API, so a missing target means no
    roster at all — `/setup` has to run before delegation can be wired up."""
    assert cma._multiagent_config(Role.AGENT, {"agents": {}}) is None


def test_roster_drift_is_detected():
    stale = _live_agent(Role.AGENT, roster=[])           # never wired up
    assert "multiagent" in cma.agent_drift(stale, Role.AGENT, _CONFIG)
    wrong = _live_agent(Role.AGENT, roster=["agent_someone_else"])
    assert "multiagent" in cma.agent_drift(wrong, Role.AGENT, _CONFIG)


def test_roster_entry_shape_is_not_drift():
    """The API resolves a bare id string into a versioned reference; comparing
    entry shapes rather than ids would report drift on every run."""
    assert cma._roster_ids({"agents": ["agent_adjuster"]}) == \
           cma._roster_ids(SimpleNamespace(
               agents=[SimpleNamespace(type="agent", id="agent_adjuster", version=3)]))


def test_update_sends_the_roster_for_the_agent_role(in_sync):
    client, config = in_sync
    client.beta.agents.by_id["agent_agent"].multiagent = None
    rows = {r["role"]: r for r in cma.update_agents(client, config)}
    assert "multiagent" in rows["agent"]["changed"]
    sent = dict(client.beta.agents.updated)["agent_agent"]
    assert sent["multiagent"] == {
        "type": "coordinator",
        "agents": [{"type": "agent", "id": "agent_adjuster", "version": 1}]}


def test_a_roster_pinned_to_a_superseded_version_is_drift(in_sync):
    """Roster entries pin a version at save time — they do not track "latest".
    Verified live: bump the adjuster and the coordinator keeps spawning the old
    one, silently. Comparing ids alone would call this in sync."""
    client, config = in_sync
    coord = client.beta.agents.by_id["agent_agent"]
    assert cma.agent_drift(coord, Role.AGENT, config, {"agent_adjuster": 1}) == []
    assert "multiagent" in cma.agent_drift(coord, Role.AGENT, config,
                                           {"agent_adjuster": 7})


def test_an_unpinned_roster_entry_matches_any_version():
    """A bare id carries no version, so it must not read as drift on its own."""
    assert cma._refs_match([("a", 3)], [("a", None)])
    assert cma._refs_match([("a", None)], [("a", 3)])
    assert not cma._refs_match([("a", 3)], [("a", 4)])
    assert not cma._refs_match([("a", 3)], [("b", 3)])


def test_delegation_targets_are_updated_before_their_coordinator(in_sync):
    """One pass must not pin the very version it is about to supersede: the
    adjuster has to be updated first so the coordinator pins the fresh one."""
    client, config = in_sync
    for a in client.beta.agents.by_id.values():
        a.system = "stale"
    cma.update_agents(client, config)

    order = [aid for aid, _ in client.beta.agents.updated]
    assert order.index("agent_adjuster") < order.index("agent_agent")
    sent = dict(client.beta.agents.updated)["agent_agent"]
    # v1 -> v2 for the adjuster, so the roster must pin v2, not v1.
    assert sent["multiagent"]["agents"] == [
        {"type": "agent", "id": "agent_adjuster", "version": 2}]


def test_update_omits_multiagent_for_roles_that_do_not_delegate(in_sync):
    """Omitted fields are preserved; sending an explicit None would *clear* a
    roster rather than leave it alone."""
    client, config = in_sync
    client.beta.agents.by_id["agent_insurer"].system = "stale"
    cma.update_agents(client, config, only=[Role.INSURER])
    sent = dict(client.beta.agents.updated)["agent_insurer"]
    assert "multiagent" not in sent


# --- session wiring for delegation ------------------------------------------

def _session(role: Role, identity: str, config: dict):
    """A CmaSession with session creation stubbed out (no network)."""
    s = object.__new__(cma.CmaSession)
    s.client, s.config, s.role, s.identity = None, config, role, identity
    s.table = cma.roles.dispatch_table(cma.roles.session_roles(role))
    return s


def test_coordinator_can_dispatch_its_subagents_tools():
    """A subagent's custom-tool calls are cross-posted to the primary thread and
    dispatched against this one table. AGENT-only would refuse every adjuster
    tool the adjuster subagent calls."""
    table = _session(Role.AGENT, "system", _CONFIG).table
    assert "update_resolution" in table, "adjuster tool missing from a delegating session"
    assert "close_stale_resolved" in table, "own tool lost"
    # A non-delegating role stays narrow.
    assert "close_stale_resolved" not in _session(Role.INSURER, "ins-1", _CONFIG).table


def test_tool_result_echoes_the_subagent_thread_id():
    """Without the echo the result is not routed to the thread waiting on it."""
    s = _session(Role.AGENT, "system", _CONFIG)
    call = SimpleNamespace(id="sevt_1", name="list_incidents", input={},
                           session_thread_id="sthr_9")
    assert s._tool_result(call)["session_thread_id"] == "sthr_9"


def test_tool_result_omits_the_thread_id_for_the_coordinators_own_calls():
    s = _session(Role.AGENT, "system", _CONFIG)
    call = SimpleNamespace(id="sevt_1", name="list_incidents", input={},
                           session_thread_id=None)
    assert "session_thread_id" not in s._tool_result(call)


def test_a_delegating_session_mounts_every_adjuster_memory_store():
    """Threads share the container and stores attach only at session-create, so
    the coordinator has to mount them or a delegated adjuster finds nothing at
    /mnt/memory and silently works off baseline policy."""
    mounted = _session(Role.AGENT, "system", _CONFIG)._memory_resources()
    assert {m["memory_store_id"] for m in mounted} == {"memstore_j", "memstore_s"}


def test_an_adjuster_session_mounts_only_its_own_store():
    mounted = _session(Role.ADJUSTER, "jaime", _CONFIG)._memory_resources()
    assert [m["memory_store_id"] for m in mounted] == ["memstore_j"]


def test_an_insurer_session_mounts_nothing():
    assert _session(Role.INSURER, "ins-1001", _CONFIG)._memory_resources() == []


# --- the turn loop: not waiting out the socket timeout ----------------------
#
# Measured before the fix: 9 SSE stream requests in one run at 603-759 s each
# (~100 min total) while the 39 tool calls inside it summed to 44 ms. On
# `session.status_idle` with `requires_action` the loop kept reading a stream the
# session had nothing more to send on, until httpx's 600 s default read timeout
# (anthropic/_constants.py) killed the socket. These tests pin the exit condition.

class _Ev(SimpleNamespace):
    """A session event. `id` doubles as the blocking event id, as the API does."""


def _idle(*blocked_on: str, stop: str = "requires_action"):
    return _Ev(type="session.status_idle",
               stop_reason=SimpleNamespace(type=stop, event_ids=list(blocked_on)))


def _call(event_id: str, name: str = "list_incidents"):
    return _Ev(type="agent.custom_tool_use", id=event_id, name=name, input={},
               session_thread_id=None)


class _Stream:
    """Yields scripted events and records how many were consumed.

    A real stream would block here rather than end, which is the whole bug: the
    test asserts on `consumed`, not on reaching the end of the script.
    """

    def __init__(self, events): self.events, self.consumed = events, 0

    def __enter__(self): return self

    def __exit__(self, *exc): return False

    def __iter__(self):
        for e in self.events:
            self.consumed += 1
            yield e


class _EventsAPI:
    def __init__(self, legs): self.legs, self.streams, self.sent = list(legs), [], []

    def stream(self, session_id, timeout=None, **kw):
        self.timeout = timeout
        s = _Stream(self.legs.pop(0) if self.legs else [])
        self.streams.append(s)
        return s

    def send(self, session_id, events, **kw): self.sent.append(events)


def _run_send(role, config, legs, monkeypatch):
    """Drive CmaSession.send over scripted stream legs, with tools stubbed out."""
    s = _session(role, "system", config)
    api = _EventsAPI(legs)
    s.client = SimpleNamespace(beta=SimpleNamespace(sessions=SimpleNamespace(events=api)))
    s.session_id = "sesn_test"
    monkeypatch.setattr(cma, "_dispatch", lambda table, name, args: '{"ok": true}')
    text = s.send("go")
    return s, api, text


def test_stops_reading_once_it_holds_every_blocking_event(in_sync, monkeypatch):
    """The exit condition is set-coverage of `stop_reason.event_ids`."""
    _, api, _ = _run_send(Role.AGENT, in_sync[1], [
        [_call("sevt_1"), _call("sevt_2"), _idle("sevt_1", "sevt_2"),
         _Ev(type="never_reached")],
        [_Ev(type="session.status_idle",
             stop_reason=SimpleNamespace(type="end_turn", event_ids=[]))],
    ], monkeypatch)
    assert api.streams[0].consumed == 3, "kept reading past the idle it could answer"
    assert len(api.sent) == 2                      # user.message, then the results
    assert {e["custom_tool_use_id"] for e in api.sent[1]} == {"sevt_1", "sevt_2"}


def test_idle_arriving_before_the_calls_it_names(in_sync, monkeypatch):
    """The naive fix gets this wrong: coverage has to be re-checked as calls
    arrive, not only when the idle does."""
    _, api, _ = _run_send(Role.AGENT, in_sync[1], [
        [_idle("sevt_1", "sevt_2"), _call("sevt_1"), _call("sevt_2"),
         _Ev(type="never_reached")],
        [_Ev(type="session.status_idle",
             stop_reason=SimpleNamespace(type="end_turn", event_ids=[]))],
    ], monkeypatch)
    assert api.streams[0].consumed == 3


def test_keeps_reading_when_blocked_on_something_it_has_not_seen(in_sync, monkeypatch):
    """Answering a partial set would strand the session, so an uncovered id means
    keep reading."""
    _, api, _ = _run_send(Role.AGENT, in_sync[1], [
        [_call("sevt_1"), _idle("sevt_1", "sevt_OTHER"), _call("sevt_2")],
        [_Ev(type="session.status_idle",
             stop_reason=SimpleNamespace(type="end_turn", event_ids=[]))],
    ], monkeypatch)
    assert api.streams[0].consumed == 3, "should have drained rather than answered early"


def test_end_turn_idle_finishes_the_turn(in_sync, monkeypatch):
    _, api, text = _run_send(Role.AGENT, in_sync[1], [
        [_Ev(type="agent.message", content=[SimpleNamespace(type="text", text="done")]),
         _Ev(type="session.status_idle",
             stop_reason=SimpleNamespace(type="end_turn", event_ids=[])),
         _Ev(type="never_reached")],
    ], monkeypatch)
    assert text == "done"
    assert api.streams[0].consumed == 2
    assert len(api.sent) == 1                      # only the user message


def test_requires_action_without_event_ids_falls_back_to_batch(in_sync, monkeypatch):
    """An API shape without `event_ids` must degrade to the old behaviour rather
    than hang."""
    _, api, _ = _run_send(Role.AGENT, in_sync[1], [
        [_call("sevt_1"), _idle()],
        [_Ev(type="session.status_idle",
             stop_reason=SimpleNamespace(type="end_turn", event_ids=[]))],
    ], monkeypatch)
    assert {e["custom_tool_use_id"] for e in api.sent[1]} == {"sevt_1"}


def test_terminated_session_ends_the_turn(in_sync, monkeypatch):
    _, api, _ = _run_send(Role.AGENT, in_sync[1], [
        [_call("sevt_1"), _Ev(type="session.status_terminated"),
         _Ev(type="never_reached")],
    ], monkeypatch)
    assert api.streams[0].consumed == 2
    assert len(api.sent) == 1                      # no results posted after termination


def test_stream_is_opened_with_an_explicit_timeout(in_sync, monkeypatch):
    """Backstop against any *other* silent stream: the SDK default is 600 s."""
    _, api, _ = _run_send(Role.AGENT, in_sync[1], [
        [_Ev(type="session.status_idle",
             stop_reason=SimpleNamespace(type="end_turn", event_ids=[]))],
    ], monkeypatch)
    assert api.timeout == cma.STREAM_TIMEOUT_S
    assert 0 < cma.STREAM_TIMEOUT_S < 600, "must be below the SDK default to help"


# --- MCP transport wiring ---------------------------------------------------
#
# The switch is `cma.MCP.active`. Off, everything below must behave exactly as it
# did — that is what keeps the existing app working.

ON = mcp_server.McpConfig(public_url="https://x.example.com", token="tok")


def test_custom_tools_are_the_default(in_sync):
    assert cma.MCP.active is False
    tools_ = cma.agent_tools_for_role(Role.INSURER)
    assert {t["type"] for t in tools_} == {"custom"}
    assert cma.mcp_servers_for_role(Role.INSURER) == []


def test_mcp_mode_swaps_the_transport_not_the_contract(monkeypatch):
    """One toolset entry replaces 8 inline declarations, but the tools the model
    can reach are the same 8 — they now live behind the endpoint."""
    monkeypatch.setattr(cma, "MCP", ON)
    tools_ = cma.agent_tools_for_role(Role.INSURER)
    assert tools_ == [{"type": "mcp_toolset", "mcp_server_name": "insurance-insurer"}]
    assert cma.mcp_servers_for_role(Role.INSURER) == [
        {"type": "url", "name": "insurance-insurer",
         "url": "https://x.example.com/mcp/insurer"}]


def test_the_adjuster_keeps_its_prebuilt_toolset_in_mcp_mode(monkeypatch):
    """The memory store is a sandbox mount, not one of our tools, so the
    read/write/glob toolset is unaffected by the transport."""
    monkeypatch.setattr(cma, "MCP", ON)
    types = [t["type"] for t in cma.agent_tools_for_role(Role.ADJUSTER)]
    assert types == ["agent_toolset_20260401", "mcp_toolset"]


def test_a_moved_url_is_drift(monkeypatch, in_sync):
    """The normal reason this changes: the server is re-homed."""
    client, config = in_sync
    monkeypatch.setattr(cma, "MCP", ON)
    live = _live_agent(Role.AGENT)                       # built with ON
    assert "mcp_servers" not in cma.agent_drift(live, Role.AGENT, config)
    monkeypatch.setattr(cma, "MCP", ON.with_overrides(public_url="https://moved.example.com"))
    assert "mcp_servers" in cma.agent_drift(live, Role.AGENT, config)


def test_switching_back_to_custom_tools_clears_the_server(monkeypatch, in_sync):
    """`mcp_servers` is always sent, including as `[]`. Omitting it would preserve
    the stale declaration and leave agents pointing at a server we stopped using."""
    client, config = in_sync
    monkeypatch.setattr(cma, "MCP", ON)
    for a in client.beta.agents.by_id.values():
        a.mcp_servers = [SimpleNamespace(type="url", name="insurance-agent",
                                         url="https://x.example.com/mcp/agent")]
    monkeypatch.setattr(cma, "MCP", mcp_server.McpConfig())      # off again
    cma.update_agents(client, config, only=[Role.AGENT])
    sent = dict(client.beta.agents.updated)["agent_agent"]
    assert sent["mcp_servers"] == []
    assert {t["type"] for t in sent["tools"]} == {"custom"}


def test_server_urls_reads_both_dict_and_model_shapes():
    """The API echoes servers back as models, not the dicts we sent."""
    as_dict = [{"type": "url", "name": "n", "url": "https://u/mcp/agent"}]
    as_model = [SimpleNamespace(type="url", name="n", url="https://u/mcp/agent")]
    assert cma._server_urls(as_dict) == cma._server_urls(as_model) == [("n", "https://u/mcp/agent")]


@pytest.mark.parametrize("adjuster_id", sorted(cma._SEED_MEMORY))
def test_seed_memory_is_text_not_a_tuple(adjuster_id):
    """Adjacent string literals concatenate; a stray comma makes a tuple instead.
    That silently produced a 4-tuple, which `memories.create(content=...)` rejects
    and which crashed `gen_cma_yaml.py`."""
    assert isinstance(cma._SEED_MEMORY[adjuster_id], str)


# --- /agent taking instructions ---------------------------------------------

@pytest.mark.parametrize("line,cmd,tokens,rest", [
    ("/agent", "/agent", [], ""),
    ("/agent   ", "/agent", [], ""),
    ("/agent close the stale claims", "/agent", ["close", "the", "stale", "claims"],
     "close the stale claims"),
    ("/update-agents adjuster --force", "/update-agents", ["adjuster", "--force"],
     "adjuster --force"),
    ("/adjuster jaime", "/adjuster", ["jaime"], "jaime"),
])
def test_split_command_gives_both_views_of_the_arguments(line, cmd, tokens, rest):
    assert cma.split_command(line) == (cmd, tokens, rest)


def test_split_command_preserves_spacing_inside_an_instruction():
    """`" ".join(tokens)` would collapse it. Instructions are prose sent verbatim
    to a model, so the author's line breaks and spacing are theirs to keep."""
    _, _, rest = cma.split_command("/agent Triage claim-1.  Then  report back.")
    assert rest == "Triage claim-1.  Then  report back."


def test_run_agent_task_sends_exactly_the_instructions_given(monkeypatch):
    """The whole point of the change: no hard-coded prompt between the caller
    and the agent."""
    sent = {}

    class _FakeSession:
        def __init__(self, client, config, role, identity):
            sent["role"], sent["identity"] = role, identity

        def send(self, text):
            sent["text"] = text
            return "report"

    monkeypatch.setattr(cma, "CmaSession", _FakeSession)
    out = cma.run_agent_task(None, _CONFIG, "Close everything resolved over a week ago.")

    assert out == "report"
    assert sent["text"] == "Close everything resolved over a week ago."
    assert sent["role"] is Role.AGENT
    assert sent["identity"] == cma.AGENT_IDENTITY


def test_the_suggested_task_is_a_suggestion_not_a_default(monkeypatch):
    """It used to be the only thing /agent could do. It must not sneak back in as
    a fallback — an empty instruction should reach the agent as given, not be
    silently replaced."""
    sent = {}

    class _FakeSession:
        def __init__(self, *a):
            pass

        def send(self, text):
            sent["text"] = text
            return ""

    monkeypatch.setattr(cma, "CmaSession", _FakeSession)

    # The empty string is the case that matters. `send(instructions or
    # SUGGESTED_AGENT_TASK)` is the obvious-looking mutation, and any test that
    # passes a non-empty instruction never triggers it.
    cma.run_agent_task(None, _CONFIG, "")
    assert sent["text"] == "", (
        "an empty instruction was silently replaced with the canned prompt")

    cma.run_agent_task(None, _CONFIG, "just do the routing")
    assert sent["text"] == "just do the routing"


def test_help_documents_both_agent_forms():
    assert "/agent <text>" in cma._HELP
    assert "/agent  " in cma._HELP        # the bare, session-starting form


# --- diagnosability of a rejected tool-result batch -------------------------
#
# A live run failed with `tool_use_id "sevt_..." does not match any
# custom_tool_use event in this session`, and the trace could not explain it:
# `cma.idle` recorded only how many ids the session was blocked on, not which,
# and nothing recorded which ids we then answered. The wire log does not capture
# SSE response bodies, so the ids were unrecoverable after the fact. These pin
# the fields that make that diagnosable.

def _traced_send(role, config, legs, monkeypatch, tmp_path):
    """Run a scripted send inside a real Observability run and return its events."""
    from agent_obs import ObsConfig, Observability

    cfg = ObsConfig(var_dir=tmp_path, enabled=True)
    with Observability.start(cfg, front_end="t"):
        _run_send(role, config, legs, monkeypatch)
    return [json.loads(line)
            for path in sorted(tmp_path.rglob("*.events.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def test_the_idle_event_records_which_ids_not_just_how_many(in_sync, monkeypatch,
                                                            tmp_path):
    rows = _traced_send(Role.AGENT, in_sync[1], [
        [_call("sevt_1"), _call("sevt_2"), _idle("sevt_1", "sevt_2")],
        [_idle(stop="end_turn")],
    ], monkeypatch, tmp_path)

    idles = [r for r in rows if r.get("event") == "cma.idle"]
    assert idles, "no cma.idle event recorded"
    first = idles[0]
    assert first["event_ids"] == ["sevt_1", "sevt_2"], (
        "the blocking ids must be in the trace — a count cannot be compared "
        "against what we answered")
    assert first["collected"] == ["sevt_1", "sevt_2"]


def test_the_answered_ids_and_their_threads_are_recorded(in_sync, monkeypatch,
                                                         tmp_path):
    """A subagent's call is answered differently from the coordinator's own, so
    the thread has to be in the record when a batch is rejected."""
    sub = _Ev(type="agent.custom_tool_use", id="sevt_sub", name="list_incidents",
              input={}, session_thread_id="sthr_9")
    rows = _traced_send(Role.AGENT, in_sync[1], [
        [_call("sevt_own"), sub, _idle("sevt_own", "sevt_sub")],
        [_idle(stop="end_turn")],
    ], monkeypatch, tmp_path)

    results = [r for r in rows if r.get("event") == "cma.tool_results"]
    assert results, "nothing recorded which ids were answered"
    assert results[0]["ids"] == ["sevt_own", "sevt_sub"]
    assert results[0]["threads"] == ["-", "sthr_9"], (
        "the thread each answered call belongs to must be visible")
