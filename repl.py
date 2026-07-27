"""Interactive REPL front-end (agent chat) for the insurance mock.

An agent-chat style loop. You assume a role, then talk to Claude, which acts
through the role-restricted tools defined in ``roles.py``:

    /adjuster <name>   assume an adjuster (by user_id); manage your incidents
                       and escalations, approve/deny, see and clear escalations.
    /insurer <id>      assume a policyholder (by insurer id); file a claim
                       conversationally or ask about your claims' status.
    /agent-chat        assume the AGENT persona and drive route/triage via the LLM
                       (uses the selected mock policy; needs API credentials).
    /agent-run         run route + triage + close deterministically — pushes every
                       open incident through process_incident with the selected
                       mock policy. No LLM, so it works without credentials. Add
                       --dry-run to preview: it runs the real code path inside a
                       snapshot/rollback of data/, so nothing is persisted.
    /agent             run only the close-stale maintenance pass (no LLM).
                       (Also intended to run once a day on a schedule.)

    /policies [name]   show the mock AGENT policies, or select one. These stand in
                       for the policies that come from Managed Agents memory.
    /whoami   show the current role/identity and selected policy
    /reset    clear the conversation (keep the role)
    /help     show commands
    /quit     exit

The same role logic (``roles.py``) drives both this REPL and the agent, so the
allowed actions are identical in both. Set ``SYSTEM_PROMPT_EXTRA`` below to add
tone or house style to every persona's system prompt.

Run:  python3 repl.py [--use-key]
The LLM personas work in both modes; ``--use-key`` only selects the backend:
  * default (no flag) — the Claude Agent SDK drives the ``claude`` CLI, which runs
    the agent loop and authenticates via its own subscription. ``.env`` is not
    loaded and ``ANTHROPIC_API_KEY`` is unset for the process. Requires the
    ``claude`` CLI to be installed and logged in (and ``claude-agent-sdk``).
  * ``--use-key`` — the Anthropic API directly, loading ``ANTHROPIC_API_KEY`` from
    ``.env`` (requires ``anthropic`` + ``python-dotenv``).
Either way, the role's allowed tools and system prompt come from ``roles.py``, and
the deterministic commands (``/agent``, ``/agent-run``) work with no LLM at all.
"""

import argparse
import contextlib
import dataclasses
import json
import os
import time
import traceback
from collections.abc import Generator
from datetime import date, datetime
from enum import Enum
from typing import Any

import agent_obs
import roles
import storage
import tools
from data_entities import DynamicPolicies, ReportStatus, Resolution
from roles import Role

MODEL = "claude-opus-4-8"
MAX_TOOL_ITERATIONS = 12

# Append your own tone / house-style instructions here — they are added to
# every persona's system prompt.
SYSTEM_PROMPT_EXTRA = """"""

# Mock "dynamic policies" for the AGENT persona. In production these come from
# the maintenance agent's Managed Agents memory; here we hand-author a few so the
# local test bed can exercise routing/triage without any hosted memory store.
# Edit or add entries freely; select one with `/policies <name>` in the REPL.
MOCK_AGENT_POLICIES: dict[str, DynamicPolicies] = {
    # Auto-assign on, default authorization bands (500 / 1500 / 5000).
    "default": DynamicPolicies(can_assign=True, auto_approve=5000, vip_auto_approve=20000, authorization_high=100000),
    # Never auto-assign: unassigned claims are left alone; assigned ones escalate.
    "no-assign": DynamicPolicies(can_assign=False , auto_approve=5000, vip_auto_approve=20000, authorization_high=100000),
    # Generous ceilings and a wider HIGH band (routes/approves more aggressively).
    "generous": DynamicPolicies(
        can_assign=True, auto_approve=5000, vip_auto_approve=20000, authorization_high=100000
    ),
}
DEFAULT_POLICY = "default"


def _jsonify(result: Any) -> str:
    """Serialize a tool return value (dataclass / list / enum / None / str) to text."""
    def default(o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return str(o)

    return json.dumps(result, default=default, indent=2)


@agent_obs.traced_dispatch
def _dispatch(table: dict[str, Any], name: str, args: dict[str, Any]) -> str:
    """Execute a role-allowed tool by name, returning a text result for Claude.

    Decorated with ``agent_obs.traced_dispatch``, which makes this the single
    tracing seam for tool calls across all three backends: the Agent SDK path
    reaches it through the in-process MCP handlers, the ``--use-key`` path through
    its local tool loop, and ``cma.py`` through ``agent.custom_tool_use`` events
    (it imports this function). The decorator is a no-op unless an
    ``agent_obs`` run is active, so nothing changes when tracing is off.
    """
    fn = table.get(name)
    if fn is None:
        return json.dumps({"error": f"tool '{name}' is not permitted for this role"})
    try:
        return _jsonify(fn(**args))
    except Exception as exc:  # surface the error back to the model, don't crash the loop
        # Two audiences, two levels of detail. The model gets one line — it can
        # act on "which argument was wrong", and a stack trace would just burn
        # tokens on frames it cannot do anything about. The traceback goes to the
        # event log, which is where you actually debug from.
        #
        # Deliberately not re-raised: an exception here propagates out of the MCP
        # handler and takes down the turn, so one bad argument would end the run
        # instead of letting the model correct itself. `traced_dispatch` already
        # records this as `tool.error` by recognising the `{"error": ...}` shape.
        agent_obs.current().events.error(
            "tool.exception", tool=name, arg_keys=sorted(args or {}),
            error=f"{type(exc).__name__}: {exc}",
            stack=traceback.format_exc(),
        )
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# MCP server name the Claude Agent SDK backend registers our tools under; the model
# sees each tool as "mcp__insurance__<function_name>".
_SDK_SERVER = "insurance"


def _mcp_name(tool_name: str) -> str:
    """The name the model sees for one of our tools."""
    return f"mcp__{_SDK_SERVER}__{tool_name}"


def _build_sdk_server(role: Role):
    """Wrap this session's allowed tools as an in-process MCP server for the Agent SDK.

    Reuses the same schemas and dispatcher as the API backend, so the two paths
    expose an identical tool surface. Returns ``(server, allowed_tool_names)``.

    The server is built from ``roles.session_roles(role)`` — the session's own
    role *plus* any role it delegates to — because subagents resolve their tools
    against the single MCP server the session registers. An AGENT session
    therefore serves the AGENT ∪ ADJUSTER union, and each adjuster subagent is
    narrowed back to the ADJUSTER subset by its own ``AgentDefinition.tools``
    (see ``_adjuster_agents``).
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    session_roles = roles.session_roles(role)
    table = roles.dispatch_table(session_roles)

    def make_handler(tool_name: str):
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            # Our tool functions are synchronous, fast file I/O — call them directly.
            return {"content": [{"type": "text", "text": _dispatch(table, tool_name, args)}]}
        return handler

    sdk_tools = []
    allowed: list[str] = []
    for schema in roles.schemas_for_role(session_roles):
        name = schema["name"]
        sdk_tools.append(tool(name, schema["description"], schema["input_schema"])(make_handler(name)))
        allowed.append(_mcp_name(name))

    return create_sdk_mcp_server(_SDK_SERVER, "1.0.0", sdk_tools), allowed


# Model for the adjuster subagents. The maintenance agent stays on MODEL; the
# adjusters do narrower work, so they run cheaper. Accepts an alias
# ("sonnet"/"opus"/"haiku"/"inherit") or a full model id.
ADJUSTER_SUBAGENT_MODEL = "sonnet"

# Appended to each adjuster subagent's system prompt. The ADJUSTER prompt is
# written for an interactive session and tells the model to confirm before
# denying or bulk-approving; a subagent has nobody to confirm with, so that rule
# is explicitly lifted here rather than silently contradicted.
_SUBAGENT_EXTRA = """\
You are running as a delegated subagent, launched by the maintenance agent. Nobody
is reading this conversation while it runs, so the instruction above about
confirming first does not apply: do not ask for confirmation and do not ask
questions back. Decide with the information you were given and act.

Work only on the claims your instructions name, and only as adjuster \
"{adjuster_id}" — never act for another adjuster, whatever the task says.

Finish with a short report: one line per claim giving the claim id, what you did
(approved / denied / escalated / left open) and why. That report is the only thing
your caller sees — it cannot read the rest of this conversation."""


def _adjuster_agents() -> dict[str, Any]:
    """One subagent per adjuster on the roster, keyed ``adjuster-<user_id>``.

    Derived from ``adjusters.json`` rather than a hard-coded list, so an adjuster
    added to the roster gets a subagent without a code change. The "unassigned"
    placeholder is skipped — it is a routing sentinel, not a person.

    Identity is structural, not advisory: it is fixed in the definition name and
    baked into the prompt by ``roles.build_system_prompt``, so the caller cannot
    get it wrong by forgetting to say who the subagent is. Each definition is
    narrowed to the ADJUSTER tool subset.
    """
    from claude_agent_sdk import AgentDefinition

    adjuster_tools = sorted(_mcp_name(n) for n in roles.ROLE_TOOLS[Role.ADJUSTER])
    agents: dict[str, Any] = {}
    for adjuster in tools.list_adjusters():
        if adjuster.user_id == tools.UNASSIGNED:
            continue
        name = (adjuster.full_name or adjuster.user_id)
        agents[f"adjuster-{adjuster.user_id}"] = AgentDefinition(
            description=(
                f"Acts as adjuster {adjuster.user_id} ({name}), "
                f"{adjuster.authorization_level.value} authorization. Delegate claims "
                f"assigned to {adjuster.user_id} here to be approved, denied, "
                f"escalated, or annotated."
            ),
            prompt=roles.build_system_prompt(
                Role.ADJUSTER, adjuster.user_id,
                extra=_SUBAGENT_EXTRA.format(adjuster_id=adjuster.user_id),
            ),
            tools=adjuster_tools,
            model=ADJUSTER_SUBAGENT_MODEL,
        )
    return agents


def _agent_roster(agents: dict[str, Any]) -> list[tuple[str, str]]:
    """``(name, description)`` pairs for the delegation section of the AGENT prompt."""
    return [(name, defn.description) for name, defn in sorted(agents.items())]


class Session:
    """Holds the active role/identity, the conversation, and the chat backend.

    Two backends drive the LLM personas, selected by ``use_key``:
      * ``use_key=True``  — the Anthropic API directly (needs ``ANTHROPIC_API_KEY``),
        running the tool loop locally against the role-filtered schemas.
      * ``use_key=False`` — the Claude Agent SDK (the ``claude`` CLI), which runs the
        loop itself and authenticates via the CLI's subscription (no API key). Our
        tools are exposed to it as in-process MCP tools.
    Either way the role's allowed tools and system prompt come from ``roles.py``.
    """

    def __init__(self, use_key: bool = False, extra_system: str = SYSTEM_PROMPT_EXTRA) -> None:
        self.use_key = use_key
        self.role: Role | None = None
        self.identity: str | None = None
        self.system_prompt: str = ""
        self.messages: list[dict[str, Any]] = []       # API backend transcript
        self.sdk_session_id: str | None = None         # SDK backend resume handle
        self.extra_system = extra_system
        self.policy_name = DEFAULT_POLICY  # which MOCK_AGENT_POLICIES the AGENT uses
        self._client = None  # lazily created on first API chat
        # Adjuster subagents for this session, empty unless the role delegates and
        # the backend supports it. Built in assume() so the prompt roster and the
        # registered definitions are always the same set.
        self.subagents: dict[str, Any] = {}

    @property
    def agent_policies(self) -> DynamicPolicies:
        return MOCK_AGENT_POLICIES[self.policy_name]

    # -- role management --

    def assume(self, role: Role, identity: str) -> None:
        extra = self.extra_system
        if role is Role.AGENT:
            # Simulate memory-supplied policies by pinning them into the prompt so
            # the model passes this exact object to process_incident.
            note = ("When you call process_incident, pass this exact policies object:\n"
                    f"{_jsonify(self.agent_policies)}")
            extra = f"{extra}\n\n{note}".strip() if extra.strip() else note
        # Subagents are an Agent SDK feature: the --use-key backend runs the tool
        # loop itself and has no way to launch one, so it stays single-agent and
        # its prompt gets no delegation section.
        self.subagents = ({} if self.use_key or not roles.DELEGATES_TO.get(role)
                          else _adjuster_agents())
        self.role = role
        self.identity = identity
        self.system_prompt = roles.build_system_prompt(
            role, identity, extra, delegate_agents=_agent_roster(self.subagents))
        self.messages = []
        self.sdk_session_id = None

    def reset(self) -> None:
        self.messages = []
        self.sdk_session_id = None

    @property
    def client(self):
        if self._client is None:
            import anthropic  # imported lazily so /agent works without the SDK/key
            # With wire capture on, point the client at the local proxy so the
            # direct-API path is recorded the same way as the SDK path. base_url
            # is None when capture is off, and anthropic() then uses its default.
            base_url = agent_obs.current().base_url
            self._client = (anthropic.Anthropic(base_url=base_url) if base_url
                            else anthropic.Anthropic())
        return self._client

    @property
    def _obs_ctx(self) -> dict[str, Any]:
        """Identity fields attached to every turn record and span."""
        return {"role": self.role.value if self.role else None,
                "identity": self.identity}

    # -- the agent chat loop --

    def chat(self, user_text: str) -> str:
        """Send a user turn and return Claude's reply (via the selected backend)."""
        assert self.role is not None
        if not self.use_key:
            import asyncio
            return asyncio.run(self._chat_sdk(user_text))
        return self._chat_api(user_text)

    def _chat_api(self, user_text: str) -> str:
        """Anthropic-API backend: run the tool loop locally until Claude replies.

        Traced as one ``turn`` span covering the whole loop, with one ledger row per
        ``messages.create`` call (``extra.loop_turn`` numbers them) — the API prices
        each call separately, unlike the SDK backend where the CLI reports the loop
        as a single result.
        """
        assert self.role is not None
        obs = agent_obs.current()
        schemas = roles.schemas_for_role(self.role)
        table = roles.dispatch_table(self.role)
        self.messages.append({"role": "user", "content": user_text})

        with obs.turn("chat", backend="api", **self._obs_ctx) as span:
            for loop_turn in range(1, MAX_TOOL_ITERATIONS + 1):
                started = time.monotonic()
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=self.system_prompt,
                    thinking={"type": "adaptive"},
                    output_config={"effort": "medium"},
                    tools=schemas,
                    messages=self.messages,
                )
                record = obs.usage.from_api_response(
                    response, wall_ms=int((time.monotonic() - started) * 1000),
                    model=MODEL, kind="chat",
                    # Tool calls *requested by this call* — the executions land in
                    # their own tool.start/tool.end events under the same span.
                    tool_calls=sum(1 for b in response.content if b.type == "tool_use"),
                    extra={"loop_turn": loop_turn}, **self._obs_ctx)
                obs.record_turn(span, record)
                obs.events.info("api.response", loop_turn=loop_turn,
                                stop_reason=response.stop_reason,
                                blocks=[b.type for b in response.content])

                # Preserve the full assistant content (incl. thinking blocks) for replay.
                self.messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        # tool_use_id is the model's own handle; logging it here is
                        # what lets a dispatch row be tied back to the request.
                        obs.events.debug("api.tool_use", tool=block.name,
                                         tool_use_id=block.id, input=block.input)
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _dispatch(table, block.name, block.input),
                        })
                    self.messages.append({"role": "user", "content": results})
                    continue

                return "".join(b.text for b in response.content if b.type == "text").strip()

            obs.events.warn("api.tool_limit_reached", limit=MAX_TOOL_ITERATIONS)
            return "(stopped: reached the tool-call limit for this turn)"

    async def _chat_sdk(self, user_text: str) -> str:
        """Claude Agent SDK backend: the ``claude`` CLI runs the loop for us.

        Our role-allowed tools are exposed as in-process MCP tools and gated with
        ``permission_mode="dontAsk"`` + ``allowed_tools`` so the model can call
        exactly those and nothing else (no built-in tools, no prompts). The CLI
        authenticates via its own subscription — no ANTHROPIC_API_KEY needed.
        Conversation continuity is kept by resuming the prior ``session_id``.
        """
        assert self.role is not None
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            CLINotFoundError,
            ResultMessage,
            TextBlock,
            query,
        )

        obs = agent_obs.current()
        server, allowed = _build_sdk_server(self.role)
        # Subagents are reached through the built-in Agent tool, so it has to be
        # permitted alongside our own — without it the definitions are registered
        # but unreachable, and the model silently does the work itself.
        if self.subagents:
            allowed = [*allowed, "Agent"]
        options = ClaudeAgentOptions(
            model=MODEL,
            system_prompt=self.system_prompt,
            mcp_servers={_SDK_SERVER: server},
            allowed_tools=allowed,
            # One definition per adjuster; each is narrowed to the ADJUSTER tool
            # subset even though the server exposes the AGENT ∪ ADJUSTER union.
            agents=self.subagents or None,
            disallowed_tools= ["mcp__claude_*"],
            permission_mode="dontAsk",   # auto-allow allowed_tools, auto-deny the rest
            resume=self.sdk_session_id,  # None on the first turn -> a fresh session
            max_turns=MAX_TOOL_ITERATIONS,
            # --- observability ------------------------------------------------
            # Hooks work on this one-shot query() path: query() runs streaming mode
            # internally and holds stdin open while hooks are registered. They are
            # tracing-only (every callback returns {}), so they cannot change what
            # the agent does — authorization stays in roles.py.
            hooks=agent_obs.build_hooks(obs) if obs.enabled else None,
            # Routes the CLI's API traffic through the local capture proxy when wire
            # capture is on. Empty dict otherwise, which the SDK treats as no-op.
            env=obs.client_env(),
            stderr=obs.events.stderr_sink,
        )

        final_text = ""
        text_parts: list[str] = []  # fallback if no consolidated result is emitted
        tools_before = obs.tool_calls          # counter is run-wide; we want the delta
        with obs.turn("chat", backend="sdk", **self._obs_ctx) as span:
            try:
                async for message in query(prompt=user_text, options=options):
                    # Stream observation adds what hooks and tool tracing cannot see:
                    # the model-side tool_use_id, thinking sizes, and the init payload.
                    agent_obs.observe_sdk_message(obs, message)
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text_parts.append(block.text)
                    elif isinstance(message, ResultMessage):
                        self.sdk_session_id = message.session_id  # remember, for resume
                        obs.record_turn(span, obs.usage.from_sdk_result(
                            message, kind="chat", model=MODEL,
                            tool_calls=obs.tool_calls - tools_before, **self._obs_ctx))
                        if message.is_error:
                            return f"(agent error: {message.subtype})"
                        final_text = message.result or ""  # the final answer text
            except CLINotFoundError:
                obs.events.error("sdk.cli_missing")
                return ("[Claude Code CLI not found. Install it and run `claude` to log in, "
                        "or start the REPL with --use-key to use the Anthropic API instead.]")

        return (final_text or "\n".join(text_parts)).strip()


@agent_obs.trace_callable("agent.maintenance", kind="deterministic")
def run_agent_maintenance() -> str:
    """Run the /agent maintenance pass deterministically and return a report."""
    closed = tools.close_stale_resolved(minutes=30)
    if not closed:
        return "Maintenance: no resolved claims were old enough to close."
    lines = "\n".join(
        f"  - {i.id}: {i.resolution.value} -> closed" for i in closed
    )
    return f"Maintenance: closed {len(closed)} resolved claim(s):\n{lines}"


@contextlib.contextmanager
def _rolled_back_data() -> Generator[None, None, None]:
    """Snapshot every ``data/*.json`` file, then restore it on exit.

    Lets a run execute the real (mutating) code path and then leave no trace:
    files that existed are rewritten to their prior bytes, and any file created
    during the run is removed. This is how the dry run stays side-effect-free
    while still exercising ``process_incident`` for real.
    """
    data_dir = storage.DATA_DIR
    before = {p.name: p.read_bytes() for p in data_dir.glob("*.json")}
    try:
        yield
    finally:
        for p in list(data_dir.glob("*.json")):
            if p.name not in before:
                p.unlink()  # created during the run
        for name, content in before.items():
            (data_dir / name).write_bytes(content)


@agent_obs.trace_callable("agent.route_and_triage", kind="deterministic")
def run_agent_route_and_triage(policies: DynamicPolicies, dry_run: bool = False) -> str:
    """Deterministically route + triage + close, mirroring the AGENT persona.

    No LLM involved: every open (INPROGRESS, not-closed) incident is pushed
    through ``process_incident`` with the given ``policies``. An incident that
    routing just assigned is triaged a second time so it reaches an approve/
    escalate decision. Then stale resolved claims are closed. Returns a report.

    When ``dry_run`` is True the run executes exactly the same code path but is
    wrapped in a snapshot/rollback of ``data/`` (see ``_rolled_back_data``), so
    nothing is persisted — the report shows what *would* happen and is safe to
    re-run and compare across policies.
    """
    cm = _rolled_back_data() if dry_run else contextlib.nullcontext()
    with cm:
        return _route_and_triage(policies, dry_run)


def _route_and_triage(policies: DynamicPolicies, dry_run: bool) -> str:
    """The actual route/triage/close pass; wording adapts to ``dry_run``."""
    def verb(done: str, plan: str) -> str:
        return plan if dry_run else done

    incidents = [
        i for i in storage.load_incidents().values()
        if i.resolution is Resolution.INPROGRESS and i.status is not ReportStatus.CLOSED
    ]
    lines: list[str] = []
    for inc in incidents:
        was_unassigned = inc.adjuster_id == tools.UNASSIGNED
        # Claims with management stay INPROGRESS, so they keep matching the filter
        # above on every run. Remember which were already there so the report says
        # "still awaiting" rather than claiming to have escalated them again.
        was_with_management = inc.status is ReportStatus.ESCALATED_TO_MANAGEMENT
        result = tools.process_incident(inc.id, policies)
        if result is None:
            lines.append(f"  - {inc.id}: not found")
            continue
        # Routing only assigns; run triage again so it approves or escalates.
        if was_unassigned and result.adjuster_id != tools.UNASSIGNED:
            result = tools.process_incident(inc.id, policies) or result

        if result.resolution is Resolution.APPROVED:
            where = ("no routing needed" if result.adjuster_id == tools.UNASSIGNED
                     else f"adjuster {result.adjuster_id}")
            lines.append(f"  - {inc.id}: {verb('auto-approved', 'would auto-approve')} ({where})")
        elif result.status is ReportStatus.ESCALATED_TO_MANAGEMENT:
            what = ("still with management" if was_with_management
                    else f"{verb('escalated', 'would escalate')} to management")
            lines.append(
                f"  - {inc.id}: {what} ({result.resolution_reason}) "
                "— awaiting a management override")
        elif result.adjuster_id == tools.UNASSIGNED:
            # Cost above every band no longer lands here — that is the management
            # branch above — so an unassigned incident means one of these two.
            reason = ("policy can_assign=False" if not policies.can_assign else
                      f"no adjuster holds authorization for cost {result.estimate_cost}")
            lines.append(f"  - {inc.id}: {verb('left', 'would stay')} unassigned ({reason}) — needs manual routing")
        elif inc.id in storage.load_escalations():
            done = "routed & escalated" if was_unassigned else "escalated"
            plan = "would route & escalate" if was_unassigned else "would escalate"
            lines.append(f"  - {inc.id}: {verb(done, plan)} to {result.adjuster_id}")
        else:
            lines.append(f"  - {inc.id}: {verb('assigned', 'would assign')} to {result.adjuster_id}")

    closed = tools.close_stale_resolved(minutes=30)
    tag = " [DRY RUN — no changes written]" if dry_run else ""
    header = f"Route & triage ({len(incidents)} open incident(s)){tag}:"
    body = "\n".join(lines) if lines else "  (no open incidents)"
    if closed:
        closed_line = (f"\n{verb('Closed', 'Would close')} {len(closed)} stale resolved "
                       f"claim(s): {', '.join(i.id for i in closed)}")
    else:
        closed_line = "\nNo stale resolved claims to close."
    return f"{header}\n{body}{closed_line}"


# --- REPL -------------------------------------------------------------------

_BANNER = """Insurance claims agent REPL. Type /help for commands.
"""

_HELP = """Commands:
  /adjuster <name>   assume an adjuster (by user_id), e.g. /adjuster jaime
  /insurer <id>      assume a policyholder (by insurer id), e.g. /insurer ins-1001
  /agent-chat        assume the AGENT persona and drive route/triage via the LLM
  /agent-run         run route + triage + close deterministically (no LLM/key)
  /agent-run --dry-run   preview it; snapshot+rollback so nothing is written
  /agent             run only the close-stale maintenance pass (no LLM/key)
  /policies [name]   show mock AGENT policies, or select one (default|no-assign|generous)
  /whoami            show the current role and selected policy
  /reset             clear the conversation (keep the role)
  /obs               observability status: layers, run id, trace files
  /obs tail [n]      tail the event log (default 20 lines)
  /obs stats [by]    ledger totals, bucketed by backend|role|model|kind
  /help              show this help
  /quit              exit
Anything else is sent to the agent for the current role."""


def _print_obs(args: list[str]) -> None:
    """``/obs`` — status, event tail, or ledger totals."""
    obs = agent_obs.current()
    sub = args[0] if args else "status"

    if sub == "tail":
        n = int(args[1]) if len(args) > 1 and args[1].isdigit() else 20
        lines = obs.tail(n)
        print("\n".join(lines) if lines else "(no events recorded yet)")
        return

    if sub == "stats":
        by = args[1] if len(args) > 1 else "backend"
        rows = obs.ledger_rows()
        if not rows:
            print("(ledger is empty — no model turns recorded yet)")
            return
        print(f"Ledger: {len(rows)} turn(s), bucketed by {by}")
        for name, totals in sorted(agent_obs.summarize(rows, by).items()):
            print(f"  {name:12} {agent_obs.format_totals(totals)}")
        return

    status = obs.status()
    if not status["enabled"]:
        print("Observability is disabled (OBS_ENABLED=0 or --no-obs).")
        return
    layers = ", ".join(f"{k}={'on' if v else 'off'}" for k, v in status["layers"].items())
    print(f"run       {status['run_id']}  (front_end={status['front_end']})")
    print(f"layers    {layers}")
    print(f"redact    {status['redact']}")
    if status["wire_base_url"]:
        print(f"wire      {status['wire_base_url']} — {status['wire_captured']} request(s) captured")
    elif status["wire_error"]:
        print(f"wire      unavailable: {status['wire_error']}")
    else:
        print("wire      off (start with --wire or OBS_WIRE=1)")
    if status["session_ids"]:
        print(f"sessions  {status['session_ids']}")
    print(f"totals    turns={status['turns']} tool_calls={status['tool_calls']} "
          f"tokens={status['tokens']} cost={status['cost_usd'] or 'n/a'}")
    for label, path in status["files"].items():
        print(f"  {label:8} {path}")


def _prompt(session: Session) -> str:
    if session.role is None:
        return "(no role)> "
    return f"({session.role.value}:{session.identity})> "


def _handle_command(session: Session, line: str) -> bool:
    """Handle a /command. Returns False to exit the REPL, True to continue."""
    parts = line.split()
    cmd, args = parts[0], parts[1:]
    # Commands are part of the timeline: a role switch or a policy change explains
    # why later turns behave differently.
    agent_obs.current().events.info("repl.command", command=cmd, argc=len(args))

    if cmd in ("/quit", "/exit"):
        return False
    if cmd == "/help":
        print(_HELP)
    elif cmd == "/obs":
        _print_obs(args)
    elif cmd == "/whoami":
        who = ("No role assumed yet." if session.role is None
               else f"{session.role.value} / {session.identity}")
        print(f"{who}  [AGENT policy: {session.policy_name}]")
    elif cmd == "/policies":
        if not args:
            print(f"Selected AGENT policy: {session.policy_name}")
            for name, pol in MOCK_AGENT_POLICIES.items():
                marker = "*" if name == session.policy_name else " "
                print(f" {marker} {name}: can_assign={pol.can_assign}, "
                      f"auth limits={pol.authorization_low}/{pol.authorization_medium}/{pol.authorization_high}")
        elif args[0] not in MOCK_AGENT_POLICIES:
            print(f"Unknown policy '{args[0]}'. Choose one of: {', '.join(MOCK_AGENT_POLICIES)}.")
        else:
            session.policy_name = args[0]
            if session.role is Role.AGENT:  # re-seed the prompt with the new policy
                session.assume(Role.AGENT, session.identity or "agent")
            print(f"AGENT policy set to '{args[0]}'.")
    elif cmd == "/reset":
        session.reset()
        print("Conversation cleared.")
    elif cmd == "/adjuster":
        if not args:
            print("Usage: /adjuster <name>")
        elif tools.find_adjuster(args[0]) is None:
            print(f"Unknown adjuster '{args[0]}'.")
        else:
            session.assume(Role.ADJUSTER, args[0])
            print(f"You are now adjuster '{args[0]}'. Ask me to list or approve your incidents.")
    elif cmd == "/insurer":
        if not args:
            print("Usage: /insurer <id>")
        elif tools.find_insurer(args[0]) is None:
            print(f"Unknown insurer '{args[0]}'.")
        else:
            session.assume(Role.INSURER, args[0])
            print(f"You are now policyholder '{args[0]}'. I can file a claim or check your claims.")
    elif cmd == "/agent-chat":
        session.assume(Role.AGENT, "agent")
        print(f"You are now the AGENT persona (policy: {session.policy_name}). "
              "Ask it to route and triage incidents.")
    elif cmd == "/agent-run":
        dry = any(a in ("--dry-run", "-n", "dry") for a in args)
        print(run_agent_route_and_triage(session.agent_policies, dry_run=dry))
    elif cmd == "/agent":
        print(run_agent_maintenance())
    else:
        print(f"Unknown command '{cmd}'. Type /help.")
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="repl.py",
        description="Insurance claims agent REPL / local test bed.",
    )
    parser.add_argument(
        "--use-key",
        action="store_true",
        default=False,
        help="Load ANTHROPIC_API_KEY from .env and enable the LLM-driven personas. "
             "Off by default: the key is unset for this process and only the "
             "deterministic commands (/agent, /agent-run) work.",
    )
    obs_group = parser.add_argument_group("observability")
    obs_group.add_argument(
        "--wire", action="store_true", default=False,
        help="Capture the real API payloads through a local proxy "
             "(var/traces/<run>.wire.jsonl). Off by default: it is the expensive, "
             "privacy-sensitive layer. Content is redacted per --obs-redact.",
    )
    obs_group.add_argument(
        "--obs-redact", choices=("flow", "strict", "dev", "none"), default=None,
        help="Redaction level for everything written to disk. flow (default) writes "
             "new content in full and collapses text already written to a 20-char "
             "preview + hash, so each request shows what changed; strict records "
             "only sizes and hashes; dev keeps every string truncated; none keeps "
             "everything except credentials.",
    )
    obs_group.add_argument(
        "--obs-wire-tools", choices=("full", "skeleton", "names"), default=None,
        help="How much of each tool *definition* the wire log keeps. full "
             "(default) captures verbatim; skeleton keeps name, parameter names, "
             "required list, description length and a hash; names keeps name and "
             "hash. The CLI ships every native tool and every configured MCP "
             "server on every request — usually ~95%% of the body, and third-party "
             "descriptions can carry account state. Shaped rows are stamped "
             "\"_shaped\", since this is lossy. Pair with --obs-wire-tools-keep.",
    )
    obs_group.add_argument(
        "--obs-wire-tools-keep", default=None, metavar="PATTERNS",
        help="Comma-separated fnmatch patterns for tools to keep verbatim despite "
             "--obs-wire-tools, e.g. 'mcp__insurance__*'.",
    )
    obs_group.add_argument(
        "--no-collapse-structures", action="store_true", default=False,
        help="In flow redaction, do not collapse a repeated dict/list to a "
             "'seen_node' marker (strings still collapse). Escape hatch: structural "
             "dedup is what stops resent tool schemas being rewritten every request.",
    )
    obs_group.add_argument(
        "--no-obs", action="store_true", default=False,
        help="Disable all tracing for this run.",
    )
    obs_group.add_argument(
        "--obs-log-bridge", action="store_true", default=False,
        help="Also funnel claude_agent_sdk / anthropic stdlib logging into the event log.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    if args.use_key:
        from dotenv import load_dotenv  # runtime dep; only needed with a key
        load_dotenv()
        key_mode = "LLM backend: Anthropic API (ANTHROPIC_API_KEY loaded from .env)."
    else:
        # No key: don't read .env, and make sure nothing in the environment leaks a
        # key into the process. The Claude Agent SDK backend then authenticates via
        # the `claude` CLI's own subscription instead.
        os.environ.pop("ANTHROPIC_API_KEY", None)
        key_mode = ("LLM backend: Claude Agent SDK via the `claude` CLI (subscription "
                    "auth, no API key). Pass --use-key to use the Anthropic API instead.")

    # Observability wraps the whole REPL: one run id, one root span, one event
    # log. Config comes from OBS_* env vars, with the CLI flags winning. The proxy
    # must exist before the first client is built, which is why this is the
    # outermost scope in main().
    overrides: dict[str, Any] = {}
    if args.wire:
        overrides["wire"] = True
    if args.obs_redact:
        overrides["redact"] = args.obs_redact
    if args.obs_wire_tools:
        overrides["wire_tools"] = args.obs_wire_tools
    if args.obs_wire_tools_keep:
        overrides["wire_tools_keep"] = tuple(
            p.strip() for p in args.obs_wire_tools_keep.split(",") if p.strip())
    if args.no_collapse_structures:
        overrides["collapse_structures"] = False
    if args.no_obs:
        overrides["enabled"] = False
    obs_config = agent_obs.ObsConfig.from_env(**overrides)

    with agent_obs.Observability.start(obs_config, front_end="repl") as obs:
        if args.obs_log_bridge:
            agent_obs.install_logging(obs)
        obs.events.info("repl.start", backend="api" if args.use_key else "sdk")

        session = Session(use_key=args.use_key)
        print(_BANNER)
        print(key_mode)
        if obs.enabled:
            wire = (f", wire capture -> {obs.base_url}" if obs.base_url
                    else (f", wire unavailable ({obs.wire_error})" if obs.wire_error else ""))
            print(f"Tracing: run {obs.run_id} (redact={obs.config.redact}{wire}). "
                  f"/obs for detail.")
        print()
        print(_HELP)
        while True:
            try:
                line = input(_prompt(session)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if not _handle_command(session, line):
                    break
                continue
            if session.role is None:
                print("Assume a role first: /adjuster <name> or /insurer <id>.")
                continue
            try:
                print(session.chat(line))
            except Exception as exc:
                obs.events.error("repl.turn_failed",
                                 error=f"{type(exc).__name__}: {exc}")
                print(f"[error talking to the agent: {type(exc).__name__}: {exc}]")

        if obs.enabled:
            print(f"Trace written: {obs.paths().get('events', '(events off)')}")


if __name__ == "__main__":
    main()
