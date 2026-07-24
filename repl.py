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

Run:  python3 repl.py
Requires the Anthropic SDK (`pip install anthropic`) and credentials
(ANTHROPIC_API_KEY or `ant auth login`). The /agent maintenance pass is
deterministic and works without any credentials.
"""

import contextlib
import dataclasses
import json
from collections.abc import Generator
from datetime import date, datetime
from enum import Enum
from typing import Any

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
    "default": DynamicPolicies(can_assign=True),
    # Never auto-assign: unassigned claims are left alone; assigned ones escalate.
    "no-assign": DynamicPolicies(can_assign=False),
    # Generous ceilings and a wider HIGH band (routes/approves more aggressively).
    "generous": DynamicPolicies(
        can_assign=True, auto_approve=1000, vip_auto_approve=2000, authorization_high=10000
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


def _dispatch(table: dict[str, Any], name: str, args: dict[str, Any]) -> str:
    """Execute a role-allowed tool by name, returning a text result for Claude."""
    fn = table.get(name)
    if fn is None:
        return json.dumps({"error": f"tool '{name}' is not permitted for this role"})
    try:
        return _jsonify(fn(**args))
    except Exception as exc:  # surface the error back to the model, don't crash the loop
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


class Session:
    """Holds the active role/identity, the conversation, and the Claude client."""

    def __init__(self, extra_system: str = SYSTEM_PROMPT_EXTRA) -> None:
        self.role: Role | None = None
        self.identity: str | None = None
        self.system_prompt: str = ""
        self.messages: list[dict[str, Any]] = []
        self.extra_system = extra_system
        self.policy_name = DEFAULT_POLICY  # which MOCK_AGENT_POLICIES the AGENT uses
        self._client = None  # lazily created on first chat

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
        self.role = role
        self.identity = identity
        self.system_prompt = roles.build_system_prompt(role, identity, extra)
        self.messages = []

    def reset(self) -> None:
        self.messages = []

    @property
    def client(self):
        if self._client is None:
            import anthropic  # imported lazily so /agent works without the SDK/key
            self._client = anthropic.Anthropic()
        return self._client

    # -- the agent chat loop --

    def chat(self, user_text: str) -> str:
        """Send a user turn and run the tool loop until Claude produces a reply."""
        assert self.role is not None
        schemas = roles.schemas_for_role(self.role)
        table = roles.dispatch_table(self.role)
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=self.system_prompt,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                tools=schemas,
                messages=self.messages,
            )
            # Preserve the full assistant content (incl. thinking blocks) for replay.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _dispatch(table, block.name, block.input),
                    }
                    for block in response.content
                    if block.type == "tool_use"
                ]
                self.messages.append({"role": "user", "content": results})
                continue

            return "".join(b.text for b in response.content if b.type == "text").strip()

        return "(stopped: reached the tool-call limit for this turn)"


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
        elif result.adjuster_id == tools.UNASSIGNED:
            reason = "policy can_assign=False" if not policies.can_assign else \
                     f"cost {result.estimate_cost} exceeds every authorization limit"
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
  /help              show this help
  /quit              exit
Anything else is sent to the agent for the current role."""


def _prompt(session: Session) -> str:
    if session.role is None:
        return "(no role)> "
    return f"({session.role.value}:{session.identity})> "


def _handle_command(session: Session, line: str) -> bool:
    """Handle a /command. Returns False to exit the REPL, True to continue."""
    parts = line.split()
    cmd, args = parts[0], parts[1:]

    if cmd in ("/quit", "/exit"):
        return False
    if cmd == "/help":
        print(_HELP)
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


def main() -> None:
    session = Session()
    print(_BANNER)
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
            print(f"[error talking to the agent: {type(exc).__name__}: {exc}]")


if __name__ == "__main__":
    main()
