"""Managed Agents (CMA) front-end for the insurance mock.

This is the hosted-agent version of the REPL. Instead of running the tool loop
locally (see ``repl.py``), Anthropic runs the agent loop in a per-session
container; our ``tools.py`` functions are declared as **custom tools** and
executed here in the orchestrator when the agent emits ``agent.custom_tool_use``
events, and each adjuster's persistent memory is a Managed Agents **memory
store** mounted into the session at ``/mnt/memory/<store>/``.

Setup (once) vs runtime (every session):
    setup()      -> creates the environment, one agent per role, and one memory
                    store per adjuster; persists their IDs in data/cma_config.json.
                    Guarded so it is safe to call repeatedly.
    CmaSession   -> per role/identity: creates a session (with the adjuster's
                    memory store attached and an identity-specific system-prompt
                    override), then streams events, dispatching custom-tool calls
                    to the role-restricted tools.

Roles map to agents:
    ADJUSTER — custom tools for its domain + the prebuilt agent toolset (so it
        can read/write its mounted memory store) + a memory store per adjuster.
    INSURER  — custom tools only (file a claim / query own claims).
    AGENT    — the maintenance agent (close stale resolved claims).

Requires the Anthropic SDK and a workspace with Managed Agents enabled
(``client.beta.agents`` / ``sessions`` / ``memory_stores``). The domain logic,
roles, and schemas are shared with the local REPL, so the two stay in lockstep.

Run:  python3 cma.py
"""

import dataclasses
import json

import agent_schemas
import roles
import storage
import tools
from data_entities import DynamicPolicies
from repl import _dispatch, _jsonify  # reuse the role-restricted dispatcher + serializer
from roles import Role

# Per-role model: cheaper/faster models for the narrower roles, the most capable
# model for the autonomous maintenance agent.
MODEL_BY_ROLE = {
    Role.INSURER: "claude-haiku-4-5",
    Role.ADJUSTER: "claude-sonnet-5",
    Role.AGENT: "claude-opus-4-8",
}
CONFIG_FILE = storage.DATA_DIR / "cma_config.json"

# Adjusters that get a personal memory store provisioned at setup.
MEMORY_ADJUSTERS = ["jaime", "sam", "jane"]

# Append your own tone / house-style here — added to every agent's system prompt.
SYSTEM_PROMPT_EXTRA = ""

# Seed content for an adjuster's memory store (only jaime ships with overrides).
_SEED_MEMORY = {
    "jaime": (
        "# Auto-approval policy overrides (Cascadia region)\n\n"
        "Apply these on top of the baseline DynamicPolicies:\n"
        "- vip_auto_approve: 800\n\n"
        "## Standing guidance\n"
        "- Lean generous with long-time VIP customers.\n"
        "- Always escalate suspected fraud regardless of cost.\n"
    ),
}


def _memory_store_name(adjuster_id: str) -> str:
    return f"adjuster-{adjuster_id}"


# --- custom tool definitions ------------------------------------------------

def custom_tools_for_role(role: Role) -> list[dict]:
    """CMA custom-tool defs for a role: our schemas wrapped as ``type: custom``."""
    allowed = roles.ROLE_TOOLS[role]
    return [
        {"type": "custom", **schema}
        for schema in agent_schemas.build_tool_schemas()
        if schema["name"] in allowed
    ]


def agent_tools_for_role(role: Role) -> list[dict]:
    """Full tool list for a role's agent (prebuilt toolset for memory + custom tools)."""
    custom = custom_tools_for_role(role)
    if role is Role.ADJUSTER:
        # The prebuilt toolset gives read/write/edit/glob so the agent can use its
        # mounted memory store.
        return [{"type": "agent_toolset_20260401"}, *custom]
    return custom


# --- system prompts (role-generic on the agent; identity injected per session) ---

_ADJUSTER_SYSTEM = """You are an AI assistant acting on behalf of an insurance adjuster.
You manage that adjuster's claims: list/inspect incidents and escalations; approve or
deny (individually or in bulk when asked); escalate; update status; notify the adjuster
and policyholder; and log history/notes. Only take adjuster actions.

You have a persistent memory store mounted under /mnt/memory/. It holds your standing
auto-approval policy overrides and working notes. ALWAYS read it before acting (use the
read/glob tools), and write updates back (write/edit) when you learn a lasting rule.

The baseline auto-approval policy (DynamicPolicies defaults) is:
{defaults_json}
Merge any overrides from your memory onto this baseline and pass the resulting object as
the `policies` argument to approve_incident / escalate_incident. Whether a policyholder is
"upset" is your judgement — read the incident and the insurer's history and set
insurer_upset accordingly. Confirm before denying a claim or approving many at once."""

_INSURER_SYSTEM = """You are an AI assistant speaking directly with an insurance
policyholder. You can only (1) help them file a new claim — gather policy, loss type(s),
a description, and an estimated cost, then create it (new claims are routed to intake
adjuster "{intake}"), and (2) answer questions about the status of THIS policyholder's
own claims. Never approve, deny, escalate, or reveal other policyholders' claims. Be warm
and clear, and confirm details before filing."""

_AGENT_SYSTEM = """You are the automated maintenance agent for the claims system. Your job
is to close out claims that have been resolved (approved or denied) for a while: call
close_stale_resolved to find and close claims resolved more than 30 minutes ago, then
report how many you closed."""


def _agent_system(role: Role) -> str:
    if role is Role.ADJUSTER:
        base = _ADJUSTER_SYSTEM.format(
            defaults_json=json.dumps(dataclasses.asdict(DynamicPolicies()), indent=2)
        )
    elif role is Role.INSURER:
        base = _INSURER_SYSTEM.format(intake=roles.DEFAULT_INTAKE_ADJUSTER)
    else:
        base = _AGENT_SYSTEM
    return f"{base}\n\n{SYSTEM_PROMPT_EXTRA.strip()}" if SYSTEM_PROMPT_EXTRA.strip() else base


def _session_system(role: Role, identity: str) -> str:
    """Identity-specific system prompt used as a per-session override."""
    base = _agent_system(role)
    if role is Role.ADJUSTER:
        adjuster = tools.find_adjuster(identity)
        name = (adjuster.full_name if adjuster else None) or identity
        store = _memory_store_name(identity)
        return (f"{base}\n\nYou are adjuster {name} (id \"{identity}\"). Your memory store is "
                f"mounted at /mnt/memory/{store}/.")
    if role is Role.INSURER:
        insurer = tools.find_insurer(identity)
        name = (insurer.full_name if insurer else None) or identity
        return (f"{base}\n\nYou are speaking with policyholder {name} (insurer id "
                f"\"{identity}\"). Use this insurer_id for their claims.")
    return base


# --- config persistence -----------------------------------------------------

def _load_config() -> dict:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {"environment_id": None, "agents": {}, "memory_stores": {}}


def _save_config(config: dict) -> None:
    storage.DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def setup(client) -> dict:
    """Idempotently provision the environment, per-role agents, and memory stores.

    Returns the config dict (also persisted to data/cma_config.json). Safe to call
    repeatedly — only missing resources are created.
    """
    config = _load_config()

    if not config.get("environment_id"):
        env = client.beta.environments.create(
            name="insurance-mock-env",
            config={"type": "cloud", "networking": {"type": "unrestricted"}},
        )
        config["environment_id"] = env.id
        _save_config(config)

    for role in Role:
        if role.value in config["agents"]:
            continue
        agent = client.beta.agents.create(
            name=f"insurance-{role.value}",
            model=MODEL_BY_ROLE[role],
            system=_agent_system(role),
            tools=agent_tools_for_role(role),
        )
        config["agents"][role.value] = agent.id
        _save_config(config)

    for adjuster_id in MEMORY_ADJUSTERS:
        if adjuster_id in config["memory_stores"]:
            continue
        store = client.beta.memory_stores.create(
            name=_memory_store_name(adjuster_id),
            description=f"Personal memory for adjuster {adjuster_id}: auto-approval "
                        "policy overrides and working notes.",
        )
        seed = _SEED_MEMORY.get(adjuster_id)
        if seed:
            client.beta.memory_stores.memories.create(
                store.id, path="/policies.md", content=seed
            )
        config["memory_stores"][adjuster_id] = store.id
        _save_config(config)

    return config


# --- session runtime --------------------------------------------------------

class CmaSession:
    """Drives one Managed Agents session for a role/identity."""

    def __init__(self, client, config: dict, role: Role, identity: str) -> None:
        self.client = client
        self.config = config
        self.role = role
        self.identity = identity
        self.table = roles.dispatch_table(role)
        self.session_id = self._create_session()

    def _create_session(self) -> str:
        resources = []
        if self.role is Role.ADJUSTER:
            store_id = self.config["memory_stores"].get(self.identity)
            if store_id:
                resources.append({
                    "type": "memory_store",
                    "memory_store_id": store_id,
                    "access": "read_write",
                    "instructions": "Your personal adjuster memory: auto-approval policy "
                                    "overrides and working notes. Read before acting.",
                })
        session = self.client.beta.sessions.create(
            agent={
                "type": "agent_with_overrides",
                "id": self.config["agents"][self.role.value],
                "system": _session_system(self.role, self.identity),
            },
            environment_id=self.config["environment_id"],
            resources=resources,
            title=f"{self.role.value}:{self.identity}",
        )
        return session.id

    def send(self, user_text: str) -> str:
        """Send a user turn, service custom-tool calls, and return the agent's reply."""
        to_send: list[dict] | None = [
            {"type": "user.message", "content": [{"type": "text", "text": user_text}]}
        ]
        text_parts: list[str] = []
        terminated = False

        while True:
            # Stream-first: open the stream, then send inside it so no early event is missed.
            with self.client.beta.sessions.events.stream(session_id=self.session_id) as stream:
                if to_send is not None:
                    self.client.beta.sessions.events.send(session_id=self.session_id, events=to_send)
                    to_send = None
                tool_calls = []
                for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "agent.message":
                        for block in getattr(event, "content", []) or []:
                            if getattr(block, "type", None) == "text":
                                text_parts.append(block.text)
                    elif etype == "agent.custom_tool_use":
                        tool_calls.append(event)
                    elif etype == "session.status_terminated":
                        terminated = True
                        break
                    elif etype == "session.status_idle":
                        # Idle waiting on us (a custom tool) -> keep going; otherwise done.
                        stop = getattr(event, "stop_reason", None)
                        if getattr(stop, "type", None) != "requires_action":
                            terminated = False
                            break

            if terminated or not tool_calls:
                break
            to_send = [
                {
                    "type": "user.custom_tool_result",
                    "custom_tool_use_id": call.id,
                    "content": [{"type": "text", "text": _dispatch(self.table, call.name, call.input)}],
                }
                for call in tool_calls
            ]

        return "".join(text_parts).strip()


def run_agent_maintenance(client, config: dict) -> str:
    """Run the maintenance persona as a CMA session and return its report."""
    session = CmaSession(client, config, Role.AGENT, "system")
    return session.send(
        "Close any claims that have been resolved (approved or denied) for more than "
        "30 minutes, then report how many you closed."
    )


# --- REPL -------------------------------------------------------------------

_HELP = """Commands:
  /setup             provision the environment, agents, and memory stores (run once)
  /adjuster <name>   start an adjuster session (by user_id), e.g. /adjuster jaime
  /insurer <id>      start a policyholder session (by insurer id), e.g. /insurer ins-1001
  /agent             run the maintenance agent now (close stale resolved claims)
  /whoami            show the current role
  /help              show this help
  /quit              exit
Anything else is sent to the current session."""


def main() -> None:
    import anthropic

    client = anthropic.Anthropic()
    config = _load_config()
    session: CmaSession | None = None

    print("Managed Agents insurance REPL. Run /setup first if you haven't. /help for commands.\n")
    print(_HELP)

    def prompt() -> str:
        return f"({session.role.value}:{session.identity})> " if session else "(no session)> "

    while True:
        try:
            line = input(prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split()
            cmd, args = parts[0], parts[1:]
            if cmd in ("/quit", "/exit"):
                break
            if cmd == "/help":
                print(_HELP)
            elif cmd == "/setup":
                config = setup(client)
                print(f"Setup complete. environment={config['environment_id']}, "
                      f"agents={list(config['agents'])}, memory_stores={list(config['memory_stores'])}")
            elif cmd == "/whoami":
                print("No session." if session is None else f"{session.role.value} / {session.identity}")
            elif cmd == "/adjuster":
                if not config.get("environment_id"):
                    print("Run /setup first.")
                elif not args or tools.find_adjuster(args[0]) is None:
                    print("Usage: /adjuster <known-user-id>")
                else:
                    session = CmaSession(client, config, Role.ADJUSTER, args[0])
                    print(f"Adjuster session started for '{args[0]}' (session {session.session_id}).")
            elif cmd == "/insurer":
                if not config.get("environment_id"):
                    print("Run /setup first.")
                elif not args or tools.find_insurer(args[0]) is None:
                    print("Usage: /insurer <known-insurer-id>")
                else:
                    session = CmaSession(client, config, Role.INSURER, args[0])
                    print(f"Insurer session started for '{args[0]}' (session {session.session_id}).")
            elif cmd == "/agent":
                if not config.get("environment_id"):
                    print("Run /setup first.")
                else:
                    print(run_agent_maintenance(client, config))
            else:
                print(f"Unknown command '{cmd}'. Type /help.")
            continue

        if session is None:
            print("Start a session first: /adjuster <name> or /insurer <id>.")
            continue
        try:
            print(session.send(line))
        except Exception as exc:
            print(f"[error: {type(exc).__name__}: {exc}]")


if __name__ == "__main__":
    main()
