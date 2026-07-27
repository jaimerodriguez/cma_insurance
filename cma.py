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

import argparse
import dataclasses
import json
import os
import time
from collections.abc import Sequence
from typing import Any

import agent_obs
import agent_schemas
import roles
import storage
import tools
from data_entities import DynamicPolicies
# Reuse the role-restricted dispatcher + serializer. `_dispatch` is already
# decorated with agent_obs.traced_dispatch there, so every custom-tool call this
# front-end services is traced with no extra wiring here.
from repl import _dispatch, _jsonify
from roles import Role

# Per-role model: cheaper/faster models for the narrower roles, the most capable
# model for the autonomous maintenance agent.
MODEL_BY_ROLE = {
    Role.INSURER: "claude-haiku-4-5",
    Role.ADJUSTER: "claude-haiku-4-5",
    Role.AGENT: "claude-sonnet-5",

    # "claude-opus-5" 
    # "claude-opus-4-8" 
}
CONFIG_FILE = storage.DATA_DIR / "cma_config.json"

# Read timeout for the session event stream, in seconds.
#
# A backstop, not the mechanism: the loop in `CmaSession.send` now stops reading
# as soon as the session says it is blocked on us, so a healthy turn never waits
# on this. It exists because the SDK's default is `httpx.Timeout(timeout=10*60)`
# (anthropic/_constants.py), and a stream that goes quiet for any *other* reason
# should cost seconds, not ten minutes.
#
# It has to clear the longest legitimate gap between events — the model can think
# for minutes without emitting — so this is deliberately generous. Tighten it only
# with evidence from the event log; if it ever fires, that is a bug to chase, not
# a number to lower.
STREAM_TIMEOUT_S = 300.0

# Adjusters that get a personal memory store provisioned at setup.
MEMORY_ADJUSTERS = ["jaime", "sam", "jane"]

# Append your own tone / house-style here — added to every agent's system prompt.
SYSTEM_PROMPT_EXTRA = ""

# Seed content for an adjuster's memory store (only jaime ships with overrides).
_SEED_MEMORY = {
    "jaime": (
        "# Auto-approval policy overrides (Cascadia region)\n\n"
        "Apply these on top of the baseline DynamicPolicies:\n"
        "- vip_auto_approve: 60000\n\n"
        "## Standing guidance\n"
        "- Lean generous with long-time VIP customers.\n"
        "- Always escalate suspected fraud regardless of cost.\n"
    ),
    "jane": ( 
        "Your auto approve (auto_approve) limit is 30000\n\n" , 
        "Your VIP auto approve (vip_auto_approve) limit is 80000\n\n" , 
        "Your default authorization limit is 75000 \n\n" ,  
        "Be diligent for fraud\n\n"
    )
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


# _INSURER_SYSTEM_PROMPT. Params: {policies_json}
_ADJUSTER_SYSTEM = """
You are an AI assistant acting on behalf of an insurance adjuster.
When you are invoked, you will always be told what your name is and your adjuster_id. 
If you are invoked without these, don't do anything as you will not know on whose behalf you need to act.  

You can ONLY take adjuster actions, and only for this adjuster. Use the tools to:
- list and inspect this adjuster's incidents and escalations,
- approve or deny incidents (individually or in bulk when asked, e.g. "approve all"),
- escalate incidents, update status, and notify the  policyholder,
- hand a claim up to management for any reason (escalate_to_management with a
  reason) — use this when the decision shouldn't be this adjuster's, e.g. a
  conflict of interest, threatened litigation, or an ambiguous exclusion,
- log history and notes on incidents, insurers, and this adjuster.

The default auto-approval policies for your adjuster (from agent memory) are:
{policies_json}, but these can be overridden by persistent memory. 

<managed_agent_instructions> 
The following instructions apply to when you run as a managed agent. Ignore them otherwise. 
You have a persistent memory store mounted under /mnt/memory/. Look for standing
auto-approval policy overrides and working notes.  ALWAYS read it before acting (use the
read/glob tools), and write updates back (write/edit) when you learn a lasting rule.
</managed_agent_instructions> 

Merge any overrides from your memory onto the baseline and use the resulting object as
the `policies` argument to approve_incident / escalate_incident. 

Whether a policyholder is "upset" is your judgement — read the incident and the insurer's history and set
insurer_upset accordingly. Confirm before denying a claim or approving many at once.

Be thorough in your resolution reasons. Specially when declining claims. 
"""
 

# _INSURER_SYSTEM_PROMPT . Params: None 
_INSURER_SYSTEM = """You are an AI assistant speaking directly with policyholder (a.k.a. insurer). 
Your goal is to do two things for this policyholder:
1. Help them file a new incident/claim. Have a natural conversation to gather what
   create_incident needs: which policy (use find_policy / find_insurer to confirm their
   policies), the type(s) of loss, a description, and an estimated cost.  
   To file a claim, you need the insurer_id, and teh details. Confirm these details with the customer before
   creating the claim.

2. Answer questions about the status of THIS policyholder's claims (use
   list_incidents_for_insurer / get_incident_details).

You should be helpful, for example, if they don't know their policy, look it up for them. 
If they want to file a claim and only have one policy, use that as the policy_id. Just confirm it before you use it. 

Do not get creative asking for police reports or similar tasks. Stay factual. 
If the insurer sounds frustrated, note it down in both the created Incident's history and the Insurer's history.  
Never approve, deny, escalate, or take adjuster actions — you don't have those tools.
Never reveal or discuss other policyholders' claims. Be warm and clear."""


# _AGENT_SYSTEM_PROMPT . Params: None 
_AGENT_SYSTEM = """You are an agent processing insurance claims.   
Your job is to triage unassigned claims and claims the adjusters could not resolve (status == ESCALATED_TO_MANAGEMENT or authorization_level == OVERRIDE_NEEDED). 

## Routing and resolving  unassigned claims. 
- You can deny any claims when they do not have enough information, or they are not in the right category.   
- You can automatically approve claims that are within your authorization budgets and within the policies'' coverage.  You have budget variance based on whether the customer is upset, or whether they are a VIP. Do not auto approve claims that are outside of policies. Escalate those to management instead (status == ESCALATED_TO_MANAGEMENT ). Add the reason it was escalated. 
- You must find out if a customer (or Insurer) is upset by looking at the claim''s history. Consider this for your decisions.  

## delegating claims to adjusters. 
- For claims that are over your authorization limits and have not been escalated, you can assign them to adjusters with higher levels. Check the adjuster''s authorization level before you assign a claim to them.  
- If a claim is above all adjuster's authorization levels, then set its status to ESCALATED_TO_MANAGEMENT and set its' authorization_level to OVERRIDE_NEEDED.  

- Process all the claims that you can, and delegate all the other claims first.  
- Once you have delegated the claims, call the adjuster subagent once for each adjuster and instruct them to process their claims.  When you invoke an adjuster, you MUST tell it what name and adjuster_id they need to act on behalf of. You can call these adjuster subagents sequentially (not in parallel). 

For escalated claims: 
- As an agent, you will only resolve the escalated claim in when the approval code ("UNICORN") is provided. 
- When the approval code is mentioned to you resolve the escakated claims regardless of elevation or approval limits, use your judgement to approve or deny them, just ignore the limits. 
- You can prompt for the code, but never share it. The code is a secret for you to validate. Don''t share it with anyone. 

## closing claims 
A different part of your job is to close out claims that have been resolved (approved or denied) for a while: call
close_stale_resolved to find and close claims resolved more than 30 minutes ago. 

## output or results. 
As you work, keep track of the claims you processed and report the outcome or action taken for each.  
When you close a claim where the insurer was upset, update them with the resolution for that claim so we know if they will be upset next time. If they were happy with the resolution, remove any notes about prior dissatisfaction. 

  
 
Once a claim has an adjuster, the decision is that adjuster''s; delegate it rather than resolving it yourself.  
After you are done triaging and routing to adjusters, 
invoke each adjuster via their own subagent in a single task, not one task per claim,
and give it the claim ids plus anything you learned that they might not infer directly from the claim. 
 
Delegate for: approving, denying, escalating to management, notifying, and logging
notes on a claim that has an assigned adjuster.

Do NOT delegate for: routing and assignment (yours), close_stale_resolved (yours),
or a claim that is already resolved. Never send a claim to an adjuster other than
the one it is assigned to.

Launch the subagents you need in a single message, but don''t run adjusters concurrently; run them sequentially.
Then, collect their reports and fold them into your own summary; say what each adjuster decided.

How delegation actually works: you have a subagent roster, and spawning from it is
a built-in capability of this session — not a tool you call. There is no
create_agent and no list_agents tool; calling one fails and wastes a turn.
The roster holds ONE adjuster agent, shared by every adjuster, and it has no
identity of its own. So each delegated task MUST open by naming the adjuster it is
to act as — their full name and adjuster_id — followed by the claim ids and the
context. The adjuster agent is instructed to refuse the work outright if you leave
those out, so a task without them is a wasted round trip.
 
"""


def _agent_system(role: Role) -> str:
    if role is Role.ADJUSTER:
        # Keyword must match the placeholder in _ADJUSTER_SYSTEM: `str.format`
        # raises KeyError for a placeholder it was not given, so a rename on one
        # side alone breaks every adjuster session at prompt-build time.
        base = _ADJUSTER_SYSTEM.format(
            policies_json=json.dumps(dataclasses.asdict(DynamicPolicies()), indent=2)
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
    repeatedly, only missing resources are created.
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
        if role.value in config["agents"] :
            continue
        else:              
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


# --- agent updates ----------------------------------------------------------
#
# `setup()` only ever *creates* what is missing, so once an agent exists it never
# picks up later edits to the prompts in this module or to the tool set in
# `roles.py`. These helpers close that gap: they locate the live agent for each
# role and push the code-defined configuration onto it.
#
# The code is the source of truth here — nothing reads config back off the server
# and merges it. An update replaces `system`, `model`, and `tools` wholesale
# (array fields in particular are replaced, not merged), which is exactly what a
# "apply what the repo says" pass wants.

def _agent_name(role: Role) -> str:
    """Deterministic agent name per role — also how an agent is found without an id."""
    return f"insurance-{role.value}"


def _multiagent_config(role: Role, config: dict,
                       versions: dict[str, int] | None = None) -> dict[str, Any] | None:
    """The coordinator roster for a role, or None when it delegates to nobody.

    Managed Agents expresses delegation as a ``multiagent`` coordinator roster —
    a top-level field on the agent, *not* a ``tools[]`` entry. Without it the
    maintenance agent has no delegation mechanism at all: told to delegate and
    given no way to, it invented ``create_agent`` / ``list_agents`` custom tools,
    which ``_dispatch`` refused as "not permitted for this role". Those names
    were hallucinated, never real tools.

    Which roles delegate to which is ``roles.DELEGATES_TO`` — the same map the
    Agent SDK front-end uses for its subagents, so the two backends cannot drift
    on who is allowed to hand work to whom.

    Roster entries are agent ids, so the delegated-to agents must already exist:
    run ``/setup`` first. A role whose targets are all missing gets no roster
    rather than an empty one, which the API rejects. Depth is capped at 1 — a
    rostered agent must not itself carry a roster — so the adjuster stays a leaf.

    **Roster entries pin a version.** Sending a bare id does not track "latest":
    the API resolves it to ``{"type": "agent", "id", "version"}`` fixed at save
    time. Update the adjuster afterwards and the coordinator keeps spawning the
    *old* one — verified live, and silently, since nothing errors. So when the
    caller knows the current versions (``versions``) they are pinned explicitly,
    which also lets ``agent_drift`` see a stale pin as drift.
    """
    targets = roles.DELEGATES_TO.get(role, ())
    ids = [(config.get("agents") or {}).get(t.value) for t in targets]
    ids = [i for i in ids if i]
    if not ids:
        return None
    entries: list[Any] = [
        {"type": "agent", "id": i, "version": versions[i]}
        if versions and i in versions else i
        for i in ids
    ]
    return {"type": "coordinator", "agents": entries}


def _desired_agent(role: Role, config: dict,
                   versions: dict[str, int] | None = None) -> dict[str, Any]:
    """The agent configuration this codebase defines for a role."""
    return {
        "name": _agent_name(role),
        "model": MODEL_BY_ROLE[role],
        "system": _agent_system(role),
        "tools": agent_tools_for_role(role),
        "multiagent": _multiagent_config(role, config, versions),
    }


def find_agent(client, config: dict, role: Role):
    """The live agent for a role, or ``None`` if it does not exist.

    Tries the id recorded in ``cma_config.json`` first, then falls back to a
    lookup by name. The fallback matters because the config file is local and
    disposable while the agents are not: without it, a lost or hand-edited
    config would silently create a second ``insurance-<role>`` agent rather than
    finding the one already there. Archived agents are excluded — they are
    read-only and cannot be updated or attached to a new session.
    """
    import anthropic

    agent_id = (config.get("agents") or {}).get(role.value)
    if agent_id:
        try:
            return client.beta.agents.retrieve(agent_id)
        except (anthropic.NotFoundError, anthropic.BadRequestError):
            # A stale id 404s; a *malformed* one (hand-edited config, truncated
            # paste) 400s with "Invalid agent ID" rather than 404. Both mean the
            # recorded id is unusable, so both fall through to the name lookup —
            # catching only NotFoundError left the second case crashing.
            pass
    name = _agent_name(role)
    return next((a for a in client.beta.agents.list(limit=100) if a.name == name), None)


def _tools_match(desired: list[dict], live: Any) -> bool:
    """True when every tool we define is present on the agent and matches.

    Compared as a subset rather than for equality, because the API echoes tools
    back with server-side defaults filled in: a prebuilt toolset sent as
    ``{"type": "agent_toolset_20260401"}`` comes back carrying ``default_config``
    and ``configs``. Demanding exact equality would report "changed" on every
    run and mint a new agent version each time. The key sets are still compared
    exactly, so an added or removed tool is caught.
    """
    def key(tool: dict) -> tuple:
        return (tool.get("type"), tool.get("name"))

    live_by = {key(t): t for t in
               (x.model_dump(exclude_none=True) for x in (live or []))}
    want_by = {key(t): t for t in desired}
    if set(live_by) != set(want_by):
        return False
    return all(live_by[k].get(field) == value
               for k, want in want_by.items()
               for field, value in want.items())


def _roster_refs(multiagent: Any) -> list[tuple[str, int | None]]:
    """``(agent_id, pinned_version)`` for each roster entry, sorted.

    Entries may be a bare id string, a ``{"type": "agent", "id", "version"}``
    reference, or ``{"type": "self"}``; the API resolves them to concrete
    references, so what comes back is not the shape that went out. Normalising
    to id + version keeps the comparison stable across forms while still
    catching a pin left behind on a superseded version. A bare id carries no
    version, which compares equal to anything via ``_refs_match``.
    """
    if multiagent is None:
        return []
    entries = getattr(multiagent, "agents", None)
    if entries is None and isinstance(multiagent, dict):
        entries = multiagent.get("agents")
    refs: list[tuple[str, int | None]] = []
    for e in entries or []:
        if isinstance(e, str):
            refs.append((e, None))
        elif isinstance(e, dict):
            if e.get("id"):
                refs.append((e["id"], e.get("version")))
        elif getattr(e, "id", None):
            refs.append((e.id, getattr(e, "version", None)))
    return sorted(refs)


def _roster_ids(multiagent: Any) -> list[str]:
    """Just the agent ids in a coordinator roster."""
    return [i for i, _ in _roster_refs(multiagent)]


def _refs_match(live: list[tuple[str, int | None]],
                want: list[tuple[str, int | None]]) -> bool:
    """Compare rosters, treating an unpinned entry as matching any version."""
    if [i for i, _ in live] != [i for i, _ in want]:
        return False
    return all(lv == wv or wv is None or lv is None
               for (_, lv), (_, wv) in zip(live, want))


def agent_drift(agent: Any, role: Role, config: dict,
                versions: dict[str, int] | None = None) -> list[str]:
    """Names of the code-defined fields that differ from the live agent.

    ``versions`` maps agent id -> current version. Supplying it lets a roster
    pinned to a superseded version count as drift; without it only membership is
    compared, so a coordinator left pointing at an old subagent looks in sync.
    """
    desired = _desired_agent(role, config, versions)
    drift = []
    if agent.system != desired["system"]:
        drift.append("system")
    if not _tools_match(desired["tools"], agent.tools):
        drift.append("tools")
    # `model` comes back as an object (id + server-filled effort/speed), so only
    # the id is ours to compare.
    if getattr(agent.model, "id", agent.model) != desired["model"]:
        drift.append("model")
    if agent.name != desired["name"]:
        drift.append("name")
    if not _refs_match(_roster_refs(getattr(agent, "multiagent", None)),
                       _roster_refs(desired["multiagent"])):
        drift.append("multiagent")
    return drift


def update_agents(client, config: dict, only: Sequence[Role] | None = None,
                  force: bool = False) -> list[dict[str, Any]]:
    """Push this module's system prompts, tools, and model onto each role's agent.

    ``version`` is deliberately *not* sent. Supplying it asks the API for
    optimistic concurrency and returns 409 on any mismatch — including when the
    fields you are sending already equal the stored ones — which is the wrong
    behaviour for a declarative apply that owns these agents outright. Omitting
    it is an unconditional last-write-wins update.

    Every update mints a new immutable agent version; the previous ones remain
    retrievable. Sessions pin their version at creation, so a running session is
    unaffected and the change lands on the *next* session for that role — no need
    to restart anything, but an open ``/adjuster`` session keeps the old prompt.

    Args:
        client: The Anthropic client.
        config: The config dict from ``_load_config`` / ``setup``. Updated in
            place (and persisted) when an agent is located by name rather than id.
        only: Restrict to these roles. Defaults to every role.
        force: Send the update even when nothing looks different — useful when
            you suspect the local drift check is wrong rather than the agent.
            It is safe to leave on: the API is itself idempotent for identical
            content (a forced no-op update returns the same version rather than
            minting a new one), so this costs a request, not version churn. Off
            by default only to keep a repeated run silent.

    Returns:
        One record per role: ``{role, agent_id, action, changed, version}`` where
        ``action`` is ``updated`` / ``unchanged`` / ``missing``.
    """
    results: list[dict[str, Any]] = []
    # Current version per agent id, filled in as we go and consulted when a
    # coordinator's roster is built. Delegation *targets* are processed first so
    # that by the time a coordinator's roster is pinned, the agent it points at
    # has already been brought up to date — otherwise a single pass would pin the
    # version it is about to supersede.
    versions: dict[str, int] = {}
    scope = sorted(only or list(Role), key=lambda r: bool(roles.DELEGATES_TO.get(r)))

    for role in scope:
        agent = find_agent(client, config, role)
        if agent is None:
            results.append({"role": role.value, "agent_id": None,
                            "action": "missing", "changed": [], "version": None})
            continue

        # Self-heal the local config when the agent was found by name.
        if (config.setdefault("agents", {})).get(role.value) != agent.id:
            config["agents"][role.value] = agent.id
            _save_config(config)

        versions.setdefault(agent.id, agent.version)
        # A coordinator whose target is outside `only` still needs that target's
        # current version, or a stale pin goes unnoticed. One extra read.
        for target in roles.DELEGATES_TO.get(role, ()):
            target_id = (config.get("agents") or {}).get(target.value)
            if target_id and target_id not in versions:
                found = find_agent(client, config, target)
                if found is not None:
                    versions[found.id] = found.version

        drift = agent_drift(agent, role, config, versions)
        if not drift and not force:
            results.append({"role": role.value, "agent_id": agent.id,
                            "action": "unchanged", "changed": [],
                            "version": agent.version})
            continue

        desired = _desired_agent(role, config, versions)
        fields: dict[str, Any] = {
            "name": desired["name"],
            "model": desired["model"],
            "system": desired["system"],
            "tools": desired["tools"],
        }
        # Only sent when this role has a roster: omitted fields are preserved,
        # whereas an explicit None would clear a roster that is already there.
        if desired["multiagent"] is not None:
            fields["multiagent"] = desired["multiagent"]
        updated = client.beta.agents.update(agent.id, **fields)
        versions[agent.id] = updated.version   # later rosters pin the fresh one
        agent_obs.current().events.info(
            "cma.agent_updated", role=role.value, agent_id=agent.id,
            changed=drift or ["(forced)"],
            version_from=agent.version, version_to=updated.version,
        )
        results.append({"role": role.value, "agent_id": agent.id,
                        "action": "updated", "changed": drift or ["(forced)"],
                        "version": updated.version})
    return results


# --- session runtime --------------------------------------------------------

class CmaSession:
    """Drives one Managed Agents session for a role/identity."""

    def __init__(self, client, config: dict, role: Role, identity: str) -> None:
        self.client = client
        self.config = config
        self.role = role
        self.identity = identity
        # The union of this role's tools and those of any role it delegates to.
        # A subagent's custom-tool calls are cross-posted to the primary thread
        # and dispatched here, against this one table — so an AGENT session whose
        # table held only AGENT tools would refuse every adjuster tool its own
        # subagent called. Each *agent* still declares only its own tools; this
        # widens what the client will execute, not what any agent may ask for.
        self.table = roles.dispatch_table(roles.session_roles(role))
        self.session_id = self._create_session()

    def _memory_resources(self) -> list[dict[str, Any]]:
        """Memory stores to mount for this session.

        An adjuster session mounts that adjuster's own store. A session whose
        agent delegates to adjusters mounts *all* of them: threads share the
        container, memory stores can only be attached at session-create time,
        and the adjuster prompt tells the model to read ``/mnt/memory`` before
        acting — so without this a delegated adjuster thread finds an empty
        mount and silently works off baseline policy. Capped at 8 per session by
        the API, which the three-adjuster roster is comfortably under.
        """
        stores = self.config.get("memory_stores") or {}
        if self.role is Role.ADJUSTER:
            wanted = [(self.identity, stores.get(self.identity))]
        elif Role.ADJUSTER in roles.DELEGATES_TO.get(self.role, ()):
            wanted = sorted(stores.items())
        else:
            return []
        return [
            {"type": "memory_store", "memory_store_id": store_id,
             "access": "read_write",
             "instructions": f"Adjuster {adjuster_id}'s memory: auto-approval policy "
                             "overrides and working notes. Read before acting as them."}
            for adjuster_id, store_id in wanted if store_id
        ]

    def _create_session(self) -> str:
        resources = self._memory_resources()
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

    def _tool_result(self, call: Any) -> dict[str, Any]:
        """Build the ``user.custom_tool_result`` for one custom-tool call.

        In a multiagent session a subagent's tool call is cross-posted to the
        primary thread — which is why watching this one stream is still enough —
        carrying a ``session_thread_id`` identifying the thread it came from. The
        result has to echo that id back or it is not routed to the subagent
        waiting on it. It is absent for the coordinator's own calls, so it is only
        included when present rather than sent as null.
        """
        result = {
            "type": "user.custom_tool_result",
            "custom_tool_use_id": call.id,
            "content": [{"type": "text",
                         "text": _dispatch(self.table, call.name, call.input)}],
        }
        thread_id = getattr(call, "session_thread_id", None)
        if thread_id:
            result["session_thread_id"] = thread_id
        return result

    def send(self, user_text: str) -> str:
        """Send a user turn, service custom-tool calls, and return the agent''s reply.

        Traced as one ``turn`` span per user message covering every stream leg. The
        session events are the only usage source on this backend - there is no
        ``ResultMessage`` - so ``session.usage``-shaped events are read defensively
        and a run with none simply records zeros (the wire log still has the truth).
        """
        obs = agent_obs.current()
        to_send: list[dict] | None = [
            {"type": "user.message", "content": [{"type": "text", "text": user_text}]}
        ]
        text_parts: list[str] = []
        terminated = False
        started = time.monotonic()
        usage: Any = None
        legs = 0

        with obs.turn("chat", backend="cma", role=self.role.value,
                      identity=self.identity, session_id=self.session_id) as span:
            obs.note_session(self.session_id, backend="cma")
            while True:
                legs += 1
                # Stream-first: open the stream, then send inside it so no early event is missed.
                with self.client.beta.sessions.events.stream(
                        session_id=self.session_id, timeout=STREAM_TIMEOUT_S) as stream:
                    if to_send is not None:
                        self.client.beta.sessions.events.send(session_id=self.session_id, events=to_send)
                        to_send = None
                    tool_calls = []
                    # Event ids the session says it is blocked on, from the most
                    # recent `requires_action` idle. Empty until it tells us.
                    awaiting: set[str] = set()
                    for event in stream:
                        etype = getattr(event, "type", None)
                        # Every event type at debug level: this is the only window
                        # into a hosted loop we do not run ourselves.
                        obs.events.debug("cma.event", event_type=etype)
                        if getattr(event, "usage", None) is not None:
                            usage = event.usage
                        if etype == "agent.message":
                            for block in getattr(event, "content", []) or []:
                                if getattr(block, "type", None) == "text":
                                    text_parts.append(block.text)
                        elif etype == "agent.custom_tool_use":
                            obs.events.info("cma.custom_tool_use", tool=event.name,
                                            tool_use_id=event.id, input=event.input,
                                            # Present only for a subagent's call.
                                            thread=getattr(event, "session_thread_id", None))
                            tool_calls.append(event)
                            # The idle can arrive before the calls it names, so
                            # coverage is re-checked here too — see the
                            # `session.status_idle` branch below.
                            if awaiting and awaiting <= {c.id for c in tool_calls}:
                                break
                        elif etype in ("session.thread_created",
                                       "session.thread_status_idle",
                                       "session.thread_status_terminated"):
                            # Delegation is otherwise invisible: the primary stream
                            # shows a condensed view of subagent activity, not their
                            # individual tool calls. Drill in with
                            # `sessions.threads.events.list(thread_id, ...)`.
                            obs.events.info("cma.thread", event_type=etype,
                                            thread=getattr(event, "session_thread_id", None),
                                            agent_name=getattr(event, "agent_name", None))
                        elif etype in ("agent.thread_message_sent",
                                       "agent.thread_message_received"):
                            obs.events.info(
                                "cma.thread_message", event_type=etype,
                                to=getattr(event, "to_agent_name", None),
                                from_=getattr(event, "from_agent_name", None))
                        elif etype == "session.status_terminated":
                            obs.events.warn("cma.terminated", session_id=self.session_id)
                            terminated = True
                            break
                        elif etype == "session.status_idle":
                            stop = getattr(event, "stop_reason", None)
                            stop_type = getattr(stop, "type", None)
                            blocked_on = list(getattr(stop, "event_ids", None) or [])
                            # Logged at info: without it the events log cannot tell
                            # "idle, waiting on us" from "idle, done" after the fact,
                            # which is what made this stall hard to see.
                            obs.events.info("cma.idle", stop=stop_type,
                                            blocked_on=len(blocked_on))
                            if stop_type != "requires_action":
                                terminated = False
                                break
                            # Idle waiting on *us*. The session emits nothing further
                            # until we answer, so reading on only waits out the
                            # client's socket timeout — which is where ~90 of every
                            # 100 minutes of a run was going. Stop as soon as we hold
                            # every event it named. Resolving fewer than all re-emits
                            # idle with the remainder, so partial batches still work.
                            awaiting = set(blocked_on)
                            if not awaiting or awaiting <= {c.id for c in tool_calls}:
                                # No ids given (older/other shape) falls back to the
                                # previous batch behaviour rather than hanging.
                                break

                if terminated or not tool_calls:
                    break
                to_send = [self._tool_result(call) for call in tool_calls]

            obs.record_turn(span, obs.usage.from_cma_usage(
                usage, session_id=self.session_id,
                wall_ms=int((time.monotonic() - started) * 1000),
                role=self.role.value, identity=self.identity,
                model=MODEL_BY_ROLE[self.role], kind="chat",
                is_error=terminated,
                extra={"stream_legs": legs}))

        return "".join(text_parts).strip()


@agent_obs.trace_callable("cma.maintenance", kind="maintenance")
def run_agent_maintenance(client, config: dict) -> str:
    """Run the maintenance persona as a CMA session and return its report."""
    session = CmaSession(client, config, Role.AGENT, "system")
    return session.send(
        "Process all the unassigned claims. Auto approve as many as you can. Delegate to adjusters as needed, then report back the summary on all actions performed."         
    )


# --- REPL -------------------------------------------------------------------

_HELP = """Commands:
  /setup             provision the environment, agents, and memory stores (run once)
  /update-agents     push this file's system prompts + tools onto the existing
                     agents (add a role to limit it, `--force` to update anyway)
  /adjuster <name>   start an adjuster session (by user_id), e.g. /adjuster jaime
  /insurer <id>      start a policyholder session (by insurer id), e.g. /insurer ins-1001
  /agent             run the maintenance agent now (close stale resolved claims)
  /whoami            show the current role
  /obs               observability status (add `tail [n]` or `stats [by]`)
  /help              show this help
  /quit              exit
Anything else is sent to the current session."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cma.py",
        description="Managed Agents insurance REPL (hosted agent loop).",
    )
    obs_group = parser.add_argument_group("observability")
    obs_group.add_argument(
        "--wire", action="store_true", default=False,
        help="Capture the real API payloads through a local proxy. Covers the "
             "Managed Agents session endpoints (/v1/beta/...) as well as /v1/messages.",
    )
    obs_group.add_argument(
        "--obs-redact", choices=("flow", "strict", "dev", "none"), default=None,
        help="Redaction level for what is written to disk (default: strict).",
    )
    obs_group.add_argument(
        "--obs-wire-tools", choices=("full", "skeleton", "names"), default=None,
        help="How much of each tool definition the wire log keeps: full (default), "
             "skeleton (name, params, required, description length, hash) or names. "
             "Shaped rows are stamped \"_shaped\".",
    )
    obs_group.add_argument(
        "--obs-wire-tools-keep", default=None, metavar="PATTERNS",
        help="Comma-separated fnmatch patterns kept verbatim despite "
             "--obs-wire-tools, e.g. 'mcp__insurance__*'.",
    )
    obs_group.add_argument(
        "--no-collapse-structures", action="store_true", default=False,
        help="In flow redaction, do not collapse a repeated dict/list to a "
             "'seen_node' marker (strings still collapse).",
    )
    obs_group.add_argument("--no-obs", action="store_true", default=False,
                           help="Disable all tracing for this run.")
    return parser.parse_args(argv)


def _load_api_key() -> str:
    """Load ``ANTHROPIC_API_KEY`` from ``.env`` and report which credential is in play.

    Managed Agents is an API-only surface — there is no subscription path like the
    ``claude`` CLI's, so this front-end always needs API credentials. ``.env`` is
    the intended source here, unlike ``repl.py`` where loading it is opt-in behind
    ``--use-key`` (its default backend authenticates through the CLI instead).

    ``load_dotenv`` does not override a variable already exported, so a key in the
    shell still wins over ``.env``. If neither supplies one we say so and continue
    rather than exiting: ``anthropic.Anthropic()`` also resolves an
    ``ant auth login`` profile, and refusing to start would break that path.
    """
    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return ("no ANTHROPIC_API_KEY — falling back to ambient credentials "
                "(`ant auth login` profile, if configured)")
    if not key.startswith("sk-ant-"):
        # A leftover placeholder is worth catching here: an API key outranks the
        # profile in the SDK's resolution order, so it does not fall through to a
        # working profile — it goes out on the wire and comes back as a 401.
        raise SystemExit(
            f"ANTHROPIC_API_KEY does not look like an API key (got {len(key)} chars "
            f"starting {key[:6]!r}). Put a real 'sk-ant-…' key in .env, or unset the "
            "variable to use an `ant auth login` profile instead."
        )
    return "ANTHROPIC_API_KEY from .env / environment"


def main() -> None:
    import anthropic

    args = _parse_args()
    # Before Observability.start: the wire proxy reads ANTHROPIC_BASE_URL at
    # construction, so anything .env sets has to be in the environment first.
    key_source = _load_api_key()
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

    with agent_obs.Observability.start(agent_obs.ObsConfig.from_env(**overrides),
                                      front_end="cma") as obs:
        # Managed Agents traffic is ordinary HTTPS to the Anthropic API, so the same
        # capture proxy that records /v1/messages records the session endpoints too —
        # base_url is all it takes. None when wire capture is off.
        client = (anthropic.Anthropic(base_url=obs.base_url) if obs.base_url
                  else anthropic.Anthropic())
        config = _load_config()
        session: CmaSession | None = None

        print("Managed Agents insurance REPL. Run /setup first if you haven't. /help for commands.")
        print(f"Auth: {key_source}.\n")
        if obs.enabled:
            wire = f", wire -> {obs.base_url}" if obs.base_url else ""
            print(f"Tracing: run {obs.run_id} (redact={obs.config.redact}{wire}). /obs for detail.")
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
                cmd, args_ = parts[0], parts[1:]
                obs.events.info("repl.command", command=cmd, argc=len(args_))
                if cmd in ("/quit", "/exit"):
                    break
                if cmd == "/help":
                    print(_HELP)
                elif cmd == "/obs":
                    from repl import _print_obs   # shared with the local REPL
                    _print_obs(args_)
                elif cmd == "/setup":
                    config = setup(client)
                    obs.events.info("cma.setup", environment=config.get("environment_id"),
                                    agents=list(config.get("agents") or {}))
                    print(f"Setup complete. environment={config['environment_id']}, "
                          f"agents={list(config['agents'])}, memory_stores={list(config['memory_stores'])}")
                elif cmd == "/update-agents":
                    force = "--force" in args_
                    named = [a for a in args_ if not a.startswith("-")]
                    try:
                        only = [Role(a) for a in named] or None
                    except ValueError:
                        print(f"Usage: /update-agents [{'|'.join(r.value for r in Role)}] [--force]")
                        continue
                    for row in update_agents(client, config, only=only, force=force):
                        detail = (f" ({', '.join(row['changed'])}) -> v{row['version']}"
                                  if row["action"] == "updated" else
                                  "" if row["action"] == "unchanged" else
                                  " — run /setup to create it")
                        print(f"  {row['role']:9} {row['action']}{detail}")
                elif cmd == "/whoami":
                    print("No session." if session is None else f"{session.role.value} / {session.identity}")
                elif cmd == "/adjuster":
                    if not config.get("environment_id"):
                        print("Run /setup first.")
                    elif not args_ or tools.find_adjuster(args_[0]) is None:
                        print("Usage: /adjuster <known-user-id>")
                    else:
                        session = CmaSession(client, config, Role.ADJUSTER, args_[0])
                        print(f"Adjuster session started for '{args_[0]}' (session {session.session_id}).")
                elif cmd == "/insurer":
                    if not config.get("environment_id"):
                        print("Run /setup first.")
                    elif not args_ or tools.find_insurer(args_[0]) is None:
                        print("Usage: /insurer <known-insurer-id>")
                    else:
                        session = CmaSession(client, config, Role.INSURER, args_[0])
                        print(f"Insurer session started for '{args_[0]}' (session {session.session_id}).")
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
                obs.events.error("cma.turn_failed", error=f"{type(exc).__name__}: {exc}")
                print(f"[error: {type(exc).__name__}: {exc}]")

        if obs.enabled:
            print(f"Trace written: {obs.paths().get('events', '(events off)')}")


if __name__ == "__main__":
    main()
