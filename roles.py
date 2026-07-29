"""Role-based access control and system-prompt seeding.

This is the single place that defines *what each role is allowed to do*. Both
the interactive REPL and the agent use it, so the two can never drift: the same
allow-list filters the tool schemas sent to Claude and the dispatch table the
REPL executes against.

Roles:
    ADJUSTER — an insurance adjuster (identified by their user_id). Manages
        their incidents and escalations: approve/deny, escalate, notify, log.
        Their auto-approval policies come from agent memory (see
        ``agent_memory``).
    INSURER  — a policyholder (identified by their insurer id). Can only file a
        new claim (conversationally) and ask about the status of their own
        claims.
    AGENT    — the scheduled/maintenance persona. Closes claims that have been
        resolved for a while.

The tool allow-lists below reference functions by name from
``tools.AGENT_TOOLS``.
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from enum import Enum
from typing import Any, Callable

import agent_memory
import agent_schemas
import prompts
import tools
from data_entities import DynamicPolicies


class Role(str, Enum):
    ADJUSTER = "adjuster"
    INSURER = "insurer"
    AGENT = "agent"


# Default adjuster an insurer-filed claim is routed to for triage.
DEFAULT_INTAKE_ADJUSTER = "unassigned"

# Which tools (by function name) each role may call.
ROLE_TOOLS: dict[Role, set[str]] = {
    Role.ADJUSTER: {
        # read / discover
        "list_incidents", "get_incident_details", "get_adjuster_details",
        "find_adjuster", "find_insurer", "find_policy", "list_escalations",
        # decide
        "process_incident", "escalate_incident", "escalate_to_management",
        "update_resolution", "update_status",
        # notify
        "notify_adjuster", "notify_insurer", "notify_update",
        # log
        "append_incident_history", "set_incident_history",
        "append_insurer_history", "append_insurer_preferences",
        "append_adjuster_history", "append_adjuster_preferences",
    },
    Role.INSURER: {
        "create_incident", "get_incident_details", "list_incidents_for_insurer",
        "find_policy", "retrieve_policies", "find_insurer",
        "append_insurer_history", "append_insurer_preferences",
    },
    Role.AGENT: {
        # AGENT only. It wipes and regenerates the whole claim set, which is a
        # setup operation, not claims work — an adjuster must never be able to
        # discard the claims it is being asked to handle.
        "generate_unassigned_incidents",
        "close_stale_resolved", "list_incidents", "get_incident_details", "update_status",
        "process_incident", "assign_incident", "escalate_to_management",
        "find_adjuster", "list_adjusters", "get_adjuster_details",
        "retrieve_policies",
    },
}


# Roles a session must *also* serve because it can delegate to subagents of that
# role. Only the Claude Agent SDK backend can delegate; the direct-API and
# Managed Agents paths ignore this and stay single-role.
#
# The consequence is worth stating plainly: an AGENT session's tool surface is
# the union of AGENT and ADJUSTER, because a subagent's tools are resolved
# against the one MCP server the session registers. Which tools each *agent*
# within that session may call is then narrowed per-agent — see
# ``repl._adjuster_agents``. That moves part of the enforcement from "the
# function is not in the dispatch table" to "the tool is not on this agent's
# allow-list"; ``_dispatch`` still refuses anything outside the table.
DELEGATES_TO: dict[Role, tuple[Role, ...]] = {
    Role.AGENT: (Role.ADJUSTER,),
}

# A role, or several — every tool lookup below accepts either.
RoleSpec = Role | Iterable[Role]


def session_roles(role: Role) -> tuple[Role, ...]:
    """The role itself plus any role it can delegate to (see ``DELEGATES_TO``)."""
    return (role, *DELEGATES_TO.get(role, ()))


def allowed_tool_names(spec: RoleSpec) -> set[str]:
    """Union of the tool names the given role (or roles) may call."""
    if isinstance(spec, Role):
        return set(ROLE_TOOLS[spec])
    return {name for role in spec for name in ROLE_TOOLS[role]}


def tools_for_role(spec: RoleSpec) -> list[Callable[..., Any]]:
    """Return the ``AGENT_TOOLS`` functions this role (or roles) may call."""
    allowed = allowed_tool_names(spec)
    return [fn for fn in tools.AGENT_TOOLS if fn.__name__ in allowed]


def schemas_for_role(spec: RoleSpec) -> list[agent_schemas.ToolSchema]:
    """Return the Anthropic tool schemas for the tools this role (or roles) may call."""
    allowed = allowed_tool_names(spec)
    return [s for s in agent_schemas.build_tool_schemas() if s["name"] in allowed]


def dispatch_table(spec: RoleSpec) -> dict[str, Callable[..., Any]]:
    """Return a name -> function map restricted to the allowed tools.

    A tool call for a name not in the table must be refused — this is the
    enforcement point that mirrors the schema filtering. Passing several roles
    widens the table to their union, which is what a session that delegates to
    subagents needs (see ``session_roles``).
    """
    return {fn.__name__: fn for fn in tools_for_role(spec)}


# --- system prompts ---------------------------------------------------------

def build_system_prompt(role: Role, identity_id: str | None = None, extra: str = "",
                        delegate_agents: Sequence[tuple[str, str]] = (), *,
                        hosted: bool = False,
                        memory_mount: str | None = None) -> str:
    """Assemble the system prompt for a role, from the text in ``prompts``.

    The single seam every front-end goes through. Wording lives in ``prompts``;
    what this owns is which blocks apply and in what order.

    Args:
        role: The active role.
        identity_id: The adjuster user_id or insurer id. ``None`` builds the
            identity-free body — which is what a *stored* Managed Agents agent
            needs, since one agent object serves every identity and the identity
            arrives later as a per-session override.
        extra: Extra instructions to append (e.g. tone), supplied by the caller.
        delegate_agents: ``(agent_name, one_line_description)`` pairs for the
            subagents this session actually registered. Only the AGENT prompt
            uses them, and only when non-empty — a backend with no subagents must
            not be told to delegate to agents that do not exist.
        hosted: True on the Managed Agents backend. Selects the memory-store
            block and the roster mechanics that match how spawning works there.
            Blocks that do not apply are omitted rather than accompanied by an
            instruction to ignore them.
        memory_mount: Where the adjuster's memory store is mounted. Only used
            when ``hosted``; falls back to the store root if not given.

    Returns:
        The system prompt string.
    """
    parts: list[str] = []

    if role is Role.ADJUSTER:
        # With an identity we can use that adjuster's effective policies; without
        # one the stored agent gets the defaults and merges its own overrides.
        policies = (agent_memory.effective_policies(identity_id) if identity_id
                    else DynamicPolicies())
        parts.append(prompts.ADJUSTER_BODY.format(
            policies_json=json.dumps(_policies_dict(policies), indent=2)))
        if hosted:
            parts.append(prompts.ADJUSTER_MEMORY_BLOCK.format(
                memory_mount=memory_mount or "/mnt/memory/"))
        if identity_id:
            adjuster = tools.find_adjuster(identity_id)
            name = (adjuster.full_name if adjuster else None) or identity_id
            parts.append(prompts.ADJUSTER_IDENTITY.format(
                name=name, adjuster_id=identity_id))
            note = agent_memory.get_adjuster_memory(identity_id).get("notes", "")
            if note:
                parts.append(f"Memory note for this adjuster: {note}")

    elif role is Role.INSURER:
        parts.append(prompts.INSURER_BODY.format(
            intake_adjuster=DEFAULT_INTAKE_ADJUSTER))
        if identity_id:
            insurer = tools.find_insurer(identity_id)
            name = (insurer.full_name if insurer else None) or identity_id
            parts.append(prompts.INSURER_IDENTITY.format(
                name=name, insurer_id=identity_id))

    else:
        parts.append(prompts.AGENT_BODY)
        if delegate_agents:
            roster = "\n".join(f"- {name} — {desc}" for name, desc in delegate_agents)
            mechanics = (prompts.DELEGATION_MECHANICS_HOSTED if hosted
                         else prompts.DELEGATION_MECHANICS_SDK)
            parts.append(prompts.AGENT_DELEGATION.format(
                roster=roster, mechanics=mechanics))

    if extra.strip():
        parts.append(extra.strip())
    return "\n\n".join(parts)


def _policies_dict(policies: DynamicPolicies) -> dict[str, Any]:
    """The DynamicPolicies fields as a plain dict for the prompt / tool argument."""
    return asdict(policies)
