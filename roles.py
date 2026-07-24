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
from enum import Enum
from typing import Any, Callable

import agent_memory
import agent_schemas
import tools


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
        "process_incident", "escalate_incident", "update_resolution", "update_status",
        # notify
        "notify_adjuster", "notify_insurer", "notify_update",
        # log
        "append_incident_history", "set_incident_history",
        "append_insurer_history", "append_insurer_preferences",
        "append_adjuster_history", "append_adjuster_preferences",
    },
    Role.INSURER: {
        "create_incident", "get_incident_details", "list_incidents_for_insurer",
        "find_policy", "find_insurer",
        "append_insurer_history", "append_insurer_preferences",
    },
    Role.AGENT: {
        "close_stale_resolved", "list_incidents", "get_incident_details", "update_status",
        "process_incident", "assign_incident",
        "find_adjuster", "list_adjusters", "get_adjuster_details",
    },
}


def tools_for_role(role: Role) -> list[Callable[..., Any]]:
    """Return the ``AGENT_TOOLS`` functions this role may call."""
    allowed = ROLE_TOOLS[role]
    return [fn for fn in tools.AGENT_TOOLS if fn.__name__ in allowed]


def schemas_for_role(role: Role) -> list[dict]:
    """Return the Anthropic tool schemas for the tools this role may call."""
    allowed = ROLE_TOOLS[role]
    return [s for s in agent_schemas.build_tool_schemas() if s["name"] in allowed]


def dispatch_table(role: Role) -> dict[str, Callable[..., Any]]:
    """Return a name -> function map restricted to this role's allowed tools.

    A tool call for a name not in the table must be refused — this is the
    enforcement point that mirrors the schema filtering.
    """
    return {fn.__name__: fn for fn in tools_for_role(role)}


# --- system prompts ---------------------------------------------------------

_ADJUSTER_PROMPT = """You are an AI assistant acting on behalf of insurance adjuster \
{name} (id: "{adjuster_id}"). You help this adjuster manage their claims.

You can ONLY take adjuster actions, and only for this adjuster. Use the tools to:
- list and inspect this adjuster's incidents and escalations,
- approve or deny incidents (individually or in bulk when asked, e.g. "approve all"),
- escalate incidents, update status, and notify the adjuster and policyholder,
- log history and notes on incidents, insurers, and this adjuster.

Auto-approval policies for this adjuster (from agent memory) are:
{policies_json}
When you call process_incident or escalate_incident, pass this exact object as the
`policies` argument so the correct ceilings and any agent override apply.
{notes}
Whether a policyholder is "upset" is your judgement — read the incident and the
insurer's history and pass insurer_upset=true/false to process_incident accordingly.

Always confirm before denying a claim or approving many at once. Report what you did
concisely."""

_INSURER_PROMPT = """You are an AI assistant speaking directly with policyholder \
{name} (insurer id: "{insurer_id}"). You represent the insurance company to this customer.

You can ONLY do two things for this policyholder:
1. Help them file a new incident/claim. Have a natural conversation to gather what
   create_incident needs: which policy (use find_policy / find_insurer to confirm their
   policies), the type(s) of loss, a description, and an estimated cost. New claims are
   routed to intake adjuster "{intake_adjuster}" (use that as adjuster_id) and this
   policyholder's insurer_id "{insurer_id}". Confirm the details with the customer before
   creating the claim.
2. Answer questions about the status of THIS policyholder's claims (use
   list_incidents_for_insurer / get_incident_details).

Never approve, deny, escalate, or take adjuster actions — you don't have those tools.
Never reveal or discuss other policyholders' claims. Be warm and clear."""

_AGENT_PROMPT = """You are the automated maintenance agent for the claims system.
You run unattended, so never ask for confirmation — just do the work and report it.

Your tasks each run:

1. Triage and route incidents. Use list_incidents("unassigned") to find incidents
   that haven't been routed yet. For each incident still awaiting a decision, call
   process_incident with a policies object that enables assignment, e.g.
   {"can_assign": true}. With can_assign set, process_incident will:
     - auto-approve the claim if it's within the approval ceilings; otherwise
     - route an unassigned incident to an adjuster whose authorization_level fits
       the claim cost (using the policy's authorization_low / authorization_medium
       / authorization_high limits); or
     - escalate an incident that already has an adjuster to that adjuster.
   Routing only assigns an adjuster — it does not decide the claim in the same
   step, so call process_incident again (with the same policies) on an incident it
   just assigned so it gets approved or escalated. If process_incident leaves an
   incident unassigned (its cost exceeds every authorization limit), report it as
   needing manual routing. You may override the authorization_* limits in the
   policies object when a run needs different cost bands.

2. Close stale claims. Use close_stale_resolved to find and close claims that have
   been resolved (approved or denied) for more than 30 minutes.

Report concisely what you routed, triaged, and closed."""


def build_system_prompt(role: Role, identity_id: str, extra: str = "") -> str:
    """Build the seed system prompt for a role + identity.

    Args:
        role: The active role.
        identity_id: The adjuster user_id or insurer id assumed for the session.
        extra: Extra instructions to append (e.g. tone) — supplied by the caller.

    Returns:
        The system prompt string.
    """
    if role is Role.ADJUSTER:
        adjuster = tools.find_adjuster(identity_id)
        name = (adjuster.full_name if adjuster else None) or identity_id
        policies = agent_memory.effective_policies(identity_id)
        policies_json = json.dumps(_policies_dict(policies), indent=2)
        note_text = agent_memory.get_adjuster_memory(identity_id).get("notes", "")
        notes = f"\nMemory note for this adjuster: {note_text}\n" if note_text else ""
        prompt = _ADJUSTER_PROMPT.format(
            name=name, adjuster_id=identity_id, policies_json=policies_json, notes=notes
        )
    elif role is Role.INSURER:
        insurer = tools.find_insurer(identity_id)
        name = (insurer.full_name if insurer else None) or identity_id
        prompt = _INSURER_PROMPT.format(
            name=name, insurer_id=identity_id, intake_adjuster=DEFAULT_INTAKE_ADJUSTER
        )
    else:
        prompt = _AGENT_PROMPT

    return f"{prompt}\n\n{extra.strip()}" if extra.strip() else prompt


def _policies_dict(policies) -> dict[str, Any]:
    """The DynamicPolicies fields as a plain dict for the prompt / tool argument."""
    from dataclasses import asdict
    return asdict(policies)
