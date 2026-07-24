"""Claude agent memory: per-adjuster overrides for the auto-approval policies.

Stores, in ``data/agent_memory.json``, a small record per adjuster that can
override fields of the default ``DynamicPolicies`` (the auto-approve ceilings
and the agent override flag) plus free-text notes the agent should keep in
mind. When an adjuster assumes their role in the REPL/agent, their effective
policies are the defaults merged with these overrides.

Shape on disk::

    {
      "jaime": {
        "policy_overrides": {"vip_auto_approve": 800},
        "notes": "Cascadia region: be generous with long-time VIP customers."
      }
    }
"""

import json
from typing import Any

import storage
from data_entities import DynamicPolicies

AGENT_MEMORY_FILE = storage.DATA_DIR / "agent_memory.json"


def load_agent_memory() -> dict[str, Any]:
    """Load the whole agent-memory store, keyed by adjuster id (empty if absent)."""
    if not AGENT_MEMORY_FILE.exists():
        return {}
    with AGENT_MEMORY_FILE.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return raw


def _write_agent_memory(memory: dict[str, Any]) -> None:
    storage.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with AGENT_MEMORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def get_adjuster_memory(adjuster_id: str) -> dict[str, Any]:
    """Return one adjuster's memory record (``{}`` if they have none)."""
    return load_agent_memory().get(adjuster_id, {})


def effective_policies(adjuster_id: str) -> DynamicPolicies:
    """Return the ``DynamicPolicies`` in effect for an adjuster.

    The defaults, with any ``policy_overrides`` from the adjuster's memory
    merged in (unknown keys are ignored by ``DynamicPolicies.coerce``).
    """
    overrides = get_adjuster_memory(adjuster_id).get("policy_overrides", {})
    return DynamicPolicies.coerce(overrides)


def set_policy_override(adjuster_id: str, field: str, value: Any) -> None:
    """Set (or clear, with ``None``) one policy override for an adjuster and save."""
    memory = load_agent_memory()
    record = memory.setdefault(adjuster_id, {})
    overrides = record.setdefault("policy_overrides", {})
    if value is None:
        overrides.pop(field, None)
    else:
        overrides[field] = value
    _write_agent_memory(memory)


def set_notes(adjuster_id: str, notes: str) -> None:
    """Set the free-text notes for an adjuster and save."""
    memory = load_agent_memory()
    memory.setdefault(adjuster_id, {})["notes"] = notes
    _write_agent_memory(memory)
