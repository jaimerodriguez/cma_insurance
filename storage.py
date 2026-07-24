"""File-backed persistence for the insurance mock.

Reads and writes the domain entities defined in ``data_entities`` to JSON
files under the ``data/`` subfolder, each keyed by the entity's id. This is the
storage layer: it is imported by ``tools.py`` but is not part of the agent tool
surface. Functions here operate on whole collections; ``tools.py`` builds the
per-id lookups and workflow on top.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from data_entities import (
    Adjuster,
    Escalation,
    Incident,
    Insurer,
    Policy,
    escalation_from_dict,
    escalation_to_dict,
    incident_from_dict,
    incident_to_dict,
    policy_from_dict,
)

# JSON stores live in the data/ subfolder next to this module.
DATA_DIR = Path(__file__).parent / "data"
INCIDENTS_FILE = DATA_DIR / "incidents.json"
ADJUSTERS_FILE = DATA_DIR / "adjusters.json"
INSURERS_FILE = DATA_DIR / "insurers.json"
POLICIES_FILE = DATA_DIR / "policies.json"
ESCALATIONS_FILE = DATA_DIR / "escalations.json"


def _dump(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as indented JSON to ``path``, creating data/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --- incidents --------------------------------------------------------------

def load_incidents() -> dict[str, Incident]:
    """Load every incident from disk, keyed by id (empty if the file is absent)."""
    if not INCIDENTS_FILE.exists():
        return {}
    with INCIDENTS_FILE.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return {incident_id: incident_from_dict(d) for incident_id, d in raw.items()}


def _write_incidents(incidents: dict[str, Incident]) -> None:
    """Serialize and overwrite ``incidents.json`` with the given incidents."""
    _dump(INCIDENTS_FILE, {iid: incident_to_dict(inc) for iid, inc in incidents.items()})


def save_incident(incident: Incident) -> None:
    """Persist a single incident, inserting or updating it by its id."""
    incidents = load_incidents()
    incidents[incident.id] = incident
    _write_incidents(incidents)


def update_incident(incident_id: str, mutate: Callable[[Incident], None]) -> Incident | None:
    """Load an incident, apply ``mutate`` in place, persist, and return it (or None)."""
    incidents = load_incidents()
    incident = incidents.get(incident_id)
    if incident is None:
        return None
    mutate(incident)
    incidents[incident_id] = incident
    _write_incidents(incidents)
    return incident


# --- adjusters --------------------------------------------------------------

def load_adjusters() -> dict[str, Adjuster]:
    """Load every adjuster from disk, keyed by first-name id (empty if absent)."""
    if not ADJUSTERS_FILE.exists():
        return {}
    with ADJUSTERS_FILE.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return {adjuster_id: Adjuster(**data) for adjuster_id, data in raw.items()}


def _write_adjusters(adjusters: dict[str, Adjuster]) -> None:
    """Serialize and overwrite ``adjusters.json`` with the given adjusters."""
    _dump(ADJUSTERS_FILE, {aid: asdict(a) for aid, a in adjusters.items()})


def update_adjuster(adjuster_id: str, mutate: Callable[[Adjuster], None]) -> Adjuster | None:
    """Load an adjuster, apply ``mutate`` in place, persist, and return it (or None)."""
    adjusters = load_adjusters()
    adjuster = adjusters.get(adjuster_id)
    if adjuster is None:
        return None
    mutate(adjuster)
    _write_adjusters(adjusters)
    return adjuster


# --- insurers (policyholders) -----------------------------------------------

def load_insurers() -> dict[str, Insurer]:
    """Load every insurer (policyholder) from disk, keyed by id (empty if absent)."""
    if not INSURERS_FILE.exists():
        return {}
    with INSURERS_FILE.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return {insurer_id: Insurer(**data) for insurer_id, data in raw.items()}


def _write_insurers(insurers: dict[str, Insurer]) -> None:
    """Serialize and overwrite ``insurers.json`` with the given insurers."""
    _dump(INSURERS_FILE, {iid: asdict(ins) for iid, ins in insurers.items()})


def update_insurer(insurer_id: str, mutate: Callable[[Insurer], None]) -> Insurer | None:
    """Load an insurer, apply ``mutate`` in place, persist, and return it (or None)."""
    insurers = load_insurers()
    insurer = insurers.get(insurer_id)
    if insurer is None:
        return None
    mutate(insurer)
    _write_insurers(insurers)
    return insurer


# --- policies ---------------------------------------------------------------

def load_policies() -> dict[str, Policy]:
    """Load every policy from disk, keyed by id (empty if the file is absent)."""
    if not POLICIES_FILE.exists():
        return {}
    with POLICIES_FILE.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return {policy_id: policy_from_dict(d) for policy_id, d in raw.items()}


# --- escalation queue -------------------------------------------------------

def load_escalations() -> dict[str, Escalation]:
    """Load the escalation queue from disk, keyed by incident id (empty if absent)."""
    if not ESCALATIONS_FILE.exists():
        return {}
    with ESCALATIONS_FILE.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return {incident_id: escalation_from_dict(d) for incident_id, d in raw.items()}


def save_escalations(escalations: dict[str, Escalation]) -> None:
    """Serialize and overwrite ``escalations.json`` with the given escalations."""
    _dump(ESCALATIONS_FILE, {iid: escalation_to_dict(e) for iid, e in escalations.items()})
