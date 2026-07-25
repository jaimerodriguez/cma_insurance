"""Shared test fixtures.

The repo has no package layout — ``storage``, ``tools`` and friends are
top-level modules imported relative to the project root — so the root goes on
``sys.path`` here rather than relying on ``python -m pytest`` putting the cwd
there. That makes a bare ``pytest tests/`` work too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import storage  # noqa: E402
from data_entities import Adjuster, AuthorizationLevel, Insurer  # noqa: E402

# Seeded ids, exported so tests read as prose rather than string literals.
LOW_ADJUSTER = "leo"        # authorization_level LOW
MED_ADJUSTER = "mia"        # MEDIUM
HIGH_ADJUSTER = "hana"      # HIGH
UNASSIGNED = "unassigned"   # the placeholder; must never be selected by routing

PLAIN_INSURER = "ins-plain"
VIP_INSURER = "ins-vip"
AUTO_POLICY = "pol-auto"    # held by PLAIN_INSURER
HOME_POLICY = "pol-home"    # held by PLAIN_INSURER
VIP_AUTO_POLICY = "pol-vip-auto"
VIP_HOME_POLICY = "pol-vip-home"


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Point the storage layer at a scratch dir seeded with a minimal roster.

    Every ``storage`` path global is redirected, not just ``DATA_DIR``: the file
    constants are bound at import time, so patching the directory alone would
    leave reads and writes pointed at the real ``data/``. A test that forgot one
    would quietly mutate the repo's own fixtures, which is the failure worth
    engineering against here.

    Yields the scratch ``data/`` path; tests mostly use the module-level id
    constants instead.
    """
    scratch = tmp_path / "data"
    scratch.mkdir()
    monkeypatch.setattr(storage, "DATA_DIR", scratch)
    for attr in ("INCIDENTS_FILE", "ADJUSTERS_FILE", "INSURERS_FILE",
                 "POLICIES_FILE", "ESCALATIONS_FILE"):
        monkeypatch.setattr(storage, attr, scratch / getattr(storage, attr).name)

    # One adjuster per authorization band, so routing has an unambiguous
    # lowest-qualifying answer at every cost, plus the placeholder.
    storage._write_adjusters({
        LOW_ADJUSTER: Adjuster(LOW_ADJUSTER, authorization_level=AuthorizationLevel.LOW),
        MED_ADJUSTER: Adjuster(MED_ADJUSTER, authorization_level=AuthorizationLevel.MEDIUM),
        HIGH_ADJUSTER: Adjuster(HIGH_ADJUSTER, authorization_level=AuthorizationLevel.HIGH),
        UNASSIGNED: Adjuster(UNASSIGNED, authorization_level=AuthorizationLevel.LOW),
    })
    storage._write_insurers({
        PLAIN_INSURER: Insurer(PLAIN_INSURER, "Pat Plain", "1 Main St", "555-0100"),
        VIP_INSURER: Insurer(VIP_INSURER, "Vi Ivy", "2 Oak Ave", "555-0200", is_VIP=True),
    })
    storage._dump(storage.POLICIES_FILE, {
        pid: {"id": pid, "policy_type": ptype, "insurer_id": holder,
              "effective_date": "2026-01-01", "expiration_date": "2027-01-01",
              "premium": 1200.0}
        for pid, ptype, holder in (
            (AUTO_POLICY, "auto", PLAIN_INSURER),
            (HOME_POLICY, "home", PLAIN_INSURER),
            (VIP_AUTO_POLICY, "auto", VIP_INSURER),
            (VIP_HOME_POLICY, "home", VIP_INSURER),
        )
    })
    storage._write_incidents({})
    storage.save_escalations({})
    return scratch
