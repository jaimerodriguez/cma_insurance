"""Tests for the triage path — ``process_incident`` and what it decides.

Weighted toward the branches where being wrong is expensive or silent: money
(which auto-approve ceiling applied), routing (who ends up able to decide a
claim), and the escalation-to-management path, where the failure mode is a claim
nobody owns rather than an exception.

Every test runs against the scratch ``data/`` from the ``world`` fixture; see
``conftest.py``.

Run with:  .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import storage
import tools
from conftest import (
    AUTO_POLICY,
    HIGH_ADJUSTER,
    HOME_POLICY,
    LOW_ADJUSTER,
    MED_ADJUSTER,
    PLAIN_INSURER,
    UNASSIGNED,
    VIP_AUTO_POLICY,
    VIP_HOME_POLICY,
    VIP_INSURER,
)
from data_entities import (
    Adjuster,
    AuthorizationLevel,
    DynamicPolicies,
    Incident,
    ReportStatus,
    ReportType,
    Resolution,
)

# Round numbers so a failure message reads as arithmetic, not as a lookup.
BANDS = DynamicPolicies(
    auto_approve=1_000,
    vip_auto_approve=2_000,
    upset_auto_approve_home=3_000,
    vip_auto_approve_home=4_000,
    can_assign=True,
    authorization_low=10_000,
    authorization_medium=20_000,
    authorization_high=30_000,
)


def _file(cost: int, *, policy: str = AUTO_POLICY, insurer: str = PLAIN_INSURER,
          adjuster: str = UNASSIGNED,
          loss: ReportType = ReportType.CAR_ACCIDENT) -> Incident:
    return tools.create_incident(adjuster, loss, "a loss occurred", insurer, policy, cost)


# --- auto-approval ceilings -------------------------------------------------

def test_claim_under_the_baseline_ceiling_auto_approves(world):
    out = tools.process_incident(_file(999).id, BANDS)
    assert out.resolution is Resolution.APPROVED
    assert "999 < 1000" in out.resolution_reason


def test_ceiling_is_exclusive_at_the_boundary(world):
    """`cost < ceiling`, not `<=` — a claim exactly at the ceiling escalates."""
    out = tools.process_incident(_file(1_000).id, BANDS)
    assert out.resolution is Resolution.INPROGRESS


def test_vip_gets_the_vip_ceiling(world):
    plain = tools.process_incident(_file(1_500).id, BANDS)
    vip = tools.process_incident(
        _file(1_500, policy=VIP_AUTO_POLICY, insurer=VIP_INSURER).id, BANDS)
    assert plain.resolution is Resolution.INPROGRESS
    assert vip.resolution is Resolution.APPROVED


def test_vip_home_policy_gets_the_highest_ceiling(world):
    out = tools.process_incident(
        _file(3_500, policy=VIP_HOME_POLICY, insurer=VIP_INSURER).id, BANDS)
    assert out.resolution is Resolution.APPROVED
    assert "< 4000" in out.resolution_reason


def test_upset_policyholder_raises_the_ceiling_only_on_home_claims(world):
    auto = tools.process_incident(_file(2_500).id, BANDS, insurer_upset=True)
    home = tools.process_incident(_file(2_500, policy=HOME_POLICY).id, BANDS,
                                  insurer_upset=True)
    assert auto.resolution is Resolution.INPROGRESS
    assert home.resolution is Resolution.APPROVED


def test_home_loss_type_earns_home_ceilings_on_a_non_home_policy(world):
    """The loss kind counts, not just the coverage: a burglary reported against an
    AUTO policy is still a dwelling loss."""
    car = tools.process_incident(
        _file(3_500, policy=VIP_AUTO_POLICY, insurer=VIP_INSURER,
              loss=ReportType.CAR_ACCIDENT).id, BANDS)
    home = tools.process_incident(
        _file(3_500, policy=VIP_AUTO_POLICY, insurer=VIP_INSURER,
              loss=ReportType.HOME_THEFT).id, BANDS)
    assert car.resolution is Resolution.INPROGRESS      # vip ceiling 2000
    assert home.resolution is Resolution.APPROVED       # vip-home ceiling 4000


@pytest.mark.parametrize("loss", [ReportType.HOME_THEFT, ReportType.HOME_ACCIDENT,
                                  ReportType.HOME_NATURAL_DISASTER])
def test_every_home_flag_counts_as_a_home_loss(world, loss):
    out = tools.process_incident(
        _file(3_500, policy=VIP_AUTO_POLICY, insurer=VIP_INSURER, loss=loss).id, BANDS)
    assert out.resolution is Resolution.APPROVED


def test_a_combined_loss_counts_as_home_if_any_flag_is_home(world):
    out = tools.process_incident(
        _file(3_500, policy=VIP_AUTO_POLICY, insurer=VIP_INSURER,
              loss=ReportType.CAR_ACCIDENT | ReportType.HOME_ACCIDENT).id, BANDS)
    assert out.resolution is Resolution.APPROVED


def test_agent_override_approves_regardless_of_cost(world):
    policies = DynamicPolicies(auto_approve=1, agent_override_autoapprove=True,
                               agent_override_autoapprove_reason="loyal customer")
    out = tools.process_incident(_file(9_999_999).id, policies)
    assert out.resolution is Resolution.APPROVED
    assert "loyal customer" in out.resolution_reason


# --- routing ----------------------------------------------------------------

@pytest.mark.parametrize("cost,expected", [
    (10_000, LOW_ADJUSTER),     # at the LOW limit — inclusive
    (10_001, MED_ADJUSTER),
    (20_000, MED_ADJUSTER),
    (20_001, HIGH_ADJUSTER),
    (30_000, HIGH_ADJUSTER),
])
def test_routes_to_the_lowest_adjuster_who_qualifies(world, cost, expected):
    out = tools.process_incident(_file(cost).id, BANDS)
    assert out.adjuster_id == expected


def test_routing_never_picks_the_unassigned_placeholder(world):
    """The placeholder sits in the roster as a LOW adjuster, so a lowest-qualifying
    search that forgot to exclude it would route real claims to nobody.

    The roster here leaves it as the *only* LOW candidate. With a real LOW
    adjuster present the tie-break on user_id ("leo" < "unassigned") picks the
    right one either way, and the test passes whether or not the guard exists —
    which is exactly the vacuous assertion this avoids.
    """
    storage._write_adjusters({
        UNASSIGNED: Adjuster(UNASSIGNED, authorization_level=AuthorizationLevel.LOW),
        MED_ADJUSTER: Adjuster(MED_ADJUSTER, authorization_level=AuthorizationLevel.MEDIUM),
    })
    out = tools.process_incident(_file(5_000).id, BANDS)
    assert out.adjuster_id == MED_ADJUSTER


def test_can_assign_false_leaves_an_unassigned_claim_alone(world):
    incident = _file(15_000)
    out = tools.process_incident(incident.id, replace(BANDS, can_assign=False))
    assert out.adjuster_id == UNASSIGNED
    assert out.status is ReportStatus.NEW
    assert incident.id not in storage.load_escalations()


def test_claim_beyond_every_adjuster_but_within_the_bands_is_left_for_manual_routing(world):
    """Distinct from OVERRIDE_NEEDED: the bands allow it, the roster does not."""
    storage._write_adjusters({
        LOW_ADJUSTER: Adjuster(LOW_ADJUSTER, authorization_level=AuthorizationLevel.LOW),
        UNASSIGNED: Adjuster(UNASSIGNED, authorization_level=AuthorizationLevel.LOW),
    })
    out = tools.process_incident(_file(25_000).id, BANDS)
    assert out.adjuster_id == UNASSIGNED
    assert out.status is ReportStatus.NEW      # not sent to management


def test_an_assigned_claim_that_cannot_auto_approve_escalates_to_its_adjuster(world):
    incident = _file(15_000, adjuster=MED_ADJUSTER)
    out = tools.process_incident(incident.id, BANDS)
    assert out.adjuster_id == MED_ADJUSTER
    assert out.resolution is Resolution.INPROGRESS
    queued = storage.load_escalations()[incident.id]
    assert queued.adjuster_id == MED_ADJUSTER


# --- escalation to management -----------------------------------------------

def test_cost_beyond_every_band_goes_to_management(world):
    out = tools.process_incident(_file(30_001).id, BANDS)
    assert out.status is ReportStatus.ESCALATED_TO_MANAGEMENT
    assert out.resolution is Resolution.INPROGRESS       # still live
    assert out.resolved_date is None
    assert "exceeds every authorization limit" in out.resolution_reason


def test_management_escalation_ignores_can_assign(world):
    """`can_assign=False` means "don't pick an adjuster", not "strand a claim
    nobody is allowed to decide"."""
    out = tools.process_incident(_file(30_001).id, DynamicPolicies(
        can_assign=False, auto_approve=1, authorization_high=30_000))
    assert out.status is ReportStatus.ESCALATED_TO_MANAGEMENT


def test_management_escalation_applies_to_an_already_assigned_claim(world):
    incident = _file(30_001, adjuster=HIGH_ADJUSTER)
    out = tools.process_incident(incident.id, BANDS)
    assert out.status is ReportStatus.ESCALATED_TO_MANAGEMENT
    assert incident.id not in storage.load_escalations()   # not the adjuster's call


def test_retriaging_a_management_claim_is_a_no_op(world):
    """These claims stay INPROGRESS, so a scheduled pass keeps selecting them.
    Without the guard each pass re-stamps the history."""
    incident = _file(30_001)
    tools.process_incident(incident.id, BANDS)
    for _ in range(3):
        tools.process_incident(incident.id, BANDS)
    history = tools.get_incident_details(incident.id).history
    assert history.count("escalated to management") == 1


def test_management_escalation_outranks_the_agent_override(world):
    """An escalation to management is exactly where the agent stops deciding."""
    incident = _file(30_001)
    tools.process_incident(incident.id, BANDS)
    out = tools.process_incident(incident.id, DynamicPolicies(
        agent_override_autoapprove=True, agent_override_autoapprove_reason="override"))
    assert out.resolution is Resolution.INPROGRESS
    assert out.status is ReportStatus.ESCALATED_TO_MANAGEMENT


def test_an_adjuster_can_escalate_for_any_reason_and_the_queue_is_withdrawn(world):
    incident = _file(15_000, adjuster=MED_ADJUSTER)
    tools.escalate_incident(incident.id, BANDS)
    assert incident.id in storage.load_escalations()

    out = tools.escalate_to_management(incident.id, "conflict of interest")
    assert out.status is ReportStatus.ESCALATED_TO_MANAGEMENT
    assert out.resolution is Resolution.INPROGRESS
    assert out.resolved_date is None
    assert incident.id not in storage.load_escalations()
    assert "conflict of interest" in out.history


def test_escalate_to_management_on_a_missing_incident_returns_none(world):
    assert tools.escalate_to_management("no-such-case", "why") is None


# --- terminal resolutions and auto-close ------------------------------------

def test_management_override_is_terminal_and_clears_the_queue(world):
    incident = _file(15_000, adjuster=MED_ADJUSTER)
    tools.escalate_incident(incident.id, BANDS)
    out = tools.update_resolution(incident.id, "management",
                                  Resolution.MANAGEMENT_OVERRIDE, reason="board call")
    assert out.resolved_date is not None
    assert incident.id not in storage.load_escalations()


@pytest.mark.parametrize("resolution,closes", [
    (Resolution.APPROVED, True),
    (Resolution.DENIED, True),
    (Resolution.MANAGEMENT_OVERRIDE, True),
    (Resolution.DISPUTED, False),
    (Resolution.INPROGRESS, False),
])
def test_only_terminal_resolutions_auto_close(world, resolution, closes):
    incident = _file(500, adjuster=MED_ADJUSTER)
    tools.update_resolution(incident.id, MED_ADJUSTER, resolution)
    closed = tools.close_stale_resolved(minutes=0)
    assert any(c.id == incident.id for c in closed) is closes


def test_a_claim_awaiting_management_is_not_auto_closed(world):
    """The status says escalated; the resolution says undecided. Closing it would
    silently drop a claim no one has ruled on."""
    incident = _file(30_001)
    tools.process_incident(incident.id, BANDS)
    assert not tools.close_stale_resolved(minutes=0)
    assert tools.get_incident_details(incident.id).status is (
        ReportStatus.ESCALATED_TO_MANAGEMENT)


def test_close_stale_respects_the_age_threshold(world):
    incident = _file(500, adjuster=MED_ADJUSTER)
    tools.update_resolution(incident.id, MED_ADJUSTER, Resolution.APPROVED)
    assert not tools.close_stale_resolved(minutes=30)   # resolved just now
    assert tools.close_stale_resolved(minutes=0)


# --- authorization levels ---------------------------------------------------

@pytest.mark.parametrize("cost,expected", [
    (0, AuthorizationLevel.LOW),
    (10_000, AuthorizationLevel.LOW),
    (10_001, AuthorizationLevel.MEDIUM),
    (20_000, AuthorizationLevel.MEDIUM),
    (20_001, AuthorizationLevel.HIGH),
    (30_000, AuthorizationLevel.HIGH),
    (30_001, AuthorizationLevel.OVERRIDE_NEEDED),
])
def test_required_authorization_maps_cost_to_a_band(cost, expected):
    assert BANDS.required_authorization(cost) is expected


def test_required_authorization_never_returns_none():
    """It used to return None above the top band, which callers could not tell
    apart from "no adjuster available" — both meant the claim was left alone."""
    assert BANDS.required_authorization(10**9) is AuthorizationLevel.OVERRIDE_NEEDED


def test_authorization_levels_are_ordered_with_override_above_high():
    ranks = [lv.rank for lv in (AuthorizationLevel.LOW, AuthorizationLevel.MEDIUM,
                                AuthorizationLevel.HIGH,
                                AuthorizationLevel.OVERRIDE_NEEDED)]
    assert ranks == sorted(ranks) == [0, 1, 2, 3]


def test_an_adjuster_cannot_hold_the_override_level():
    """Otherwise an adjuster ranked above HIGH becomes the routing candidate for
    exactly the claims nobody is allowed to decide."""
    with pytest.raises(ValueError, match="not a level an adjuster can hold"):
        Adjuster("mallory", authorization_level="override_needed")


# --- guards -----------------------------------------------------------------

def test_process_incident_on_a_missing_incident_returns_none(world):
    assert tools.process_incident("no-such-case", BANDS) is None


def test_the_fixture_isolates_the_real_data_directory(world):
    """If this regresses, every other test in this file starts writing to the
    repo's own data/ files."""
    _file(500)
    assert storage.INCIDENTS_FILE.parent == world
    assert "tmp" in str(storage.INCIDENTS_FILE) or "pytest" in str(storage.INCIDENTS_FILE)
