"""Tests for ``tools.generate_unassigned_incidents``.

The tool is destructive and role-restricted, so the cases that matter are the
guard rails — the bounds, the wipe, the role gate — and the internal consistency
of what it produces. A generated claim that references a policy the insurer does
not hold, or a dwelling loss on an auto policy, would flow straight into triage
and produce nonsense the tests downstream could not distinguish from a bug.
"""

from __future__ import annotations

import json
import random

import pytest

import roles
import storage
import tools
from data_entities import PolicyType, ReportStatus, ReportType, Resolution
from repl import _dispatch
from roles import Role


@pytest.fixture(autouse=True)
def seeded():
    """Deterministic batches: the tool draws ids and choices from ``random``."""
    random.seed(20260727)


def _generate(count: int) -> dict:
    return tools.generate_unassigned_incidents(count)


# --- bounds -----------------------------------------------------------------

@pytest.mark.parametrize("count", [tools.MIN_GENERATED_INCIDENTS, 17,
                                   tools.MAX_GENERATED_INCIDENTS])
def test_accepts_the_documented_range_inclusive(world, count):
    assert _generate(count)["count"] == count
    assert len(storage.load_incidents()) == count


@pytest.mark.parametrize("count", [-1, 0, 1, 4, 51, 100, 10_000])
def test_rejects_counts_outside_the_range(world, count):
    with pytest.raises(ValueError, match="between 5 and 50"):
        _generate(count)


def test_a_rejected_count_leaves_the_existing_claims_alone(world):
    """The bound is checked before the wipe, not after — a caller who asks for
    51 must not lose the claims they already had."""
    _generate(6)
    before = storage.load_incidents()
    assert len(before) == 6

    with pytest.raises(ValueError):
        _generate(51)

    assert storage.load_incidents().keys() == before.keys()


def test_a_bool_is_not_an_acceptable_count(world):
    """``True`` is an ``int`` in Python and would sail past a range check as 1."""
    with pytest.raises(ValueError, match="must be an integer"):
        tools.generate_unassigned_incidents(True)


def test_refuses_when_there_are_no_policies_to_file_against(world):
    storage._dump(storage.POLICIES_FILE, {})
    with pytest.raises(ValueError, match="no policies on file"):
        _generate(5)


# --- the destructive contract -----------------------------------------------

def test_replaces_rather_than_appends(world):
    first = _generate(8)
    first_ids = set(storage.load_incidents())
    assert first["replaced"] == 0

    second = _generate(6)
    second_ids = set(storage.load_incidents())

    assert len(second_ids) == 6, "second batch must replace, not accumulate"
    assert second["replaced"] == 8
    assert not (first_ids & second_ids)


# --- what it generates ------------------------------------------------------

def test_every_claim_is_unassigned_and_awaiting_triage(world):
    _generate(20)
    for incident in storage.load_incidents().values():
        assert incident.adjuster_id == tools.UNASSIGNED
        assert incident.status is ReportStatus.NEW
        assert incident.resolution is Resolution.INPROGRESS
        assert incident.resolution_reason is None
        assert incident.resolved_date is None


def test_policy_and_insurer_are_consistent_with_the_real_records(world):
    """The claim's insurer must be the one that actually holds the policy —
    otherwise `find_policy` and `find_insurer` disagree during triage."""
    _generate(25)
    policies = storage.load_policies()
    for incident in storage.load_incidents().values():
        assert incident.policy_id in policies
        assert incident.insurer_id == policies[incident.policy_id].insurer_id


def test_the_loss_type_matches_the_policy_type(world):
    """A dwelling loss on an auto policy is the kind of nonsense that reaches
    triage looking like a real coverage question."""
    _generate(40)
    policies = storage.load_policies()
    for incident in storage.load_incidents().values():
        policy_type = policies[incident.policy_id].policy_type
        vehicle = ReportType.STOLEN_CAR | ReportType.CAR_ACCIDENT
        if policy_type is PolicyType.AUTO:
            assert incident.incident_type & vehicle
            assert not incident.incident_type & ReportType.HOME
        elif policy_type is PolicyType.HOME:
            assert incident.incident_type & ReportType.HOME
            assert not incident.incident_type & vehicle


def test_costs_are_positive_whole_hundreds(world):
    _generate(30)
    for incident in storage.load_incidents().values():
        assert incident.estimate_cost > 0
        assert incident.estimate_cost % 100 == 0


def test_a_large_batch_spans_more_than_one_authorization_band(world):
    """The point of the generator is to give triage something to do. A batch
    that lands entirely in one band exercises a single code path."""
    from data_entities import DynamicPolicies

    _generate(tools.MAX_GENERATED_INCIDENTS)
    limits = DynamicPolicies()
    bands = {limits.required_authorization(i.estimate_cost)
             for i in storage.load_incidents().values()}
    assert len(bands) >= 3, f"only reached {bands}"


def test_details_and_history_are_populated_and_reference_the_claim(world):
    _generate(12)
    for incident in storage.load_incidents().values():
        assert incident.policy_id in incident.incident_details
        assert f"${incident.estimate_cost:,}" in incident.incident_details
        assert incident.history
        assert "Awaiting assignment" in incident.history


def test_ids_are_unique_across_a_full_batch(world):
    result = _generate(tools.MAX_GENERATED_INCIDENTS)
    assert result["count"] == len(storage.load_incidents()) == 50


def test_the_summary_describes_what_was_written(world):
    result = _generate(15)
    incidents = storage.load_incidents()
    costs = [i.estimate_cost for i in incidents.values()]

    assert result["count"] == 15
    assert result["adjuster_id"] == tools.UNASSIGNED
    assert sum(result["by_type"].values()) == 15
    assert result["cost_range"] == [min(costs), max(costs)]
    assert all(i in incidents for i in result["sample_ids"])
    # Named flags, not the integer bitmask str() would produce.
    assert all(not k.isdigit() for k in result["by_type"]), result["by_type"]


def test_the_summary_is_not_the_claims_themselves(world):
    """A 50-claim payload is thousands of tokens the caller did not ask for."""
    result = _generate(50)
    assert len(json.dumps(result)) < 1000
    assert len(result["sample_ids"]) == 3


def test_seeding_reproduces_a_batch_exactly(world):
    """Ids come from ``random``, not ``uuid4``, so a seeded demo is repeatable."""
    def batch() -> dict[str, str]:
        random.seed(4242)
        _generate(10)
        return {k: v.incident_details for k, v in storage.load_incidents().items()}

    assert batch() == batch()


# --- role enforcement -------------------------------------------------------

def test_only_the_agent_role_may_call_it():
    assert "generate_unassigned_incidents" in roles.ROLE_TOOLS[Role.AGENT]
    assert "generate_unassigned_incidents" not in roles.ROLE_TOOLS[Role.ADJUSTER]
    assert "generate_unassigned_incidents" not in roles.ROLE_TOOLS[Role.INSURER]


@pytest.mark.parametrize("role", [Role.ADJUSTER, Role.INSURER])
def test_dispatch_refuses_it_for_other_roles(world, role):
    """Enforcement is the dispatch table, not just the advertised schema."""
    out = json.loads(_dispatch(roles.dispatch_table(role),
                               "generate_unassigned_incidents", {"count": 10}))
    assert "not permitted for this role" in out["error"]
    assert storage.load_incidents() == {}


def test_it_reaches_the_agent_role_through_dispatch(world):
    out = json.loads(_dispatch(roles.dispatch_table(Role.AGENT),
                               "generate_unassigned_incidents", {"count": 7}))
    assert out["count"] == 7
    assert len(storage.load_incidents()) == 7


def test_an_out_of_range_count_comes_back_as_the_error_contract(world):
    """`_dispatch` turns the ValueError into `{"error": ...}` so the model can
    correct itself rather than the turn dying."""
    out = json.loads(_dispatch(roles.dispatch_table(Role.AGENT),
                               "generate_unassigned_incidents", {"count": 500}))
    assert "ValueError" in out["error"]
    assert "between 5 and 50" in out["error"]
