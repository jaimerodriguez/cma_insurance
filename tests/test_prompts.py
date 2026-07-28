"""Tests for the shared prompt text and its assembly.

`roles.py` and `cma.py` each used to hold a full set of prompts. They diverged
silently, and the cost was concrete: a delegation fix made in `roles.py` never
reached the Managed Agents path, because `cma.py` read only its own copy. The
first test here is the one that matters — it fails if prompt text reappears in a
front-end.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import cma
import prompts
import roles
from roles import Role

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A front-end assembling its own persona text is the regression to catch: scan
# every triple-quoted literal, and flag the substantial ones that address a model
# in the second person. Order-independent, unlike a single regex — the first
# version of this required the marker *after* 200 characters of body and so
# missed a prompt that opens with one, which is the likely shape.
_TRIPLE_QUOTED = re.compile(r'"""(.*?)"""', re.S)
_PERSONA_MARKERS = (
    "You are an AI assistant", "You are an agent", "You are the automated",
    "acting on behalf of an insurance", "speaking directly with a policyholder",
)


def _persona_literals(src: str) -> list[str]:
    return [body[:90].strip() for body in _TRIPLE_QUOTED.findall(src)
            if len(body) > 200 and any(m in body for m in _PERSONA_MARKERS)]


@pytest.mark.parametrize("module", ["cma.py", "repl.py", "roles.py"])
def test_no_front_end_defines_its_own_persona_prompt(module):
    """Wording lives in prompts.py. This is the guard that keeps it there."""
    found = _persona_literals((PROJECT_ROOT / module).read_text())
    assert not found, (
        f"{module} appears to define persona prompt text again: {found[:1]!r}. "
        f"Put the wording in prompts.py and assemble it through "
        f"roles.build_system_prompt.")


def test_the_guard_itself_detects_a_planted_prompt():
    """A detector that cannot detect is worse than none — it reads as coverage."""
    planted = 'X = """You are an AI assistant acting on behalf of an insurance adjuster. ' \
              + "detail " * 40 + '"""'
    assert _persona_literals(planted), "the guard would not notice a reintroduced prompt"
    assert not _persona_literals('X = """short docstring"""')


def test_cma_builds_its_prompts_through_the_shared_seam():
    src = (PROJECT_ROOT / "cma.py").read_text()
    assert "roles.build_system_prompt" in src, (
        "cma.py stopped going through the shared builder — this is exactly how "
        "the two copies diverged last time")


# --- identity handling ------------------------------------------------------

def test_the_stored_agent_prompt_names_nobody():
    """A Managed Agents agent is a stored, versioned object serving every
    identity. A name baked in would mean a new agent version per person."""
    for role, identity in ((Role.ADJUSTER, "jaime"), (Role.INSURER, "ins-1001")):
        body = roles.build_system_prompt(role, None, hosted=True)
        assert identity not in body, f"{role.value} body leaks an identity"


def test_the_identity_block_appears_only_when_an_identity_is_given(world):
    from conftest import HIGH_ADJUSTER
    without = roles.build_system_prompt(Role.ADJUSTER, None)
    with_id = roles.build_system_prompt(Role.ADJUSTER, HIGH_ADJUSTER)
    assert HIGH_ADJUSTER not in without
    assert HIGH_ADJUSTER in with_id
    assert len(with_id) > len(without)


# --- conditional blocks, not "ignore this" ----------------------------------

def test_the_memory_block_is_omitted_off_the_hosted_backend():
    """Telling a local adjuster to read /mnt/memory/ sends it after a directory
    that does not exist. Omit the block rather than asking it to be ignored."""
    local = roles.build_system_prompt(Role.ADJUSTER, None, hosted=False)
    hosted = roles.build_system_prompt(Role.ADJUSTER, None, hosted=True)
    assert "/mnt/memory" not in local
    assert "managed_agent_instructions" not in local
    assert "/mnt/memory" in hosted


def test_the_memory_mount_points_at_this_adjusters_store():
    p = roles.build_system_prompt(Role.ADJUSTER, "jaime", hosted=True,
                                  memory_mount="/mnt/memory/store-jaime/")
    assert "/mnt/memory/store-jaime/" in p


def test_delegation_mechanics_match_the_backend():
    """The two backends spawn differently, and describing the wrong one wastes a
    turn: the SDK gives each adjuster its own subagent, a Managed Agents roster
    holds one shared agent that must be told who it is."""
    roster = [("insurance-adjuster", "acts as any adjuster")]
    hosted = roles.build_system_prompt(Role.AGENT, None, delegate_agents=roster,
                                       hosted=True)
    sdk = roles.build_system_prompt(Role.AGENT, None, delegate_agents=roster,
                                    hosted=False)
    assert "no create_agent" in hosted and "shared by every adjuster" in hosted
    assert "no create_agent" not in sdk
    assert "already knows which adjuster it is" in sdk


def test_no_delegation_block_without_a_roster():
    """A single-role session must not be told to delegate to subagents it has
    no way to launch."""
    p = roles.build_system_prompt(Role.AGENT, None, delegate_agents=())
    assert "Delegating to adjuster subagents" not in p
    assert "{roster}" not in p and "{mechanics}" not in p


# --- content that had to survive the merge ----------------------------------

def test_the_agent_keeps_the_escalation_approval_code():
    """cma.py's copy was the only one with it; losing it in the merge would have
    silently disabled resolving escalated claims."""
    p = roles.build_system_prompt(Role.AGENT, None)
    assert "UNICORN" in p
    assert "never share it" in p or "do not reveal it" in p


def test_the_delegation_fix_reaches_the_hosted_backend():
    """The bug that motivated the merge: this text existed only in roles.py, and
    the Managed Agents path never saw it."""
    p = cma._agent_system(Role.AGENT)
    assert "before you launch a single subagent" in p
    assert "A brief is complete when the adjuster needs nothing further" in p


def test_the_adjuster_delegation_rule_reaches_the_hosted_backend():
    p = cma._agent_system(Role.ADJUSTER)
    assert "its brief is your authorization" in p
    assert "Do not ask it to confirm" in p


def test_the_insurer_is_told_the_intake_adjuster():
    """cma.py's copy called .format(intake=...) against a template with no such
    placeholder, so the hosted policyholder was never told which adjuster_id to
    file against."""
    p = roles.build_system_prompt(Role.INSURER, None)
    assert roles.DEFAULT_INTAKE_ADJUSTER in p


# --- rendering --------------------------------------------------------------

@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("hosted", [True, False])
def test_every_prompt_renders_with_no_placeholders_left(role, hosted):
    """A stray `{policies_json}` reaching the model is invisible until it does
    something strange; `str.format` also raises KeyError on a renamed field."""
    roster = [("insurance-adjuster", "acts as any adjuster")]
    for identity in (None, "jaime" if role is Role.ADJUSTER else "ins-1001"):
        p = roles.build_system_prompt(role, identity, delegate_agents=roster,
                                      hosted=hosted, memory_mount="/mnt/memory/x/")
        leftovers = re.findall(r"\{[a-z_]+\}", p)
        assert not leftovers, f"{role.value} left {leftovers} unrendered"


def test_prompts_carry_no_doubled_apostrophes():
    """cma.py's copies contained `adjuster''s`, `don''t`, `policies''` — a paste
    artifact the model read literally."""
    text = "\n".join(v for k, v in vars(prompts).items()
                     if k.isupper() and isinstance(v, str))
    assert "''" not in text, "doubled apostrophes are back in the prompt text"


def test_extra_is_appended_last():
    p = roles.build_system_prompt(Role.AGENT, None, extra="HOUSE STYLE: be terse.")
    assert p.rstrip().endswith("HOUSE STYLE: be terse.")
