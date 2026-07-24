"""Data model for the insurance incident-report mock.

Defines the enums and dataclasses for the domain and the (de)serialization
helpers that convert them to/from JSON-ready dicts. This module has no I/O and
no knowledge of files — see ``storage.py`` for persistence and ``tools.py`` for
the agent-facing operations.

Entities and how they relate:
    Incident --policy_id--> Policy --insurer_id--> Insurer (policyholder)
    Incident --adjuster_id--> Adjuster
    Incident --insurer_id--> Insurer   (secondary/redundant direct link)

Terminology note: ``Insurer`` here models the *insured party*
(the policyholder/customer), not the insurance company.
"""

import uuid
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timezone
from enum import IntFlag, StrEnum, auto
from typing import Any


def _create_unique_identifier() -> str:
    """Return a fresh random unique id as a string (uuid4)."""
    return str(uuid.uuid4())


# --- enums ------------------------------------------------------------------

class ReportType(IntFlag):
    """Kind(s) of loss an incident reports.

    This is a *bit flag*, so a single incident may combine several types with
    the ``|`` operator, e.g. ``ReportType.STOLEN_CAR | ReportType.CAR_ACCIDENT``.
    On disk it is stored as the integer bitmask (1, 2, 4, or a sum such as 5).

    Members:
        STOLEN_CAR (1): A vehicle was stolen.
        STOLEN_OTHER (2): Non-vehicle property was stolen (e.g. burglary).
        CAR_ACCIDENT (4): A vehicle collision occurred.
    """

    STOLEN_CAR = auto()
    STOLEN_OTHER = auto()
    CAR_ACCIDENT = auto()


class PolicyType(StrEnum):
    """The single kind of coverage a policy provides.

    A policy is exactly one of these. Serialized as its lowercase string value
    ("auto", "home", "other").

    Members:
        AUTO: Vehicle coverage.
        HOME: Property/dwelling coverage.
        OTHER: Any other coverage type.
    """

    AUTO = auto()
    HOME = auto()
    OTHER = auto()


class Resolution(StrEnum):
    """Outcome of an incident/claim's review.

    Serialized as its lowercase string value ("approved", "denied",
    "inprogress", "disputed").

    Members:
        APPROVED: The claim was approved for payout.
        DENIED: The claim was rejected.
        INPROGRESS: Still under review. Default for new incidents.
        DISPUTED: Under dispute / contested.
    """

    APPROVED = auto()
    DENIED = auto()
    INPROGRESS = auto()
    DISPUTED = auto()


class ReportStatus(StrEnum):
    """Lifecycle state of an incident/claim.

    Serialized as its lowercase string value ("new", "open", "notified",
    "closed").

    Members:
        NEW: Just filed, not yet triaged. This is the default for new incidents.
        OPEN: Under active investigation.
        NOTIFIED: The adjuster and insurer have been notified of an update.
        CLOSED: Resolved; no further action.
    """

    NEW = auto()
    OPEN = auto()
    NOTIFIED = auto()
    CLOSED = auto()


class AuthorizationLevel(StrEnum):
    """How much approval authority an adjuster has.

    Used by the agent when deciding which adjuster to assign an incident to
    (e.g. higher-cost or higher-risk claims go to a higher-authority adjuster).
    Serialized as its lowercase string value ("low", "medium", "high").

    Members:
        LOW: Baseline authority. Default for adjusters.
        MEDIUM: Elevated authority.
        HIGH: Full authority.
    """

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


# --- entities ---------------------------------------------------------------

@dataclass
class Adjuster:
    """An insurance adjuster — the staff member who handles a claim.

    Adjusters are keyed in ``adjusters.json`` by a short first-name id
    (``user_id``), e.g. "jaime".

    Attributes:
        user_id: Unique id / lookup key for the adjuster (e.g. "jaime").
        location: Human-readable base location (e.g. "Seattle WA"). Optional.
        phone_number: Contact phone number. Optional.
        timezone: The adjuster's IANA time-zone name, e.g.
            "America/Los_Angeles". Optional.
        full_name: The adjuster's full display name. Optional.
        preferences: Free-text notes on how to work with this person, e.g.
            "do not share my phone number, tell them I will contact them".
            Optional.
        history: Free-text running log of interactions, typically
            date-stamped, e.g. "7/7/2026 - frustrated with lack of progress".
            Optional.
        authorization_level: How much approval authority this adjuster has
            (see ``AuthorizationLevel``). Used by the agent when assigning
            incidents. Defaults to ``AuthorizationLevel.LOW``.
    """

    user_id: str
    location: str | None = None
    phone_number: str | None = None
    timezone: str | None = None  # IANA name, e.g. "America/Los_Angeles"
    full_name: str | None = None
    preferences: str | None = None
    history: str | None = None
    authorization_level: AuthorizationLevel = AuthorizationLevel.LOW

    def __post_init__(self) -> None:
        # Coerce a plain string (e.g. loaded from JSON) into the enum.
        self.authorization_level = AuthorizationLevel(self.authorization_level)


@dataclass
class Insurer:
    """The insured party (policyholder / customer).

    Despite the name, this represents the *person insured*, not the insurance
    company. Keyed in ``insurers.json`` by ``id`` (e.g. "ins-1001").

    Attributes:
        id: Unique id / lookup key for the insured party (e.g. "ins-1001").
        full_name: The policyholder's full name.
        address: Mailing/street address as a single line.
        phone_number: Contact phone number.
        preferences: Free-text notes on how to work with this person, e.g.
            "Their name is Joseph, but goes by Joe". Optional.
        history: Free-text running log of interactions with this
            policyholder, typically date-stamped, e.g.
            "7/7/2026 - frustrated with lack of progress". Optional.
        is_VIP: Whether this is a VIP policyholder (gets higher auto-approve
            ceilings during triage). Defaults to False.
    """

    id: str
    full_name: str
    address: str
    phone_number: str
    preferences: str | None = None
    history: str | None = None
    is_VIP: bool = False


@dataclass
class Policy:
    """An insurance policy held by an insured party.

    Keyed in ``policies.json`` by ``id`` (e.g. "pol-2001"). Links back to its
    holder through ``insurer_id``.

    Attributes:
        policy_type: The coverage type (see ``PolicyType``).
        insurer_id: Id of the ``Insurer`` (policyholder) who holds this policy.
        effective_date: Date coverage begins.
        expiration_date: Date coverage ends.
        premium: Annual premium amount in dollars.
        id: Unique policy id. Auto-generated (uuid4) if not supplied.
    """

    policy_type: PolicyType
    insurer_id: str
    effective_date: date
    expiration_date: date
    premium: float
    id: str = field(default_factory=_create_unique_identifier)


@dataclass
class Incident:
    """An incident report (insurance claim) — the central record.

    Keyed in ``incidents.json`` by ``id``. Cross-references the policy it was
    filed against, the adjuster handling it, and (redundantly) the insurer.

    Attributes:
        policy_id: Id of the ``Policy`` this claim is filed against.
        adjuster_id: Id of the ``Adjuster`` handling this claim.
        insurer_id: Id of the ``Insurer`` (policyholder). Redundant with the
            policy's ``insurer_id``; kept as a direct shortcut.
        incident_type: One or more ``ReportType`` flags describing the loss.
        incident_details: Free-text description of what happened.
        submitted_date: When the claim was filed. Defaults to the current
            UTC time at creation.
        status: Current ``ReportStatus``. Defaults to ``ReportStatus.NEW``.
        resolution: Review outcome (see ``Resolution``). Defaults to
            ``Resolution.INPROGRESS``.
        resolution_reason: Free-text explanation of the resolution (or of why
            the claim was escalated), e.g. "auto-approved: cost 250 < 300
            ceiling" or "agent override: loyal customer". Optional.
        resolved_date: When the resolution was last set to APPROVED or DENIED
            (UTC). Used by the daily maintenance job to auto-close claims that
            have been resolved for a while. None until first resolved.
        estimate_cost: Estimated cost of the incident, in whole dollars.
            Drives the ``approve_incident`` triage. Defaults to 0.
        history: Free-text running log of activity on this claim, typically
            date-stamped, e.g. "7/7/2026 - left voicemail for policyholder".
            Optional.
        id: Unique incident/case id. Auto-generated (uuid4) if not supplied.
    """

    policy_id: str
    adjuster_id: str
    insurer_id: str
    incident_type: ReportType
    incident_details: str
    submitted_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ReportStatus = ReportStatus.NEW
    resolution: Resolution = Resolution.INPROGRESS
    resolution_reason: str | None = None
    resolved_date: datetime | None = None
    estimate_cost: int = 0
    history: str | None = None
    id: str = field(default_factory=_create_unique_identifier)


@dataclass
class Escalation:
    """A queued request for an adjuster to approve or deny an escalated claim.

    Created by ``escalate_incident`` when a claim's cost is too high to
    auto-approve. Stored in ``escalations.json`` keyed by ``incident_id``.

    Attributes:
        incident_id: Id of the escalated ``Incident``.
        adjuster_id: Id of the ``Adjuster`` responsible for deciding it.
        escalated_date: When the escalation was created (UTC).
    """

    incident_id: str
    adjuster_id: str
    escalated_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DynamicPolicies:
    """Tunable auto-approval rules applied by ``approve_incident``.

    Each ``*_auto_approve*`` field is a dollar *ceiling*: a claim is
    auto-approved when its cost is strictly below the highest ceiling that
    applies to it. Which ceilings apply depends on the policyholder's VIP
    status, whether the policyholder is upset, and the policy type:

        - ``auto_approve``           always applies (baseline).
        - ``vip_auto_approve``       applies when the insurer is a VIP.
        - ``vip_auto_approve_home``  applies when VIP *and* the policy is HOME.
        - ``upset_auto_approve_home``applies when the insurer is upset *and*
                                     the policy is HOME.

    ``agent_override_autoapprove`` short-circuits everything: when True the
    claim is auto-approved regardless of cost or any other field, and
    ``agent_override_autoapprove_reason`` is recorded on the incident. The
    agent is expected to set these two.

    When a claim is *not* auto-approved, ``can_assign`` controls what happens to
    an as-yet *unassigned* incident: if True it may be routed to an adjuster
    (see ``required_authorization``); if False it is left untouched. The three
    ``authorization_*`` fields are dollar limits mapping a claim's cost to the
    minimum adjuster ``AuthorizationLevel`` able to handle it.

    Attributes:
        auto_approve: Baseline auto-approve ceiling in dollars.
        vip_auto_approve: Ceiling for VIP policyholders.
        upset_auto_approve_home: Ceiling for upset policyholders on HOME claims.
        vip_auto_approve_home: Ceiling for VIP policyholders on HOME claims.
        agent_override_autoapprove: When True, force auto-approval.
        agent_override_autoapprove_reason: Why the agent forced approval.
        can_assign: When True, an unassigned incident that isn't auto-approved
            may be routed to a suitable adjuster instead of being left alone.
        authorization_low: Max claim cost a LOW-authorization adjuster handles.
        authorization_medium: Max claim cost a MEDIUM-authorization adjuster handles.
        authorization_high: Max claim cost a HIGH-authorization adjuster handles.
    """

    auto_approve: int = 300
    vip_auto_approve: int = 500
    upset_auto_approve_home: int = 1500
    vip_auto_approve_home: int = 5000
    agent_override_autoapprove: bool = False
    agent_override_autoapprove_reason: str | None = None
    can_assign: bool = False
    authorization_low: int = 500
    authorization_medium: int = 1500
    authorization_high: int = 5000

    def required_authorization(self, cost: int) -> "AuthorizationLevel | None":
        """Lowest ``AuthorizationLevel`` able to handle a claim of ``cost``.

        Maps the cost against the ``authorization_*`` limits (inclusive):
        ``cost <= authorization_low`` -> LOW, then MEDIUM, then HIGH.

        Args:
            cost: The incident's estimated cost in dollars.

        Returns:
            The minimum ``AuthorizationLevel`` that qualifies, or ``None`` when
            the cost exceeds even ``authorization_high`` (no adjuster qualifies).
        """
        if cost <= self.authorization_low:
            return AuthorizationLevel.LOW
        if cost <= self.authorization_medium:
            return AuthorizationLevel.MEDIUM
        if cost <= self.authorization_high:
            return AuthorizationLevel.HIGH
        return None

    @classmethod
    def coerce(cls, value: "DynamicPolicies | dict[str, Any] | None") -> "DynamicPolicies":
        """Normalize ``None`` / a plain dict / a ``DynamicPolicies`` into an instance.

        Lets callers (e.g. an agent tool-call dispatcher) pass the policies as a
        JSON object without constructing the dataclass. ``None`` yields the
        defaults; unknown dict keys are ignored.

        Args:
            value: ``None``, a dict of field values, or a ``DynamicPolicies``.

        Returns:
            A ``DynamicPolicies`` instance.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            valid = {f.name for f in fields(cls)}
            return cls(**{k: v for k, v in value.items() if k in valid})
        raise TypeError(
            f"policies must be DynamicPolicies, dict, or None; got {type(value).__name__}"
        )

    def auto_approve_ceiling(self, is_vip: bool, is_home: bool, insurer_upset: bool) -> int:
        """Return the highest auto-approve ceiling that applies to a claim."""
        ceiling = self.auto_approve
        if is_vip:
            ceiling = max(ceiling, self.vip_auto_approve)
            if is_home:
                ceiling = max(ceiling, self.vip_auto_approve_home)
        if insurer_upset and is_home:
            ceiling = max(ceiling, self.upset_auto_approve_home)
        return ceiling

    def should_auto_approve(
        self, cost: int, is_vip: bool, is_home: bool, insurer_upset: bool
    ) -> tuple[bool, str]:
        """Decide whether a claim auto-approves, with a human-readable reason.

        Args:
            cost: The incident's estimated cost in dollars.
            is_vip: Whether the policyholder is a VIP.
            is_home: Whether the claim's policy is a HOME policy.
            insurer_upset: Whether the agent judged the policyholder upset.

        Returns:
            ``(approve, reason)`` — ``approve`` is True to auto-approve; if
            False the claim should be escalated. ``reason`` explains either
            outcome and is suitable for the incident's ``resolution_reason``.
        """
        if self.agent_override_autoapprove:
            reason = self.agent_override_autoapprove_reason or "agent override"
            return True, f"agent override: {reason}"
        ceiling = self.auto_approve_ceiling(is_vip, is_home, insurer_upset)
        if cost < ceiling:
            return True, f"auto-approved: cost {cost} < {ceiling} ceiling"
        return False, f"escalated: cost {cost} >= {ceiling} auto-approve ceiling"


# --- (de)serialization ------------------------------------------------------

def incident_to_dict(incident: Incident) -> dict[str, Any]:
    """Convert an ``Incident`` to a JSON-serializable dict.

    The ``incident_type`` flags are stored as an integer bitmask and the
    enums/datetime as their string forms.
    """
    return {
        "id": incident.id,
        "policy_id": incident.policy_id,
        "adjuster_id": incident.adjuster_id,
        "insurer_id": incident.insurer_id,
        "incident_type": int(incident.incident_type),  # IntFlag -> bitmask
        "incident_details": incident.incident_details,
        "submitted_date": incident.submitted_date.isoformat(),
        "status": str(incident.status),
        "resolution": str(incident.resolution),
        "resolution_reason": incident.resolution_reason,
        "resolved_date": incident.resolved_date.isoformat() if incident.resolved_date else None,
        "estimate_cost": incident.estimate_cost,
        "history": incident.history,
    }


def incident_from_dict(data: dict[str, Any]) -> Incident:
    """Rebuild an ``Incident`` from its JSON dict (inverse of ``incident_to_dict``)."""
    return Incident(
        id=data["id"],
        policy_id=data["policy_id"],
        adjuster_id=data["adjuster_id"],
        insurer_id=data["insurer_id"],
        incident_type=ReportType(data["incident_type"]),
        incident_details=data["incident_details"],
        submitted_date=datetime.fromisoformat(data["submitted_date"]),
        status=ReportStatus(data["status"]),
        resolution=Resolution(data["resolution"]) if "resolution" in data else Resolution.INPROGRESS,
        resolution_reason=data.get("resolution_reason"),
        resolved_date=datetime.fromisoformat(data["resolved_date"]) if data.get("resolved_date") else None,
        estimate_cost=data.get("estimate_cost", 0),
        history=data.get("history"),
    )


def policy_from_dict(data: dict[str, Any]) -> Policy:
    """Rebuild a ``Policy`` from its JSON dict, parsing the enum and ISO dates."""
    return Policy(
        id=data["id"],
        policy_type=PolicyType(data["policy_type"]),
        insurer_id=data["insurer_id"],
        effective_date=date.fromisoformat(data["effective_date"]),
        expiration_date=date.fromisoformat(data["expiration_date"]),
        premium=data["premium"],
    )


def escalation_to_dict(escalation: Escalation) -> dict[str, Any]:
    """Convert an ``Escalation`` to a JSON-serializable dict."""
    return {
        "incident_id": escalation.incident_id,
        "adjuster_id": escalation.adjuster_id,
        "escalated_date": escalation.escalated_date.isoformat(),
    }


def escalation_from_dict(data: dict[str, Any]) -> Escalation:
    """Rebuild an ``Escalation`` from its JSON dict."""
    return Escalation(
        incident_id=data["incident_id"],
        adjuster_id=data["adjuster_id"],
        escalated_date=datetime.fromisoformat(data["escalated_date"]),
    )
