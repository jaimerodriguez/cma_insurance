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
    On disk it is stored as the integer bitmask (1, 2, 4, …, or a sum such as 5).

    Members:
        STOLEN_CAR (1): A vehicle was stolen.
        STOLEN_OTHER (2): Non-vehicle property was stolen (e.g. burglary).
        CAR_ACCIDENT (4): A vehicle collision occurred.
        HOME_THEFT (8): Property was stolen from the dwelling.
        HOME_ACCIDENT (16): Accidental damage to the dwelling (e.g. burst pipe,
            kitchen fire).
        HOME_NATURAL_DISASTER (32): Dwelling damage from a natural event
            (e.g. storm, flood, wildfire, earthquake).

    ``HOME`` is a composite alias for the three dwelling-loss flags, so a caller
    can test "is this a home loss at all?" with ``incident_type & ReportType.HOME``
    instead of naming each member. It is an alias rather than a member: it does
    not appear in iteration, so the generated tool schema still advertises only
    the six real flags a caller may combine.
    """

    STOLEN_CAR = auto()
    STOLEN_OTHER = auto()
    CAR_ACCIDENT = auto()
    HOME_THEFT = auto()
    HOME_ACCIDENT = auto()
    HOME_NATURAL_DISASTER = auto()

    HOME = HOME_THEFT | HOME_ACCIDENT | HOME_NATURAL_DISASTER



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
    "inprogress", "disputed", "management_override").

    Members:
        APPROVED: The claim was approved for payout.
        DENIED: The claim was rejected.
        INPROGRESS: Still under review. Default for new incidents.
        DISPUTED: Under dispute / contested.
        MANAGEMENT_OVERRIDE: Management settled the claim outside the normal
            authorization bands. Terminal (see ``is_terminal``).

    Some outcomes end a claim's review and some do not; ``is_terminal`` is the
    single place that distinguishes them.
    """

    APPROVED = auto()
    DENIED = auto()
    INPROGRESS = auto()
    DISPUTED = auto()
    MANAGEMENT_OVERRIDE = auto()

    @property
    def is_terminal(self) -> bool:
        """Whether this outcome ends the claim's review.

        A terminal resolution is what stamps ``Incident.resolved_date``, clears
        any pending ``Escalation``, and makes the claim eligible for the
        auto-close maintenance pass. INPROGRESS and DISPUTED are *not* terminal:
        both mean the claim is still live.

        Exists so the rule lives next to the enum rather than being re-spelled as
        an ``in (APPROVED, DENIED, …)`` tuple at each call site — there are three
        in ``tools.py`` alone, and adding a member used to mean remembering all
        of them.
        """
        return self in (Resolution.APPROVED, Resolution.DENIED,
                        Resolution.MANAGEMENT_OVERRIDE)


class ReportStatus(StrEnum):
    """Lifecycle state of an incident/claim.

    Serialized as its lowercase string value ("new", "open", "notified",
    "escalated_to_management", "closed").

    Members:
        NEW: Just filed, not yet triaged. This is the default for new incidents.
        OPEN: Under active investigation.
        NOTIFIED: The adjuster and insurer have been notified of an update.
        ESCALATED_TO_MANAGEMENT: Handed up to management. An adjuster may set
            this for any reason via ``tools.escalate_to_management``; the agent
            sets it automatically when a claim's cost exceeds every
            authorization band (see ``AuthorizationLevel.OVERRIDE_NEEDED``).
            The claim is still live — its ``resolution`` stays ``INPROGRESS``
            until management decides, typically as
            ``Resolution.MANAGEMENT_OVERRIDE``.
        CLOSED: Resolved; no further action.
    """

    NEW = auto()
    OPEN = auto()
    NOTIFIED = auto()
    ESCALATED_TO_MANAGEMENT = auto()
    CLOSED = auto()


class AuthorizationLevel(StrEnum):
    """How much approval authority an adjuster has.

    Used by the agent when deciding which adjuster to assign an incident to
    (e.g. higher-cost or higher-risk claims go to a higher-authority adjuster).
    Serialized as its lowercase string value ("low", "medium", "high",
    "override_needed").

    Members:
        LOW: Baseline authority. Default for adjusters.
        MEDIUM: Elevated authority.
        HIGH: Full authority.
        OVERRIDE_NEEDED: Not an authority anyone holds — the marker for a claim
            whose cost exceeds *every* band, so no adjuster can decide it and
            management must. Returned by
            ``DynamicPolicies.required_authorization``; never a valid value for
            ``Adjuster.authorization_level`` (``Adjuster`` rejects it).

    The members are ordered by ``rank``, lowest authority first. Note the two
    distinct readings: on an ``Adjuster`` a level is authority *held*, while as a
    ``required_authorization`` result it is authority *needed*. OVERRIDE_NEEDED
    only ever makes sense in the second reading, which is what ``is_assignable``
    exists to enforce.
    """

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    OVERRIDE_NEEDED = auto()

    @property
    def rank(self) -> int:
        """Ordering position; higher means more authority required or held.

        Lets callers compare levels (``adjuster.authorization_level.rank >=
        required.rank``) without a lookup table each site. OVERRIDE_NEEDED ranks
        above HIGH, so the comparison naturally finds no qualifying adjuster.
        """
        return _AUTHORIZATION_ORDER.index(self)

    @property
    def is_assignable(self) -> bool:
        """Whether an adjuster can actually hold this level.

        False only for OVERRIDE_NEEDED, which denotes "beyond every band" rather
        than an authority a person has.
        """
        return self is not AuthorizationLevel.OVERRIDE_NEEDED


# Lowest authority first. Defined next to the enum rather than derived from
# declaration order so that reordering the members cannot silently change the
# comparison semantics that routing depends on.
_AUTHORIZATION_ORDER: tuple[AuthorizationLevel, ...] = (
    AuthorizationLevel.LOW,
    AuthorizationLevel.MEDIUM,
    AuthorizationLevel.HIGH,
    AuthorizationLevel.OVERRIDE_NEEDED,
)


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
            incidents. Defaults to ``AuthorizationLevel.LOW``. Must be a level a
            person can hold — ``OVERRIDE_NEEDED`` is rejected, since it marks a
            claim that is beyond every band rather than an authority anyone has.

    Raises:
        ValueError: If ``authorization_level`` is ``OVERRIDE_NEEDED``.
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
        # Fail loudly rather than let an unassignable level reach the routing
        # code, where an adjuster ranked above HIGH would silently become the
        # candidate for claims no one is allowed to decide.
        if not self.authorization_level.is_assignable:
            raise ValueError(
                f"adjuster {self.user_id!r}: authorization_level "
                f"{self.authorization_level.value!r} is not a level an adjuster "
                "can hold; it marks a claim that needs a management override"
            )


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
        resolved_date: When the resolution was last set to a *terminal* outcome
            (see ``Resolution.is_terminal``) — UTC. Used by the daily maintenance
            job to auto-close claims that have been resolved for a while. None
            until first resolved.
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
    status, whether the policyholder is upset, and whether the claim is a
    *home* claim:

        - ``auto_approve``           always applies (baseline).
        - ``vip_auto_approve``       applies when the insurer is a VIP.
        - ``vip_auto_approve_home``  applies when VIP *and* the claim is home.
        - ``upset_auto_approve_home``applies when the insurer is upset *and*
                                     the claim is home.

    "Home" is broader than the policy type: a claim counts as home when its
    policy is ``PolicyType.HOME`` **or** its ``incident_type`` carries any
    ``ReportType.HOME`` flag, so a dwelling loss reported against a non-HOME
    policy still gets the home ceilings. ``tools.approve_incident`` computes it.

    ``agent_override_autoapprove`` short-circuits everything: when True the
    claim is auto-approved regardless of cost or any other field, and
    ``agent_override_autoapprove_reason`` is recorded on the incident. The
    agent is expected to set these two.

    When a claim is *not* auto-approved, ``can_assign`` controls what happens to
    an as-yet *unassigned* incident: if True it may be routed to an adjuster
    (see ``required_authorization``); if False it is left untouched. The three
    ``authorization_*`` fields are dollar limits mapping a claim's cost to the
    minimum adjuster ``AuthorizationLevel`` able to handle it. A cost above
    ``authorization_high`` maps to ``OVERRIDE_NEEDED``, which routes the claim to
    ``ReportStatus.ESCALATED_TO_MANAGEMENT`` rather than to any adjuster —
    ``can_assign`` does not gate that, because leaving such a claim unassigned
    would strand it with no one able to decide it.

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

    auto_approve: int = 3000
    vip_auto_approve: int = 5000
    upset_auto_approve_home: int = 15000
    vip_auto_approve_home: int = 25000
    agent_override_autoapprove: bool = False
    agent_override_autoapprove_reason: str | None = None
    can_assign: bool = False
    authorization_low: int = 5000
    authorization_medium: int = 15000
    authorization_high: int = 50000

    def required_authorization(self, cost: int) -> AuthorizationLevel:
        """Lowest ``AuthorizationLevel`` able to handle a claim of ``cost``.

        Maps the cost against the ``authorization_*`` limits (inclusive):
        ``cost <= authorization_low`` -> LOW, then MEDIUM, then HIGH. Above
        ``authorization_high`` no adjuster qualifies and the claim needs
        management, which is ``OVERRIDE_NEEDED``.

        Args:
            cost: The incident's estimated cost in dollars.

        Returns:
            The minimum ``AuthorizationLevel`` that qualifies, or
            ``AuthorizationLevel.OVERRIDE_NEEDED`` when the cost exceeds even
            ``authorization_high``. Never ``None`` — "beyond every band" is a
            real answer the caller must route on (to
            ``ReportStatus.ESCALATED_TO_MANAGEMENT``), not a missing one it might
            quietly treat as "leave it alone".
        """
        if cost <= self.authorization_low:
            return AuthorizationLevel.LOW
        if cost <= self.authorization_medium:
            return AuthorizationLevel.MEDIUM
        if cost <= self.authorization_high:
            return AuthorizationLevel.HIGH
        return AuthorizationLevel.OVERRIDE_NEEDED

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
            is_home: Whether this is a home claim — HOME policy type or any
                ``ReportType.HOME`` flag on the incident.
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
