"""Agent-callable tools for the insurance incident-report mock.

This module is the *agent tool surface*. Every function here is meant to be
invoked by an AI agent; the ones to expose are listed in the ``AGENT_TOOLS``
registry at the bottom of the file (the single source of truth). The data model
lives in ``data_entities`` and file persistence in ``storage`` — neither is part
of the agent surface.

Conventions:
    - Every ``find_*``/``get_*`` tool returns ``None`` when the id is unknown,
      so callers should handle a missing result rather than assume success.
    - ``append_*`` adds to a history/preferences field (history entries are
      date-stamped); ``set_*`` overwrites it (pass ``None`` to clear).
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from data_entities import (
    Adjuster,
    AuthorizationLevel,
    DynamicPolicies,
    Escalation,
    Incident,
    Insurer,
    Policy,
    PolicyType,
    ReportStatus,
    ReportType,
    Resolution,
)
import storage

# Placeholder adjuster id for incidents that have not been routed yet.
UNASSIGNED = "unassigned"

def _select_adjuster(policies: DynamicPolicies, cost: int) -> str | None:
    """Pick the id of the lowest-authorization adjuster able to handle ``cost``.

    Uses ``policies.required_authorization(cost)`` to find the minimum level,
    then returns the qualifying adjuster with the lowest authorization (ties
    broken by user_id for determinism). The "unassigned" placeholder is never
    selected.

    Returns ``None`` when no adjuster has sufficient authorization. Callers must
    not read that as "cost is beyond every band" — those two cases are different
    and are distinguished by ``required_authorization`` returning
    ``OVERRIDE_NEEDED``, which ``process_incident`` checks *before* calling here.
    An adjuster can never hold OVERRIDE_NEEDED (``Adjuster`` rejects it), so if
    one did reach this function the rank comparison would find no candidate
    anyway.
    """
    required = policies.required_authorization(cost)
    candidates = [
        a for a in list_adjusters()
        if a.user_id != UNASSIGNED
        and a.authorization_level.is_assignable
        and a.authorization_level.rank >= required.rank
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda a: (a.authorization_level.rank, a.user_id))
    return best.user_id


def _stamp(note: str) -> str:
    """Prefix a note with today's date, e.g. "7/23/2026 - <note>"."""
    now = datetime.now()
    return f"{now.month}/{now.day}/{now.year} - {note}"


def _append_line(existing: str | None, line: str) -> str:
    """Append ``line`` to ``existing`` on its own line (no leading blank line)."""
    return line if not existing else f"{existing}\n{line}"


# --- reads / discovery ------------------------------------------------------

def get_incident_details(incident_id: str) -> Incident | None:
    """Look up a single incident by its id.

    Args:
        incident_id: The incident/case id to fetch (e.g. "case-3001").

    Returns:
        The matching ``Incident``, or ``None`` if no incident has that id.
    """
    return storage.load_incidents().get(incident_id)


def list_incidents(adjuster_id: str) -> list[Incident]:
    """List the incidents assigned to a given adjuster.

    Use this to discover incidents when you only know the adjuster (rather
    than a specific case id).

    Args:
        adjuster_id: The adjuster's id whose incidents to list (e.g. "jaime").

    Returns:
        A list of ``Incident`` objects handled by that adjuster (empty if the
        adjuster has none).
    """
    return [i for i in storage.load_incidents().values() if i.adjuster_id == adjuster_id]


def list_incidents_for_insurer(insurer_id: str) -> list[Incident]:
    """List the incidents (claims) belonging to a given insurer (policyholder).

    Use this to answer a policyholder's questions about the status of their
    own claims.

    Args:
        insurer_id: The insurer's id whose incidents to list (e.g. "ins-1001").

    Returns:
        A list of ``Incident`` objects filed by that insurer (empty if none).
    """
    return [i for i in storage.load_incidents().values() if i.insurer_id == insurer_id]


def find_adjuster(adjuster_id: str) -> Adjuster | None:
    """Look up a single adjuster by their first-name id.

    Args:
        adjuster_id: The adjuster's id / lookup key (e.g. "jaime").

    Returns:
        The matching ``Adjuster``, or ``None`` if no adjuster has that id.
    """
    return storage.load_adjusters().get(adjuster_id)


def list_adjusters() -> list[Adjuster]:
    """List every adjuster on the roster.

    Useful for routing an incident to an adjuster with a suitable
    ``authorization_level`` (see ``assign_incident``).

    Returns:
        All ``Adjuster`` records (may include the "unassigned" placeholder).
    """
    return list(storage.load_adjusters().values())


def find_insurer(insurer_id: str) -> Insurer | None:
    """Look up a single insurer (policyholder) by id.

    Args:
        insurer_id: The insurer's id / lookup key (e.g. "ins-1001").

    Returns:
        The matching ``Insurer``, or ``None`` if no insurer has that id.
    """
    return storage.load_insurers().get(insurer_id)


def find_policy(policy_id: str) -> Policy | None:
    """Look up a single policy by id.

    Args:
        policy_id: The policy's id / lookup key (e.g. "pol-2001").

    Returns:
        The matching ``Policy``, or ``None`` if no policy has that id.
    """
    return storage.load_policies().get(policy_id)


def retrieve_policies(insurer_id: str) -> list[Policy]:
    """List every policy held by a given insurer (policyholder).

    Use this to see which policies a policyholder holds (e.g. before filing a
    claim against one of them).

    Args:
        insurer_id: The insurer's id whose policies to list (e.g. "ins-1001").

    Returns:
        A list of ``Policy`` objects held by that insurer (empty if none).
    """
    return [p for p in storage.load_policies().values() if p.insurer_id == insurer_id]


def get_adjuster_details(incident_id: str) -> Adjuster | None:
    """Resolve the adjuster assigned to a given incident.

    Follows the chain: incident id -> the incident's ``adjuster_id`` -> the
    adjuster record.

    Args:
        incident_id: The incident/case id to resolve (e.g. "case-3001").

    Returns:
        The assigned ``Adjuster``, or ``None`` if the incident does not exist
        or its ``adjuster_id`` matches no known adjuster.
    """
    incident = get_incident_details(incident_id)
    if incident is None:
        return None
    return find_adjuster(incident.adjuster_id)


def list_escalations(adjuster_id: str) -> list[Escalation]:
    """List the escalations awaiting a given adjuster's decision.

    Args:
        adjuster_id: The adjuster's id (e.g. "jaime").

    Returns:
        A list of ``Escalation`` entries assigned to that adjuster (empty if
        none are pending).
    """
    return [e for e in storage.load_escalations().values() if e.adjuster_id == adjuster_id]


# --- create -----------------------------------------------------------------

def create_incident(
    adjuster_id: str,
    type: ReportType,
    details: str,
    insurer_id: str,
    policy_id: str,
    estimate_cost: int = 0,
) -> Incident:
    """Create a new incident (claim) and persist it.

    A fresh unique id is generated, ``submitted_date`` is set to now (UTC),
    ``status`` defaults to ``ReportStatus.NEW`` and ``resolution`` to
    ``Resolution.INPROGRESS``. The new incident is saved before being returned.

    The referenced ids are stored as-is and not validated here; use
    ``find_adjuster``/``find_insurer``/``find_policy`` first if you need to
    confirm they exist.

    Args:
        adjuster_id: Id of the adjuster who will handle the claim (e.g. "jaime").
        type: One or more ``ReportType`` flags describing the loss. Combine
            with ``|`` (or by adding the values) when a single event caused
            several kinds of loss, e.g. ``STOLEN_CAR | CAR_ACCIDENT`` = 5, or
            ``HOME_ACCIDENT | HOME_NATURAL_DISASTER`` = 48 for storm damage that
            also burst a pipe. A dwelling loss (any HOME_* flag) raises the
            auto-approval ceilings during triage, so pick the home flags for
            home losses even when the claim sits on a non-HOME policy.
        details: Free-text description of the incident.
        insurer_id: Id of the insurer/policyholder (e.g. "ins-1001").
        policy_id: Id of the policy the claim is filed against (e.g. "pol-2001").
        estimate_cost: Estimated cost of the incident in whole dollars. Used
            by ``process_incident``. Defaults to 0.

    Returns:
        The newly created and persisted ``Incident`` (including its generated id).
    """
    incident = Incident(
        policy_id=policy_id,
        adjuster_id=adjuster_id,
        insurer_id=insurer_id,
        incident_type=type,
        incident_details=details,
        estimate_cost=estimate_cost,
    )
    storage.save_incident(incident)
    return incident


# --- history: append (date-stamped) -----------------------------------------

def append_incident_history(incident_id: str, note: str) -> Incident | None:
    """Append a date-stamped note to an incident's history log and save it.

    The note is prefixed with today's date and added on a new line, e.g.
    "left voicemail for policyholder" becomes
    "7/23/2026 - left voicemail for policyholder".

    Args:
        incident_id: The incident/case id to update (e.g. "case-3001").
        note: The activity to record (without a date; it is added for you).

    Returns:
        The updated ``Incident``, or ``None`` if no incident has that id.
    """
    return storage.update_incident(
        incident_id,
        lambda i: setattr(i, "history", _append_line(i.history, _stamp(note))),
    )


def append_adjuster_history(adjuster_id: str, note: str) -> Adjuster | None:
    """Append a date-stamped note to an adjuster's history log and save it.

    Args:
        adjuster_id: The adjuster's id (e.g. "jaime").
        note: The interaction to record (without a date; it is added for you).

    Returns:
        The updated ``Adjuster``, or ``None`` if no adjuster has that id.
    """
    return storage.update_adjuster(
        adjuster_id,
        lambda a: setattr(a, "history", _append_line(a.history, _stamp(note))),
    )


def append_insurer_history(insurer_id: str, note: str) -> Insurer | None:
    """Append a date-stamped note to an insurer's history log and save it.

    Args:
        insurer_id: The insurer's id (e.g. "ins-1001").
        note: The interaction to record (without a date; it is added for you).

    Returns:
        The updated ``Insurer``, or ``None`` if no insurer has that id.
    """
    return storage.update_insurer(
        insurer_id,
        lambda ins: setattr(ins, "history", _append_line(ins.history, _stamp(note))),
    )


# --- preferences: append (not stamped) --------------------------------------

def append_adjuster_preferences(adjuster_id: str, note: str) -> Adjuster | None:
    """Append a preference line to an adjuster and save it.

    Unlike history, preferences are *not* date-stamped — they describe
    standing wishes. The note is added on its own line.

    Args:
        adjuster_id: The adjuster's id (e.g. "jaime").
        note: The preference to add, e.g. "Prefers email over phone".

    Returns:
        The updated ``Adjuster``, or ``None`` if no adjuster has that id.
    """
    return storage.update_adjuster(
        adjuster_id,
        lambda a: setattr(a, "preferences", _append_line(a.preferences, note)),
    )


def append_insurer_preferences(insurer_id: str, note: str) -> Insurer | None:
    """Append a preference line to an insurer (policyholder) and save it.

    Unlike history, preferences are *not* date-stamped — they describe
    standing wishes. The note is added on its own line.

    Args:
        insurer_id: The insurer's id (e.g. "ins-1001").
        note: The preference to add, e.g. "Their name is Joseph, but goes by Joe".

    Returns:
        The updated ``Insurer``, or ``None`` if no insurer has that id.
    """
    return storage.update_insurer(
        insurer_id,
        lambda ins: setattr(ins, "preferences", _append_line(ins.preferences, note)),
    )


# --- history / preferences: set (overwrite or clear) ------------------------

def set_incident_history(incident_id: str, value: str | None) -> Incident | None:
    """Overwrite an incident's entire history field (pass None to clear).

    Replaces whatever is there; nothing is date-stamped. Use
    ``append_incident_history`` to add a dated entry instead.

    Args:
        incident_id: The incident/case id to update (e.g. "case-3001").
        value: The new full history text, or ``None`` to clear it.

    Returns:
        The updated ``Incident``, or ``None`` if no incident has that id.
    """
    return storage.update_incident(incident_id, lambda i: setattr(i, "history", value))


def set_adjuster_history(adjuster_id: str, value: str | None) -> Adjuster | None:
    """Overwrite an adjuster's entire history field (pass None to clear).

    Args:
        adjuster_id: The adjuster's id (e.g. "jaime").
        value: The new full history text, or ``None`` to clear it.

    Returns:
        The updated ``Adjuster``, or ``None`` if no adjuster has that id.
    """
    return storage.update_adjuster(adjuster_id, lambda a: setattr(a, "history", value))


def set_insurer_history(insurer_id: str, value: str | None) -> Insurer | None:
    """Overwrite an insurer's entire history field (pass None to clear).

    Args:
        insurer_id: The insurer's id (e.g. "ins-1001").
        value: The new full history text, or ``None`` to clear it.

    Returns:
        The updated ``Insurer``, or ``None`` if no insurer has that id.
    """
    return storage.update_insurer(insurer_id, lambda ins: setattr(ins, "history", value))


def set_adjuster_preferences(adjuster_id: str, value: str | None) -> Adjuster | None:
    """Overwrite an adjuster's entire preferences field (pass None to clear).

    Args:
        adjuster_id: The adjuster's id (e.g. "jaime").
        value: The new full preferences text, or ``None`` to clear it.

    Returns:
        The updated ``Adjuster``, or ``None`` if no adjuster has that id.
    """
    return storage.update_adjuster(adjuster_id, lambda a: setattr(a, "preferences", value))


def set_insurer_preferences(insurer_id: str, value: str | None) -> Insurer | None:
    """Overwrite an insurer's entire preferences field (pass None to clear).

    Args:
        insurer_id: The insurer's id (e.g. "ins-1001").
        value: The new full preferences text, or ``None`` to clear it.

    Returns:
        The updated ``Insurer``, or ``None`` if no insurer has that id.
    """
    return storage.update_insurer(insurer_id, lambda ins: setattr(ins, "preferences", value))


# --- incident workflow ------------------------------------------------------

def update_resolution(
    incident_id: str,
    adjuster: str,
    resolution: Resolution | str,
    reason: str | None = None,
) -> Incident | None:
    """Set an incident's resolution and record who decided it (and why).

    Appends a date-stamped line to the incident's history noting the new
    resolution, the deciding adjuster, and the reason if given. When ``reason``
    is provided it is also stored on the incident's ``resolution_reason``. If
    the resolution is *terminal* (see ``Resolution.is_terminal`` — APPROVED,
    DENIED or MANAGEMENT_OVERRIDE) the incident's ``resolved_date`` is stamped
    and any pending escalation for this incident is removed from the queue (this
    is how an adjuster clears an escalation they were asked to decide).

    Args:
        incident_id: The incident/case id to update (e.g. "case-3001").
        adjuster: Id of the adjuster making the decision (use "system" for
            automated approvals).
        resolution: The new ``Resolution`` (or its string value, e.g. "approved",
            "management_override").
        reason: Optional free-text explanation, stored on the incident and
            noted in its history.

    Returns:
        The updated ``Incident``, or ``None`` if no incident has that id.
    """
    resolution = Resolution(resolution)

    def mutate(incident: Incident) -> None:
        incident.resolution = resolution
        if reason is not None:
            incident.resolution_reason = reason
        if resolution.is_terminal:
            incident.resolved_date = datetime.now(timezone.utc)
        note = f"resolution set to {resolution.value} by {adjuster}"
        if reason:
            note += f" ({reason})"
        incident.history = _append_line(incident.history, _stamp(note))

    incident = storage.update_incident(incident_id, mutate)
    if incident is None:
        return None

    if resolution.is_terminal:
        escalations = storage.load_escalations()
        if escalations.pop(incident_id, None) is not None:
            storage.save_escalations(escalations)
    return incident


def update_status(incident_id: str, status: ReportStatus | str) -> Incident | None:
    """Set an incident's lifecycle status and log the change.

    Args:
        incident_id: The incident/case id to update (e.g. "case-3001").
        status: The new ``ReportStatus`` (or its string value, e.g. "closed").

    Returns:
        The updated ``Incident``, or ``None`` if no incident has that id.
    """
    status = ReportStatus(status)

    def mutate(incident: Incident) -> None:
        incident.status = status
        incident.history = _append_line(
            incident.history, _stamp(f"status set to {status.value}")
        )

    return storage.update_incident(incident_id, mutate)


def assign_incident(incident_id: str, adjuster_id: str) -> Incident | None:
    """Assign an incident to an adjuster and log the reassignment.

    Sets the incident's ``adjuster_id`` and appends a date-stamped note to its
    history recording the new assignee. The adjuster must exist in
    ``adjusters.json``; use ``find_adjuster`` first if unsure.

    Args:
        incident_id: The incident/case id to assign (e.g. "case-3001").
        adjuster_id: Id of the adjuster to assign it to (e.g. "jaime").

    Returns:
        The updated ``Incident``, or ``None`` if no incident has that id or no
        adjuster has ``adjuster_id``.
    """
    if find_adjuster(adjuster_id) is None:
        return None

    def mutate(incident: Incident) -> None:
        incident.adjuster_id = adjuster_id
        incident.history = _append_line(
            incident.history, _stamp(f"assigned to {adjuster_id}")
        )

    return storage.update_incident(incident_id, mutate)


def notify_adjuster(incident_id: str) -> str | None:
    """Notify the incident's adjuster that its resolution status changed.

    Appends a date-stamped note to the adjuster's history and returns a
    human-readable confirmation.

    Args:
        incident_id: The incident/case id (e.g. "case-3001").

    Returns:
        A confirmation message, or ``None`` if the incident or its adjuster
        can't be found.
    """
    incident = get_incident_details(incident_id)
    if incident is None:
        return None
    adjuster = find_adjuster(incident.adjuster_id)
    if adjuster is None:
        return None
    append_adjuster_history(
        incident.adjuster_id,
        f"notified: incident {incident_id} resolution is now {incident.resolution.value}",
    )
    who = adjuster.full_name or adjuster.user_id
    return f"Notified adjuster {who}: incident {incident_id} resolution is now {incident.resolution.value}."


def notify_insurer(incident_id: str) -> str | None:
    """Notify the incident's insurer (policyholder) that its resolution changed.

    Appends a date-stamped note to the insurer's history and returns a
    human-readable confirmation.

    Args:
        incident_id: The incident/case id (e.g. "case-3001").

    Returns:
        A confirmation message, or ``None`` if the incident or its insurer
        can't be found.
    """
    incident = get_incident_details(incident_id)
    if incident is None:
        return None
    insurer = find_insurer(incident.insurer_id)
    if insurer is None:
        return None
    append_insurer_history(
        incident.insurer_id,
        f"notified: incident {incident_id} resolution is now {incident.resolution.value}",
    )
    return f"Notified policyholder {insurer.full_name}: incident {incident_id} resolution is now {incident.resolution.value}."


def notify_update(incident_id: str) -> str | None:
    """Notify both the adjuster and the insurer of an incident update.

    Calls ``notify_adjuster`` and ``notify_insurer``, then sets the incident's
    status to ``ReportStatus.NOTIFIED``.

    Args:
        incident_id: The incident/case id (e.g. "case-3001").

    Returns:
        A combined confirmation message, or ``None`` if the incident does not
        exist.
    """
    incident = get_incident_details(incident_id)
    if incident is None:
        return None
    messages = [notify_adjuster(incident_id), notify_insurer(incident_id)]
    update_status(incident_id, ReportStatus.NOTIFIED)
    return " ".join(m for m in messages if m) or "No parties could be notified."


def escalate_incident(
    incident_id: str, policies: DynamicPolicies | dict[str, Any] | None = None
) -> Escalation | None:
    """Escalate an incident to its adjuster for an approve/deny decision.

    Adds an entry to the escalation queue keyed by the incident id and assigned
    to the incident's adjuster, and logs the escalation on the incident's
    history. The adjuster later resolves it via ``update_resolution`` (which
    clears it from the queue).

    Args:
        incident_id: The incident/case id to escalate (e.g. "case-3004").
        policies: The ``DynamicPolicies`` in effect (a ``DynamicPolicies``, a
            plain dict of its fields, or None). Used only to annotate the
            history note with the baseline auto-approve ceiling. Defaults to a
            fresh ``DynamicPolicies()`` when omitted.

    Returns:
        The created ``Escalation``, or ``None`` if no incident has that id.
    """
    policies = DynamicPolicies.coerce(policies)
    incident = get_incident_details(incident_id)
    if incident is None:
        return None
    escalations = storage.load_escalations()
    escalation = Escalation(incident_id=incident_id, adjuster_id=incident.adjuster_id)
    escalations[incident_id] = escalation
    storage.save_escalations(escalations)
    append_incident_history(
        incident_id,
        f"escalated to adjuster {incident.adjuster_id} for approval "
        f"(cost {incident.estimate_cost} exceeds auto-approve ceilings; "
        f"baseline {policies.auto_approve})",
    )
    return escalation


def escalate_to_management(incident_id: str, reason: str) -> Incident | None:
    """Hand an incident up to management, for any reason.

    Sets ``status`` to ``ReportStatus.ESCALATED_TO_MANAGEMENT``, stores ``reason``
    on the incident's ``resolution_reason``, and logs a date-stamped line to its
    history. Any pending adjuster escalation for the incident is withdrawn from
    the queue, because the decision is no longer the adjuster's to make.

    The claim stays live: ``resolution`` is left at ``INPROGRESS`` until
    management decides, typically by calling ``update_resolution`` with
    ``Resolution.MANAGEMENT_OVERRIDE``. This is deliberately *not* a resolution
    of its own — an escalation is a routing step, and treating it as an outcome
    would stamp ``resolved_date`` and let the auto-close pass close a claim
    nobody has decided.

    An adjuster may call this for any reason (a conflict of interest, a
    policyholder threatening litigation, an ambiguous exclusion). The agent calls
    it automatically when a claim's cost exceeds every authorization band — see
    ``AuthorizationLevel.OVERRIDE_NEEDED`` and ``process_incident``.

    Args:
        incident_id: The incident/case id to escalate (e.g. "case-3001").
        reason: Why this needs management. Free text, stored on the incident and
            noted in its history.

    Returns:
        The updated ``Incident``, or ``None`` if no incident has that id.
    """
    incident = get_incident_details(incident_id)
    if incident is None:
        return None

    escalations = storage.load_escalations()
    if escalations.pop(incident_id, None) is not None:
        storage.save_escalations(escalations)

    def mutate(updated: Incident) -> None:
        updated.status = ReportStatus.ESCALATED_TO_MANAGEMENT
        updated.resolution_reason = reason
        updated.history = _append_line(
            updated.history, _stamp(f"escalated to management ({reason})")
        )

    return storage.update_incident(incident_id, mutate)


def process_incident(
    incident_id: str,
    policies: DynamicPolicies | dict[str, Any] | None = None,
    insurer_upset: bool = False,
) -> Incident | None:
    """Triage an incident for approval using the given dynamic policies.

    The decision is delegated to ``policies.should_auto_approve`` (see
    ``DynamicPolicies``), which considers the incident cost, whether the
    policyholder is a VIP (read from the insurer record), whether this is a home
    claim, whether the policyholder is upset, and any agent override. A claim
    counts as *home* when its policy is ``PolicyType.HOME`` or its
    ``incident_type`` carries any ``ReportType.HOME`` flag.

    If it auto-approves, the resolution is set to APPROVED via
    ``update_resolution`` (adjuster "system") with the reason recorded.

    An incident already at ``ESCALATED_TO_MANAGEMENT`` is returned untouched:
    management owns the decision, and the pass is run repeatedly.

    Otherwise, in order:

    1. If the cost exceeds every authorization band
       (``required_authorization`` returns ``OVERRIDE_NEEDED``), the incident is
       sent to management via ``escalate_to_management`` — status
       ``ESCALATED_TO_MANAGEMENT``, resolution still ``INPROGRESS``. No adjuster
       could decide it, so this happens regardless of ``can_assign`` or whether
       the incident is already assigned.
    2. Else if it is unassigned, it is routed to the lowest-authorization
       adjuster who qualifies — or left untouched when ``can_assign`` is False
       or nobody qualifies.
    3. Else it is escalated to its own adjuster (resolution stays
       ``INPROGRESS``) and the escalation reason is stored on the incident.

    Whether the insurer is upset is a judgement call, so it is supplied by the
    caller rather than computed here: the agent should read the incident and
    the insurer's history and decide. It defaults to False when unknown.

    Args:
        incident_id: The incident/case id to triage (e.g. "case-3001").
        policies: The ``DynamicPolicies`` (ceilings + agent override) to apply,
            as a ``DynamicPolicies``, a plain dict of its fields, or None.
            Defaults to a fresh ``DynamicPolicies()`` when omitted.
        insurer_upset: True if the agent judges the policyholder to be upset.
            Defaults to False.

    Returns:
        The incident after triage — APPROVED if auto-approved, status
        ESCALATED_TO_MANAGEMENT if it needs an override, newly assigned if it was
        routed, otherwise unchanged and now in the adjuster escalation queue.
        ``None`` if no incident has that id.
    """
    policies = DynamicPolicies.coerce(policies)
    incident = get_incident_details(incident_id)
    if incident is None:
        return None

    # Already with management: not the agent's to re-triage. Returning early
    # keeps this idempotent — the claim stays selected by "still INPROGRESS"
    # filters, so a scheduled pass revisits it every run, and without this guard
    # each visit would re-stamp the history and re-decide over management's head.
    # It also outranks agent_override_autoapprove on purpose: an escalation to
    # management is precisely the case where the agent stops deciding.
    if incident.status is ReportStatus.ESCALATED_TO_MANAGEMENT:
        return incident

    insurer = find_insurer(incident.insurer_id)
    policy = find_policy(incident.policy_id)
    is_vip = bool(insurer and insurer.is_VIP)
    # A home claim by *either* measure: the coverage it was filed against, or the
    # kind of loss reported. A dwelling loss reported against a non-HOME policy
    # still gets the home ceilings — the alternative is that HOME_THEFT on an
    # AUTO policy is silently treated as a car claim.
    is_home = bool(
        (policy and policy.policy_type == PolicyType.HOME)
        or (incident.incident_type & ReportType.HOME)
    )

    approve, reason = policies.should_auto_approve(
        incident.estimate_cost, is_vip, is_home, insurer_upset
    )
    if approve:
        return update_resolution(incident_id, "system", Resolution.APPROVED, reason=reason)

    # Not auto-approved. Before any routing: if the cost is beyond every
    # authorization band, no adjuster may decide it however it is assigned, so it
    # goes straight to management. This is checked ahead of `can_assign` on
    # purpose — `can_assign=False` means "don't pick an adjuster", not "strand a
    # claim nobody is allowed to decide".
    required = policies.required_authorization(incident.estimate_cost)
    if required is AuthorizationLevel.OVERRIDE_NEEDED:
        return escalate_to_management(
            incident_id,
            f"cost {incident.estimate_cost} exceeds every authorization limit "
            f"(high {policies.authorization_high}); needs a management override",
        )

    # An unassigned incident must not be escalated to the "unassigned"
    # placeholder: route it to an adjuster if the policy allows, otherwise leave
    # it untouched for a later pass.
    if incident.adjuster_id == UNASSIGNED:
        if not policies.can_assign:
            return incident  # do nothing
        adjuster_id = _select_adjuster(policies, incident.estimate_cost)
        if adjuster_id is None:
            return incident  # no adjuster has sufficient authorization; leave it
        return assign_incident(incident_id, adjuster_id)

    escalate_incident(incident_id, policies)
    # record why it escalated without changing the (still INPROGRESS) resolution
    return storage.update_incident(
        incident_id, lambda i: setattr(i, "resolution_reason", reason)
    )


# --- maintenance ------------------------------------------------------------

def close_stale_resolved(minutes: int = 30) -> list[Incident]:
    """Auto-close incidents that have been resolved a while.

    Scans all incidents and sets ``status`` to ``ReportStatus.CLOSED`` for any
    whose ``resolution`` is terminal (see ``Resolution.is_terminal`` — APPROVED,
    DENIED or MANAGEMENT_OVERRIDE), whose ``resolved_date`` is more than
    ``minutes`` minutes ago, and which are not already CLOSED. Intended to be run
    on a schedule (e.g. once a day) or on demand.

    Args:
        minutes: Age threshold in minutes since ``resolved_date``. Defaults to 30.

    Returns:
        The list of incidents that were closed by this call (empty if none
        were due).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    closed: list[Incident] = []
    for incident in storage.load_incidents().values():
        if (
            incident.resolution.is_terminal
            and incident.status is not ReportStatus.CLOSED
            and incident.resolved_date is not None
            and incident.resolved_date < cutoff
        ):
            updated = update_status(incident.id, ReportStatus.CLOSED)
            if updated is not None:
                closed.append(updated)
    return closed


# --- agent tool registry ----------------------------------------------------
#
# The single source of truth for which functions should be exposed to an AI
# agent. A schema generator should iterate over this list. Everything not here
# (the ``data_entities`` model and the ``storage`` persistence layer) is
# internal and must NOT be turned into a tool.

AGENT_TOOLS: list[Callable[..., Any]] = [
    # -- reads / discovery --
    get_incident_details,
    list_incidents,
    list_incidents_for_insurer,
    get_adjuster_details,
    find_adjuster,
    list_adjusters,
    find_insurer,
    find_policy,
    retrieve_policies,
    list_escalations,
    # -- create --
    create_incident,
    # -- incident workflow --
    process_incident,
    escalate_incident,
    escalate_to_management,
    update_resolution,
    update_status,
    assign_incident,
    notify_adjuster,
    notify_insurer,
    notify_update,
    # -- maintenance --
    close_stale_resolved,
    # -- history --
    append_incident_history,
    append_adjuster_history,
    append_insurer_history,
    set_incident_history,
    set_adjuster_history,
    set_insurer_history,
    # -- preferences --
    append_adjuster_preferences,
    append_insurer_preferences,
    set_adjuster_preferences,
    set_insurer_preferences,
]
