"""Claude Agent SDK integration: hooks and message-stream observation.

Hooks here are **tracing only** — every callback returns ``{}`` (pass through) and
none of them can change what the agent does. That is a deliberate constraint: an
observability layer that can veto tool calls is no longer an observability layer,
and this project already enforces authorization in ``roles.py`` where it belongs.

``QuotaGuard`` is provided for completeness (it is the one hook use that has to
mutate control flow to be worth anything) but it is **not installed by default** —
pass ``quota=`` explicitly to opt in.

All 10 hook events the SDK exposes are registered. Note that hooks work on the
one-shot ``query()`` path as well as ``ClaudeSDKClient``: ``query()`` runs
streaming mode internally and keeps stdin open while hooks are present. Only
``can_use_tool`` requires an ``AsyncIterable`` prompt.
"""

from __future__ import annotations

from typing import Any

from .usage import Totals

# Every hook event name in claude-agent-sdk 0.2.126.
HOOK_EVENTS = (
    "UserPromptSubmit", "PreToolUse", "PostToolUse", "PostToolUseFailure",
    "PermissionRequest", "Notification", "PreCompact",
    "SubagentStart", "SubagentStop", "Stop",
)


class QuotaGuard:
    """Optional token ceiling on top of the SDK's own ``max_turns``.

    Off by default. Enforcement lands at turn boundaries because it is fed by
    completed-turn ledger totals, so a turn already in flight always finishes.
    """

    def __init__(self, totals: Totals, token_quota: int):
        self.totals = totals
        self.token_quota = token_quota

    @property
    def used(self) -> int:
        return self.totals.all_tokens

    @property
    def fraction(self) -> float:
        return self.used / self.token_quota if self.token_quota else 0.0

    @property
    def exceeded(self) -> bool:
        return self.token_quota > 0 and self.used >= self.token_quota


def build_hooks(obs: Any, *, quota: QuotaGuard | None = None) -> dict[str, list[Any]]:
    """All 10 hook events, wired to the event log.

    Returns a value for ``ClaudeAgentOptions.hooks``. Import of ``HookMatcher`` is
    deferred so this module can be imported without the SDK installed.
    """
    from claude_agent_sdk import HookMatcher

    events = obs.events

    async def user_prompt_submit(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        prompt = hook_input.get("prompt", "") or ""
        events.info("hook.prompt_submitted", chars=len(prompt), prompt=prompt)
        obs.note_session(hook_input.get("session_id"))
        if quota is None:
            return {}
        # The one case where a hook adds context rather than just observing.
        # additionalContext lands after the cached prefix, so per-turn text costs
        # tokens but never invalidates the prompt cache.
        return {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": (
                    f"[harness] token quota used: {quota.used}/{quota.token_quota}"),
            }
        }

    async def pre_tool_use(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        tool = hook_input.get("tool_name", "?")
        events.info("hook.tool_pre", tool=tool, tool_use_id=tool_use_id,
                    input_keys=sorted((hook_input.get("tool_input") or {}).keys()))
        if quota is not None and quota.exceeded:
            events.warn("hook.quota_denied", tool=tool, used=quota.used,
                        quota=quota.token_quota)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Token quota exhausted ({quota.used}/{quota.token_quota}). "
                        "Stop calling tools and summarize what you have."),
                }
            }
        return {}

    async def post_tool_use(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        response = hook_input.get("tool_response")
        events.info("hook.tool_post", tool=hook_input.get("tool_name", "?"),
                    tool_use_id=tool_use_id,
                    response_bytes=len(str(response) if response is not None else ""))
        return {}

    async def post_tool_use_failure(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        events.error("hook.tool_failure", tool=hook_input.get("tool_name", "?"),
                     tool_use_id=tool_use_id, error=str(hook_input.get("error", "")))
        return {}

    async def permission_request(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        events.info("hook.permission_requested", tool=hook_input.get("tool_name", "?"),
                    tool_use_id=tool_use_id)
        return {}

    async def notification(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        events.info("hook.notification", message=str(hook_input.get("message", "")))
        return {}

    async def pre_compact(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        # Worth an explicit level: compaction means the conversation outgrew the
        # window, which changes what later turns can see.
        events.warn("hook.compact", trigger=hook_input.get("trigger"))
        return {}

    async def subagent_start(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        events.info("hook.subagent_start",
                    agent=hook_input.get("agent_type") or hook_input.get("agent_name"))
        return {}

    async def subagent_stop(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        events.info("hook.subagent_stop",
                    agent=hook_input.get("agent_type") or hook_input.get("agent_name"))
        return {}

    async def stop(hook_input: dict, tool_use_id, context) -> dict[str, Any]:
        events.info("hook.turn_stop",
                    **({"quota_used": quota.used} if quota else {}))
        return {}

    callbacks = {
        "UserPromptSubmit": user_prompt_submit,
        "PreToolUse": pre_tool_use,
        "PostToolUse": post_tool_use,
        "PostToolUseFailure": post_tool_use_failure,
        "PermissionRequest": permission_request,
        "Notification": notification,
        "PreCompact": pre_compact,
        "SubagentStart": subagent_start,
        "SubagentStop": subagent_stop,
        "Stop": stop,
    }
    return {event: [HookMatcher(hooks=[cb])] for event, cb in callbacks.items()}


def observe_sdk_message(obs: Any, message: Any) -> None:
    """Trace one message from a ``query()`` / client stream.

    Complements hooks and tool tracing with what only the stream carries: the
    model-side ``tool_use_id`` (so hook rows and dispatch rows can be correlated),
    tool calls that never reached dispatch, and thinking-block sizes.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    events = obs.events
    if isinstance(message, SystemMessage):
        data = message.data or {}
        if message.subtype == "init":
            sid = data.get("session_id")
            obs.note_session(sid)
            events.info("sdk.init", session_id=sid,
                        model=data.get("model"),
                        tools=len(data.get("tools") or []),
                        mcp_servers=[s.get("name") for s in (data.get("mcp_servers") or [])
                                     if isinstance(s, dict)])
        else:
            events.debug("sdk.system", subtype=message.subtype)
    elif isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                events.info("sdk.tool_use", tool=block.name, tool_use_id=block.id,
                            input=block.input)
            elif isinstance(block, ThinkingBlock):
                events.debug("sdk.thinking", chars=len(block.thinking or ""))
            elif isinstance(block, TextBlock):
                events.debug("sdk.text", chars=len(block.text or ""))
    elif isinstance(message, UserMessage):
        content = message.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, ToolResultBlock):
                    events.debug("sdk.tool_result", tool_use_id=block.tool_use_id,
                                 is_error=bool(getattr(block, "is_error", False)))
    elif isinstance(message, ResultMessage):
        obs.note_session(message.session_id)
        events.info("sdk.result", subtype=getattr(message, "subtype", None),
                    is_error=bool(getattr(message, "is_error", False)),
                    num_turns=getattr(message, "num_turns", None))
