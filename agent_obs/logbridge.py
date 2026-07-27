"""Bridge stdlib ``logging`` into the event log.

So that a library's ``logger.warning(...)`` and our ``obs.events.warn(...)`` end up
in the same ordered stream instead of two places you have to correlate by hand.
The source project had no ``logging`` integration at all and left bare ``print()``
calls in library code; this is the replacement.

Off by default — call ``install_logging(obs)`` to attach it. Useful targets are
``claude_agent_sdk`` (control-protocol and transport diagnostics), ``anthropic``
(retries, rate limits), and ``httpx``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

_LEVEL_MAP = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warn",
    logging.ERROR: "error",
    logging.CRITICAL: "error",
}

DEFAULT_LOGGERS = ("claude_agent_sdk", "anthropic")


class ObsHandler(logging.Handler):
    """Forwards records to an ``EventLog`` as ``log.<logger name>`` events."""

    def __init__(self, obs: Any, level: int = logging.INFO):
        super().__init__(level)
        self.obs = obs

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = "<unformattable log record>"
        level = _LEVEL_MAP.get(record.levelno, "info")
        try:
            self.obs.events.event(
                "log", level=level, logger=record.name,
                where=f"{record.module}:{record.lineno}", message=message,
            )
        except Exception:
            # A logging handler that raises breaks the caller. Never propagate.
            pass


def install_logging(obs: Any, *, loggers: Iterable[str] = DEFAULT_LOGGERS,
                    level: int = logging.INFO) -> ObsHandler:
    """Attach an ``ObsHandler`` to the named loggers and return it.

    ``propagate`` is left alone: this adds a destination, it does not take the
    logger away from whatever else the app has configured.
    """
    handler = ObsHandler(obs, level)
    for name in loggers:
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        if logger.level == logging.NOTSET or logger.level > level:
            logger.setLevel(level)
    # `threshold`, not `level`: `level` is an envelope key, and while EventLog now
    # survives the collision it renames the field to `field.level`, which reads as
    # an accident rather than a choice.
    obs.events.debug("logging.installed", loggers=list(loggers),
                     threshold=logging.getLevelName(level))
    return handler


def uninstall_logging(handler: ObsHandler,
                      loggers: Iterable[str] = DEFAULT_LOGGERS) -> None:
    for name in loggers:
        logging.getLogger(name).removeHandler(handler)
