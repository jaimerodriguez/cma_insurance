"""Tests for REPL line editing (`repl.enable_line_editing`).

Without `readline` imported, `input()` leaves the terminal in canonical mode and
an arrow key arrives as its raw escape sequence — Up echoes `^[[A` into the line
instead of recalling the previous command, Shift-Left echoes `^[[1;2D`. Both
REPLs used bare `input()` and nothing imported `readline`.

The main test drives a real pseudo-terminal and sends an actual `\\x1b[A`,
because that is the only way to test the thing that was broken: asserting
`"readline" in sys.modules` would pass even if the import happened after the
first prompt, which is too late to matter.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pty = pytest.importorskip("pty", reason="pseudo-terminals are POSIX-only")


# pty.fork() warns when the parent is multi-threaded, which pytest is. The
# deadlock it guards against needs the child to run Python between fork and
# exec; this one execv's immediately, so the window does not exist.
def _drive(code: str, keystrokes: list[bytes], settle: float = 0.6) -> str:
    """Run `code` under a pty, send `keystrokes`, return everything it printed."""
    pid, fd = pty.fork()
    if pid == 0:                                    # child
        os.chdir(PROJECT_ROOT)
        os.execv(sys.executable, [sys.executable, "-c", code])

    time.sleep(1.2)                                 # let the interpreter reach the prompt
    for keys in keystrokes:
        os.write(fd, keys)
        time.sleep(settle)

    out = b""
    try:
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            out += chunk
    except OSError:                                 # pty closed on child exit
        pass
    os.waitpid(pid, 0)
    return out.decode(errors="replace")


@pytest.fixture
def history_name():
    """A unique history file, removed afterwards.

    The file lands next to `repl.py` rather than in `tmp_path`, because the path
    is derived inside the child process from `repl.__file__`.
    """
    name = f"pytest{os.getpid()}"
    yield name
    (PROJECT_ROOT / "var" / f"{name}_history").unlink(missing_ok=True)


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_up_arrow_recalls_the_previous_line(history_name):
    code = (f"import repl; repl.enable_line_editing({history_name!r});"
            "a=input('1> '); b=input('2> ');"
            "print('FIRST='+a); print('SECOND='+b)")
    out = _drive(code, [b"hello world\n", b"\x1b[A", b"\n"])

    assert "SECOND=hello world" in out, (
        f"Up arrow did not recall the previous line.\n--- terminal output ---\n{out}")
    # The symptom being fixed: the escape sequence must never reach the buffer.
    assert "\x1b[A" not in out.split("FIRST=")[0].replace("1> ", ""), (
        "the raw escape sequence leaked into the input line")


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_editing_survives_an_unwritable_history_location(history_name):
    """History is a convenience; line editing is the point. A read-only var/
    must not stop the REPL starting."""
    code = ("import pathlib, repl;"
            # Point the history at a path that cannot be created.
            "repl.Path = pathlib.Path;"
            f"repl.enable_line_editing('/nonexistent-dir/nope');"
            "a=input('1> '); print('GOT='+a)")
    out = _drive(code, [b"typed\n"])
    assert "GOT=typed" in out, f"REPL failed to run.\n{out}"
    assert "Traceback" not in out


def test_both_repls_enable_line_editing():
    """A cheap guard against the wiring being dropped from either entry point,
    since the pty tests only cover the helper itself."""
    repl_src = (PROJECT_ROOT / "repl.py").read_text()
    cma_src = (PROJECT_ROOT / "cma.py").read_text()

    assert "enable_line_editing(\"repl\")" in repl_src or \
           "enable_line_editing('repl')" in repl_src
    assert "enable_line_editing(\"cma\")" in cma_src or \
           "enable_line_editing('cma')" in cma_src


def test_the_helper_is_not_wrapped_in_the_dispatch_tracer():
    """It sits directly above `_dispatch`, whose `@agent_obs.traced_dispatch`
    decorator this function was once accidentally inserted underneath — which
    silently moved tool tracing onto the wrong function."""
    import repl
    assert not hasattr(repl.enable_line_editing, "__wrapped__")
    assert hasattr(repl._dispatch, "__wrapped__"), "_dispatch lost its tracer"
