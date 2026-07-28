"""Tests for the `mcp_call.py` command-line MCP client.

The end-to-end paths (a real call over HTTP, the role gate, the error contract)
are covered against a live server in `test_mcp_server.py`. What is worth testing
here is the CLI's own logic — argument coercion, endpoint resolution, and the two
things that were actually wrong the first time it ran: an `ExceptionGroup` hiding
the real HTTP error, and the destructive default target.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

import mcp_call
from mcp_server import McpConfig
from roles import Role

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- argument coercion ------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("20", 20),                       # the count= case, must not stay a string
    ("0", 0),
    ("-3", -3),
    ("2.5", 2.5),
    ("true", True),
    ("false", False),
    ("null", None),
    ('"20"', "20"),                   # explicitly quoted stays a string
    ('["a","b"]', ["a", "b"]),
    ('{"k":1}', {"k": 1}),
    ("ins-1001", "ins-1001"),         # bare id, not JSON
    ("Maria Gonzalez", "Maria Gonzalez"),
    ("", ""),
])
def test_parse_value_coerces_the_way_a_caller_expects(raw, expected):
    assert mcp_call.parse_value(raw) == expected


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity"])
def test_json_non_standard_float_literals_stay_strings(raw):
    """Python's json parser accepts these as floats even though JSON-the-spec has
    no such literals. No tool argument here wants a float infinity, and silently
    turning a value into `nan` is worse than leaving it a string."""
    got = mcp_call.parse_value(raw)
    assert got == raw and isinstance(got, str)


def test_pairs_split_on_the_first_equals_only():
    """A value containing `=` — a token, a query string — must survive intact."""
    got = mcp_call.parse_args_pairs(["reason=cost=too high", "count=5"])
    assert got == {"reason": "cost=too high", "count": 5}


def test_a_pair_without_equals_is_rejected_with_a_usable_message():
    with pytest.raises(SystemExit) as exc:
        mcp_call.parse_args_pairs(["20"])
    assert "key=value" in str(exc.value)


# --- endpoint resolution ----------------------------------------------------

def _cfg(**kw) -> McpConfig:
    return McpConfig(token="t", **kw)


def test_public_url_is_the_default_target_when_set():
    cfg = _cfg(public_url="https://claims.example.com")
    url = mcp_call.resolve_url(cfg, Role.AGENT, local=False, url=None)
    assert url == "https://claims.example.com/mcp/agent"


def test_local_flag_overrides_a_configured_public_url():
    """The escape hatch that keeps a destructive tool off the deployment."""
    cfg = _cfg(public_url="https://claims.example.com", host="127.0.0.1", port=8787)
    url = mcp_call.resolve_url(cfg, Role.AGENT, local=True, url=None)
    assert url == "http://127.0.0.1:8787/mcp/agent"


def test_falls_back_to_local_when_no_public_url_is_configured():
    url = mcp_call.resolve_url(_cfg(port=9999), Role.INSURER, local=False, url=None)
    assert url == "http://127.0.0.1:9999/mcp/insurer"


def test_explicit_url_wins_over_everything():
    cfg = _cfg(public_url="https://configured.example.com")
    url = mcp_call.resolve_url(cfg, Role.ADJUSTER, local=False,
                               url="https://other.example.com")
    assert url == "https://other.example.com/mcp/adjuster"


def test_a_full_endpoint_url_is_not_doubled():
    """Pasting the URL you just curled is the obvious thing to try."""
    cfg = _cfg()
    url = mcp_call.resolve_url(cfg, Role.AGENT, local=False,
                               url="https://x.example.com/mcp/agent")
    assert url == "https://x.example.com/mcp/agent"


def test_the_role_selects_the_endpoint_path():
    cfg = _cfg(public_url="https://x.example.com")
    paths = {r: mcp_call.resolve_url(cfg, r, local=False, url=None) for r in Role}
    assert paths[Role.AGENT].endswith("/mcp/agent")
    assert paths[Role.ADJUSTER].endswith("/mcp/adjuster")
    assert paths[Role.INSURER].endswith("/mcp/insurer")
    assert len(set(paths.values())) == 3


# --- ExceptionGroup unwrapping ---------------------------------------------

def test_root_cause_unwraps_the_taskgroup_exception_group():
    """`streamable_http_client` runs inside a task group, so a plain 401 arrives
    wrapped. Catching httpx.HTTPError alone misses every one of them and prints a
    twenty-line traceback where one line belongs — which is what it did."""
    request = httpx.Request("POST", "https://x.example.com/mcp/agent")
    response = httpx.Response(401, request=request)
    real = httpx.HTTPStatusError("401", request=request, response=response)

    wrapped = BaseExceptionGroup("unhandled errors in a TaskGroup", [real])
    assert mcp_call.root_cause(wrapped) is real

    nested = BaseExceptionGroup("outer", [BaseExceptionGroup("inner", [real])])
    assert mcp_call.root_cause(nested) is real


def test_root_cause_passes_a_bare_exception_through():
    err = ValueError("plain")
    assert mcp_call.root_cause(err) is err


# --- CLI surface ------------------------------------------------------------

def _run(*argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    import os
    base = {k: v for k, v in os.environ.items()
            if k not in ("MCP_PUBLIC_URL", "MCP_BEARER_TOKEN", "MCP_TOOLS")}
    return subprocess.run([sys.executable, "mcp_call.py", *argv],
                          cwd=PROJECT_ROOT, capture_output=True, text=True,
                          env={**base, **(env or {})})


def test_missing_token_is_refused_before_any_network_call():
    result = _run("--local", "agent", "list_incidents",
                  env={"MCP_PUBLIC_URL": "", "MCP_BEARER_TOKEN": ""})
    assert result.returncode != 0
    assert "MCP_BEARER_TOKEN" in result.stderr


def test_a_bad_argument_pair_reports_before_announcing_a_target():
    """A key=value typo must not print a line implying we contacted the server."""
    result = _run("--local", "agent", "generate_unassigned_incidents", "20",
                  env={"MCP_BEARER_TOKEN": "t"})
    assert result.returncode == 1
    assert "key=value" in result.stderr
    assert "→" not in result.stderr


def test_the_target_url_is_echoed_to_stderr_not_stdout():
    """Destructive tools default to the *deployed* server when MCP_PUBLIC_URL is
    set, so the target has to be visible — but on stderr, so piping the JSON
    result still works."""
    result = _run("--local", "agent", "list_incidents",
                  env={"MCP_BEARER_TOKEN": "t", "MCP_PORT": "1"})
    assert "→ http://127.0.0.1:1/mcp/agent" in result.stderr
    assert "→" not in result.stdout


def test_an_unreachable_server_gives_one_line_not_a_traceback():
    """Port 1 refuses instantly. Before the ExceptionGroup fix this printed a
    twenty-line anyio traceback."""
    result = _run("--local", "agent", "list_incidents",
                  env={"MCP_BEARER_TOKEN": "t", "MCP_PORT": "1"})
    assert result.returncode == 2
    assert "Could not reach" in result.stderr
    assert "Traceback" not in result.stderr
    assert "ExceptionGroup" not in result.stderr


def test_omitting_the_tool_without_list_is_a_usage_error():
    result = _run("--local", "agent", env={"MCP_BEARER_TOKEN": "t"})
    assert result.returncode == 2
    assert "--list" in result.stderr


def test_an_unknown_role_is_rejected_by_the_parser():
    result = _run("--local", "manager", "list_incidents",
                  env={"MCP_BEARER_TOKEN": "t"})
    assert result.returncode == 2
    assert "manager" in result.stderr
