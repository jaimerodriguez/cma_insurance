"""Tests for the container deployment surface (Dockerfile, entrypoint, config).

These guard failure modes that are invisible locally and only surface after a
push to a registry and a cold start in Azure: a module that joined
``mcp_server``'s import closure but never joined the image, a third-party
import missing from ``requirements-mcp.txt``, or a seeding script that quietly
overwrites a live volume. See ACA_Deploy.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
ENTRYPOINT = PROJECT_ROOT / "entrypoint.sh"
MCP_REQUIREMENTS = PROJECT_ROOT / "requirements-mcp.txt"

# Client-side entry points. They are excluded from the image on purpose, so if
# one ever turns up in the server's import closure that is a real regression —
# the server would start needing the Anthropic client or the Agent SDK.
CLIENT_ONLY = {"cma", "gen_cma_yaml", "manual_setup"}


def _import_closure() -> tuple[set[str], set[str]]:
    """First-party and third-party top-level modules ``mcp_server`` really needs.

    Measured by importing it in a clean interpreter and reading ``sys.modules``,
    rather than by parsing ``import`` statements: that way a function-local
    import stays out (``repl``'s ``claude_agent_sdk`` imports are exactly this,
    and keeping them out is why the image needs no Agent SDK) and a re-export
    picked up through another module stays in.
    """
    probe = (
        "import sys, mcp_server;"
        "root = __import__('pathlib').Path(sys.argv[1]).resolve();"
        "first, third = set(), set();"
        "[(first if str(getattr(m, '__file__', '') or '').startswith(str(root))"
        "  and '/.venv/' not in str(m.__file__)"
        "  else third).add(n.split('.')[0])"
        " for n, m in list(sys.modules.items())"
        " if getattr(m, '__file__', None) and not n.startswith('_')];"
        "print(','.join(sorted(first)) + '|' + ','.join(sorted(third)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe, str(PROJECT_ROOT)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-1]
    first, third = out.split("|")
    stdlib = set(sys.stdlib_module_names)
    return ({m for m in first.split(",") if m},
            {m for m in third.split(",") if m and m not in stdlib})


def _copied_into_image() -> str:
    """The text of every COPY instruction in the Dockerfile, joined.

    Line continuations are folded so a module on a wrapped COPY still counts.
    """
    text = DOCKERFILE.read_text().replace("\\\n", " ")
    return " ".join(line for line in text.splitlines()
                    if line.strip().startswith("COPY"))


def test_every_first_party_module_the_server_imports_is_copied_into_the_image():
    first_party, _ = _import_closure()
    copied = _copied_into_image()
    missing = sorted(
        name for name in first_party
        if f"{name}.py" not in copied and f"{name}/" not in copied
    )
    assert not missing, (
        f"mcp_server imports {missing}, which the Dockerfile never COPYs. "
        f"The image will crash on startup with ModuleNotFoundError."
    )


def test_the_server_does_not_import_the_client_side_entry_points():
    first_party, _ = _import_closure()
    assert not (first_party & CLIENT_ONLY)


def test_the_server_never_pulls_in_the_client_side_packages():
    """`requirements-mcp.txt` omits `anthropic`, `claude-agent-sdk` and `PyYAML`
    because the modules that use them import them inside functions the server
    never calls. That is an assumption about `repl.py`'s internals, so check it
    rather than trust it: promoting one of those to a module-level import would
    otherwise pass every test here and fail on the first cold start in Azure."""
    _, third_party = _import_closure()
    leaked = third_party & {"anthropic", "claude_agent_sdk", "yaml"}
    assert not leaked, (
        f"{sorted(leaked)} reached the server's import closure. Either make the "
        f"import function-local again or add it to requirements-mcp.txt.")


def _direct_third_party_imports(first_party: set[str]) -> set[str]:
    """Top-level packages our own modules import by name, stdlib excluded.

    Deliberately *direct* imports only, parsed from source: the runtime closure
    also contains everything ``mcp`` and ``starlette`` drag in, and pip installs
    those automatically. A transitive dependency missing from the requirements
    file is not a bug; a direct import missing from it is one waiting for the
    day that dependency edge goes away.

    Function-local imports count — ``mcp_server._validation_error`` imports
    ``jsonschema`` inside the function, and the image needs it all the same.
    """
    import ast

    found: set[str] = set()
    files = [PROJECT_ROOT / f"{n}.py" for n in first_party
             if (PROJECT_ROOT / f"{n}.py").exists()]
    files += [p for n in first_party if (PROJECT_ROOT / n).is_dir()
              for p in (PROJECT_ROOT / n).rglob("*.py")]

    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])

    return found - first_party - set(sys.stdlib_module_names)


def test_every_direct_third_party_import_is_declared_in_the_mcp_requirements():
    first_party, _ = _import_closure()
    declared = {
        line.split(">=")[0].split("==")[0].split("[")[0].strip().lower()
        for line in MCP_REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    # Import name != distribution name for these three.
    aliases = {"dotenv": "python-dotenv", "opentelemetry": "opentelemetry-sdk",
               "anyio": "mcp"}          # anyio ships as a hard dep of mcp
    # Imported only by code paths the server never reaches: repl.py's REPL and
    # SDK front-ends are function-local imports that the image never executes.
    client_only_imports = {"claude_agent_sdk", "anthropic", "yaml"}

    missing = sorted(
        name for name in _direct_third_party_imports(first_party)
        if name not in client_only_imports
        and aliases.get(name, name).lower() not in declared
    )
    assert not missing, (
        f"{missing} imported directly by a module in the image but absent from "
        f"requirements-mcp.txt — the container will fail on import.")


def test_claims_data_dir_relocates_every_storage_path():
    """The override must reach the file constants, not just ``DATA_DIR``.

    They are bound at import time from ``DATA_DIR``, so an override applied to
    the directory alone would leave every read and write pointed at the baked-in
    ``/app/data`` while the mounted volume sat empty.
    """
    probe = (
        "import storage, agent_memory;"
        "print(storage.DATA_DIR);"
        "print(storage.INCIDENTS_FILE);"
        "print(storage.ADJUSTERS_FILE);"
        "print(storage.INSURERS_FILE);"
        "print(storage.POLICIES_FILE);"
        "print(storage.ESCALATIONS_FILE);"
        "print(agent_memory.AGENT_MEMORY_FILE)"
    )
    env = {**os.environ, "CLAIMS_DATA_DIR": "/mnt/claims"}
    out = subprocess.run([sys.executable, "-c", probe], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, check=True, env=env)
    lines = out.stdout.strip().splitlines()
    assert lines[0] == "/mnt/claims"
    assert all(line.startswith("/mnt/claims/") for line in lines[1:]), lines


def test_unset_claims_data_dir_keeps_the_repo_data_dir():
    env = {k: v for k, v in os.environ.items() if k != "CLAIMS_DATA_DIR"}
    out = subprocess.run([sys.executable, "-c", "import storage; print(storage.DATA_DIR)"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True,
                         check=True, env=env)
    assert out.stdout.strip() == str(PROJECT_ROOT / "data")


# --- entrypoint.sh ----------------------------------------------------------

def _patched_entrypoint(tmp_path, seed: Path) -> Path:
    """entrypoint.sh with the image's absolute SEED_DIR pointed at a fixture."""
    text = ENTRYPOINT.read_text().replace("SEED_DIR=/app/seed", f"SEED_DIR={seed}")
    path = tmp_path / "entrypoint.sh"
    path.write_text(text)
    return path


def _run(script: Path, data_dir: Path, stub_bin: Path, cwd: Path):
    return subprocess.run(
        ["sh", str(script)], capture_output=True, text=True, cwd=cwd,
        env={"PATH": f"{stub_bin}:{os.environ['PATH']}",
             "CLAIMS_DATA_DIR": str(data_dir)})


@pytest.fixture
def stub_bin(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    (d / "python").write_text("#!/bin/sh\necho STUB_SERVER_STARTED\n")
    (d / "python").chmod(0o755)
    return d


@pytest.fixture
def seed_dir(tmp_path):
    d = tmp_path / "seed"
    d.mkdir()
    (d / "incidents.json").write_text('{"src": "seed"}')
    (d / "adjusters.json").write_text('{"src": "seed"}')
    return d


def test_entrypoint_seeds_an_empty_volume(tmp_path, stub_bin, seed_dir):
    script = _patched_entrypoint(tmp_path, seed_dir)
    data = tmp_path / "vol"
    result = _run(script, data, stub_bin, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "STUB_SERVER_STARTED" in result.stdout
    assert (data / "incidents.json").read_text() == '{"src": "seed"}'
    assert (data / "adjusters.json").read_text() == '{"src": "seed"}'


def test_entrypoint_never_overwrites_a_populated_volume(tmp_path, stub_bin, seed_dir):
    """The live-data case: a restart must not reset the world to the seed."""
    script = _patched_entrypoint(tmp_path, seed_dir)
    data = tmp_path / "vol"
    data.mkdir()
    (data / "incidents.json").write_text('{"src": "live"}')
    (data / "adjusters.json").write_text('{"src": "live"}')

    result = _run(script, data, stub_bin, tmp_path)

    assert result.returncode == 0, result.stderr
    assert (data / "incidents.json").read_text() == '{"src": "live"}'
    assert (data / "adjusters.json").read_text() == '{"src": "live"}'
    assert "seeding" not in result.stdout


def test_entrypoint_fills_only_the_gaps_in_a_partial_volume(tmp_path, stub_bin, seed_dir):
    """A half-written volume — the case a directory-level empty check misses."""
    script = _patched_entrypoint(tmp_path, seed_dir)
    data = tmp_path / "vol"
    data.mkdir()
    (data / "incidents.json").write_text('{"src": "live"}')

    result = _run(script, data, stub_bin, tmp_path)

    assert result.returncode == 0, result.stderr
    assert (data / "incidents.json").read_text() == '{"src": "live"}'   # kept
    assert (data / "adjusters.json").read_text() == '{"src": "seed"}'   # filled
