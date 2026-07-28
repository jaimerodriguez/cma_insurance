"""Streamable-HTTP MCP server exposing the role-restricted domain tools.

This is a **facade**. It owns no domain logic: every call lands on
``repl._dispatch`` against a ``roles.dispatch_table``, exactly as the other three
backends do, so ``tools.py``, ``storage.py`` and ``agent_schemas.py`` are
untouched and every call is traced by ``@agent_obs.traced_dispatch`` for free.

Why it exists
-------------
On the Managed Agents front-end our tools are declared as **custom tools**, which
means Anthropic hands each call back to *us* over the session event stream and
waits. That works, but it puts our client in the middle of every tool call and
requires a process sitting on the session for the whole run. Served over MCP
instead, Anthropic's orchestration layer calls the tools directly and the session
never blocks on us — which is what makes unattended and scheduled runs possible.

It is *not* a latency win for this workload: the tools are local JSON file I/O at
2-9 ms, and MCP replaces that with a network round trip. Adopt it for
availability, not speed.

Shape
-----
One endpoint per role — ``/mcp/adjuster``, ``/mcp/insurer``, ``/mcp/agent`` —
each serving only ``roles.schemas_for_role(role)``. A single endpoint with
per-agent ``mcp_toolset`` filtering would have been simpler to host, but that
filter is advisory: it changes what the model is *offered*, not what the server
will *execute*. Per-role paths keep the guarantee ``roles.dispatch_table``'s
docstring makes — "a tool call for a name not in the table must be refused".

Run:  python3 mcp_server.py --help
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hmac
import json
import os
import secrets
import threading
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import anyio
import mcp.types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

import agent_obs
import roles
from repl import _dispatch
from roles import Role

# Model-visible server name per role. Distinct names rather than one shared
# "insurance": in a coordinator session the AGENT and its ADJUSTER subagent each
# declare a server, and two servers with the same name at different URLs is a
# question best not asked.
SERVER_NAME_PREFIX = "insurance"


def server_name(role: Role) -> str:
    return f"{SERVER_NAME_PREFIX}-{role.value}"


def endpoint_roles(role: Role) -> tuple[Role, ...]:
    """Roles whose tools one endpoint serves — deliberately ``(role,)``.

    With custom tools an AGENT session had to serve the AGENT ∪ ADJUSTER union
    (``roles.session_roles``), because a subagent's calls were cross-posted to the
    primary thread and dispatched against one client-side table. Over MCP a
    subagent is its own agent with its own ``mcp_servers``, so its calls arrive at
    ``/mcp/adjuster`` and the AGENT endpoint can be exactly the 11 agent tools —
    which *recovers* the enforcement the union gave up.

    If a live run shows subagent calls arriving at the coordinator's endpoint
    instead, return ``roles.session_roles(role)`` here and the union comes back.
    """
    return (role,)


# --- config -----------------------------------------------------------------

@dataclass(frozen=True)
class McpConfig:
    """Server + client-side settings, mirroring ``agent_obs.ObsConfig``'s shape.

    ``enabled`` drives whether ``cma.py`` points its agents at MCP at all. It
    auto-derives from having both a public URL and a token, so an untouched
    ``.env`` leaves the custom-tool path exactly as it is.
    """

    enabled: bool | None = None      # None -> auto: bool(public_url and token)
    host: str = "127.0.0.1"
    port: int = 8787
    path_prefix: str = "/mcp"
    public_url: str = ""             # MCP_PUBLIC_URL, e.g. https://host.example.com
    token: str = ""                  # MCP_BEARER_TOKEN
    json_response: bool = False

    @property
    def active(self) -> bool:
        if self.enabled is not None:
            return self.enabled
        return bool(self.public_url and self.token)

    def path_for(self, role: Role) -> str:
        return f"{self.path_prefix}/{role.value}"

    def url_for(self, role: Role) -> str:
        return f"{self.public_url.rstrip('/')}{self.path_for(role)}"

    def local_url_for(self, role: Role) -> str:
        return f"http://{self.host}:{self.port}{self.path_for(role)}"

    def with_overrides(self, **kw: Any) -> McpConfig:
        return replace(self, **kw)

    def describe(self) -> dict[str, Any]:
        """Safe to print — the token is fingerprinted, never shown."""
        return {
            "active": self.active, "host": self.host, "port": self.port,
            "public_url": self.public_url or "(unset)",
            "token": f"set ({len(self.token)} chars)" if self.token else "(unset)",
            "urls": {r.value: (self.url_for(r) if self.public_url
                               else self.local_url_for(r)) for r in Role},
        }

    @classmethod
    def from_env(cls, **overrides: Any) -> McpConfig:
        """Build from ``MCP_*`` env vars; explicit kwargs win.

        Recognised: MCP_TOOLS (on/off, overrides the auto-derivation),
        MCP_HOST, MCP_PORT, MCP_PATH_PREFIX, MCP_PUBLIC_URL, MCP_BEARER_TOKEN,
        MCP_JSON_RESPONSE.
        """
        base = cls()
        raw = os.environ.get("MCP_TOOLS")
        cfg = cls(
            enabled=(None if raw is None
                     else raw.strip().lower() in ("1", "true", "yes", "on")),
            host=os.environ.get("MCP_HOST", base.host),
            port=int(os.environ.get("MCP_PORT") or base.port),
            path_prefix=os.environ.get("MCP_PATH_PREFIX", base.path_prefix),
            public_url=os.environ.get("MCP_PUBLIC_URL", base.public_url),
            token=os.environ.get("MCP_BEARER_TOKEN", base.token),
            json_response=(os.environ.get("MCP_JSON_RESPONSE", "").strip().lower()
                           in ("1", "true", "yes", "on")),
        )
        return replace(cfg, **overrides) if overrides else cfg


# --- the facade -------------------------------------------------------------
#
# Everything below wraps existing code. The one place a plain wrapper is not
# enough is the lock: see `_storage_guard`.

# `storage.py` is whole-file read-modify-write with no locking, which is safe
# today only because every backend dispatches serially. An HTTP server does not,
# so the *entire* tool call is serialized here. Locking just the write would not
# help: two callers that each did `load_incidents()` before either wrote would
# still lose one another's changes across unrelated records.
#
# This is the one thing the facade cannot delegate — it is new behaviour wrapped
# *around* the existing functions, not a call into them.
_STORAGE_LOCK = threading.Lock()

# One worker thread. The tool functions are blocking file I/O, so they must not
# run on the event loop; and since they serialize on the lock anyway, more
# threads would only queue up blocked. Total serialized work is ~50 ms per run.
_LIMITER = anyio.CapacityLimiter(1)


@contextlib.contextmanager
def _storage_guard() -> Iterator[None]:
    with _STORAGE_LOCK:
        yield


def dispatch_guarded(table: dict[str, Any], name: str, args: dict[str, Any]) -> str:
    """``repl._dispatch`` with the storage lock held for the whole call."""
    with _storage_guard():
        return _dispatch(table, name, args)


@functools.lru_cache(maxsize=None)
def tool_specs(role: Role) -> tuple[dict[str, Any], ...]:
    """The Anthropic-shaped tool schemas this endpoint serves."""
    return tuple(roles.schemas_for_role(endpoint_roles(role)))


@functools.lru_cache(maxsize=None)
def mcp_tools(role: Role) -> tuple[mcp_types.Tool, ...]:
    """Our schemas as MCP tools — the same bytes, only the key name differs.

    ``FastMCP`` is deliberately *not* used. Its ``add_tool`` derives the input
    schema from the Python signature via pydantic and offers no override, which
    would silently replace ``agent_schemas.build_tool_schemas()`` — losing the
    Google-docstring parsing, the ``ReportType`` bitmask descriptions, the
    ``DynamicPolicies`` object schema, and the nullable-optional fix recorded in
    the CHANGELOG as a live defect. The low-level ``Server`` publishes ours
    verbatim, which is the only way this stays a facade.
    """
    return tuple(
        mcp_types.Tool(name=s["name"], description=s["description"],
                       inputSchema=s["input_schema"])
        for s in tool_specs(role)
    )


def _validation_error(role: Role, name: str, args: dict[str, Any]) -> str | None:
    """Validate against our own schema, returning ``_dispatch``'s error shape.

    Done here rather than by the MCP server (``call_tool(validate_input=True)``)
    on purpose: that path rejects *before* the handler runs, so the call never
    reaches `_dispatch` and produces no `tool.start`/`tool.end` pair — the exact
    blind spot that made the nullable-schema defect so hard to see. Validating
    inside the seam keeps every failure mode traced and identically shaped.
    """
    import jsonschema

    spec = next((s for s in tool_specs(role) if s["name"] == name), None)
    if spec is None:
        return None                      # unknown name: let _dispatch refuse it
    try:
        jsonschema.validate(args, spec["input_schema"])
    except jsonschema.ValidationError as exc:
        return f"Input validation error: {exc.message}"
    return None


def build_role_server(role: Role) -> Server:
    """A low-level MCP server exposing exactly one role's tools."""
    table = roles.dispatch_table(endpoint_roles(role))
    server: Server = Server(server_name(role), version="1.0.0")

    @server.list_tools()
    async def _list() -> list[mcp_types.Tool]:
        return list(mcp_tools(role))

    @server.call_tool(validate_input=False)
    async def _call(name: str, args: dict[str, Any]) -> list[mcp_types.ContentBlock]:
        obs = agent_obs.current()
        err = _validation_error(role, name, args)
        if err is not None:
            obs.events.warn("mcp.invalid_input", role=role.value, tool=name, error=err)
            return [mcp_types.TextContent(type="text",
                                          text=json.dumps({"error": err}))]
        text = await anyio.to_thread.run_sync(
            functools.partial(dispatch_guarded, table, name, args),
            limiter=_LIMITER)
        return [mcp_types.TextContent(type="text", text=text)]

    return server


# --- HTTP -------------------------------------------------------------------

class BearerAuth:
    """Pure-ASGI bearer check.

    Not ``BaseHTTPMiddleware``: that buffers the response body, which breaks the
    streamable-HTTP transport this exists to protect.
    """

    def __init__(self, app: Any, token: str, exempt: tuple[str, ...] = ("/healthz",)):
        self.app, self.token, self.exempt = app, token, exempt

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path", "") in self.exempt:
            return await self.app(scope, receive, send)
        header = ""
        for key, value in scope.get("headers", []):
            if key == b"authorization":
                header = value.decode("latin-1")
                break
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not (self.token and hmac.compare_digest(supplied, self.token)):
            body = json.dumps({"error": "unauthorized"}).encode()
            await send({"type": "http.response.start", "status": 401, "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="mcp"'),
                (b"content-length", str(len(body)).encode()),
            ]})
            return await send({"type": "http.response.body", "body": body})
        await self.app(scope, receive, send)


def build_app(config: McpConfig) -> Starlette:
    """One Starlette app mounting a session manager per role, behind bearer auth."""
    managers = {
        role: StreamableHTTPSessionManager(
            app=build_role_server(role), json_response=config.json_response,
            stateless=True,          # no session affinity to preserve across calls
            security_settings=None,  # host/origin checks off: we sit behind a proxy
        )
        for role in Role
    }

    async def healthz(_: Request) -> JSONResponse:
        obs = agent_obs.current()
        return JSONResponse({
            "ok": True,
            "run_id": obs.run_id,
            "tools": {r.value: len(tool_specs(r)) for r in Role},
        })

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette):
        async with contextlib.AsyncExitStack() as stack:
            for manager in managers.values():
                await stack.enter_async_context(manager.run())
            yield

    by_path = {config.path_for(r): managers[r] for r in Role}

    async def route_by_role(scope: Any, receive: Any, send: Any) -> None:
        """Dispatch ``/mcp/<role>`` to that role's session manager.

        The prefix is mounted once and the role segment matched here, rather than
        one ``Mount`` per role: a ``Mount`` on the full path 307-redirects the
        bare URL to a trailing slash, and an MCP client that does not follow
        redirects — including the current SDK's — simply fails. Registering the
        URL *with* a trailing slash would work too, but a config value whose
        final character is load-bearing is a trap worth not setting.
        """
        manager = by_path.get(scope.get("path", "").rstrip("/"))
        if manager is None:
            body = json.dumps({"error": "unknown role endpoint"}).encode()
            await send({"type": "http.response.start", "status": 404, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode())]})
            return await send({"type": "http.response.body", "body": body})
        await manager.handle_request(scope, receive, send)

    app = Starlette(
        routes=[Route("/healthz", healthz),
                Mount(config.path_prefix, app=route_by_role)],
        lifespan=lifespan,
    )
    app.add_middleware(BearerAuth, token=config.token)
    return app


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="mcp_server.py",
        description="Serve the role-restricted domain tools over streamable HTTP.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--public-url", default=None,
                        help="Externally reachable base URL, recorded on the agents.")
    parser.add_argument("--print-config", action="store_true",
                        help="Show the resolved config (token fingerprinted) and exit.")
    parser.add_argument("--new-token", action="store_true",
                        help="Print a fresh token for .env and exit.")
    parser.add_argument("--no-obs", action="store_true")
    args = parser.parse_args()

    if args.new_token:
        print(f"MCP_BEARER_TOKEN={secrets.token_urlsafe(32)}")
        return

    overrides = {k: v for k, v in (("host", args.host), ("port", args.port),
                                   ("public_url", args.public_url)) if v is not None}
    config = McpConfig.from_env(**overrides)

    if args.print_config:
        print(json.dumps(config.describe(), indent=2))
        return
    if not config.token:
        raise SystemExit(
            "MCP_BEARER_TOKEN is unset — refusing to serve the claims tools "
            "unauthenticated. Generate one with `python3 mcp_server.py --new-token`.")

    import uvicorn

    obs_cfg = agent_obs.ObsConfig.from_env(**({"enabled": False} if args.no_obs else {}))
    # The server needs its own run: `agent_obs.current()` is a process global, so
    # in a separate process it would otherwise be the disabled stand-in and every
    # tool call would go untraced.
    with agent_obs.Observability.start(obs_cfg, front_end="mcp") as obs:
        agent_obs.install_logging(obs)
        for role in Role:
            print(f"  {role.value:9} {config.local_url_for(role)}  "
                  f"({len(tool_specs(role))} tools)")
        if config.public_url:
            print(f"  public   {config.public_url.rstrip('/')}{config.path_prefix}/<role>")
        print(f"Tracing: run {obs.run_id}\n")
        uvicorn.run(build_app(config), host=config.host, port=config.port,
                    log_config=None)


if __name__ == "__main__":
    main()
