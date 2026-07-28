"""Call a tool on the MCP server from the command line, with no model in the loop.

For driving the server directly — seeding a demo, checking a deployment, poking
one tool — where going through an agent would be slow, expensive, and
non-deterministic.

    python3 mcp_call.py --list
    python3 mcp_call.py agent generate_unassigned_incidents count=20
    python3 mcp_call.py insurer find_insurer insurer_id=ins-1001

The endpoint comes from ``McpConfig`` (so ``MCP_PUBLIC_URL`` / ``MCP_BEARER_TOKEN``
in ``.env``), falling back to the local server when no public URL is set.
``--local`` forces local, ``--url`` overrides both.

Why a real MCP client rather than a raw POST
--------------------------------------------
This drives ``mcp.ClientSession`` over ``streamable_http_client`` — the same
stack a Managed Agents session uses — instead of hand-rolling the JSON-RPC
envelope with ``curl``. A raw POST would be shorter, but it would also succeed in
cases where a real client fails: proper streamable-HTTP support through the
ingress is the whole reason this is hosted on Container Apps rather than Azure
Functions, and a tool used to verify a deployment should exercise the thing that
can actually break.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx

from mcp_server import McpConfig
from roles import Role


def parse_value(raw: str) -> Any:
    """Interpret one ``key=value`` argument the way a caller would expect.

    JSON first, so ``count=20`` is the integer 20, ``flag=true`` is a boolean,
    ``x=null`` is None, and ``ids=["a","b"]`` is a list. Anything JSON rejects is
    a plain string, which is what makes ``name=Maria Gonzalez`` work without
    quoting gymnastics.

    ``NaN`` / ``Infinity`` stay strings. Python's parser accepts them as floats
    even though JSON-the-spec has no such literals, and no tool argument here
    wants a float infinity — a policyholder named "Infinity" is likelier.
    """
    def _reject(constant: str) -> Any:
        raise ValueError(constant)

    try:
        return json.loads(raw, parse_constant=_reject)
    except (json.JSONDecodeError, ValueError):
        return raw


def parse_args_pairs(pairs: list[str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(
                f"bad argument {pair!r} — expected key=value "
                f"(e.g. count=20, insurer_id=ins-1001)")
        key, _, raw = pair.partition("=")
        args[key] = parse_value(raw)
    return args


def resolve_url(config: McpConfig, role: Role, *, local: bool, url: str | None) -> str:
    """The endpoint for one role.

    ``--url`` is a *base* URL, like ``MCP_PUBLIC_URL`` — the role path is added.
    A full endpoint URL is accepted too rather than silently doubling the path,
    because pasting the URL you just curled is the obvious thing to try.
    """
    if url:
        base = url.rstrip("/")
        suffix = config.path_for(role)
        return base if base.endswith(suffix) else f"{base}{suffix}"
    if local or not config.public_url:
        return config.local_url_for(role)
    return config.url_for(role)


def root_cause(exc: BaseException) -> BaseException:
    """Unwrap anyio's ``ExceptionGroup`` down to the error that actually happened.

    ``streamable_http_client`` runs inside a task group, so an ordinary 401 comes
    back as ``ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)``
    wrapping the ``HTTPStatusError``. Catching ``httpx.HTTPError`` alone silently
    misses every one of them and dumps a twenty-line traceback where a one-line
    message belongs.
    """
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


@contextlib.asynccontextmanager
async def session(url: str, token: str, timeout: float) -> AsyncIterator[Any]:
    """An authenticated MCP session against one role endpoint."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # Auth rides on the httpx client, which is how `streamable_http_client` takes
    # it — the older `streamablehttp_client` spelling with `headers=` is
    # deprecated.
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as sess:
                await sess.initialize()
                yield sess


async def do_list(url: str, token: str, timeout: float) -> int:
    async with session(url, token, timeout) as sess:
        tools = sorted((await sess.list_tools()).tools, key=lambda t: t.name)
    print(f"{len(tools)} tools at {url}\n")
    for tool in tools:
        required = tool.inputSchema.get("required", [])
        params = ", ".join(
            f"{name}{'' if name in required else '?'}"
            for name in tool.inputSchema.get("properties", {}))
        summary = (tool.description or "").split(".")[0].strip()
        print(f"  {tool.name}({params})")
        if summary:
            print(f"      {summary}.")
    return 0


async def do_call(url: str, token: str, timeout: float,
                  tool: str, args: dict[str, Any]) -> int:
    async with session(url, token, timeout) as sess:
        result = await sess.call_tool(tool, args)

    text = "\n".join(getattr(b, "text", "") for b in result.content)
    # Tool results are JSON strings inside the MCP text block. Re-indent when we
    # can so the output is readable; pass through untouched when we cannot.
    try:
        payload = json.loads(text)
        print(json.dumps(payload, indent=2))
    except json.JSONDecodeError:
        payload = None
        print(text)

    # `_dispatch` reports failures as `{"error": ...}` with `isError` unset, so
    # the exit code has to come from the payload, not from the protocol. A script
    # that pipes this needs `&&` to mean something.
    failed = bool(getattr(result, "isError", False)) or (
        isinstance(payload, dict) and "error" in payload)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="mcp_call.py",
        description="Call one MCP tool directly, with no model in the loop.",
        epilog="examples:\n"
               "  python3 mcp_call.py --list\n"
               "  python3 mcp_call.py agent generate_unassigned_incidents count=20\n"
               "  python3 mcp_call.py insurer find_insurer insurer_id=ins-1001",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", nargs="?", default="agent",
                        choices=[r.value for r in Role],
                        help="which role endpoint to call (default: agent)")
    parser.add_argument("tool", nargs="?", help="tool name; omit with --list")
    parser.add_argument("args", nargs="*", metavar="key=value",
                        help="tool arguments, JSON-parsed (count=20, x=null, s=text)")
    parser.add_argument("--list", action="store_true",
                        help="list the tools this role serves and exit")
    parser.add_argument("--local", action="store_true",
                        help="target the local server even if MCP_PUBLIC_URL is set")
    parser.add_argument("--url", help="explicit base URL, overriding both")
    parser.add_argument("--token", help="bearer token (default: MCP_BEARER_TOKEN)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="HTTP timeout in seconds (default: 60)")
    opts = parser.parse_args(argv)

    if not opts.list and not opts.tool:
        parser.error("give a tool name, or --list to see what is available")

    config = McpConfig.from_env()
    role = Role(opts.role)
    url = resolve_url(config, role, local=opts.local, url=opts.url)
    token = opts.token or config.token
    if not token:
        raise SystemExit(
            "No bearer token. Set MCP_BEARER_TOKEN in .env or pass --token.")

    # Before announcing a target: a key=value typo should not print a line that
    # implies we contacted the server.
    call_args = parse_args_pairs(opts.args) if not opts.list else {}

    # Always, to stderr so it never pollutes piped output. `MCP_PUBLIC_URL` in
    # `.env` makes the *deployed* server the default target, and some of these
    # tools are destructive — `generate_unassigned_incidents` discards every
    # claim on file. Seeing the host before the call is what stops that being a
    # surprise.
    print(f"→ {url}", file=sys.stderr)

    try:
        if opts.list:
            return asyncio.run(do_list(url, token, opts.timeout))
        return asyncio.run(do_call(url, token, opts.timeout, opts.tool, call_args))
    except (httpx.HTTPError, BaseExceptionGroup) as exc:
        cause = root_cause(exc)
        if isinstance(cause, httpx.HTTPStatusError):
            # 401 is by far the most common, and the SDK's message never
            # mentions the token, which sends people looking in the wrong place.
            hint = ("\n  The token does not match the server's MCP_BEARER_TOKEN."
                    if cause.response.status_code == 401 else "")
            print(f"HTTP {cause.response.status_code} from {url}{hint}",
                  file=sys.stderr)
            return 2
        if isinstance(cause, httpx.HTTPError):
            print(f"Could not reach {url}: {type(cause).__name__}: {cause}\n"
                  f"  Is the server running? `python3 mcp_server.py` for local, "
                  f"or check MCP_PUBLIC_URL for a deployment.", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
