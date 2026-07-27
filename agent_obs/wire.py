"""Local capture proxy: records what actually goes over the wire.

Why this exists at all: on the Claude Agent SDK path the ``claude`` CLI assembles
the real ``/v1/messages`` payload itself — system prompt, message array, tool
schemas, ``cache_control`` breakpoints — so no amount of SDK-level instrumentation
can see it. Pointing the client's base URL at this proxy makes it visible, then
forwards the request upstream verbatim.

Generalised from the single-purpose version it was adapted from:

* **Any path, not just ``/v1/messages``.** ``wire_paths`` is a list of substrings,
  which is what lets it cover the Managed Agents control plane (``/v1/agents``,
  ``/v1/sessions``, ``/v1/environments``, ``/v1/memory_stores``, …) as well as the
  Messages API. Note those are plain ``/v1/<resource>`` paths — the beta is a
  ``?beta=true`` query parameter plus a header, not a ``/v1/beta/`` prefix — so a
  path list that does not name them captures nothing on the ``cma.py`` front-end.
* **Responses too**, at three levels (``none``/``summary``/``full``). The original
  captured requests only, so you could see the cache breakpoints you sent but not
  what came back. SSE streams are summarised by event-type counts rather than
  reassembled, unless ``full``.
* **Redaction at the boundary** — bodies pass through the configured redactor, so
  strict mode records structure and sizes without policyholder text.
* **Optional shaping before redaction** — ``shapers`` drop payload that is known
  boilerplate (see ``shape.py``; tool definitions are 95% of a request here). Off
  by default, because a shaped capture is no longer a faithful one.
* **Upstream resolved at construction, never at import.** The original read
  ``ANTHROPIC_BASE_URL`` at import time while the same variable was then set to
  the proxy's own address; a second proxy in the same process could forward to
  itself. Here the upstream is passed in and a self-target is rejected.

Auth headers (``authorization``, ``x-api-key``, cookies) are forwarded upstream
but never written to disk.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .events import EventLog
from .redact import Redactor
from .shape import Shaper
from .sinks import Sink

# Hop-by-hop and recomputed headers that must not be forwarded verbatim.
_SKIP_FORWARD = {"host", "connection", "keep-alive", "proxy-connection",
                 "transfer-encoding", "content-length", "upgrade", "te"}

# Request headers worth keeping. No credentials in this list, by construction:
# `anthropic-beta` is the interesting one — it is how you find out which betas the
# CLI negotiated on your behalf.
_CAPTURE_HEADERS = ("content-type", "anthropic-version", "anthropic-beta",
                    "user-agent", "x-stainless-lang", "x-stainless-package-version")

_CAPTURE_RESPONSE_HEADERS = ("content-type", "request-id", "anthropic-ratelimit-requests-remaining",
                             "anthropic-ratelimit-tokens-remaining", "retry-after")

# A response body larger than this is summarised even in `full` mode, so one
# runaway reply cannot write hundreds of MB.
_MAX_FULL_RESPONSE_BYTES = 2 * 1024 * 1024


class ProxyTargetError(ValueError):
    """Raised when the configured upstream would make the proxy call itself."""


class CaptureProxy:
    """Forwarding HTTP proxy on 127.0.0.1 that records matching traffic."""

    def __init__(self, sink: Sink, redactor: Redactor, *, upstream: str,
                 paths: tuple[str, ...] = ("/v1/messages",),
                 response_mode: str = "summary",
                 events: EventLog | None = None,
                 shapers: tuple[Shaper, ...] = ()):
        self.sink = sink
        self.redactor = redactor
        self.shapers = shapers
        self.paths = paths
        self.response_mode = response_mode
        self.events = events
        self.captured = 0
        self._lock = threading.Lock()

        parsed = urlsplit(upstream)
        self._host = parsed.hostname or "api.anthropic.com"
        self._port = parsed.port or (80 if parsed.scheme == "http" else 443)
        self._tls = parsed.scheme != "http"
        self.upstream = upstream

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        # Self-target check happens *after* binding, because our own port is only
        # known once we have one. The original of this code read the upstream from
        # ANTHROPIC_BASE_URL at import time while that same variable was later set
        # to the proxy's address — a loop that manifests as a hang, not an error.
        # Note this rejects only the exact self address: forwarding to a *different*
        # local port is legitimate (a mock upstream, or a proxy chain).
        if (self._host in ("127.0.0.1", "localhost", "::1")
                and self._port == self._server.server_address[1]):
            self._server.server_close()
            raise ProxyTargetError(
                f"refusing to forward to {upstream!r}: that is this proxy's own "
                "address, so every request would loop back to itself. Unset "
                "ANTHROPIC_BASE_URL or set OBS_UPSTREAM to the real API."
            )
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="agent-obs-capture", daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        """Hand this to the client as its base URL (``ANTHROPIC_BASE_URL`` for the
        CLI, ``base_url=`` for ``anthropic.Anthropic``)."""
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def shutdown(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass

    # --- recording -------------------------------------------------------

    def _matches(self, path: str) -> bool:
        return any(p in path for p in self.paths)

    def _record(self, row: dict[str, Any]) -> None:
        # Shape first, redact last: the redactor must be the final thing to touch
        # a row so credential stripping cannot be bypassed by a shaper, and a
        # shaper must see the unredacted structure to fingerprint it. A broken
        # shaper degrades to a verbatim capture rather than losing the row.
        for shaper in self.shapers:
            try:
                row = shaper.shape(row)
            except Exception as exc:
                row = {**row, "shape_error": f"{type(exc).__name__}: {exc}"[:300]}
                break
        self.sink.write(self.redactor.wire(row))
        with self._lock:
            self.captured += 1

    @staticmethod
    def _parse_body(body: bytes) -> Any:
        if not body:
            return None
        try:
            return json.loads(body)
        except ValueError:
            return {"_unparsed_bytes": len(body)}

    @staticmethod
    def _sse_summary(chunks: list[bytes]) -> dict[str, Any]:
        """Event-type counts for an SSE stream, without keeping the payloads.

        This is the cheap way to answer "did it stream, how much, and did it end
        cleanly" — `message_delta` carries the output token count, and its presence
        distinguishes a completed stream from a truncated one.
        """
        counts: dict[str, int] = {}
        total = 0
        for chunk in chunks:
            total += len(chunk)
            for line in chunk.split(b"\n"):
                if line.startswith(b"event:"):
                    name = line[6:].strip().decode("utf-8", "replace")
                    counts[name] = counts.get(name, 0) + 1
        return {"_t": "sse", "bytes": total, "events": counts}


    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):   # keep the REPL quiet
                pass

            def _proxy(self) -> None:
                started = time.monotonic()
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                interesting = proxy._matches(self.path)

                row: dict[str, Any] | None = None
                if interesting:
                    row = {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "method": self.command,
                        "path": self.path,
                        "headers": {k: self.headers.get(k) for k in _CAPTURE_HEADERS
                                    if self.headers.get(k)},
                        "request_bytes": len(body),
                        "body": proxy._parse_body(body),
                    }

                fwd = {k: v for k, v in self.headers.items()
                       if k.lower() not in _SKIP_FORWARD}
                conn_cls = (http.client.HTTPSConnection if proxy._tls
                            else http.client.HTTPConnection)
                conn = conn_cls(proxy._host, proxy._port, timeout=900)
                collected: list[bytes] = []
                status = None
                try:
                    conn.request(self.command, self.path, body=body or None, headers=fwd)
                    resp = conn.getresponse()
                    status = resp.status
                    self.send_response(resp.status)
                    resp_headers = {}
                    for key, value in resp.getheaders():
                        if key.lower() in ("transfer-encoding", "content-length",
                                           "connection"):
                            continue
                        if key.lower() in _CAPTURE_RESPONSE_HEADERS:
                            resp_headers[key.lower()] = value
                        self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.close_connection = True
                    self.end_headers()

                    want_body = interesting and proxy.response_mode != "none"
                    kept = 0
                    # read1 returns whatever has arrived, so SSE deltas relay
                    # immediately instead of waiting for a full block.
                    while chunk := resp.read1(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        if want_body and kept < _MAX_FULL_RESPONSE_BYTES:
                            collected.append(chunk)
                            kept += len(chunk)

                    if row is not None:
                        row["status"] = status
                        row["duration_ms"] = int((time.monotonic() - started) * 1000)
                        if want_body:
                            row["response_headers"] = resp_headers
                            row["response_bytes"] = sum(len(c) for c in collected)
                            row["response_body"] = _shape_response(
                                collected, resp_headers.get("content-type", ""),
                                proxy.response_mode)
                        proxy._record(row)

                except (BrokenPipeError, ConnectionResetError):
                    # Client went away mid-stream (an interrupt). Still record.
                    if row is not None:
                        row["status"] = status
                        row["client_disconnected"] = True
                        proxy._record(row)
                except Exception as exc:
                    if row is not None:
                        row["status"] = status
                        row["proxy_error"] = f"{type(exc).__name__}: {exc}"[:300]
                        proxy._record(row)
                    if proxy.events:
                        proxy.events.error("wire.proxy_error", path=self.path,
                                           error=f"{type(exc).__name__}: {exc}"[:300])
                    try:
                        err = json.dumps({"type": "error", "error": {
                            "type": "capture_proxy_error",
                            "message": str(exc)[:300]}}).encode()
                        self.send_response(502)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err)))
                        self.end_headers()
                        self.wfile.write(err)
                    except Exception:
                        pass
                finally:
                    conn.close()

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = _proxy

        return Handler


def _shape_response(chunks: list[bytes], content_type: str, mode: str) -> Any:
    """Turn collected response bytes into whatever the mode asks for."""
    raw = b"".join(chunks)
    is_sse = "event-stream" in content_type
    if mode == "summary":
        if is_sse:
            return CaptureProxy._sse_summary(chunks)
        return {"_t": "json", "bytes": len(raw)}
    # full
    if is_sse:
        return {"_t": "sse_raw", "bytes": len(raw),
                "stream": raw.decode("utf-8", "replace"),
                "summary": CaptureProxy._sse_summary(chunks)}
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return {"_unparsed_bytes": len(raw)}
