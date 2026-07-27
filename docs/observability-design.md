# Observability design — tracing, logging, usage accounting

**Status:** Implemented. `agent_obs/` is built and wired into `repl.py` (both
backends) and `cma.py`; 28 tests in `tests/test_agent_obs.py`. Sections 1–3 and 5
are the findings that motivated it, Section 4 the proposal as written before
building, Section 6 the questions it had to answer, **Section 7 the decisions
actually taken** (read this one first if you want the current design), and Section
8 the first thing the wire layer surfaced.

Deltas from the Section 4 proposal, all recorded in Section 7: files are keyed on a
run id rather than a session id (no pending-file renames at all), `wire` covers all
three backends rather than just the SDK path, and diagnostic strings are truncated
rather than hashed in strict mode.

**Goal:** bring the tracing/logging capability of `../agent_transcriber2` into this
project (`claude_agent_transcriber2`) in a programmatic, controlled, repeatable,
**reusable** form — a package we can drop into either project rather than a second
copy-paste.

**Source reviewed:** `../agent_transcriber2` @ commit as of 2026-07-24 —
`src/agent_transcriber/{tracing,otel,capture,ledger,hooks,stats,agent,config}.py`
(~700 lines of observability code), plus `config/default.yaml` and the README's
"Usage tracking" / "Hooks & tracing" sections.

---

## 1. What exists in the source project

Four independent subsystems bundled behind one facade (`TraceLogger`) and constructed
at three call sites (`repl.py:86`, `cli.py:150`, `experiments/runner.py:40`).

| Layer | File | Output | Question it answers |
|---|---|---|---|
| Semantic event log | `tracing.py` (79 L) | `var/traces/<sid>.jsonl` | what happened, in order |
| OTel spans | `otel.py` (115 L) | `var/traces/<sid>.otel.jsonl` | how long, how much, nested |
| Wire capture | `capture.py` (151 L) | `var/traces/<sid>.requests.jsonl` | what was literally sent to the API |
| Usage ledger | `ledger.py` (141 L) | `var/ledger.jsonl` (global, append-only) | what it cost, over all time |

Two consumers: `hooks.py` (all 10 SDK hook events → events + quota enforcement) and
`stats.py` (rich tables).

### 1.1 `TraceLogger` — event log + facade

`event(name, **fields)` appends `{ts, event, session_id, **fields}` as one JSON line,
reopening the file per write.

The notable mechanism is the **pending-file rename idiom**: the SDK session id is not
known until the CLI's `init` message arrives, so writes start in
`pending-<pid>-<epoch>.jsonl` and `set_session_id()` renames the file once the id is
known (`tracing.py:43-54`). It fans the id out to the OTel tracer and the capture proxy.

It also owns:
- `stderr_sink`, passed as `ClaudeAgentOptions.stderr`, keeping only lines matching
  `error|warn|fatal`, truncated to 500 chars (`tracing.py:71-74`).
- `cli_transcripts_dir()`, which computes where the spawned CLI auto-persists its own
  raw transcripts (`~/.claude/projects/<cwd-slug>/`) — a free data source they point
  at rather than reimplement.

### 1.2 `OtelSessionTracer` — real OTel, file-backed

Does **not** touch the global tracer provider. Constructs a private `TracerProvider` +
`SimpleSpanProcessor` + a ~15-line custom `SpanExporter` writing
`span.to_json(indent=None)` per line (`otel.py:27-62`). That is what keeps multiple
sessions in one process in separate files, and why this is ~100 lines instead of an
OTLP collector dependency.

One span per turn, opened in `agent.py:115` around the whole message-drain loop:
- open attributes: `request.kind`, `prompt.label`, `session.name`, `persona`,
  `gen_ai.request.model`, `effort`
- `ToolUseBlock` → `span.add_event("tool_use", {...})` (`agent.py:143`) — tool calls
  are span *events*, not child spans
- close attributes via `record_turn_result(span, record)` (`otel.py:98`), copying the
  ledger record's tokens/cost/durations onto the span; `StatusCode.ERROR` on `is_error`

### 1.3 `CaptureProxy` — the genuinely novel piece

Solves a real problem: the Claude Code CLI assembles the actual `/v1/messages` payload
internally, so the SDK layer never sees the system prompt, the messages array, the tool
schemas, or the `cache_control` breakpoints.

Mechanism: a `ThreadingHTTPServer` on `127.0.0.1:0`; its URL is handed to the spawned
CLI as `ANTHROPIC_BASE_URL` (`options.py:145`); every POST body to `/v1/messages` is
recorded; everything is forwarded upstream verbatim.

Details that matter:
- **Auth forwarded, never written** — only `content-type`, `anthropic-version`,
  `anthropic-beta`, `user-agent` are captured (`capture.py:33`).
- **SSE relays correctly** via `resp.read1(65536)` + flush per chunk instead of a
  fixed-size `read()`, so streaming deltas are not buffered (`capture.py:130`).
- Hop-by-hop headers stripped; `Connection: close`; broken pipe on interrupt swallowed.

This is what produced the numbers quoted in that project's README: 116 tools / ~63k
tokens of schemas on the wire before trimming, 25 tools / ~7k after
(`options.py:30-47`). There is no other way to obtain that measurement.

### 1.4 `UsageLedger` + `QuotaTracker`

`TurnRecord` is a 22-field dataclass built from `ResultMessage` with defensive
`getattr(..., None) or 0` throughout. `Totals.add()` accumulates; `all_tokens` and
`cache_hit_ratio` are derived properties.

`QuotaTracker` (`hooks.py:18`) reads `ledger.session_totals.all_tokens`; the
`PreToolUse` hook returns `permissionDecision: "deny"` once exhausted — so enforcement
lands at **turn boundaries**, and a one-shot turn always completes.

The `UserPromptSubmit` hook injects quota status via `additionalContext` specifically
because that lands *after* the cached prefix and therefore does not invalidate the
prompt cache (`hooks.py:61-63`).

---

## 2. Worth keeping

1. **JSONL everywhere, one file per session, append-only.** No DB, no migrations,
   `jq`-able. Correct call at this size.
2. **The pending-rename idiom** — clean answer to "the id arrives after the first write".
3. **Private TracerProvider** — no global state, composable, testable.
4. **Capture proxy as the escape hatch** for everything the SDK abstraction hides.
5. **Layer separation by question asked** — events / spans / wire / cost are genuinely
   different questions; merging them into one stream would be worse.
6. **Documenting the negative space** (that README's "not available at the SDK layer"
   table) so nobody re-litigates it.

---

## 3. Problems to fix before replicating

### 3.1 Structural

- `TraceLogger` is a misnamed god-object: it owns an event log, an OTel provider, and
  an HTTP proxy. `trace.capture_base_url`, `trace.otel.span(...)`, `trace.event(...)`
  all hang off one handle. It is a *session observability bundle*, not a trace logger.
- **The pending-rename block is copy-pasted three times** (`tracing.py:49-53`,
  `otel.py:71-75`, `capture.py:64-68`) with three slightly different implementations —
  `capture.py` takes a lock, the other two do not.
- No stdlib `logging` anywhere in `src/`. No level, no filter, no way to turn it down.
- Debug `print()` left in library code: `ledger.py:121` (cache stats every turn),
  `agent.py:89` (context usage). Belongs in `stats.py` or behind a level.

### 3.2 Correctness / robustness

- `TraceLogger.event()` takes no lock and reopens the file per call. Fine while hooks
  run on a single loop, but that invariant is undocumented.
- `stderr_sink` **discards** everything not matching `error|warn|fatal`. Full CLI
  stderr goes nowhere.
- Orphaned `pending-*.jsonl` files accumulate when a session dies before init (one is
  sitting in that project's `var/traces/` now).
- `capture.py:27` reads `UPSTREAM` from `ANTHROPIC_BASE_URL` **at import time**, and
  the same var is then set to the proxy's own URL for the child. Any path constructing
  the proxy in a process whose env was already mutated self-loops.
- `stats._bucket` (`stats.py:44-52`) reconstructs `TurnRecord` from raw dicts via
  `__dataclass_fields__` default archaeology. A field rename silently breaks all-time
  history. Ledger rows carry no `schema_version`.
- One global `ledger.jsonl` for all users and processes; `read_all()` slurps the whole
  file. Cross-process append atomicity holds only under `PIPE_BUF`.

### 3.3 Scale / privacy — highest priority for this project

- **`requests.jsonl` grows quadratically.** Every turn resends the whole messages
  array, and each is captured whole: 631 KB for a *short* session. No cap, no
  truncation, no rotation, no retention policy.
- **Zero redaction.** Full prompts, tool results, and conversation history land on disk
  in plaintext. This project carries policyholder/incident data — that is a compliance
  question, not a preference.
- **Responses are not captured at all** — only request bodies. You get the
  cache-breakpoint layout but not what came back.
- No sampling switch; capture is all-or-nothing per session
  (`config/default.yaml:25`).

### 3.4 Coverage

- Zero tests for `tracing.py`, `otel.py`, `capture.py`. Only two ledger tests exist.
  The proxy especially — header filtering, SSE relay, auth non-leakage — wants tests.
- Spans are flat: `parent_id` always `null`, every turn a root span, tool calls as
  events rather than child spans → no per-tool latency, no session roll-up.
- Attribute naming is half-migrated: `gen_ai.usage.input_tokens` (OTel GenAI semconv)
  next to ad-hoc `cost.usd`, `cache.read_input_tokens`, `duration.wall_ms`.

---

## 4. Proposed shape (NOT yet decided)

The four layers have exactly one dependency on the host app: `record_turn_result` reads
a `TurnRecord`. Everything else is generic. Proposal is a standalone package —
installable as a path/git dependency, or vendored as one subpackage here:

```
agent_obs/
  __init__.py        SessionObservability — the single facade you construct
  sinks.py           JsonlSink: pending-rename + lock + rotation, ONE implementation
  events.py          EventLog (uses JsonlSink)
  spans.py           OtelSessionTracer, + optional OTLP exporter alongside the file one
  wire.py            CaptureProxy (uses JsonlSink)
  usage.py           TurnRecord / Totals / UsageLedger — schema_version on every row
  redact.py          pluggable Redactor protocol, applied at the sink boundary
  sdk.py             build_hooks(...) → the 10 SDK hook matchers
  config.py          ObsConfig dataclass: one toggle + one level per layer
```

Changes to make during extraction, in priority order:

1. **One `JsonlSink`** owning pending-rename, locking, size cap, and rotation. Deletes
   three copies, fixes the lock inconsistency, puts retention in one place.
2. **A `Redactor` protocol applied at the sink**, not per call site. Default drops or
   hashes message content in `requests.jsonl` and truncates tool payloads. Ship a
   permissive dev redactor; strict is the default.
3. **`ObsConfig` with per-layer enable + verbosity**, so `events=on, spans=on,
   wire=off` is one line rather than a constructor arg threaded through option-building.
4. **Rename the facade** to `SessionObservability` with `.events`, `.spans`, `.wire`,
   `.usage`; the `trace.otel.span()` chain disappears.
5. **Bridge to stdlib `logging`** — an `ObsHandler` so `logging.getLogger("agent_obs")`
   and the event log are one stream, and library code stops calling `print()`.
6. **Span hierarchy**: session root span → per-turn children → per-tool grandchildren.
   Fixes the flat-trace problem and per-tool latency together.
7. **Semconv pass** on attribute names so traces work in any OTel backend without a
   mapping layer.
8. **Tests on the proxy**: auth headers never written, SSE chunks relayed unbuffered,
   upstream self-loop rejected.

---

## 5. Tool-call tracing in this project — settled findings

Verified against the installed SDK, **`claude-agent-sdk 0.2.126`**
(`.venv/lib/python3.14/site-packages/claude_agent_sdk`), 2026-07-24.

### 5.1 Hooks are available on the one-shot `query()` path

`query()` **always runs streaming mode internally** — `_internal/client.py:172-179`
constructs `Query(is_streaming_mode=True, ...)` with the comment *"Always use streaming
mode internally (matching TypeScript SDK)"*, and passes `hooks` through unconditionally.
`_internal/query.py:812-826` (`wait_for_result_and_end_input`) explicitly holds stdin
open when hooks are present so the bidirectional control protocol can run.

So a string-prompt `query()` call supports all 10 hook events. The **only** callback
that requires an `AsyncIterable` prompt is `can_use_tool`, which raises
`ValueError("can_use_tool callback requires streaming mode…")` at
`_internal/client.py:101-106`.

This corrects the earlier framing: hooks are not a reason to adopt `ClaudeSDKClient`.
`can_use_tool`, `interrupt()`, live `set_model()`, and `get_context_usage()` are.

### 5.2 Every tool call in this project already funnels through one function

Unlike the source project — whose tools lived in an external MCP subprocess, making
hooks the only interception point — **all three backends here execute tools in our own
process, through `repl._dispatch(table, name, args)` (`repl.py:97`)**:

| Backend | Path to `_dispatch` |
|---|---|
| Agent SDK (default) | `_build_sdk_server` → per-tool MCP `handler` → `_dispatch` (`repl.py:126`) |
| Direct API (`--use-key`) | local tool loop → `_dispatch` (`repl.py:216`) |
| Managed Agents (`cma.py`) | `agent.custom_tool_use` event → `_dispatch` (`cma.py:293`, imported from `repl` at `cma.py:40`) |

That makes the tool-function boundary the primary tracing seam, and it is
**backend-independent**: one wrapper yields tool name, arguments, result, duration,
and exceptions across all three paths, with no SDK coupling at all.

Implication for the package: tool tracing belongs in a generic decorator/wrapper, not
in `sdk.py`. Hooks become *enrichment* on the SDK path (permission decisions,
`tool_use_id` correlation, `PreToolUse` veto, subagent start/stop, compaction) rather
than the source of truth for "a tool ran".

### 5.3 The three tool-tracing sources, ranked

1. **`_dispatch` wrapper** — all backends, has args + result + duration + errors.
   Lacks the model-side `tool_use_id` and any call the model made that never reached
   dispatch (denied, malformed).
2. **Message-stream observation** — `ToolUseBlock` in `AssistantMessage`; on the direct
   API path also `stop_reason == "tool_use"`. Gives `tool_use_id` and captures calls
   that never executed. Available on the SDK and direct-API paths.
3. **Hooks** (SDK path only) — `PreToolUse`/`PostToolUse`/`PostToolUseFailure` give
   permission decisions and the SDK's own `tool_use_id`, plus the events that have no
   other source: `PreCompact`, `SubagentStart`/`SubagentStop`, `Notification`.

Layers 1 and 2 correlate by `tool_use_id` where both are present; layer 1 alone is
enough for a complete tool timeline.

---

## 6. Open questions — to work through before committing to a shape

### 6.1 `ClaudeSDKClient` vs `query()` — the blocking one

The two codebases do not have the same shape, and the difference lands on the
observability design.

| | `agent_transcriber2` (source) | this project |
|---|---|---|
| Layout | `src/` package | flat modules |
| SDK usage | long-lived streaming `ClaudeSDKClient` | one-shot `query()` per turn (`repl.py:273`) |
| Other LLM paths | none | direct `messages.create()` (`repl.py:211`), Managed Agents (`cma.py`) |
| Hooks | all 10 registered | none registered yet (**available** — see §5.1) |
| Tool execution | external MCP subprocess | in-process, via `_dispatch` (**see §5.2**) |
| Session continuity | client holds it | `resume=<session_id>` per turn |

Consequences to resolve:

- **Session id timing.** `SystemMessage`-init-driven `set_session_id` does not fire the
  same way on the one-shot `query()` path — ids arrive on `ResultMessage`. Either the
  pending-rename does more work here, or files key on our own run id instead.
  → *Decide: own run id vs SDK session id as the primary file key.* (Leaning own run
  id, since it is the only key all three backends share.)
- **Ledger feeding differs per backend.** `ResultMessage` on the SDK path; the raw
  `Message.usage` on the direct-API path; session events on Managed Agents.
  → *Decide: an adapter per backend producing one `TurnRecord`.*
- **Capture proxy is redundant for `--use-key`** (we already own that payload) and
  probably inapplicable to Managed Agents.
  → *Decide: is `wire` an SDK-path-only layer?*
- **What `ClaudeSDKClient` would actually buy us** — now narrowed to `can_use_tool`,
  `interrupt()`, live `set_model()`, `get_context_usage()`, and a stable session
  without per-turn `resume`. Cost: a persistent connection and a restructured REPL.
  Needs its own investigation before the package API is fixed.

### 6.2 Secondary questions

- Redaction default: drop content entirely, hash it, or keep a length/shape summary?
- Retention: size cap per file, run count, or age?
- Do we want OTLP export at all, or is file-only enough for now?
- Package delivery: vendored subpackage in this repo, or a separate repo both projects
  depend on? (Affects how much API stability we owe ourselves.)
- Where does the ledger live when two front-ends (`repl.py`, `cma.py`) run
  concurrently?

---

## 7. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-24 | **Run id, not session id, is the primary file key.** Files are `var/traces/<run_id>.{events,otel,wire}.jsonl`; `var/runs.jsonl` indexes run → session ids. | The three backends' session ids arrive at different times and in different shapes, and Managed Agents has no per-turn id. A run id we mint is available before the first write and is common to all three. Removes the pending-file rename **and** the orphaned `pending-*.jsonl` files it left on crashes (§3.2). |
| 2026-07-24 | **One `JsonlSink`** with lock, size cap, and rotation; every layer uses it. | Deletes the three divergent copies (§3.1) and bounds `wire.jsonl`, whose growth is quadratic in turn count (§3.3). |
| 2026-07-24 | **Redaction at the sink**, four modes: `flow` (default), `strict`, `dev`, `none`. | A call site can never forget to redact; the only way to log raw content is to configure it deliberately. |
| 2026-07-24 | **`flow` replaced `strict` as the default: first sight in full, repeats collapsed to a 20-char preview + `sha`.** Dedup is by content hash across the run, not by position. | `strict` was the wrong trade in practice. It hashed our *own* tool schemas and system prompts — static app text, never user data — and made request flows unreadable, which is the entire purpose of the wire log. `flow` keeps every message in the array visible, shows what is new in each request, and collapses the resent history to a followable skeleton. Hash-based (not positional) because the same text moves index between turns, and because it collapses resent schemas for the same reason it collapses history. Also bounds the quadratic growth that motivated rotation. `strict` remains for traces that might leave this machine. |
| 2026-07-24 | **Exception: diagnostic strings are truncated, not hashed** (`error`, `proxy_error`, `line`; 300 chars). | A hashed error message makes a trace worthless for debugging. Deliberate, documented hole in the strict guarantee, bounded by truncation. Caught by a test that asserted the readable form. |
| 2026-07-24 | **Tool tracing lives in a `@traced_dispatch` decorator on `repl._dispatch`**, resolving the active run via `agent_obs.current()`. | §5.2: all three backends converge there. Resolving at call time (not via an argument) means the three call sites are unchanged and `cma.py`'s `from repl import _dispatch` gets the traced version for free. |
| 2026-07-24 | **Hooks are tracing-only**; all 10 registered, every callback returns `{}`. `QuotaGuard` exists but is not installed. | An observability layer that can veto tool calls stops being one; authorization belongs in `roles.py`. Confirms §5.1 — hooks work on the one-shot `query()` path. |
| 2026-07-24 | **`wire` covers all three backends**, not just the SDK. Path list is configurable (`/v1/messages`, `/v1/beta`). | Managed Agents is ordinary HTTPS to the same API, so `base_url=obs.base_url` on the `anthropic` client captures it. Reverses the §6.1 leaning that `wire` would be SDK-only. |
| 2026-07-25 | **Corrected the path list.** `/v1/beta` was wrong: Managed Agents resources are plain `/v1/<resource>` paths (`/v1/agents?beta=true`, `/v1/sessions?beta=true`, …) — the beta is a query parameter and a header, not a path prefix. The default now names each resource. | The decision above was right and its implementation was not: `cma.py --wire` matched zero requests, so the proxy forwarded everything and recorded nothing, producing no wire file. The unit test asserted `/v1/beta/sessions/...` — a path the SDK never sends — so it encoded the same wrong assumption and passed. Resources are listed explicitly rather than globbing `/v1/` so `/v1/oauth/token` (refresh tokens, client secrets) is never captured; a test derives the required set from the installed SDK so a new resource fails loudly. |
| 2026-07-24 | **Wire capture also records responses**, at `none`/`summary`/`full`. SSE is summarised by event-type counts unless `full`. | The original captured requests only — you could see the breakpoints you sent but not what came back. |
| 2026-07-24 | **Self-target check happens after the socket binds**, and rejects only the exact self address. | Our own port is unknown until bound; a *different* local port is a legitimate upstream (mock, proxy chain). Fixes the import-time hazard in §3.2. |
| 2026-07-24 | **Span hierarchy: session → turn → tool.** GenAI semconv names where they exist, `claude.*` otherwise. | Fixes the flat traces of §3.4 and yields per-tool latency for free. |
| 2026-07-24 | **`schema_version` on every ledger row**; `summarize()` copies only known keys. | An unknown field version degrades to partial totals instead of corrupting history (§3.2). |
| 2026-07-24 | **`opentelemetry-sdk` is an optional dependency**; missing → `NullTracer` with the same API. | Spans are the only layer that needs it. No call site guards. |
| 2026-07-24 | **Instrumentation never breaks the app.** A failed proxy sets `wire_error` and the run continues; sinks swallow write errors and count `dropped`. | Tested (`test_wire_failure_does_not_disable_the_run`). |
| 2026-07-24 | **`ClaudeSDKClient` deferred.** Staying on one-shot `query()`. | §5.1 removed the main reason to migrate — hooks work on `query()`. What remains (`can_use_tool`, `interrupt()`, live `set_model()`, `get_context_usage()`) is not needed for tracing. Revisit if the REPL wants interrupts. |

### Still open

* Retention policy beyond per-file rotation (age-based cleanup of `var/traces/`).
* Whether to enable `otlp_endpoint` and view traces in a real backend.
* Whether `agent_obs` becomes its own repo (both projects would depend on it) or
  stays vendored here. Currently vendored.
* Whether to act on the tool-schema finding below.

---

## 8. First finding from the wire layer

The layer paid for itself on the first live run. A single `list my incidents` turn
on the Agent SDK backend:

```
POST /v1/messages   req=  2,288 B   tools=  0
POST /v1/messages   req=109,414 B   tools= 49
POST /v1/messages   req=291,853 B   tools=130
→ ledger: cache_creation=153,126 tokens, cost=$1.55, one tool call
```

130 tool schemas shipped on the third request, against the ~10 this role is
allowed. `allowed_tools` gates **calling**, not **shipping**: it denies a call
after the schema has already been paid for. The levers that remove schemas from
the request are `ClaudeAgentOptions.tools` (built-ins) and `disallowed_tools`
(the only one that reaches account-bound claude.ai connectors) — the same
conclusion the source project reached and documented in its `options.py`.

Not changed here, because it alters what the agent can do and that is a
behavioural decision, not an instrumentation one. Flagged for a decision.
