# Local chat and API

The native runtime now has a terminal chat client and a server with responsive
startup, status and cancellation. These use the existing `Session` execution
path. Experimental MTP, streamed embeddings and benchmark-only kernel settings
are not selected by chat.

The implementation is available; real-model API, coding, request-latency and
20-minute session qualification remain pending. See the
[implementation report](benchmarks/2026-09-17-usability/README.md).

## Launch

From the project folder, use an interactive terminal with existing verified assets:

```sh
build/qwen/bin/zerocool chat
```

The default model directory is `.cache/models/qwen38-flash-next` and prepared
directory is `.cache/prepared/q4-records-v1`. Paths are relative to the current
directory. Both must already contain verified assets; startup reports missing
or invalid assets. Use `--model DIR` and `--prepared DIR` to override either path.

The client immediately shows its interface and starts a child `serve` process.
The server binds an OS-assigned loopback port before loading weights. The
endpoint is displayed at the top; another client can use it while chat is idle.
A private inherited socket communicates that endpoint, the server instance ID,
and parent lifetime. Chat pins the instance and refuses a replacement process
on the same port until you explicitly quit and reconnect.
Quitting or losing the parent stops new work, cancels generation, drains users
of buffers, and shuts down the child. Shutdown can take longer than cancellation
acknowledgement. A failed startup stays visible until quit; it does not restart
or silently attach to a different server.

To manage the server separately:

```sh
build/qwen/bin/zerocool serve --model .cache/models/qwen38-flash-next \
  --prepared .cache/prepared/q4-records-v1 --memory-gb 12 --port 8080
build/qwen/bin/zerocool chat --connect http://127.0.0.1:8080
```

Connected chat never terminates that server. The current client accepts IPv4
loopback HTTP endpoints only. Server and client do not download assets.

Locally launched chat requests 12GiB, 8,192 total context tokens, 256 output
tokens, greedy sampling, thinking off and Q4. Admission may reduce memory.
`--artifact mixed-4_8bit` explicitly selects the verified mixed artifact and
defaults its model directory to `.cache/qwen-mixed-reference`. It uses the same
verified prepared records. An explicit `--model` always overrides the default.
Local options also include `--memory-gb`, `--context`
and `--prepared`; both launch forms accept `--max-tokens`, `--temperature` and
`--thinking`. Connected mode cannot change the server's artifact or limits.
Existing `run`, `bench` and `serve` defaults are unchanged.

## Controls and history

| Action | Control |
|---|---|
| Insert newline | Enter |
| Send | Ctrl+D |
| Cancel active response | Ctrl+C |
| Clear composition | Ctrl+U |
| Quit | Ctrl+Q |
| Scroll transcript | PageUp / PageDown |
| Show build and reuse details | F2 |
| Clear retained state and start over | `/new`, then Ctrl+D |
| Save transcript explicitly | `/save PATH`, then Ctrl+D |
| Toggle help | `/help`, then Ctrl+D |

Bracketed paste never sends automatically. Code blocks and reasoning are
visually separate. Model output cannot issue terminal control sequences.
Interrupted output remains visible, but is excluded from the next request;
the user prompt returns to the editor for retry. The input remains locked during
an active response. Reset is allowed only when the server is idle.

Exports are JSON with content, reasoning and interrupted flags. They are created
with owner-only permissions and never overwrite an existing path. There is no
automatic persistence. Composition is limited to 256KiB; the transcript to 1MiB
and 1,024 attempted turns. At the limit, save and start a new conversation; no
history is silently removed. Rendering builds only a window of transcript lines.
Lines longer than 4,096 displayed bytes are visibly clipped; export and request
history retain the complete text. Context overflow is a server error, not an
automatic compaction request.

The footer shows phase, processed input, context usage, engine physical footprint
and observed generation rate. The first generated token has no decode rate yet.
Missing measurements appear as a dash. Memory snapshots are sampled at most once
per second; F2 shows the sample's monotonic timestamp. A matching token prefix
and actually reused computation are different measurements.

## API contract

The listener binds `127.0.0.1`. There is one inference worker and one retained
session. It owns the model, tokenizer, session, Metal operations and destruction.
Four network workers handle a bounded queue of eight sockets. Stream queues are
bounded to 256KiB; disconnected or blocked clients cancel their request. Network
workers read immutable status snapshots and never touch Metal objects.

| Endpoint | Behavior |
|---|---|
| `GET /v1/models` | One selected model; available during initialization |
| `GET /health` | 503 while loading or failed, 200 after initialization; includes current phase |
| `POST /v1/chat/completions` | Nonstreaming JSON or SSE; busy requests receive 429 |
| `GET /zerocool/status` | Version 1 snapshot; available while loading, generating, cancelling and draining |
| `POST /zerocool/session/reset` | Reset retained state while idle; 409 while busy; 503 before ready |

The listener is reachable from a browser, so every request is filtered before
any work: a request carrying an `Origin` header is refused with 403, a `Host`
that is not loopback is refused with 403, and a POST without
`Content-Type: application/json` is refused with 415. The last of these forces a
CORS preflight, which fails because no CORS headers are ever returned. Ordinary
API clients already send that header. Text-part content
(`[{"type":"text","text":...}]`) is accepted alongside plain string content;
other modalities still fail explicitly. Numeric and boolean parameters are
validated by type rather than coerced, so a float `max_tokens` or a non-boolean
`stream` is a named 400 rather than a silently different setting.

Message text is never allowed to become a control token. `Tokenizer::encode_chat`
replaces control-token text inside message content, reasoning, tool calls and
tool declarations with a private marker before rendering, then encodes those
bytes as ordinary text. Only the chat template itself produces turn boundaries,
so a file pasted by a coding client cannot forge a system or assistant turn.
The prompt text is preserved exactly; `<tool_response>` keeps its meaning inside
a tool result because the template itself tests for that wrapper.

Chat supports text-only system/user/assistant/tool messages, `model`, `stream`,
`stream_options.include_usage`, `max_tokens` or `max_completion_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, `n=1`, `tools`, and `tool_choice` of
`auto` or `none`. Thinking uses `enable_thinking` (also accepted under
`chat_template_kwargs`) and the existing `reasoning_effort` template control.
No tool executes inside ZeroCool. Tool calls appear as structured function calls;
the client supplies tool-result messages on continuation.

The model IDs are `qwen3.8-flash-next:4bit` and
`qwen3.8-flash-next:mixed-4_8bit`. Unknown model IDs, nontext content, multiple
choices, forced tool choice, `stop`, `response_format`, `logit_bias`, `logprobs`
and nonzero frequency/presence penalties fail explicitly. This is a documented
Chat Completions subset, not a complete OpenAI API implementation. Responses API,
embeddings, images, automatic tool execution and automatic compaction are absent.

Responses include `id`, `object`, `created`, `model`, `choices` and usage.
Reasoning uses `reasoning_content`. Tool calls preserve indices in streamed
deltas and receive request-specific IDs. Streaming finishes with a finish-reason
chunk and `[DONE]`; when requested, a final `choices: []` chunk contains usage,
and preceding chunks have `usage: null`. Diagnostics are in the nonstreaming
`zerocool` extension and status, never ordinary content deltas. The finish-reason
and final usage chunks are released only after GPU/I/O draining succeeds.
Errors before
headers use JSON HTTP errors; errors after headers use an SSE error object.

Status includes an `instance_id`. Chat sends it in `X-ZeroCool-Instance-Id` on
subsequent requests; a mismatch returns 409 before reset or generation admission.
Ordinary API clients do not need this header.

Retained history survives a rejected or cancelled request whenever it can still
be continued. A request refused before generation, a malformed tool call in a
completed response, and a cancellation between decode steps all leave the
committed prefix intact, so the next turn does not re-ingest the prompt. Only a
failure that leaves the recurrence unrollable clears the session.

Disconnect cancels even during prompt ingestion. Busy status remains until I/O
and GPU users drain and invalid session state is cleared. Exact continuation
can reuse state; shortened or edited history rebuilds it. Startup or drain
failure is visible as `failed` and requires restarting the server. Failed startup
releases partial model resources on the inference worker. A failed
ordinary generation can be retried after drain; no second inference is queued.

## Build and model-free tests

FTXUI v7.0.3 is pinned at commit
`f921fad208912747c17d129a8ef75ec7624b6eec`; HTTP uses macOS system libcurl.
These dependencies stay outside `zerocool_lib`. Build with
`-DZEROCOOL_BUILD_TUI=OFF` to omit the client. FetchContent needs network access
on the first build or an explicit `FETCHCONTENT_SOURCE_DIR_FTXUI` checkout.
The test-only `chat_fixture` uses the same transport and TUI with an injected
executor; it is not installed, and production has no fake-inference option.

```sh
ctest --preset release   # native, transport and model-free Python checks
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/qwen/test_qwen

# A separate Python 3.12 environment for client checks:
python3.12 -m venv .cache/usability-client
.cache/usability-client/bin/pip install aider-chat==0.86.2 pyte==0.8.2
.cache/usability-client/bin/python scripts/qwen/check_chat_terminal.py \
  --out .cache/chat-terminal-check
```

## Real-model qualification

Start an explicitly configured server with uncompressed engine memory, then run:

```sh
.venv/bin/python scripts/qwen/api_check.py --url http://127.0.0.1:8080 \
  --out .cache/chat-api-check.json
.cache/usability-client/bin/python scripts/qwen/check_aider.py \
  --url http://127.0.0.1:8080 --aider .cache/usability-client/bin/aider \
  --out .cache/chat-aider-check
```

The API checker discovers the selected model, tests streams, retained/changed
history, context rejection, tool/result continuation, ingestion/generation
cancellation and recovery. It saves partial failures. Aider is pinned to 0.86.2,
with auxiliary model selection directed at the same local model, repository maps,
analytics, update checks, shell suggestions, automatic tests and automatic commits
disabled. It works in a disposable repository: fix a broken addition function,
independently test it, inject a regression, ask for recovery, and test again.
The checker bounds executable output to a tiny arithmetic AST. Two Aider
invocations do not establish a retained multi-turn coding session.

`check_api_performance.py` preserves old/new binary hashes, asset receipt/manifest
hashes, exact workload and settings, host snapshots, response reports and status
samples. It requires AC power, nominal thermal state, Low Power Mode off, 13.5GiB
available for the fixed 12GiB budget, equal admitted plans and identical output
tokens. Compression, swap growth or missing memory data prevents a clean result.
It never promotes a candidate automatically.

Create a workload JSON with `messages`, `max_tokens` (up to 256), and optional
`followups` strings for a soak. Use the same workload for both binaries:

```sh
.venv/bin/python scripts/qwen/check_api_performance.py paired \
  --baseline .cache/usability-stage/baseline/zerocool \
  --model .cache/models/qwen38-flash-next --prepared .cache/prepared/q4-records-v1 \
  --workload /path/to/fixed-chat-workload.json --out .cache/api-paired-screen

.venv/bin/python scripts/qwen/check_api_performance.py soak \
  --model .cache/models/qwen38-flash-next --prepared .cache/prepared/q4-records-v1 \
  --workload /path/to/fixed-chat-workload.json --seconds 1200 --out .cache/api-soak
```

The screen starts with two alternating old/new pairs (`--pairs 5` extends it).
New-server status polling is included at one second intervals; latency regression
over 3% requires investigation. The soak preserves history and fails on overflow;
it does not compact. Review memory growth after cache warmup. The scripted soak
is distinct from the required 20-minute interactive coding session.

Still required before usability promotion: a real API pass, independently checked
Aider workflow, paired latency evidence, actual TUI-versus-direct-request overhead,
status p95 below 500ms during real inference, cancellation acknowledgement below
500ms with drain measured separately, and a 20-minute interactive session with
no progressive memory or sustained swap growth. Target TUI footprint is below
128MiB. This stage does not establish 5 tokens/s.
