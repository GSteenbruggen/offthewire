# llama.cpp / LM Studio backend — implementation plan

Status: implemented on master · Target release: 1.5.0

## Goal

Let OffTheWire drive a model served by **llama.cpp's `llama-server`** or **LM
Studio** instead of Ollama, selected with `--backend`. Both expose the same
OpenAI-compatible HTTP API, so one new client covers both. Everything above
the client — the agent loop, session persistence, compaction, truncation
recovery, the tool registry, the UI — stays untouched and backend-agnostic.

Why this is worth doing first among the roadmap items: connecting to
`llama-server` directly turns the measured speculative-decoding win
(1.27× on qwen3.8:27b via draft-MTP, using the `llama-server.exe` Ollama
already ships) into a supported configuration instead of a lab experiment.

## Constraints (non-negotiable)

- **Loopback only.** The new client enforces the same rule as
  `OllamaClient`: host is validated at construction, non-local is a hard
  `NonLocalHostError`, no call site can opt out. `scripts/verify_offline.py`
  must scan the new module and prove it.
- **Session files stay backend-agnostic.** Messages are stored in the
  existing (Ollama-shaped) format; translation to the OpenAI wire format
  happens at request time only. A session started on Ollama must resume on
  llama.cpp and vice versa.
- **No behavior change for Ollama users.** `--backend ollama` remains the
  default; the existing client is not modified.

## Architecture

```
agent.py ──(duck-typed client)──> OllamaClient        (src/ollama_client.py, unchanged)
                              └─> OpenAICompatClient  (src/openai_compat.py, new)
```

The agent already uses the client through a narrow surface:
`ping, chat_stream, generate (compaction), load, unload` plus
`models.py`'s `list_models / model_capabilities / pick`. The new client
implements the same surface and returns the same `GenResult`.

### src/openai_compat.py (new module)

- `OpenAICompatClient(host, backend_name)` — reuses `normalize_host`-style
  validation (shared helper, parameterized default port: llama.cpp `8080`,
  LM Studio `1234`).
- `ping()` → `GET /v1/models`.
- `list_models()` → `GET /v1/models`, mapped to the dict shape
  `models.py` expects.
- `capabilities(model)` → best effort:
  - llama.cpp: `GET /props` gives `n_ctx` (the real, launch-time context
    size) and model metadata. `supports_tools` is assumed true (the server
    must be launched with `--jinja` for tools — documented, and probed with
    a tiny tools request at startup so a missing `--jinja` is a clear error,
    not a mid-turn surprise).
  - LM Studio: `/props` doesn't exist; fall back to `/v1/models` metadata
    and the user-supplied `--context`.
- `chat_stream(...)` → `POST /v1/chat/completions` with `stream: true`,
  parsing SSE (`data: {...}` lines, `[DONE]` sentinel).
- `generate(...)` → non-streaming wrapper (used by compaction).
- `load()/unload()` → no-ops (the server owns the model's lifecycle).

### Message translation (request time)

| Session format (stored) | OpenAI wire format (sent) |
|---|---|
| `content` string on user/system/assistant | unchanged |
| user msg with `images: [b64,...]` | `content` parts: `{type:"text"}` + `{type:"image_url", image_url:{url:"data:image/png;base64,..."}}` |
| assistant `tool_calls: [{function:{name, arguments<dict>}}]` | `{id:"call_<i>_<j>", type:"function", function:{name, arguments:<JSON string>}}` |
| `{role:"tool", tool_name, content}` | `{role:"tool", tool_call_id:"call_<i>_<j>", content}` |

Tool-call ids don't exist in the session format; they are synthesized
deterministically by position while walking the message list, so a tool
result always pairs with the assistant call that preceded it.

Responses are translated back **into** the session shape (arguments JSON
string → dict) before the agent sees them, so `run_turn` and the session
log never know which backend produced them.

### Streaming specifics

- Deltas: `choices[0].delta.content`, `.reasoning_content` (llama.cpp with
  `--reasoning-format deepseek`), `.tool_calls` (indexed fragments whose
  `arguments` strings are concatenated per index, then parsed once at end).
- `stream_options: {include_usage: true}` so the final chunk carries
  `usage.prompt_tokens` — this feeds `Session.note_actual()` and keeps the
  measured token budget (and therefore compaction and truncation recovery)
  working identically. llama.cpp's `timings` object, when present, feeds
  tok/s display; otherwise rates fall back to wall-clock.
- `finish_reason` mapping: `"length"` → `done_reason "length"` (truncation
  recovery just works), everything else → `"stop"`.
- Mid-stream `{"error": ...}` objects raise, same as the Ollama client.

### Thinking toggle

Per-request `chat_template_kwargs: {"enable_thinking": <bool>}` — honored
by llama.cpp for qwen3-style templates, ignored harmlessly elsewhere.
Effort strings ("low"/"high"…) don't map to this API; any truthy value
enables thinking. `ThinkingPolicy` itself is unchanged.

### Context length

llama.cpp fixes the context size when the server is launched (`-c`);
it cannot be changed per request. So:
- `session.context_limit` initializes from `/props` `n_ctx` when smaller
  than `--context`.
- `/maxtokens` explains the situation instead of silently failing: raising
  beyond `n_ctx` requires relaunching the server.

## Integration (src/agent.py)

- `--backend {ollama,llamacpp,lmstudio}` (default `ollama`) and `--host`
  (loopback-only override; error message names the backend).
- Startup: construct the right client; model = `--model`, else the single
  model the server reports (llama.cpp), else the existing picker (Ollama,
  LM Studio multi-model).
- Capability gate: same "supports tools" refusal, with the llama.cpp probe
  above; vision assumed absent unless the probe says otherwise (attaching
  an image warns, as today).
- `/model`, `/maxtokens` display backend-appropriate facts.
- MCP server (`server.py`) stays Ollama-only for now — it exposes Ollama
  model management, which has no llama.cpp equivalent.

## Testing

New `scripts/test_backends.py`, all offline via `httpx.MockTransport`:
1. Loopback refusal — non-local hosts raise at construction, env can't
   bypass.
2. Translation round-trip — images, tool calls, tool-result pairing ids,
   multi-call turns.
3. SSE parsing — content/reasoning/tool-call fragments, `[DONE]`,
   mid-stream error, usage extraction, finish_reason mapping.
4. Compaction path — `generate()` through the compat client.
5. Capability probe — `--jinja` missing produces the clear startup error.

Plus: suite registered in CI's single-line list (all three OS), README
command/flag tables synced (test_docs enforces), verify_offline extended to
the new module.

## Verification against real servers (manual, Windows)

Ollama ships `llama-server.exe`; the launch quirks are known from the
rung-0 experiment (PowerShell launch, `GGML_BACKEND_PATH` → the
`ggml-cuda.dll` file itself, PATH prepend of the cuda dir). Smoke:
a real tool-loop turn (read → edit → test), a truncation-recovery turn,
and a compaction. LM Studio same flow if installed.

## Out of scope (deliberately)

- A `scripts/launch_llamacpp.ps1` helper that resolves an installed Ollama
  model's GGUF blob and launches `llama-server` with the right CUDA env and
  speculative-decoding flags. That's the "turbo mode" follow-up, gated on
  the placement-parity spike; this change only *connects* to a server the
  user already runs.
- MCP server support for non-Ollama backends.
- Auto-detecting a running server by port scanning (explicit `--backend`
  only; guessing wrong is worse than asking).

## Milestones

1. `src/openai_compat.py` + `scripts/test_backends.py` green. ✅ (34 checks)
2. Agent integration (`--backend`, `--host`, startup, capability gate). ✅
3. Docs: README backends section, help text, roadmap update. ✅
4. Full 12-suite gate on Windows ✅ + CI (3 OS).
5. Manual smoke against llama-server ✅ — real tool loop (write_file +
   read_file round-trip) against Ollama's shipped `llama-server.exe` with
   qwen3:0.6b, streaming, live token accounting from `/props` n_ctx, tok/s
   from `timings`. LM Studio path exercised by mocked tests only (no
   install on this machine). Ship as 1.5.0.
