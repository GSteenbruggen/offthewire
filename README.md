<p align="center">
  <img src="assets/logo.png" alt="OffTheWire" width="480">
</p>

<p align="center">
  <a href="https://github.com/GSteenbruggen/offthewire/releases/download/v1.3.0/OffTheWire-Setup-1.3.0.exe"><strong>Windows (x64)</strong></a>
  ·
  <a href="https://github.com/GSteenbruggen/offthewire/releases/download/v1.3.0/OffTheWire-1.3.0-linux-x86_64.tar.gz"><strong>Linux (x86_64)</strong></a>
  ·
  <a href="https://github.com/GSteenbruggen/offthewire/releases/download/v1.3.0/OffTheWire-1.3.0-macos-arm64.tar.gz"><strong>macOS (Apple&nbsp;Silicon)</strong></a>
  <br>
  <sub>Windows: installer, per-user, no administrator · Linux and macOS: self-contained tarballs · <a href="https://github.com/GSteenbruggen/offthewire/releases">all releases</a></sub>
</p>

OffTheWire brings agentic coding to your local Ollama models. It can inspect
and modify code, run tests, shell commands, search repositories, analyze
images, and optionally perform web research — <em><ins>all while inference
stays on your own machine</ins></em>. Existing Ollama models are recognized
automatically, so
anything installed before OffTheWire continues working with no migration or
reinstallation required. A bundled MCP server also exposes your local models
to other compatible tools — check `/help` [here](#in-session-commands).

**Why it exists.** A local model is only one part of a truly local coding
workflow. General-purpose agent frameworks may still depend on cloud
services, telemetry, external APIs, or configuration choices that can
unintentionally reintroduce network access. They are also commonly optimized
for frontier-scale models, which can make them inefficient or unreliable when
paired with smaller local models. OffTheWire was designed specifically for
this environment: local execution is treated as a system-level constraint,
and the agent is tuned around the realities of Ollama models running on
consumer and workstation hardware. The result is a coding agent built for
environments where privacy, control, and independence from the cloud are
first-class requirements.

**How it differs from similar tools:**

- **Containment is enforced in code and independently verifiable.** The
  model client refuses non-loopback hosts at construction — no
  configuration, including the `OLLAMA_HOST` environment variable, can
  redirect it. The opt-in web channel refuses any URL that resolves to a
  private or local address. `scripts/verify_offline.py` proves both
  properties and scans every source file for stray egress, so the privacy
  claim can be checked in seconds rather than taken on trust.
- **Private on disk, not just on the network.** No transcript, input
  history, or image is written to disk unless persistence is explicitly
  enabled with `--save`.
- **Engineered for small local models, from measurement.** Reasoning is
  toggled per turn (planning thinks; mechanical steps do not — a measured
  ~2× difference per step), the token budget is calibrated against the
  runtime's actual counts with automatic compaction, edits are addressed by
  line number rather than string matching, and tool schemas are kept small
  and flat. These decisions came from benchmarking real agent workloads on
  consumer hardware, not from adapting a cloud harness.
- **No account, no telemetry, no update phone-home, no cloud dependency of
  any kind.**

```powershell
.venv\Scripts\python.exe scripts\verify_offline.py
```

**Platforms:** Windows x64, Linux x86_64, and macOS (Apple Silicon).
Requires [Ollama](https://ollama.com/download) and a model with the `tools`
capability. The Windows build is exercised interactively; Linux and macOS
pass the full test suite in CI — treat both as beta.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Security model](#security-model)
- [Session persistence](#session-persistence)
- [Image attachments](#image-attachments)
- [Web lookup](#web-lookup)
- [Design notes](#design-notes)
- [Benchmarks](#benchmarks)
- [MCP server](#mcp-server)
- [Repository layout](#repository-layout)
- [Building the installer](#building-the-installer)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **Fully local agent loop** — read, write, edit, search, and shell tools over
  a workspace, driven by any tool-capable Ollama model.
- **Per-turn thinking policy** — reasoning is enabled when planning or
  recovering from errors and disabled for mechanical steps, roughly halving
  wall-clock time on typical agent turns (see [Benchmarks](#benchmarks)).
- **Context management** — token budget tracked against Ollama's reported
  counts, with automatic multi-pass compaction at 75% of the window.
- **Image attachments** — paste a screenshot from the clipboard, drag a file
  into the terminal, or reference a path; requires a vision-capable model.
- **Opt-in persistence** — nothing is written to disk unless `--save` is
  passed; sessions can be resumed across runs.
- **Opt-in web lookup** — search and page reading through a local SearXNG
  container, with results condensed by the local model and attributed to
  sources.
- **Terminal UI** — streaming markdown rendering, syntax-highlighted file
  previews before writes, approval prompts for mutating operations, and clean
  degradation when output is piped.
- **Windows installer, Linux and macOS tarballs** — a per-user setup
  executable on Windows (no administrator rights), self-contained tarballs
  elsewhere; all three platforms run the full test suite in CI on every
  push.

---

## Installation

### From the installer

Download `OffTheWire-Setup-<version>.exe` from
[Releases](https://github.com/GSteenbruggen/offthewire/releases) and run it.

- Installs per-user to `%LOCALAPPDATA%\Programs\OffTheWire`; no administrator
  rights are requested and only `HKCU` is modified.
- Optionally adds the install directory to `PATH` and registers an
  uninstaller in Add/Remove Programs.
- Checks for Ollama after installation and reports what is missing rather
  than failing later.
- An optional checkbox configures [web lookup](#web-lookup); it requires
  Docker Desktop and downloads a ~1 GB container image, and is unchecked by
  default.

The installer is not code-signed. Windows SmartScreen will display an
"unrecognized app" warning; choose *More info → Run anyway*, or build from
source.

The installer does not bundle Ollama or any model. Ollama is a separate
product with its own installer and update channel, and models are tens of
gigabytes; both must be installed independently.

### Linux

```bash
tar -xzf OffTheWire-1.3.0-linux-x86_64.tar.gz
./OffTheWire/OffTheWire            # the current directory becomes the workspace
```

Optional extras: `wl-clipboard` (Wayland) or `xclip` (X11) enable pasting
images from the clipboard with `Alt+V` — without one of them, images still
attach by typed path or drag-and-drop. Web lookup uses
`scripts/setup_searxng.sh`, which requires Docker.

### macOS (Apple Silicon)

```bash
curl -LO https://github.com/GSteenbruggen/offthewire/releases/download/v1.3.0/OffTheWire-1.3.0-macos-arm64.tar.gz
tar -xzf OffTheWire-1.3.0-macos-arm64.tar.gz
./OffTheWire/OffTheWire
```

The build is not notarized. Downloading with `curl` (as above) avoids
Gatekeeper's quarantine entirely; a browser download triggers it, in which
case clear the attribute once after extracting:

```bash
xattr -dr com.apple.quarantine OffTheWire
```

Clipboard image paste works out of the box (`osascript` ships with the OS).

### From source

```powershell
git clone https://github.com/GSteenbruggen/offthewire
cd offthewire
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# verify containment
.venv\Scripts\python.exe scripts\verify_offline.py

# run the agent on a directory
.venv\Scripts\python.exe src\agent.py C:\path\to\project
```

```bash
# Linux
git clone https://github.com/GSteenbruggen/offthewire
cd offthewire
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/verify_offline.py    # verify containment
.venv/bin/python src/agent.py ~/path/to/project
```

To install a global `OffTheWire` command without modifying environment
variables:

```powershell
.\scripts\install_launcher.ps1            # -Name <alias> to rename, -Uninstall to remove
```
```bash
./scripts/install_launcher.sh             # Linux: --name <alias>, --uninstall
```

This writes a launcher shim into `%LOCALAPPDATA%\Microsoft\WindowsApps`,
which is already on the user `PATH`. The working directory at launch becomes
the agent's workspace.

---

## Usage

```powershell
OffTheWire [workspace] [options]
```

| Option | Description |
|---|---|
| `--model NAME` | Use a specific model instead of the interactive picker. |
| `--context N` | Context window in tokens. Default 32768, capped to the model's maximum. |
| `--think auto\|always\|never` | When to enable reasoning. Default `auto`. |
| `--think-level low\|medium\|high\|max\|default` | Reasoning effort when a turn does think. Default `low`; `default` sends a plain boolean for models without effort levels. |
| `--yes` | Auto-approve file writes and shell commands. |
| `--save` | Persist the conversation to disk. Off by default. |
| `--max-steps N` | Tool calls allowed per turn. Default 25. |
| `--hide-reasoning` | Show progress indicators instead of streaming the reasoning block. |
| `--web` | Enable internet lookup. Off by default. |
| `--searxng URL` | SearXNG endpoint. Default `http://localhost:8080`. |
| `--prompt "..."` | Run one task non-interactively and exit. |
| `--resume [ID]` | Continue a saved session; bare `--resume` selects the most recent. |

### In-session commands

Anything that does not begin with `/` is sent to the model as a request.

| Command | Description |
|---|---|
| `/help` | List available commands. |
| `/save` | Report whether the conversation is being written to disk, and where. |
| `/savesession` | Start saving a conversation launched without `--save`, retroactively — every turn and pasted image so far is backfilled. |
| `/model` | Show the current model and its capabilities (context, `tools` / `thinking` / `vision`). |
| `/think <arg>` | Set when to reason (`auto` \| `always` \| `never`) or at what effort (`low` \| `medium` \| `high` \| `max` \| `default`). |
| `/reasoning` | Toggle streaming the model's reasoning to the terminal. |
| `/approve` | Toggle auto-approval of file writes and shell commands. |
| `/env` | Show what the model has been told about the environment. |
| `/paste` | Attach the image on the clipboard (equivalent to `Alt+V`). |
| `/folder [path]` | Show or change the directory the agent works in. |
| `/web [on\|off]` | Toggle internet lookup without restarting; bare `/web` shows status and the queries made this session. |
| `/maxtokens [n]` | Show or change the context window (accepts `65536`, `64k`, `128K`). |
| `/maxsteps [n]` | Show or change the tool-call limit per turn. |
| `/tokens` | Show current context usage. |
| `/compact` | Summarize older turns to free context. |
| `/history [full]` | Replay the conversation; `full` includes tool calls and results. |
| `/sessions` | List saved sessions. |
| `/resume [id]` | Continue a saved session; without an id, the most recent. A unique prefix suffices. |
| `/clear` | Start a fresh conversation. |
| `/quit` | Exit. |

`/folder <path>` moves the agent to a different working directory
mid-conversation; tools, shell working directory, and the environment block
follow. Because the system prompt names the workspace, switching folders costs
a one-time reprocess of the cached conversation prefix on the next turn.

### Input

| Key | Action |
|---|---|
| `Enter` | Submit |
| `Alt+Enter` | Insert a newline |
| `Alt+V` | Attach the image on the clipboard |
| Paste | Inserted verbatim, including newlines; never auto-submits |
| `↑` / `↓` | Input history (persisted only with `--save`) |

Multi-line paste uses bracketed paste mode, so pasted blocks arrive as a
single turn. Input is sanitized of byte-order marks and null characters,
which Windows PowerShell can prepend when piping into a native command.

### Interrupting

`Ctrl+C` (or `Ctrl+Break`) cancels the current turn without ending the
session; pressing it twice within two seconds exits. Interruption is
processed in approximately 0.1 s mid-generation. The partial response is
retained and annotated so the model does not resume the cancelled answer.

**Known limitation:** `run_command` executes synchronously, so `Ctrl+C` is
not processed during a long-running shell command until it completes or
reaches its 120 s timeout.

### Example session

Given a `calc.py` in which `add()` returns `a - b` and a failing test:

```
[1] thinking=on  (planning)         32.9s   -> list_dir()
[2] thinking=off (mechanical step)   6.9s   -> read_file(calc.py), read_file(test_calc.py)
[3] thinking=off (mechanical step)   7.0s   -> run_command(python test_calc.py)   [exit 1]
[4] thinking=off (mechanical step)   9.7s   -> edit_lines(calc.py, 2, 2, "    return a + b")
[5] thinking=off (mechanical step)   4.2s   -> run_command(python test_calc.py)   all tests passed
[6] thinking=off (mechanical step)   5.7s   -> done
```

Approximately 66 seconds end to end on the reference configuration (see
[Benchmarks](#benchmarks)). The single planning turn dominates; with
reasoning forced on for every turn, the same task takes roughly three
minutes.

---

## Security model

Containment is enforced in code, not by convention, and
`scripts/verify_offline.py` checks the properties below plus scans every
source file for stray network destinations and networking imports.

**The model path accepts only loopback.** `normalize_host()` in
`ollama_client.py` raises `NonLocalHostError` for any non-loopback host, and
it is the single place the HTTP client is constructed — no call site can opt
out, including via the `OLLAMA_HOST` environment variable. Prompts, code, and
files never leave the machine. There is no telemetry, crash reporting, update
check, or account.

**The web channel cannot reach private networks.** When web lookup is
enabled, every fetched URL is resolved and refused if any resulting address
is non-public (loopback, RFC 1918, link-local, or otherwise reserved).
Redirects are followed manually so each hop is re-checked. This prevents
search results or fetched page content from steering the agent into local
services — including Ollama's own unauthenticated API.

**Nothing is written to disk by default.** See
[Session persistence](#session-persistence).

**Mutating operations require approval.** File writes, edits, and shell
commands prompt before executing unless `--yes` is passed. The full content
of every file write is displayed before it runs, even under `--yes`.

**Workspace confinement.** File tools resolve paths (including symlinks and
relative components) before checking containment; anything resolving outside
the workspace root is refused. `run_command` is intentionally not confined
beyond its working directory and is gated by approval instead.

**Residual risks.** Prompt injection is inherent to any agent that reads
files and web pages; the approval gate is the mitigation. The installer is
unsigned. DNS rebinding is out of scope for the web-channel guard.

> The MCP server is deliberately not registered with cloud-hosted MCP
> clients. Doing so would keep inference local while routing prompts and
> results through a third party. The agent is self-contained and requires no
> external orchestration.

---

## Session persistence

By default the conversation exists only in memory and is discarded on exit.
The `--save` flag enables persistence, which covers all three artifacts a
conversation produces: the transcript (`sessions/<id>.jsonl`), the input
history, and pasted clipboard images. Without `--save`, pasted images are
written to the temporary directory only for the moment they are encoded.

`/save` reports the current mode; `/sessions`, `/history`, and `/resume`
state explicitly when the current conversation is not being recorded.

```powershell
OffTheWire --save                      # persist this conversation
OffTheWire --save --resume             # continue the most recent saved session
OffTheWire --save --resume 20260810    # by id or unique prefix
```

Forgot the flag? `/savesession` opts in mid-conversation and backfills
everything accumulated so far — turns and pasted images both — after which
the session behaves exactly as if `--save` had been passed at launch.

`/resume` also works without `--save`: the previous conversation is loaded
read-only and continued in memory without further writes.

Saved sessions are append-only JSONL, so a crash cannot corrupt earlier
turns. On resume:

- The system prompt is rebuilt from the current environment rather than
  restored from the log, so a resumed session reflects the current date and
  machine state.
- Token accounting falls back to estimation until the next model response
  reports an exact prompt size.
- Incomplete trailing tool exchanges (from a session terminated mid-turn)
  are trimmed, since replaying an unanswered tool call causes models to
  repeat the call or invent its result.
- Oversized conversations are compacted automatically before the first
  request rather than sent to be silently truncated.

Transcripts can be reviewed with `/history` in-session or
`scripts/show_session.py` (`--full` includes tool calls and results; `--md`
renders markdown).

Base64 image data is never written to session files; only file paths are
recorded, and images are re-read from disk on resume. A 2 MB screenshot
would otherwise add ~2.7 MB of encoded text to the log per reference.

---

## Image attachments

Requires a model with the `vision` capability (`/model` reports
capabilities). Three input methods converge on a file path:

| Method | Usage |
|---|---|
| Clipboard | `Win+Shift+S` to capture, then `Alt+V` (or `/paste`) |
| Drag and drop | Drop a file onto the terminal window |
| Typed path | Any existing `.png` / `.jpg` / `.gif` / `.webp`; quote paths containing spaces |

```
> what is wrong with this layout? [image: pasted-20260821_130433.png]
  ⧉ pasted-20260821_130433.png  (png · 1920×1080 · 284.1 KB)
```

Implementation notes:

- **Format is determined by file content, not extension.** A `.png` that is
  actually WEBP (common with browser downloads) is reported clearly rather
  than producing an opaque decode error from Ollama.
- **Clipboard capture converts `CF_DIB` directly** — the raw bottom-up BGR
  pixel buffer produced by the Snipping Tool — to PNG using only the
  standard library. A 4K capture converts in ~0.25 s.
- **A path only becomes an attachment if the file exists**, so prose that
  merely resembles a filename is left untouched.
- **Reference labels always match what was sent.** Attachments are numbered
  in the message text; if one of several images fails to load, the
  remainder renumber and the failure is reported.

If the active model lacks vision support, the image is not sent; the path
remains in the text and a message recommends a vision-capable model.

`Alt+V` is used rather than `Ctrl+V` because Windows Terminal and VS Code
intercept `Ctrl+V` and produce nothing when the clipboard holds an image.
`Ctrl+V` is also bound for terminals that forward it, and `/paste` covers
those that forward neither.

---

## Web lookup

Disabled by default. Enable it at startup with `--web`, or at any point
mid-session with `/web on` — no restart required. Enabling adds two tools —
`search_web(query, pages)` and `fetch_url(url)` — both treated as mutating
and therefore subject to approval; `/web off` removes them again.

One-time backend setup (SearXNG in a local Docker container):

```powershell
.\scripts\setup_searxng.ps1     # Windows; requires Docker Desktop
```
```bash
./scripts/setup_searxng.sh      # Linux/macOS; requires Docker
```

Then either:

```
OffTheWire --web        # enabled from the first turn
```
```
> /web on               # or enabled mid-conversation
  web lookup ON via http://localhost:8080
```

Toggling mid-session updates the model's instructions, so it costs a one-time
reprocess of the cached conversation prefix — the command output says so when
it happens. `/web on` also checks that SearXNG is actually reachable and
reports plainly when it is not, rather than letting the first search discover
it. Bare `/web` shows status and lists every query made in the session.

Search runs through a local [SearXNG](https://github.com/searxng/searxng)
container bound to `127.0.0.1`, so no API key or account is required and no
third party accumulates a query log. The setup script enables SearXNG's JSON
output format, which is off by default and whose absence is the usual cause
of agent search failing with 403 responses.

### Condensation

A web page is 5,000–15,000 tokens; two would exhaust a 32k session. Fetched
pages are stripped to article text and reduced by the local model to a short
digest (~200 tokens per source) that answers the query. Page content is
processed entirely locally. A representative documentation query returned a
correct, sourced answer in 136 tokens where the underlying page contained
5,605 characters of article text.

### Source handling

Search-and-condense pipelines fail dangerously when failures are silent: an
unreadable primary source drops out, a low-quality source fills the gap, and
condensation rewrites its errors into authoritative prose. The pipeline is
built to keep failures visible:

| Safeguard | Behavior |
|---|---|
| Failures are reported | Unreadable, blocked, and non-answering sources are listed by URL with the reason. |
| Answers carry quotes | The verbatim source sentence accompanies each answer, making mislabelled data visible. |
| Sources are ranked | First-party documentation, MDN, GitHub, and Stack Overflow rank above tutorial farms, which are tagged as low-quality; nothing is excluded, only reordered and labelled. |
| Concerns are flagged | The condenser reports implausible figures, ambiguous units, and contradictions rather than smoothing them over. |
| Refusals are not sources | Responses amounting to "cannot be determined from the text" are detected and excluded from source counts. |
| Reserve fetching | If fewer than two sources answer, additional results are fetched rather than reporting a lone answer. |
| Single sources are marked | An uncorroborated answer is labelled as such; with multiple sources, disagreement is reported rather than resolved silently. |

The system prompt instructs the model that reporting an uncertain answer as
certain is worse than reporting that verification failed. This approach
suits documentation, APIs, and stable reference material; it is not suited
to live data rendered by JavaScript.

---

## Design notes

Decisions specific to running agents on small local models:

- **Per-turn thinking policy.** Reasoning is enabled for planning, after a
  tool error, and periodically in long chains; disabled otherwise. Measured
  benefit is ~2× on mechanical steps for `qwen3.8` (it was ~13× on
  `qwen3.6`); re-measure when changing models.
- **Explicit reasoning effort.** `qwen3.8`'s chat template defaults to its
  maximum effort (`xhigh`) when sent a bare `think: true`. The agent
  requests an explicit level (default `low`), which in testing reduced a
  planning turn's reasoning output by ~29% and wall time by ~33% with no
  observed quality loss.
- **Explicit context length.** Ollama loads models with a 4096-token window
  regardless of the model's capability and truncates silently beyond it. The
  agent always sends `num_ctx`, and `/maxtokens` reloads the model at a new
  size, reporting the resulting GPU/CPU split.
- **Line-range edits rather than string replacement.** Local models
  reproduce whitespace unreliably; addressing lines by number is more
  robust, and `read_file` numbers its output accordingly.
- **Few tools, flat arguments.** Small models mis-select from large tool
  menus and mishandle nested objects.
- **All tool output truncates.** A single large file dump would otherwise
  consume the session.
- **Shell is PowerShell, and the model is told so** — including a list of
  PowerShell equivalents for common POSIX commands, and the `&` call
  operator prepended to leading quoted paths, both of which otherwise
  produce consistent failures.
- **Command output forced to UTF-8** in both directions; Windows code pages
  otherwise corrupt non-ASCII output into text the model cannot read.
- **Repeated identical tool calls are refused.** After three identical calls
  with no intervening state change, the agent returns an explanatory error
  instead of executing again, converting a silent loop into a visible one.
- **Tool calls arrive atomically from Ollama** — measured at 60 s of
  complete silence for a 238-token file write — so file content is shown in
  full before execution, and an elapsed-time indicator runs during the
  silent window.
- **Token budget is measured, not estimated.** Accounting is calibrated
  against the `prompt_eval_count` Ollama reports; only messages appended
  since the last response are estimated. Compaction triggers at 75% of the
  window and summarizes in bounded passes so the summarization prompt itself
  always fits the context window.

### Environment grounding

A local model does not know the date, the shell it is driving, or which
interpreter to invoke. The agent supplies these facts, split by volatility
to preserve Ollama's prefix cache (warm prompt ingestion measures 3–5×
faster than cold):

| Volatility | Placement | Contents |
|---|---|---|
| Static per session | System prompt (stays cached) | OS, shell, interpreter path, tool versions |
| Volatile | Tail of conversation | Date, git branch and status, project type, instruction files |

Volatile facts are re-probed each turn (~25 ms) but injected only when
changed. The grounding block is capped under 300 tokens by a regression
test. The date is accompanied by an instruction that the model's training
data predates it, which converts "today is X" into verify-instead-of-guess
behavior. Instruction files (`AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`)
are surfaced when present.

---

## Benchmarks

All figures in this document were measured on a single reference
configuration: Windows 11 x64, RTX 4070 Ti SUPER (16 GB VRAM),
`qwen3.8:27b` (27.3B, Q4_K_M) at 32k context, Ollama 0.32.5,
Python 3.11.9. Results vary with hardware, model, and quantization;
`scripts/benchmark.py` reproduces them on any setup.

```
qwen3.8:27b  27.3B Q4_K_M, 32k context
  10.9 GB of 17.6 GB in VRAM (62% GPU)

  generation       19.5 tok/s  (mean of 3 runs)
  prompt ingest     721 tok/s  (9,024 uncached tokens; 2,400-3,900 tok/s warm)
```

Reasoning cost by prompt class:

| Prompt class | Thinking off | Thinking on | Overhead |
|---|---|---|---|
| Trivial | 10 tok / 2.6 s | 39 tok / 3.8 s | 1.4× |
| Mechanical | 16 tok / 3.4 s | 81 tok / 7.2 s | 2.2× |
| Hard | 400 tok / 38.3 s | 400 tok / 39.0 s | — (both capped) |

The mechanical row is the shape of most agent turns — select a tool, read a
result, make an edit — and motivates the default thinking policy.

Context window cost: doubling the window from 32k to 64k added 3.8 GB of KV
cache and reduced GPU residency from 79% to 63% on the reference
configuration. `OLLAMA_FLASH_ATTENTION=1` with `OLLAMA_KV_CACHE_TYPE=q8_0`
approximately halves KV cache memory.

---

## MCP server

`src/server.py` exposes model management as MCP tools over stdio:

| Tool | Description |
|---|---|
| `ollama_status` | Reachability, host, and loaded models. |
| `list_models` | Installed models with size, parameters, quantization. |
| `model_info` | Capabilities of one model: maximum context, `tools` / `thinking` / `vision` support. |
| `recommend_agent_model` | The installed model best suited to an agent loop, with rationale. |
| `loaded_models` | Resident models with VRAM/RAM split; warns when a model is partially on CPU. |
| `load_model` | Warm a model, optionally at a specific context length. |
| `unload_model` | Evict a model and free VRAM immediately. |
| `generate` | Run a prompt locally; returns the response with timing counters. |
| `benchmark` | Measure throughput with reasoning on and off. |

The server is written against **mcp 2.0.0**, which is a breaking change from
1.x: `FastMCP` is replaced by `mcp.server.MCPServer`, `CallToolResult.isError`
is now `is_error`, and the client supports in-memory transport for testing.
`scripts/smoke_test.py` exercises the server end-to-end over both in-memory
and stdio transports.

---

## Repository layout

```
LICENSE               MIT
assets/               logo and branding
src/
  agent.py            agent loop, REPL, thinking policy
  ollama_client.py    async Ollama client: loopback guard, timing, think toggle
  models.py           model listing, inspection, load/unload, benchmarking
  server.py           MCP stdio server
  tools.py            agent tools: read / write / edit / find / search / shell
  session.py          persistence, token accounting, compaction
  websearch.py        web lookup (the only module with internet access)
  environment.py      environment grounding
  repl_input.py       terminal input: bracketed paste, history, key bindings
  images.py           image attachment: clipboard, drag-and-drop, paths
  paths.py            read/write path separation for installed builds
  transcript.py       conversation rendering
  ui.py               terminal presentation
scripts/
  verify_offline.py   verifies the containment guarantees
  smoke_test.py       MCP server end-to-end test (requires Ollama)
  benchmark.py        throughput and reasoning-cost measurement
  test_*.py           regression suites (no network required)
  setup_searxng.ps1   SearXNG container setup, Windows (requires Docker)
  setup_searxng.sh    SearXNG container setup, Linux
  install_launcher.sh PATH launcher installation, Linux
  install_launcher.ps1  PATH launcher installation
  build_installer.ps1 application and installer build
  show_session.py     saved-session viewer
.github/workflows/
  ci.yml              tests on Windows, Linux, macOS; tarball build artifacts
installer/
  OffTheWire.spec        PyInstaller configuration
  OffTheWire.iss         Inno Setup configuration
```

---

## Building the installer

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
winget install --id JRSoftware.InnoSetup
.\scripts\build_installer.ps1        # -> build\OffTheWire-Setup-<version>.exe
```

The build runs in two stages — PyInstaller produces a one-directory
application (~66 MB, ~0.4 s cold start), and Inno Setup wraps it into a
~25 MB installer — with a smoke test between them, so a non-functional
bundle is never packaged. Neither tool is a runtime dependency.

The Linux and macOS tarballs are produced by CI (`.github/workflows/ci.yml`)
on every version tag, from the same `.spec` file — PyInstaller cannot
cross-compile, so each platform's binary is built on that platform's runner. To build locally on
Linux:

```bash
.venv/bin/pip install pyinstaller
cd installer && ../.venv/bin/python -m PyInstaller OffTheWire.spec
tar -czf OffTheWire-linux-x86_64.tar.gz -C dist OffTheWire
```

Notes:

- The build fails with a clear message if the checkout path is deep enough
  to push bundled file paths past the Windows 260-character `MAX_PATH`
  limit.
- Installed builds write data to `%LOCALAPPDATA%\OffTheWire` (override with
  the `OFFTHEWIRE_HOME` environment variable); the install directory is
  treated as read-only. Source checkouts keep data beside the repository.
- The uninstaller removes its `PATH` entry exactly, asks before deleting
  saved conversations, and offers to remove the SearXNG container (which
  otherwise restarts with Docker indefinitely). Silent uninstalls preserve
  user data and the container rather than accepting destructive defaults.

---

## Testing

```powershell
.venv\Scripts\python.exe scripts\test_accounting.py    # context budget and compaction
.venv\Scripts\python.exe scripts\test_images.py        # image handling and persistence
.venv\Scripts\python.exe scripts\test_repl_input.py    # input handling via a real PromptSession
.venv\Scripts\python.exe scripts\test_websearch.py     # source ranking, refusal detection, SSRF guard
.venv\Scripts\python.exe scripts\test_environment.py   # environment probing
.venv\Scripts\python.exe scripts\test_interrupt.py     # interrupt handling
.venv\Scripts\python.exe scripts\test_ask.py           # clarifying-question tool
.venv\Scripts\python.exe scripts\test_tools.py         # workspace confinement, command kill-tree
.venv\Scripts\python.exe scripts\test_models.py        # VRAM-aware model selection
.venv\Scripts\python.exe scripts\test_docs.py          # /help-vs-README and version drift
.venv\Scripts\python.exe scripts\verify_offline.py     # containment verification
```

All suites run without network access or a model. `scripts/smoke_test.py`
additionally exercises the MCP server against a live Ollama instance.

---

## Roadmap

- Offline documentation layer (self-hosted DevDocs) so library reference
  queries require no egress at all
- MCP client support in the agent, for consuming `server.py` and
  third-party MCP servers
- Parallel tool execution within a step
- Additional backends (LM Studio, llama.cpp) via their OpenAI-compatible
  APIs

---

## License

MIT — see [LICENSE](LICENSE).

Third-party components (Ollama, SearXNG, and the Python dependencies listed
in `requirements.txt`) are licensed separately by their respective projects.
