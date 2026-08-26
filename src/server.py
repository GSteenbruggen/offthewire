"""ollama-mcp -- expose local Ollama models to any MCP client.

Runs over stdio. Register it with any MCP-capable client or anything else that
speaks MCP, and that client can then hand work to models running on this
machine instead of a cloud API.

Everything is local: this process talks to http://localhost:11434 and nowhere
else.

    python src/server.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# src/ modules import each other flatly, so make sure this directory is
# importable no matter what cwd the MCP client launches us from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server import MCPServer  # noqa: E402

import models as M  # noqa: E402
from ollama_client import NonLocalHostError, OllamaClient  # noqa: E402

# NOTE: mcp 2.x replaced the 1.x `mcp.server.fastmcp.FastMCP` entry point with
# `MCPServer`. The decorator and run() surface are near-identical, but the old
# import path no longer exists at all.
server = MCPServer(
    "ollama-mcp",
    version="0.1.0",
    instructions=(
        "Runs prompts on Ollama models hosted on this machine. Use for work that "
        "should stay on-device, or to offload bulk/repetitive generation from the "
        "cloud model. Call ollama_status first if anything misbehaves."
    ),
)
# A non-local OLLAMA_HOST should read as a configuration error, not a raw
# import-time traceback in the MCP client's log.
try:
    client = OllamaClient()
except NonLocalHostError as e:
    print(f"ollama-mcp: {e}", file=sys.stderr)
    sys.exit(2)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


async def _require_ollama() -> str | None:
    if await client.ping():
        return None
    return (
        f"Cannot reach Ollama at {client.host}. Is it running? "
        f"Start it with 'ollama serve' or launch the Ollama desktop app."
    )


# --------------------------------------------------------------- inspection


@server.tool()
async def list_models() -> str:
    """List every Ollama model installed on this machine, with size, parameter
    count and quantization. Use this first to see what is available locally."""
    if err := await _require_ollama():
        return err
    return _json(await M.list_models(client))


@server.tool()
async def model_info(model: str) -> str:
    """Get the capabilities of one model: max context length, and whether it
    supports tool calling, thinking, and vision.

    Check `supports_tools` before trying to use a model to drive an agent loop.

    Args:
        model: Model name as shown by list_models, e.g. 'qwen3.8:27b'.
    """
    if err := await _require_ollama():
        return err
    return _json(await M.model_capabilities(client, model))


@server.tool()
async def loaded_models() -> str:
    """Show which models are currently resident in memory, how much of each is
    in VRAM versus system RAM, and the context window they were loaded with.

    A `gpu_percent` below 100 means part of the model is running on CPU, which
    is the usual explanation for slow generation."""
    if err := await _require_ollama():
        return err
    loaded = await M.loaded_models(client)
    if not loaded:
        return "No models are currently loaded. Use load_model to warm one."
    return _json(loaded)


@server.tool()
async def recommend_agent_model() -> str:
    """Recommend which installed model is best suited to driving an agent /
    tool-calling loop, and explain why. Only models advertising the 'tools'
    capability are eligible."""
    if err := await _require_ollama():
        return err
    return _json(await M.recommend_agent_model(client))


# ---------------------------------------------------------------- lifecycle


@server.tool()
async def load_model(
    model: str, context_length: int | None = None, keep_alive: str = "30m"
) -> str:
    """Warm a model into memory so the next generate call has no load delay.

    Ollama loads models at a 4096-token context by default regardless of what
    the model supports, and silently truncates anything longer. Pass
    context_length explicitly for long-context work.

    Args:
        model: Model to load.
        context_length: Context window in tokens, e.g. 32768. Larger windows
            consume more VRAM and may push the model onto the CPU.
        keep_alive: How long to keep it resident, e.g. '30m', '1h'.
    """
    if err := await _require_ollama():
        return err
    await client.load(model, context_length=context_length, keep_alive=keep_alive)
    return _json(
        {
            "loaded": model,
            "context_length": context_length or "server default (likely 4096)",
            "keep_alive": keep_alive,
            "status": await M.loaded_models(client),
        }
    )


@server.tool()
async def unload_model(model: str) -> str:
    """Immediately evict a model from memory and free its VRAM. Useful before
    loading a larger model on a card that cannot hold both.

    Args:
        model: Model to unload.
    """
    if err := await _require_ollama():
        return err
    await client.unload(model)
    return f"Unloaded {model}; VRAM released."


# --------------------------------------------------------------- generation


@server.tool()
async def generate(
    prompt: str,
    model: str,
    system: str | None = None,
    think: bool | str = False,
    max_tokens: int = 1024,
    temperature: float | None = None,
    context_length: int | None = None,
) -> str:
    """Run a prompt on a local Ollama model and return its response plus timing.

    Use this to delegate work to this machine's GPU rather than a cloud model:
    bulk summarization, classification, drafting, or anything where the data
    should not leave the device.

    Thinking is OFF by default and that is usually right. Reasoning models can
    spend hundreds of tokens deliberating before answering, which on a locally
    hosted model is often an order of magnitude more wall-clock time for no
    benefit on straightforward work. Turn it on for genuinely hard reasoning --
    and prefer an effort level over `true`: a bare `true` inherits the model
    template's default, which on qwen3.8 is its maximum and overthinks badly.

    Generation is slow relative to cloud APIs -- keep max_tokens modest, or the
    calling client may time out waiting.

    Args:
        prompt: The user prompt.
        model: Which local model to run it on.
        system: Optional system prompt.
        think: Enable the model's reasoning block. Costs significant time.
            Accepts false, true, or an effort level: "low" | "medium" |
            "high" | "max". Use "low" unless the problem is genuinely hard.
        max_tokens: Cap on generated tokens.
        temperature: Sampling temperature; omit for the model's default.
        context_length: Context window for this request.
    """
    if err := await _require_ollama():
        return err
    r = await client.generate(
        model,
        prompt,
        system=system,
        think=think,
        max_tokens=max_tokens,
        temperature=temperature,
        context_length=context_length,
    )
    parts = [r.content.strip()]
    if r.thinking:
        parts.append(f"\n--- reasoning ({len(r.thinking)} chars, hidden by default) ---")
    parts.append(f"\n[{r.stats_line()}]")
    if r.done_reason == "length":
        parts.append(f"[truncated at max_tokens={max_tokens}]")
    return "\n".join(parts)


@server.tool()
async def benchmark(
    model: str, prompt_tokens: int = 2000, gen_tokens: int = 100
) -> str:
    """Measure this machine's real throughput for a model, with thinking on and
    off, and report the difference.

    Reports both generation tok/s and prompt-ingestion tok/s. The prompt rate
    is the one that predicts whether an agent loop is usable, because every
    turn re-processes the whole conversation. Short test prompts cannot measure
    it meaningfully, so this pads the prompt to prompt_tokens.

    Takes a minute or two on a large model.

    Args:
        model: Model to benchmark.
        prompt_tokens: Approximate prompt size to test ingestion speed with.
        gen_tokens: How many tokens to generate per run.
    """
    if err := await _require_ollama():
        return err
    return _json(
        await M.benchmark(
            client, model, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens
        )
    )


@server.tool()
async def ollama_status() -> str:
    """Check whether Ollama is reachable and report the host in use, plus what
    is currently loaded. Start here when something is not working."""
    if not await client.ping():
        return _json(
            {
                "reachable": False,
                "host": client.host,
                "fix": "Run 'ollama serve' or start the Ollama desktop app.",
            }
        )
    return _json(
        {
            "reachable": True,
            "host": client.host,
            "installed": len(await client.list_models()),
            "loaded": await M.loaded_models(client),
        }
    )


if __name__ == "__main__":
    server.run(transport="stdio")
