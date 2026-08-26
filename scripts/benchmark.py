"""Measure what this machine actually does, so the README stops guessing.

Re-run this after changing model, quantization, context size or KV cache
settings -- every number the design leans on comes from here.

    .venv\\Scripts\\python.exe scripts\\benchmark.py
    .venv\\Scripts\\python.exe scripts\\benchmark.py --model qwen3.8:27b --context 32768

Warms the model first: a cold first call includes loading ~17 GB from disk and
will otherwise be misattributed to generation speed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.getLogger("httpx").setLevel(logging.WARNING)

import models as M  # noqa: E402
import ui  # noqa: E402
from ollama_client import OllamaClient  # noqa: E402

# Three shapes of work, because the cost of reasoning depends heavily on how
# hard the question is -- averaging them into one number hides the thing you
# actually want to know.
PROMPTS = [
    ("trivial", "Say hello.", 200),
    (
        "mechanical",
        "A Python function `add(a, b)` returns `a - b`. State the one-line fix.",
        250,
    ),
    (
        "reasoning",
        "A test passes locally but fails in CI with 'ModuleNotFoundError: no "
        "module named app'. List the three most likely causes, most likely first, "
        "and how to tell them apart.",
        400,
    ),
]


async def warm(client: OllamaClient, model: str, ctx: int) -> None:
    ui.note("warming the model (loading from disk is not generation speed)…")
    t = time.time()
    await client.generate(model, "hi", think=False, max_tokens=5, context_length=ctx)
    ui.note(f"warm after {time.time() - t:.1f}s")


async def residency(client: OllamaClient, model: str) -> None:
    for entry in await M.loaded_models(client):
        if not entry["name"].startswith(model.split(":")[0]):
            continue
        pct = entry.get("gpu_percent", 0)
        line = (
            f"{entry['in_vram']} of {entry['total']} in VRAM "
            f"({pct}% GPU), context {entry.get('context')}"
        )
        (ui.warn if pct < 100 else ui.note)(
            line + (f" — {100 - pct}% on CPU" if pct < 100 else "")
        )


async def thinking_table(client: OllamaClient, model: str, ctx: int) -> None:
    rows = []
    for label, prompt, cap in PROMPTS:
        result = {}
        for mode, think in (("off", False), ("on", True)):
            t = time.time()
            r = await client.generate(
                model, prompt, think=think, max_tokens=cap, context_length=ctx
            )
            result[mode] = {
                "tok": r.gen_tokens,
                "s": time.time() - t,
                "tps": r.gen_tps,
                "reason": len(r.thinking),
            }
        off, on = result["off"], result["on"]
        ratio = on["s"] / off["s"] if off["s"] else 0
        rows.append([
            label,
            f"{off['tok']}",
            f"{off['s']:.1f}s",
            f"{on['tok']}",
            f"{on['s']:.1f}s",
            f"{on['reason']}",
            f"{ratio:.1f}x",
        ])
        ui.note(
            f"  {label:<11} off {off['tok']:>4} tok / {off['s']:>5.1f}s   "
            f"on {on['tok']:>4} tok / {on['s']:>5.1f}s   ratio {ratio:.1f}x"
        )

    ui.console.print()
    ui.table(
        "thinking cost",
        [("prompt", "left"), ("off tok", "right"), ("off time", "right"),
         ("on tok", "right"), ("on time", "right"), ("reasoning ch", "right"),
         ("slower by", "right")],
        rows,
    )


async def speeds(client: OllamaClient, model: str, ctx: int) -> None:
    # Generation: several short runs, averaged.
    total_tok = total_s = 0.0
    for _ in range(3):
        r = await client.generate(
            model, "Count from 1 to 40 as digits separated by spaces.",
            think=False, max_tokens=120, context_length=ctx,
        )
        total_tok += r.gen_tokens
        total_s += r.gen_ns / 1e9
    gen_tps = total_tok / total_s if total_s else 0

    # Prompt ingestion: needs a large prompt. A short one measures fixed
    # overhead, which is why a 12-token 'hello world' reports nonsense.
    filler = "The quick brown fox jumps over the lazy dog. " * 900
    r = await client.generate(
        model, f"{filler}\n\nIgnore the above. Reply with one word: ready.",
        think=False, max_tokens=10, context_length=ctx,
    )

    ui.console.print()
    ui.key_values([
        ("generation", f"{gen_tps:.1f} tok/s   (mean of 3 runs)"),
        ("prompt ingest", f"{r.prompt_tps:,.0f} tok/s   ({r.prompt_tokens:,} tokens)"),
        ("first token", f"~{r.load_ns / 1e9:.2f}s load"),
    ])
    ui.console.print()
    ui.note(
        "Prompt ingestion is the number that decides whether an agent loop is "
        "bearable — every turn re-reads the conversation."
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the local model.")
    ap.add_argument("--model", help="Defaults to the recommended installed model.")
    ap.add_argument("--context", type=int, default=32768)
    args = ap.parse_args()

    client = OllamaClient()
    if not await client.ping():
        ui.error(f"Ollama unreachable at {client.host}")
        return 2

    model = args.model
    if not model:
        rec = await M.recommend_agent_model(client)
        model = rec.get("recommended")
        if not model:
            ui.error(rec.get("reason", "no tool-capable model installed"))
            return 2

    caps = await M.model_capabilities(client, model)
    max_ctx = caps["max_context"]
    ui.console.print()
    ui.console.print(f"[head]{model}[/head]  [meta]{caps['parameters']} "
                     f"{caps['quantization']}, max ctx "
                     f"{f'{max_ctx:,}' if max_ctx else '?'}[/meta]")
    ui.note(f"benchmarking at context {args.context:,}")
    ui.console.print()

    await warm(client, model, args.context)
    await residency(client, model)
    ui.console.print()
    await speeds(client, model, args.context)
    ui.console.print()
    await thinking_table(client, model, args.context)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
