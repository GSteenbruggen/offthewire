"""Model selection, inspection and benchmarking.

Sits between the raw client and whatever is driving it (MCP server today,
agent tomorrow). Everything here returns plain dicts/strings so it can be
handed straight back through an MCP tool result.
"""

from __future__ import annotations

from typing import Any

from ollama_client import GenResult, OllamaClient

# Fixed VRAM cost of running a model beyond its weights, against the totals
# nvidia-smi reports: compute buffers (~1.2 GB), recurrent state on hybrid
# architectures (~0.75 GB), the desktop compositor on the display card
# (~1 GB), and what the driver holds back from the total (~1.5 GB). The KV
# cache is deliberately *not* in here -- it scales with context and is
# estimated per model by kv_bytes_per_token().
VRAM_FIXED_OVERHEAD_BYTES = 4_500_000_000

# Ollama's own layer fitter keeps a margin on every device it places on --
# observed as several GB left idle on the second card at the point a model
# starts spilling. Charged per GPU so a pooled estimate lands where Ollama
# actually does. Calibrated on the reference machine (16 + 12 GB cards,
# qwen3.8:27b at 17.74 GB on disk): the measured full-GPU ceiling is 49,152
# context and 57,344 spills; with this margin the estimate agrees, and the
# verified search in agent.resolve_auto_context starts one load from done.
VRAM_PER_GPU_MARGIN_BYTES = 2_000_000_000

# Kept for callers that budget without a model card (no per-token estimate).
VRAM_HEADROOM_BYTES = VRAM_FIXED_OVERHEAD_BYTES

# Context sizes tried, largest first, when the window is chosen automatically.
# Powers-of-two-ish steps: the KV cache is linear in context, so the ladder
# is coarse enough that each rung is a visible change and fine enough that
# a machine is never stuck far below what it could hold.
CONTEXT_LADDER = (131072, 98304, 65536, 49152, 32768)

# The floor of the ladder: the window any machine gets when nothing larger
# fits or no measurement is possible. The same value agent.py uses as its
# --context default.
CONTEXT_FLOOR = 32768


def human_bytes(n: int | float | None) -> str:
    if not n:
        return "0 B"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _details(m: dict[str, Any]) -> dict[str, Any]:
    return m.get("details") or {}


def summarize_model(m: dict[str, Any]) -> dict[str, Any]:
    d = _details(m)
    return {
        "name": m.get("model") or m.get("name", ""),
        "size": human_bytes(m.get("size")),
        "size_bytes": m.get("size", 0),
        "family": d.get("family", ""),
        "parameters": d.get("parameter_size", ""),
        "quantization": d.get("quantization_level", ""),
        "modified": (m.get("modified_at") or "")[:10],
    }


async def list_models(client: OllamaClient) -> list[dict[str, Any]]:
    models = await client.list_models()
    out = [summarize_model(m) for m in models]
    out.sort(key=lambda x: x["name"])
    return out


async def model_capabilities(client: OllamaClient, model: str) -> dict[str, Any]:
    """What the model can do and how big a window it really has.

    ``supports_tools`` is the gate for agent use -- a model without it cannot
    drive a tool loop no matter how good it is at prose.

    A client that knows its own answer (the OpenAI-compat backends, whose
    API has no /api/show) supplies it directly; the Ollama path below reads
    the model card.
    """
    if hasattr(client, "capabilities"):
        return await client.capabilities(model)
    info = await client.show(model)
    caps = info.get("capabilities") or []
    model_info = info.get("model_info") or {}
    d = info.get("details") or {}

    ctx = None
    for key, val in model_info.items():
        if key.endswith(".context_length"):
            ctx = val
            break

    return {
        "name": model,
        "architecture": d.get("family", ""),
        "parameters": d.get("parameter_size", ""),
        "quantization": d.get("quantization_level", ""),
        "max_context": ctx,
        "kv_bytes_per_token": kv_bytes_per_token(model_info),
        "capabilities": caps,
        "supports_tools": "tools" in caps,
        "supports_thinking": "thinking" in caps,
        "supports_vision": "vision" in caps,
    }


def kv_bytes_per_token(model_info: dict[str, Any], cache_bytes: int = 2) -> int | None:
    """KV cache cost per context token, from the model card; None if unknown.

    The keys are GGUF metadata as Ollama's /api/show and a raw GGUF header
    both present them (``<arch>.block_count`` etc.), so this reads either.

    Per attention layer, a token stores one key and one value vector for
    each KV head: ``heads_kv * (key_length + value_length) * cache_bytes``.
    ``cache_bytes`` is 2 for the f16 default and 1 for q8_0.

    Hybrid architectures (qwen3.5/3.8, Jamba-style) interleave recurrent
    layers that keep no KV cache with full-attention layers that do;
    ``full_attention_interval`` says one in every N is attention. Ignoring
    it overstates the cache four-fold on the reference model -- measured
    2.0 GB at 32k for 16 attention layers, where all 65 would predict 8 GB.
    """
    arch = model_info.get("general.architecture")
    if not arch:
        return None
    layers = model_info.get(f"{arch}.block_count")
    heads_kv = model_info.get(f"{arch}.attention.head_count_kv")
    if not layers or not heads_kv:
        return None
    if isinstance(heads_kv, list):  # per-layer counts on a few architectures
        heads_kv = max(heads_kv) if heads_kv else 0
    key_len = model_info.get(f"{arch}.attention.key_length")
    val_len = model_info.get(f"{arch}.attention.value_length")
    if not key_len or not val_len:
        embed = model_info.get(f"{arch}.embedding_length")
        heads = model_info.get(f"{arch}.attention.head_count")
        if not embed or not heads:
            return None
        key_len = val_len = embed // heads
    interval = model_info.get(f"{arch}.full_attention_interval") or 1
    attention_layers = max(1, int(layers) // int(interval))
    return int(attention_layers * heads_kv * (key_len + val_len) * cache_bytes)


def vram_needed(
    size_bytes: int, kv_per_token: int | None, context: int, n_gpus: int = 1
) -> int:
    """Bytes a model needs resident at a given window: weights, the fixed
    overhead, a margin per card it will be spread over, and the KV cache
    when the per-token cost is known."""
    return (
        size_bytes
        + VRAM_FIXED_OVERHEAD_BYTES
        + VRAM_PER_GPU_MARGIN_BYTES * max(1, n_gpus)
        + (kv_per_token or 0) * context
    )


def fit_context(
    size_bytes: int,
    kv_per_token: int | None,
    vram: int | None,
    ladder: tuple[int, ...] = CONTEXT_LADDER,
    ceiling: int | None = None,
    n_gpus: int = 1,
) -> int:
    """The largest ladder rung whose KV cache fits alongside the weights.

    Returns the floor (the smallest rung) when nothing measures: no VRAM
    reading, or a model card without the fields the estimate needs. Both
    mean "no basis for going bigger", not "go bigger anyway".

    ``ceiling`` is the model's own maximum; rungs above it are skipped.
    """
    rungs = [r for r in sorted(ladder, reverse=True) if ceiling is None or r <= ceiling]
    if not rungs:
        return min(ladder)
    if not vram or not kv_per_token or not size_bytes:
        return rungs[-1]
    for rung in rungs:
        if vram_needed(size_bytes, kv_per_token, rung, n_gpus) <= vram:
            return rung
    return rungs[-1]


async def loaded_models(client: OllamaClient) -> list[dict[str, Any]]:
    """Resident models plus the GPU/CPU split.

    ``size_vram`` smaller than ``size`` means part of the model is executing on
    CPU, which is usually the single biggest cause of slow local inference.
    """
    out = []
    for m in await client.ps():
        total = m.get("size", 0) or 0
        vram = m.get("size_vram", 0) or 0
        pct_gpu = round(100 * vram / total) if total else 0
        entry = {
            "name": m.get("model") or m.get("name", ""),
            "total": human_bytes(total),
            "in_vram": human_bytes(vram),
            "gpu_percent": pct_gpu,
            "context": m.get("context_length"),
            "expires_at": m.get("expires_at", ""),
        }
        if pct_gpu < 100:
            entry["warning"] = (
                f"{100 - pct_gpu}% running on CPU -- expect materially slower "
                f"generation. Reduce context, set OLLAMA_KV_CACHE_TYPE=q8_0 "
                f"(halves the KV cache; needs OLLAMA_FLASH_ATTENTION=1), or "
                f"use a smaller quant."
            )
        out.append(entry)
    return out


async def benchmark(
    client: OllamaClient,
    model: str,
    *,
    prompt_tokens: int = 2000,
    gen_tokens: int = 100,
    context_length: int | None = None,
) -> dict[str, Any]:
    """Measure both speeds that matter, with and without thinking.

    A short prompt cannot measure prompt throughput -- the timer is swamped by
    fixed overhead, which is why a 12-token 'hello world' reports a nonsense
    prompt rate. We pad to ``prompt_tokens`` (~4 chars/token) to get a figure
    that actually predicts agent-loop latency.
    """
    filler = ("The quick brown fox jumps over the lazy dog. " * ((prompt_tokens // 9) + 1))[
        : prompt_tokens * 4
    ]
    prompt = (
        f"Here is some reference text:\n\n{filler}\n\n"
        "Ignore the text above. Reply with the single word: ready."
    )

    results: dict[str, Any] = {"model": model, "prompt_tokens_requested": prompt_tokens}

    for label, think in (("thinking_off", False), ("thinking_on", True)):
        try:
            r: GenResult = await client.generate(
                model,
                prompt,
                think=think,
                max_tokens=gen_tokens,
                context_length=context_length,
            )
            results[label] = {
                "gen_tps": round(r.gen_tps, 2),
                "prompt_tps": round(r.prompt_tps, 1),
                "prompt_tokens": r.prompt_tokens,
                "gen_tokens": r.gen_tokens,
                "thinking_tokens_est": len(r.thinking.split()) if r.thinking else 0,
                "total_s": round(r.total_s, 2),
            }
        except Exception as e:  # a model may simply not support thinking
            results[label] = {"error": f"{type(e).__name__}: {e}"}

    off, on = results.get("thinking_off", {}), results.get("thinking_on", {})
    if "total_s" in off and "total_s" in on and off["total_s"] > 0:
        speedup = on["total_s"] / off["total_s"]
        results["verdict"] = (
            f"Disabling thinking is {speedup:.1f}x faster for this prompt "
            f"({on['total_s']:.1f}s -> {off['total_s']:.1f}s). "
            + (
                "Leave thinking off for mechanical agent turns."
                if speedup > 1.5
                else "Thinking overhead is modest here."
            )
        )
    return results


async def recommend_agent_model(
    client: OllamaClient, context: int = CONTEXT_FLOOR
) -> dict[str, Any]:
    """Pick the best installed model to drive a tool loop.

    Tool support is mandatory; among those, prefer the largest that still
    plausibly fits in VRAM *at the requested context*, since spilling to CPU
    costs more than the extra parameters buy.
    """
    installed = await client.list_models()
    candidates = []
    for m in installed:
        name = m.get("model") or m.get("name", "")
        try:
            caps = await model_capabilities(client, name)
        except Exception:
            continue
        if caps["supports_tools"]:
            caps["size_bytes"] = m.get("size", 0)
            caps["size"] = human_bytes(m.get("size"))
            candidates.append(caps)

    if not candidates:
        return {
            "recommended": None,
            "reason": (
                "No installed model reports the 'tools' capability, so none can "
                "drive an agent loop. Try: ollama pull qwen3-coder:30b"
            ),
            "candidates": [],
        }

    per_gpu = vram_per_gpu()
    best, why = choose_candidate(
        candidates, sum(per_gpu) if per_gpu else None, context, n_gpus=len(per_gpu),
    )
    return {
        "recommended": best["name"],
        "reason": (
            f"{best['parameters']} {best['quantization']}, "
            f"max context {best['max_context']}, "
            f"thinking={'yes' if best['supports_thinking'] else 'no'}{why}"
        ),
        "candidates": [
            {k: c[k] for k in ("name", "size", "parameters", "max_context")}
            for c in sorted(candidates, key=lambda c: c["size_bytes"], reverse=True)
        ],
    }


def choose_candidate(
    candidates: list[dict[str, Any]],
    vram: int | None,
    context: int = CONTEXT_FLOOR,
    n_gpus: int = 1,
) -> tuple[dict[str, Any], str]:
    """Pick the model to recommend, VRAM permitting.

    The docstring above always promised "the largest that still plausibly fits
    in VRAM"; until this function existed the code simply took the largest,
    which on a small card recommends the one model guaranteed to run mostly on
    CPU. Fitting is weights plus the fixed overhead plus the KV cache at
    ``context`` -- a candidate may carry ``kv_bytes_per_token`` from its model
    card; without it only the fixed overhead is charged, which understates
    the need at large windows but never rejects a model that would fit.

    When nothing fits, the *smallest* spills least, so it wins -- the exact
    inverse of the no-information order. With no VRAM reading at all (non-NVIDIA
    GPU, no nvidia-smi), the old largest-first behaviour stands, because
    guessing a number would be worse than admitting we do not have one.
    """
    by_size = sorted(candidates, key=lambda c: c["size_bytes"], reverse=True)
    if not vram:
        return by_size[0], ""

    fitting = [
        c for c in by_size
        if vram_needed(c["size_bytes"], c.get("kv_bytes_per_token"), context, n_gpus) <= vram
    ]
    if fitting:
        best = fitting[0]
        return best, f", fits in {human_bytes(vram)} VRAM at {context:,} context"
    smallest = by_size[-1]
    return smallest, (
        f"; NOTE: no installed model fits this GPU's {human_bytes(vram)} at "
        f"{context:,} context — picked the smallest to minimize CPU spill"
    )


def vram_per_gpu() -> list[int]:
    """Total VRAM of each NVIDIA GPU in device order; empty when unreadable.

    nvidia-smi is the only probe: AMD and Apple report through interfaces this
    project has no test hardware for, and a wrong VRAM number silently skews
    the recommendation, which is worse than falling back to size order.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return parse_vram_per_gpu(proc.stdout)


def total_vram_bytes() -> int | None:
    """Combined VRAM across all NVIDIA GPUs, or None when it cannot be read.

    Summed, not the largest card: Ollama splits a model's layers across every
    CUDA device it finds, so for fitting purposes two cards' memory pools --
    a 16 GB + 12 GB machine really can hold a 24 GB model entirely on GPU.
    """
    readings = vram_per_gpu()
    return sum(readings) if readings else None


def parse_vram_readings(text: str) -> int | None:
    """Sum of per-GPU MiB lines from nvidia-smi, in bytes. Pure, for tests."""
    readings = parse_vram_per_gpu(text)
    return sum(readings) if readings else None


def parse_vram_per_gpu(text: str) -> list[int]:
    """One byte count per GPU line from nvidia-smi, in device order. Pure."""
    return [
        int(line.strip()) * 1024 * 1024
        for line in text.splitlines()
        if line.strip().isdigit()
    ]
