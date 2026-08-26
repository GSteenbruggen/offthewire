"""Model selection, inspection and benchmarking.

Sits between the raw client and whatever is driving it (MCP server today,
agent tomorrow). Everything here returns plain dicts/strings so it can be
handed straight back through an MCP tool result.
"""

from __future__ import annotations

from typing import Any

from ollama_client import GenResult, OllamaClient

# Rough VRAM headroom to leave for the KV cache and the desktop compositor.
# Below this margin a model will page into system RAM and generation speed
# falls off a cliff.
VRAM_HEADROOM_BYTES = 1_500_000_000


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
    """
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
        "capabilities": caps,
        "supports_tools": "tools" in caps,
        "supports_thinking": "thinking" in caps,
        "supports_vision": "vision" in caps,
    }


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
                f"generation. Use a smaller quant or reduce context to fit."
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


async def recommend_agent_model(client: OllamaClient) -> dict[str, Any]:
    """Pick the best installed model to drive a tool loop.

    Tool support is mandatory; among those, prefer the largest that still
    plausibly fits in VRAM, since spilling to CPU costs more than the extra
    parameters buy.
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

    candidates.sort(key=lambda c: c["size_bytes"], reverse=True)
    best = candidates[0]
    return {
        "recommended": best["name"],
        "reason": (
            f"{best['parameters']} {best['quantization']}, "
            f"max context {best['max_context']}, "
            f"thinking={'yes' if best['supports_thinking'] else 'no'}"
        ),
        "candidates": [
            {k: c[k] for k in ("name", "size", "parameters", "max_context")}
            for c in candidates
        ],
    }
