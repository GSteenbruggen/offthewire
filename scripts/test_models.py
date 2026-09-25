"""Tests for model selection logic. No network, no model.

The one decision that matters here: recommend_agent_model must not send a
small GPU the largest model. The VRAM probe itself needs hardware and is
exercised only for its failure modes; the choice function is pure and gets
the real coverage.

    .venv\\Scripts\\python.exe scripts\\test_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models import (  # noqa: E402
    CONTEXT_LADDER, VRAM_FIXED_OVERHEAD_BYTES, VRAM_HEADROOM_BYTES,
    VRAM_PER_GPU_MARGIN_BYTES, choose_candidate, fit_context, kv_bytes_per_token,
    parse_vram_per_gpu, parse_vram_readings, total_vram_bytes, vram_needed,
)

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0

GB = 1024**3


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def cand(name: str, size_gb: float) -> dict:
    return {"name": name, "size_bytes": int(size_gb * GB)}


def test_choice() -> None:
    print("\n1. VRAM-aware candidate choice")

    big, mid, small = cand("big:27b", 17.5), cand("mid:7b", 4.5), cand("small:1b", 0.9)

    # No VRAM reading: the old order stands, largest first.
    best, why = choose_candidate([small, big, mid], None)
    check("no VRAM info -> largest (legacy order)", best["name"] == "big:27b")
    check("no VRAM info -> no claim about fitting", why == "", repr(why))

    # Plenty of VRAM: still the largest, now with the fit stated.
    best, why = choose_candidate([small, big, mid], 24 * GB)
    check("24GB -> largest fits and wins", best["name"] == "big:27b")
    check("fit is stated", "fits" in why, why)

    # The bug this exists to fix: a 16GB card must NOT get the 17.5GB model.
    best, why = choose_candidate([small, big, mid], 16 * GB)
    check("16GB -> largest *fitting* model, not largest", best["name"] == "mid:7b", best["name"])

    # Nothing fits: the smallest spills least.
    best, why = choose_candidate([big, cand("big2:30b", 19.0)], 8 * GB)
    check("nothing fits -> smallest", best["name"] == "big:27b")
    check("spill is admitted", "NOTE" in why, why)

    # Headroom is real: a model equal to VRAM minus half the headroom does not fit.
    tight = cand("tight", (16 * GB - VRAM_HEADROOM_BYTES // 2) / GB)
    best, why = choose_candidate([tight, small], 16 * GB)
    check("headroom is enforced", best["name"] == "small:1b", best["name"])


def test_multi_gpu_pooling() -> None:
    """Ollama splits layers across all CUDA devices, so capacity is the sum.

    The first version took the largest card, which would tell a 16+12 GB
    machine that a 24 GB model cannot fit -- understating exactly the setup
    a second GPU is bought to create.
    """
    print("\n2. Multi-GPU VRAM pooling")

    one = parse_vram_readings("16376\n")
    two = parse_vram_readings("16376\n12288\n")
    check("single GPU parses", one == 16376 * 1024 * 1024, str(one))
    check("two GPUs sum, not max", two == (16376 + 12288) * 1024 * 1024, str(two))
    check("garbage lines ignored",
          parse_vram_readings("N/A\n16376\n") == 16376 * 1024 * 1024)
    check("no readings -> None", parse_vram_readings("") is None)

    # The scenario the fix exists for: 17.5 GB model, 16 GB card + 12 GB card.
    big = {"name": "big:27b", "size_bytes": int(17.5 * GB)}
    small = {"name": "small:7b", "size_bytes": int(4.5 * GB)}
    best, why = choose_candidate([small, big], parse_vram_readings("16376\n12288\n"))
    check("pooled VRAM fits the big model", best["name"] == "big:27b", best["name"])


# The reference model's card, as Ollama's /api/show reports it: a hybrid
# architecture where one layer in four keeps a KV cache.
QWEN38_27B = {
    "general.architecture": "qwen35",
    "qwen35.block_count": 65,
    "qwen35.attention.head_count": 24,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
    "qwen35.embedding_length": 5120,
    "qwen35.full_attention_interval": 4,
    "qwen35.context_length": 262144,
}


def test_kv_estimate() -> None:
    """Measured on the reference machine: 2048 MiB of KV cache at 32k
    context for this model. The estimate must land on that, and must not
    predict the 8 GB a naive all-layers formula gives."""
    print("\n4. KV cache per token from the model card")

    per_tok = kv_bytes_per_token(QWEN38_27B)
    check("hybrid model: only attention layers counted",
          per_tok == 16 * 4 * 512 * 2, str(per_tok))
    check("32k window lands on the measured 2 GiB",
          per_tok * 32768 == 2048 * 1024 * 1024, f"{per_tok * 32768 / GB:.2f} GiB")
    check("q8_0 cache halves it",
          kv_bytes_per_token(QWEN38_27B, cache_bytes=1) == per_tok // 2)

    # The estimate must follow the user's cache setting, or the auto-context
    # search starts an f16-sized rung below the room q8_0 actually freed.
    import os

    from models import kv_cache_bytes

    saved = os.environ.get("OLLAMA_KV_CACHE_TYPE")
    try:
        os.environ.pop("OLLAMA_KV_CACHE_TYPE", None)
        check("no cache setting -> f16 bytes", kv_cache_bytes() == 2.0)
        os.environ["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        check("q8_0 setting -> 1 byte", kv_cache_bytes() == 1.0)
        os.environ["OLLAMA_KV_CACHE_TYPE"] = "q4_0"
        check("q4_0 setting -> half a byte", kv_cache_bytes() == 0.5)
        os.environ["OLLAMA_KV_CACHE_TYPE"] = "something-new"
        check("unknown setting -> conservative f16", kv_cache_bytes() == 2.0)
    finally:
        if saved is None:
            os.environ.pop("OLLAMA_KV_CACHE_TYPE", None)
        else:
            os.environ["OLLAMA_KV_CACHE_TYPE"] = saved

    dense = {**QWEN38_27B}
    del dense["qwen35.full_attention_interval"]
    check("dense model counts every layer",
          kv_bytes_per_token(dense) == 65 * 4 * 512 * 2)

    no_lengths = {
        k: v for k, v in QWEN38_27B.items()
        if not k.endswith((".key_length", ".value_length"))
    }
    check("key/value length fall back to embedding / heads",
          kv_bytes_per_token(no_lengths) == 16 * 4 * (5120 // 24 * 2) * 2)

    check("no architecture -> None", kv_bytes_per_token({}) is None)
    check("no kv heads -> None",
          kv_bytes_per_token({"general.architecture": "x", "x.block_count": 4}) is None)


def test_context_fit() -> None:
    """The ladder decision the auto window starts from. Numbers are the
    reference machine: 17 GB weights, 16 + 12 GB cards."""
    print("\n5. Context ladder fit")

    per_tok = kv_bytes_per_token(QWEN38_27B)
    weights = 17_741_872_154  # qwen3.8:27b, as /api/tags reports it
    two_cards = parse_vram_readings("16376\n12288\n")
    one_card = parse_vram_readings("16376\n")

    check("need grows linearly with context",
          vram_needed(weights, per_tok, 65536) - vram_needed(weights, per_tok, 32768)
          == per_tok * 32768)
    check("fixed overhead and one card's margin charged even at zero context",
          vram_needed(weights, per_tok, 0)
          == weights + VRAM_FIXED_OVERHEAD_BYTES + VRAM_PER_GPU_MARGIN_BYTES)
    check("a second card adds its margin",
          vram_needed(weights, per_tok, 0, n_gpus=2) - vram_needed(weights, per_tok, 0)
          == VRAM_PER_GPU_MARGIN_BYTES)

    # Measured on the reference machine: 49,152 is the largest ladder rung
    # that runs 100% on GPU; 57,344 already spills. The estimate must agree,
    # because it decides where the verified search starts.
    fit = fit_context(weights, per_tok, two_cards, n_gpus=2)
    check("two cards: 48k fits, 64k does not (matches measurement)", fit == 49152, f"{fit:,}")
    check("one card: floor (the model does not even fit the weights)",
          fit_context(weights, per_tok, one_card) == 32768)
    check("no VRAM reading -> floor", fit_context(weights, per_tok, None) == 32768)
    check("no KV estimate -> floor", fit_context(weights, None, two_cards) == 32768)
    check("ceiling caps the ladder",
          fit_context(int(1 * GB), per_tok, two_cards, ceiling=40000) == 32768)
    check("a small model on a big pool takes the top rung",
          fit_context(int(1 * GB), per_tok, two_cards, n_gpus=2) == max(CONTEXT_LADDER))
    check("ladder is descending and ends at the floor",
          CONTEXT_LADDER == tuple(sorted(CONTEXT_LADDER, reverse=True))
          and CONTEXT_LADDER[-1] == 32768)


def test_context_aware_choice() -> None:
    """The picker charges the KV cache at the requested window: a model
    that fits at 32k can be the wrong answer at 128k."""
    print("\n6. Context-aware candidate choice")

    per_tok = kv_bytes_per_token(QWEN38_27B)
    big = {"name": "big:27b", "size_bytes": 17_741_872_154, "kv_bytes_per_token": per_tok}
    small = {"name": "small:7b", "size_bytes": int(4.5 * GB), "kv_bytes_per_token": per_tok}
    pool = parse_vram_readings("16376\n12288\n")

    best, why = choose_candidate([small, big], pool, context=32768, n_gpus=2)
    check("27B fits the pool at 32k", best["name"] == "big:27b", best["name"])
    check("the window is stated", "32,768" in why, why)
    best, why = choose_candidate([small, big], pool, context=131072, n_gpus=2)
    check("at 128k the 27B no longer fits; the 7B wins", best["name"] == "small:7b", best["name"])
    best, _ = choose_candidate([small, big], parse_vram_readings("16376\n"), context=131072)
    check("nothing fits -> smallest, as before", best["name"] == "small:7b")

    per_gpu = parse_vram_per_gpu("16376\n12288\n")
    check("per-GPU readings keep device order",
          per_gpu == [16376 * 1024 * 1024, 12288 * 1024 * 1024])


def test_probe_failure_modes() -> None:
    """The probe must return None, never raise, on machines without nvidia-smi
    or with a broken one -- a wrong number would silently skew every
    recommendation."""
    print("\n3. VRAM probe failure modes")
    try:
        v = total_vram_bytes()
        check("probe never raises", True, f"reading: {v}")
        check("reading is None or plausible", v is None or 1 * GB < v < 1024 * GB, str(v))
    except Exception as e:
        check("probe never raises", False, f"{type(e).__name__}: {e}")


def main() -> int:
    print("=" * 68)
    print("MODEL SELECTION TESTS")
    print("=" * 68)
    test_choice()
    test_multi_gpu_pooling()
    test_probe_failure_modes()
    test_kv_estimate()
    test_context_fit()
    test_context_aware_choice()
    print("\n" + "=" * 68)
    print("All model selection tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
