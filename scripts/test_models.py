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

from models import VRAM_HEADROOM_BYTES, choose_candidate, total_vram_bytes  # noqa: E402

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


def test_probe_failure_modes() -> None:
    """The probe must return None, never raise, on machines without nvidia-smi
    or with a broken one -- a wrong number would silently skew every
    recommendation."""
    print("\n2. VRAM probe failure modes")
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
    test_probe_failure_modes()
    print("\n" + "=" * 68)
    print("All model selection tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
