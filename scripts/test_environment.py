"""Tests for environment probing. No network, no model.

The version parser got this wrong three times during development -- once taking
Docker's build hash, once truncating node's "v22.13.1" to "13.1" -- so the real
banners live here as fixtures.

    .venv\\Scripts\\python.exe scripts\\test_environment.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from environment import (  # noqa: E402
    _version_of, describe_delta, probe_static, probe_volatile, static_block,
    volatile_block,
)

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def test_version_parsing() -> None:
    print("\n1. Version banners (real output from this machine)")
    cases = [
        ("Docker version 28.3.0, build 38b7060", "28.3.0"),   # build hash trap
        ("v22.13.1", "22.13.1"),                              # no word boundary
        ("git version 2.51.0.windows.1", "2.51.0"),
        ("Python 3.11.9", "3.11.9"),
        ("go version go1.22.5 windows/amd64", "1.22.5"),
        ("npm 10.9.2", "10.9.2"),
        ("cargo 1.79.0 (ffa9cf99a 2024-06-03)", "1.79.0"),
    ]
    for banner, expected in cases:
        got = _version_of(banner)
        check(f"{banner[:44]:44} -> {got}", got == expected, f"expected {expected}")


def test_blocks_stay_small() -> None:
    """These ride along on every request; regressions here are permanent."""
    print("\n2. Grounding blocks stay within budget")
    ws = ROOT
    static = static_block(probe_static(ws))
    volatile = volatile_block(probe_volatile(ws), ws)

    s_tok, v_tok = len(static) // 4, len(volatile) // 4
    check(f"static block ~{s_tok} tokens", s_tok < 180, "should stay well under 180")
    check(f"volatile block ~{v_tok} tokens", v_tok < 180, "should stay well under 180")
    check(
        f"combined ~{s_tok + v_tok} tokens",
        s_tok + v_tok < 300,
        "permanent per-request overhead",
    )
    check("static names the shell", "PowerShell" in static or "bash" in static or "sh" in static)
    check("volatile states the date", "Today is" in volatile)
    check("volatile warns about staleness", "stale" in volatile.lower())


def test_delta_is_quiet() -> None:
    print("\n3. Deltas only fire on real change")
    ws = ROOT
    snap = probe_volatile(ws)

    check("no delta on first turn", describe_delta({}, snap) is None)
    check("no delta when nothing moved", describe_delta(snap, dict(snap)) is None)

    # time-of-day churns constantly and must not trigger an injection
    later = dict(snap)
    later["time"] = "23:59 EDT"
    check("time-of-day alone does not fire", describe_delta(snap, later) is None)

    changed = dict(snap)
    changed["git"] = "branch feature/x, 2 uncommitted change(s)"
    d = describe_delta(snap, changed)
    check("git change fires", d is not None and "feature/x" in d, str(d))

    newday = dict(snap)
    newday["date"] = "Tuesday, 11 August 2026"
    d = describe_delta(snap, newday)
    check("date rollover fires", d is not None and "11 August" in d, str(d))

    gained = dict(snap)
    gained["instructions"] = list(snap.get("instructions") or []) + ["AGENTS.md"]
    d = describe_delta(snap, gained)
    check("new instruction file fires", d is not None and "AGENTS.md" in d, str(d))


def test_non_repo_and_empty_dir() -> None:
    print("\n4. Degrades cleanly outside a repo")
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        snap = probe_volatile(ws)
        block = volatile_block(snap, ws)
        check("reports 'not a repository'", "not a repository" in block, block[-80:])
        check("no project type claimed", snap["project"] is None)
        check("no instruction files claimed", snap["instructions"] == [])
        check("still states the date", "Today is" in block)


def main() -> int:
    print("=" * 70)
    print("ENVIRONMENT PROBE TESTS")
    print("=" * 70)
    test_version_parsing()
    test_blocks_stay_small()
    test_delta_is_quiet()
    test_non_repo_and_empty_dir()
    print("\n" + "=" * 70)
    print("All environment tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
