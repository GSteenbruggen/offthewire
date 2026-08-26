"""Documentation drift checks. No network, no model.

Two facts are stated in more than one place, and both have already drifted
once in this project's history:

  * The in-session command list lives in /help (agent.py) and in the README
    table. A command added to one and not the other ships a reference that
    lies.
  * The version lives in _version.py, the installer config, and the README's
    pinned download links. A release cut with any of them stale offers users
    the previous build from the front page.

This suite makes both a test failure instead of a code-review hope.

    .venv\\Scripts\\python.exe scripts\\test_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from _version import __version__  # noqa: E402

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def help_commands() -> set[str]:
    """Command names as /help presents them, parsed from the source text."""
    src = (ROOT / "src" / "agent.py").read_text(encoding="utf-8")
    block = re.search(r'HELP = f?"""(.*?)"""', src, re.DOTALL)
    assert block, "HELP block not found in agent.py"
    return set(re.findall(r"^\s{2}(/[a-z]+)", block.group(1), re.MULTILINE))


def readme_commands() -> set[str]:
    """Command names from the README's in-session commands table."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = re.search(
        r"### In-session commands(.*?)(?=\n### |\n## )", text, re.DOTALL
    )
    assert section, "In-session commands section not found in README"
    return set(re.findall(r"^\| `(/[a-z]+)", section.group(1), re.MULTILINE))


def test_command_sync() -> None:
    print("\n1. /help and the README describe the same commands")
    in_help = help_commands()
    in_readme = readme_commands()
    check("README is not missing commands", in_help <= in_readme,
          f"missing: {sorted(in_help - in_readme)}")
    check("README has no phantom commands", in_readme <= in_help,
          f"phantom: {sorted(in_readme - in_help)}")
    check("a sane number of commands parsed", len(in_help) >= 15, str(len(in_help)))


def test_version_sync() -> None:
    print(f"\n2. One version everywhere: {__version__}")

    iss = (ROOT / "installer" / "OffTheWire.iss").read_text(encoding="utf-8")
    m = re.search(r'#define AppVersion\s+"([\d.]+)"', iss)
    check("installer config matches", bool(m) and m.group(1) == __version__,
          m.group(1) if m else "not found")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pinned = set(re.findall(r"OffTheWire-(?:Setup-)?(\d+\.\d+\.\d+)", readme))
    pinned |= set(re.findall(r"/releases/download/v(\d+\.\d+\.\d+)/", readme))
    stale = {v for v in pinned if v and v != __version__}
    check("README download links match", not stale, f"stale versions: {sorted(stale)}")


def main() -> int:
    print("=" * 68)
    print("DOCUMENTATION SYNC TESTS")
    print("=" * 68)
    test_command_sync()
    test_version_sync()
    print("\n" + "=" * 68)
    print("All documentation checks passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
