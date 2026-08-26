"""Tests for the clarifying-question tool. No network, no model.

The failure that matters most is hanging: in a --prompt or piped run there is
nobody to answer, and a blocking input() would wait forever.

    .venv\\Scripts\\python.exe scripts\\test_ask.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ui  # noqa: E402
from tools import ToolRegistry, Workspace  # noqa: E402

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def registry(interactive: bool) -> ToolRegistry:
    tmp = Path(tempfile.mkdtemp(prefix="ask-"))
    reg = ToolRegistry(Workspace(tmp))
    reg.add_ask_tool(interactive=interactive)
    return reg


def test_non_interactive_never_blocks() -> None:
    print("\n1. Non-interactive runs do not hang")
    reg = registry(interactive=False)
    out = asyncio.run(reg.call("ask_user", {"question": "Which database?"}))
    check("returns immediately", "non-interactive" in out, out[:70])
    check("tells the model to assume and continue", "assumption" in out.lower())
    check("marked as an error so it is noticed", out.startswith("ERROR"))


def test_option_selection() -> None:
    print("\n2. Answering by number picks that option")
    reg = registry(interactive=True)
    answers = iter(["2"])
    original = ui.question

    captured: dict = {}

    def fake(text, options):
        captured["text"], captured["options"] = text, options
        raw = next(answers)
        return options[int(raw) - 1] if raw.isdigit() and options else raw

    ui.question = fake
    try:
        out = asyncio.run(reg.call("ask_user", {
            "question": "Which database?",
            "options": "Postgres|SQLite|MySQL",
        }))
    finally:
        ui.question = original

    check("options parsed from the pipe-separated string",
          captured["options"] == ["Postgres", "SQLite", "MySQL"], str(captured.get("options")))
    check("chosen option returned", "SQLite" in out, out)
    check("question passed through", captured["text"] == "Which database?")


def test_free_text_and_skip() -> None:
    print("\n3. Free text and skipping")
    reg = registry(interactive=True)
    original = ui.question

    ui.question = lambda text, options: "use whatever postgres is already there"
    try:
        out = asyncio.run(reg.call("ask_user", {"question": "Which database?",
                                                "options": "Postgres|SQLite"}))
    finally:
        ui.question = original
    check("free text beats the options list", "already there" in out, out)

    ui.question = lambda text, options: ""
    try:
        out = asyncio.run(reg.call("ask_user", {"question": "Which database?"}))
    finally:
        ui.question = original
    check("empty answer reads as skipped", "skipped" in out.lower(), out)
    check("skipping still tells it to continue", "assumption" in out.lower())


def test_registration() -> None:
    print("\n4. Registration")
    reg = registry(interactive=True)
    check("tool is present", "ask_user" in reg)
    names = [t["function"]["name"] for t in reg.ollama_schemas()]
    check("appears in the schemas sent to the model", "ask_user" in names)
    schema = next(t for t in reg.ollama_schemas() if t["function"]["name"] == "ask_user")
    props = schema["function"]["parameters"]["properties"]
    check("arguments stay flat for a small model",
          all(p["type"] == "string" for p in props.values()), str(props))
    check("only question is required",
          schema["function"]["parameters"]["required"] == ["question"])
    check("not gated behind approval", not reg.get("ask_user").mutating)


def main() -> int:
    print("=" * 68)
    print("ASK-USER TOOL TESTS")
    print("=" * 68)
    test_non_interactive_never_blocks()
    test_option_selection()
    test_free_text_and_skip()
    test_registration()
    print("\n" + "=" * 68)
    print("All ask tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
