"""Tests for REPL input handling, driving a real PromptSession over a pipe.

The bug: pasting five lines ran line one as a prompt and the other four as four
more prompts, because input() returns at the first newline. These tests feed
actual bracketed-paste escape sequences -- the same bytes a terminal emits on
paste -- and assert the whole block comes back as one turn.

    .venv\\Scripts\\python.exe scripts\\test_repl_input.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402

from repl_input import (  # noqa: E402
    HAVE_PROMPT_TOOLKIT, InputReader, clean_input, summarize_input,
)

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0

# What a terminal actually sends when you paste, with bracketed paste enabled.
PASTE_START, PASTE_END = "\x1b[200~", "\x1b[201~"
ENTER = "\r"


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def read_with(keys: str) -> str:
    """Feed raw key bytes to a real PromptSession and return one turn.

    Deliberately routed through asyncio.run + read_async, because that is how
    the REPL actually calls it. An earlier version of this helper used the
    synchronous read() from sync context, so every test passed while the real
    program crashed on the first keystroke: PromptSession.prompt() calls
    asyncio.run() internally, which cannot run inside an existing loop.
    """

    async def go() -> str:
        with create_pipe_input() as pipe:
            pipe.send_text(keys)
            reader = InputReader(_input=pipe, _output=DummyOutput())
            assert not reader.using_fallback, "expected a real PromptSession"
            return await reader.read_async("> ")

    return asyncio.run(go())


def test_paste_is_one_turn() -> None:
    print("\n1. Bracketed paste arrives as a single turn")

    block = "def add(a, b):\n    return a + b\n\nprint(add(1, 2))"
    got = read_with(PASTE_START + block + PASTE_END + ENTER)

    check("returns the whole block", got == block, f"got {got!r}")
    check("newlines preserved", got.count("\n") == 3, f"{got.count(chr(10))} newlines")
    check("did not stop at line 1", got.splitlines()[0] == "def add(a, b):")
    check("last line survived", got.splitlines()[-1] == "print(add(1, 2))")


def test_paste_with_trailing_newline() -> None:
    print("\n2. Paste ending in a newline")
    block = "line one\nline two\n"
    got = read_with(PASTE_START + block + PASTE_END + ENTER)
    check("trailing newline kept in buffer", got.startswith("line one\nline two"), repr(got))
    check("both lines present", len(got.strip().splitlines()) == 2, repr(got))


def test_typed_enter_submits() -> None:
    print("\n3. Typed Enter still submits immediately")
    got = read_with("hello world" + ENTER)
    check("single line submits", got == "hello world", repr(got))


def test_alt_enter_inserts_newline() -> None:
    print("\n4. Alt+Enter inserts a newline instead of submitting")
    # ESC then Enter is how terminals encode Alt+Enter
    got = read_with("first" + "\x1b\r" + "second" + ENTER)
    check("two lines from one turn", got == "first\nsecond", repr(got))


def test_slash_command_through_paste() -> None:
    print("\n5. Commands still work")
    got = read_with("/help" + ENTER)
    check("slash command unchanged", got == "/help", repr(got))


def test_running_loop_regression() -> None:
    """The REPL lives inside asyncio.run(); the sync API cannot.

    This is the exact crash that shipped: 'asyncio.run() cannot be called from
    a running event loop', raised on the very first prompt.
    """
    print("\n6. Works inside a running event loop")

    async def go():
        with create_pipe_input() as pipe:
            pipe.send_text("hello" + ENTER)
            reader = InputReader(_input=pipe, _output=DummyOutput())
            got = await reader.read_async("> ")
            try:
                reader.read("> ")
            except RuntimeError as e:
                return got, str(e)
            return got, None

    got, err = asyncio.run(go())
    check("read_async works from a running loop", got == "hello", repr(got))
    check(
        "sync read() refuses with a useful message",
        err is not None and "read_async" in err,
        str(err),
    )


def test_bom_stripped() -> None:
    """PowerShell prepends a UTF-8 BOM when piping into a native command.

    Observed: the first piped line arrived as '\\ufeff/tokens', so it did not
    start with '/' and was sent to the model as a prompt instead of being
    handled as a command. strip() does not help -- U+FEFF is not whitespace.
    """
    print("\n7. Invisible leading characters")

    check("BOM removed", clean_input("\ufeff/help") == "/help")
    check("BOM+text still a command", clean_input("\ufeff/help").startswith("/"))
    check("plain text untouched", clean_input("/help") == "/help")
    check("nulls dropped", clean_input("a\x00b") == "ab")
    check(
        "internal BOM left alone",
        clean_input("keep \ufeff inside") == "keep \ufeff inside",
    )
    # and end to end through a real session
    got = read_with("\ufeff/tokens" + ENTER)
    check("piped BOM line dispatches as a command", got == "/tokens", repr(got))


def test_summary_helper() -> None:
    print("\n8. Multi-line echo summary")
    check("single line produces no note", summarize_input("just one line") == "")
    note = summarize_input("a\nb\nc")
    check("multi-line reports count", "3 lines" in note, note)


def main() -> int:
    print("=" * 70)
    print("REPL INPUT TESTS")
    print("=" * 70)
    if not HAVE_PROMPT_TOOLKIT:
        print("\nprompt_toolkit not installed -- paste handling is degraded.")
        return 1

    test_paste_is_one_turn()
    test_paste_with_trailing_newline()
    test_typed_enter_submits()
    test_alt_enter_inserts_newline()
    test_slash_command_through_paste()
    test_running_loop_regression()
    test_bom_stripped()
    test_summary_helper()

    print("\n" + "=" * 70)
    print("All REPL input tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
