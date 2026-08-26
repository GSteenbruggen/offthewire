"""Regression tests for context accounting.

The bug these lock down: compaction fired after roughly 2k real tokens instead
of 24k, because the old calibrate() derived a chars-per-token ratio from
message *content* only. Tool-call arguments (which carry entire file bodies for
write_file) and the per-request tool schemas were excluded from the numerator
but present in the denominator, so the ratio collapsed and every estimate
inflated with it.

No network: Session's only outbound call is compaction, and that takes a client
object we fake here.

    .venv\\Scripts\\python.exe scripts\\test_accounting.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from session import (  # noqa: E402
    Session, find_session, latest_session, trim_dangling_tool_calls,
)

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def new_session(tmp: Path, limit: int = 32768) -> Session:
    # persist=True explicitly: writing to disk is opt-in since conversations
    # stopped being saved by default, and these tests are about the on-disk
    # accounting. test_not_saved_by_default below covers the other side.
    return Session(
        model="test",
        context_limit=limit,
        system_prompt="sys",
        sessions_dir=tmp,
        persist=True,
    )


class FakeClient:
    """Stands in for OllamaClient during compaction. Sends nothing anywhere."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, model, prompt, **kwargs):
        self.calls += 1

        class R:
            content = "Summary of earlier work."
            gen_tokens = 12
            total_s = 0.1

        return R()


def big_tool_call(chars: int) -> dict:
    """An assistant message whose tool_call carries a large file body -- the
    exact shape that broke the old estimator."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "write_file",
                    "arguments": {"path": "big.py", "content": "x" * chars},
                }
            }
        ],
    }


def test_ground_truth_wins(tmp: Path) -> None:
    print("\n1. Measured tokens beat estimates")
    s = new_session(tmp)
    s.tool_overhead_tokens = 1200

    s.note_actual(2000)
    check("uses the reported count", s.used_tokens() == 2000, f"got {s.used_tokens()}")

    s.append(big_tool_call(10_000))
    used = s.used_tokens()
    # 10k chars of arguments is roughly 2.8k tokens on top of the measured 2000
    check(
        "large tool_call raises the estimate proportionally",
        2000 < used < 7000,
        f"got {used:,}",
    )
    check(
        "does not trigger compaction at ~5k of a 32k window",
        not s.needs_compaction(),
        f"usage {100 * s.usage_ratio():.0f}%",
    )

    s.note_actual(4900)
    check("re-measuring resets the delta", s.used_tokens() == 4900, f"got {s.used_tokens()}")


def test_no_ratio_collapse(tmp: Path) -> None:
    """The original failure: many turns of small content + huge tool_calls."""
    print("\n2. Repeated write_file turns do not inflate the budget")
    s = new_session(tmp)
    s.tool_overhead_tokens = 1200

    # Mirrors the observed session: content tiny, tool_calls enormous.
    for i, actual in enumerate([1500, 3200, 5100, 6800], 1):
        s.append(big_tool_call(9000))
        s.append({"role": "tool", "tool_name": "write_file", "content": "Created big.py."})
        s.note_actual(actual)
        ratio = s.usage_ratio()
        check(
            f"turn {i}: reported {actual:,} tok -> {100 * ratio:.0f}% of window",
            abs(s.used_tokens() - actual) < 50 and not s.needs_compaction(),
            f"used {s.used_tokens():,}",
        )


def test_compaction_resets_measurement(tmp: Path) -> None:
    print("\n3. Compaction clears the stale measurement (the loop)")
    s = new_session(tmp, limit=8000)
    client = FakeClient()

    for _ in range(10):
        s.append({"role": "user", "content": "do something"})
        s.append({"role": "assistant", "content": "y" * 800})

    s.note_actual(7000)  # 87% of 8000
    check("over the threshold before compacting", s.needs_compaction(), f"{s.budget_line()}")

    msg = asyncio.run(s.compact(client))
    check("compaction ran", client.calls == 1, msg)
    check(
        "stale measurement dropped so it cannot re-fire immediately",
        s.last_prompt_tokens == 0,
        f"last_prompt_tokens={s.last_prompt_tokens}",
    )
    check(
        "post-compaction usage is back under the threshold",
        not s.needs_compaction(),
        f"{s.budget_line()}",
    )


def test_tool_call_pairs_stay_together(tmp: Path) -> None:
    print("\n4. Compaction never orphans a tool result from its call")
    s = new_session(tmp, limit=8000)
    for _ in range(6):
        s.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {}}}]})
        s.append({"role": "tool", "tool_name": "read_file", "content": "data"})

    _, recent = s._split_for_compaction()
    check(
        "retained tail does not start with an orphaned tool result",
        not recent or recent[0].get("role") != "tool",
        f"first kept role = {recent[0].get('role') if recent else 'none'}",
    )


def test_resume(tmp: Path) -> None:
    print("\n5. Resuming a saved session")

    s = new_session(tmp)
    s.append({"role": "user", "content": "add a docstring to calc.py"})
    s.append({"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "read_file", "arguments": {"path": "calc.py"}}}
    ]})
    s.append({"role": "tool", "tool_name": "read_file", "content": "def add(a, b): ..."})
    s.append({"role": "assistant", "content": "Done."})
    path = s.path

    back = Session.resume(
        path, model="test", context_limit=32768, system_prompt="sys", persist=True
    )
    check("all messages restored", len(back.messages) == 4, f"got {len(back.messages)}")
    check("same file continues", back.session_id == s.session_id)
    check("measurement reset for re-estimation", back.last_prompt_tokens == 0)
    check(
        "appending continues the same log",
        (lambda: (back.append({"role": "user", "content": "thanks"}),
                  len(path.read_text(encoding="utf-8").strip().splitlines()) == 5)[1])(),
    )


def test_dangling_tool_call(tmp: Path) -> None:
    """A session killed mid-turn ends with an unanswered tool call."""
    print("\n6. Interrupted sessions resume at a safe point")

    complete = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file"}}]},
        {"role": "tool", "tool_name": "read_file", "content": "data"},
    ]
    check("complete exchange is untouched", trim_dangling_tool_calls(complete) == complete)

    interrupted = complete + [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_command"}}]}
    ]
    trimmed = trim_dangling_tool_calls(interrupted)
    check(
        "unanswered trailing call is dropped",
        len(trimmed) == 3 and trimmed[-1]["role"] == "tool",
        f"kept {len(trimmed)}",
    )

    check("empty history is fine", trim_dangling_tool_calls([]) == [])

    plain_tail = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    check("normal assistant reply is kept", trim_dangling_tool_calls(plain_tail) == plain_tail)


def test_session_lookup(tmp: Path) -> None:
    print("\n7. Finding sessions")
    d = tmp / "lookup"
    d.mkdir(exist_ok=True)
    for name in ("20260810_120000", "20260810_130000", "20260811_090000"):
        (d / f"{name}.jsonl").write_text('{"role":"user","content":"x"}\n', encoding="utf-8")

    check("exact id resolves", find_session(d, "20260810_130000") is not None)
    check("unique prefix resolves", find_session(d, "20260811") is not None)
    check("ambiguous prefix refuses", find_session(d, "20260810") is None)
    check("unknown id refuses", find_session(d, "nope") is None)

    latest = latest_session(d)
    check("latest_session returns a file", latest is not None, str(latest))

    (d / "empty.jsonl").write_text("", encoding="utf-8")
    check("empty files are skipped", latest_session(d).stem != "empty")


def test_multipass_compaction() -> None:
    """A grossly oversized conversation must fold in bounded passes.

    The bug class this guards: the summarisation prompt itself overflowing
    the window (observed at 6.6x on a real 1627-message session), which
    Ollama truncates silently -- producing a "summary" of whichever fragment
    survived. The fake client records every prompt so the budget claim is
    checked against what would actually go on the wire, and threads a marker
    through the chained summaries to prove pass N sees pass N-1's output.
    """
    import asyncio

    from session import COMPACT_PROMPT_BUDGET, MAX_COMPACT_PASSES, estimate_tokens

    print("\n8. Multi-pass compaction of an oversized conversation")

    tmp = Path(tempfile.mkdtemp(prefix="acct-compact-"))
    s = new_session(tmp, limit=32768)
    filler = "def loader(row):\n    return row * 1.07\n" * 40
    s.append({"role": "user", "content": "codename BANANA-47, region eu-west-3"})
    for i in range(900):
        s.append({"role": "assistant", "content": f"working on chunk {i}",
                  "tool_calls": [{"function": {"name": "read_file",
                                               "arguments": {"path": f"f{i}.py"}}}]})
        s.append({"role": "tool", "tool_name": "read_file", "content": filler})
    check("setup is far over budget", s.usage_ratio() > 5, f"{s.usage_ratio():.1f}x")

    class FakeClient:
        def __init__(self):
            self.prompts = []

        async def generate(self, model, prompt, **kw):
            self.prompts.append(prompt)
            n = len(self.prompts)

            class R:
                content = f"[summary v{n}] key facts retained; based on: " + (
                    f"summary v{n-1}" if n > 1 else "the transcript"
                )
                gen_tokens = 120
                total_s = 0.5

            return R()

    fake = FakeClient()
    progress = []
    result = asyncio.run(s.compact(fake, on_progress=progress.append))

    check("bounded pass count", 1 <= len(fake.prompts) <= MAX_COMPACT_PASSES,
          f"{len(fake.prompts)} passes")
    budget = int(32768 * COMPACT_PROMPT_BUDGET)
    worst = max(estimate_tokens(pr) for pr in fake.prompts)
    check("every pass prompt fits its budget", worst <= budget * 1.15,
          f"worst {worst:,} vs budget {budget:,}")
    if len(fake.prompts) > 1:
        check("passes are chained, not independent",
              "summary v1" in fake.prompts[1])
        check("progress reported per pass", len(progress) == len(fake.prompts),
              f"{len(progress)} notes")
    check("conversation actually shrank", s.usage_ratio() < 1.0,
          f"now {s.usage_ratio():.2f}x")
    check("summary message leads the transcript",
          "compacted" in s.messages[0]["content"].lower()
          or "summary" in s.messages[0]["content"].lower())
    check("result reports the pass count", "passes" in result or len(fake.prompts) == 1,
          result[:90])


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="acct-"))
    try:
        print("=" * 68)
        print("SESSION + CONTEXT ACCOUNTING TESTS")
        print("=" * 68)
        test_ground_truth_wins(tmp)
        test_no_ratio_collapse(tmp)
        test_compaction_resets_measurement(tmp)
        test_tool_call_pairs_stay_together(tmp)
        test_resume(tmp)
        test_dangling_tool_call(tmp)
        test_session_lookup(tmp)
        test_multipass_compaction()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 68)
    print("All accounting tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
