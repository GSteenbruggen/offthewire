"""Tests for workspace confinement and command execution. No network, no model.

The path jail is the security boundary between "the agent may edit this
project" and "the agent may edit this machine", and until this file existed it
had no tests at all -- the SSRF guard got ten while the filesystem guard got
zero. The run_command tests cover the other long-standing gap: a cancelled or
timed-out command must die with its whole process tree, not keep running
orphaned behind a returned prompt.

    .venv\\Scripts\\python.exe scripts\\test_tools.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tools as TL  # noqa: E402
from tools import ToolError, ToolRegistry, Workspace  # noqa: E402

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0

IS_WINDOWS = os.name == "nt"


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def refused(ws: Workspace, path: str) -> bool:
    try:
        ws.resolve(path)
        return False
    except ToolError:
        return True


def test_confinement() -> None:
    print("\n1. Workspace confinement")

    root = Path(tempfile.mkdtemp(prefix="otw-jail-"))
    outside = Path(tempfile.mkdtemp(prefix="otw-outside-"))
    (root / "inner").mkdir()
    (root / "inner" / "a.txt").write_text("in")
    (outside / "loot.txt").write_text("out")

    ws = Workspace(root)

    check("plain relative path resolves", ws.resolve("inner/a.txt").is_file())
    check("dot path resolves to root", ws.resolve(".") == root.resolve())
    check("absolute path inside is allowed", not refused(ws, str(root / "inner" / "a.txt")))

    check("parent traversal refused", refused(ws, "../"))
    check("deep traversal refused", refused(ws, "inner/../../" + outside.name))
    check("absolute path outside refused", refused(ws, str(outside / "loot.txt")))
    check(
        "traversal hidden mid-path refused",
        refused(ws, f"inner/../../{outside.name}/loot.txt"),
    )
    if IS_WINDOWS:
        # "C:file" is drive-relative and pathlib versions differ on the join:
        # some treat it as a same-drive relative segment (lands inside the
        # workspace), others resolve it against the drive's CWD (lands outside
        # and must be refused). Both outcomes keep the jail intact, because
        # containment is checked after resolution -- so the invariant to
        # assert is refused-or-contained, not one specific behavior.
        try:
            p = ws.resolve("C:loot.txt")
            check(
                "drive-relative path is contained when allowed",
                p.is_relative_to(root.resolve()),
                str(p),
            )
        except ToolError:
            check("drive-relative path is contained when allowed", True, "refused")

    # A symlink inside the root pointing outside must not be followable.
    # Windows requires privilege to create symlinks, so this leg is
    # POSIX-only; resolve()-then-check is the mechanism on both.
    link = root / "sneaky"
    try:
        link.symlink_to(outside)
    except OSError:
        print("       (symlink creation unavailable here; escape leg skipped)")
    else:
        check("symlink escape refused", refused(ws, "sneaky/loot.txt"))


def test_run_command_basics() -> None:
    print("\n2. run_command basics")

    root = Path(tempfile.mkdtemp(prefix="otw-cmd-"))
    reg = ToolRegistry(Workspace(root))

    async def run(cmd: str) -> str:
        return await reg.call("run_command", {"command": cmd})

    out = asyncio.run(run("echo hello-jail"))
    check("output captured", "hello-jail" in out, out[:60])

    out = asyncio.run(run("exit 3" if IS_WINDOWS else "exit 3"))
    check("exit code reported", "[exit 3]" in out, out[:60])


def test_cancellation_kills_the_tree() -> None:
    """Ctrl+C cancels the tool task; the process tree must die with it.

    The marker-file scheme proves it: the command sleeps and then writes a
    file. If cancellation killed the tree, the file never appears -- if it
    merely abandoned the await, the orphan finishes its sleep and writes.
    """
    print("\n3. Cancellation kills the process tree")

    root = Path(tempfile.mkdtemp(prefix="otw-kill-"))
    reg = ToolRegistry(Workspace(root))
    marker = root / "survived.txt"

    if IS_WINDOWS:
        cmd = f'Start-Sleep 4; New-Item -ItemType File "{marker}"'
    else:
        cmd = f'sleep 4 && touch "{marker}"'

    async def go() -> float:
        task = asyncio.create_task(reg.call("run_command", {"command": cmd}))
        await asyncio.sleep(1.0)  # let the shell actually start
        started = time.monotonic()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return time.monotonic() - started

    took = asyncio.run(go())
    check("cancel returns promptly", took < 5, f"{took:.1f}s")
    time.sleep(4.5)  # long enough for an orphan to have finished its sleep
    check("no orphan outlived the cancel", not marker.exists())


def test_timeout_kills_the_tree() -> None:
    print("\n4. Timeout kills the process tree")

    root = Path(tempfile.mkdtemp(prefix="otw-tmo-"))
    reg = ToolRegistry(Workspace(root))
    marker = root / "survived.txt"

    if IS_WINDOWS:
        cmd = f'Start-Sleep 4; New-Item -ItemType File "{marker}"'
    else:
        cmd = f'sleep 4 && touch "{marker}"'

    saved = TL.SHELL_TIMEOUT
    TL.SHELL_TIMEOUT = 1
    try:
        out = asyncio.run(reg.call("run_command", {"command": cmd}))
    finally:
        TL.SHELL_TIMEOUT = saved

    check("timeout reported to the model", "timed out" in out, out[:80])
    time.sleep(4.5)
    check("no orphan outlived the timeout", not marker.exists())


def test_workspace_memory() -> None:
    """The last /folder destination is remembered — with two guard rails.

    An explicit path must always win, and non-interactive runs must never be
    redirected: a --prompt automation silently operating on last week's
    project instead of the current directory would be a correctness bug
    wearing a convenience's clothes.
    """
    import importlib

    print("\n5. Workspace memory")

    import agent as A
    import paths

    here = Path(tempfile.mkdtemp(prefix="otw-remembered-"))

    ws, restored = A.resolve_workspace("C:/explicit", str(here), True)
    check("explicit path always wins", ws == "C:/explicit" and not restored)

    ws, restored = A.resolve_workspace(None, str(here), True)
    check("interactive launch restores", ws == str(here) and restored)

    ws, restored = A.resolve_workspace(None, str(here), False)
    check("non-interactive never restores", ws == "." and not restored)

    ws, restored = A.resolve_workspace(None, str(here / "gone-away"), True)
    check("vanished path falls back to cwd", ws == "." and not restored)

    ws, restored = A.resolve_workspace(None, None, True)
    check("no memory means cwd", ws == "." and not restored)

    # State round-trip, isolated from the real data dir via the env override.
    saved = os.environ.get(paths.HOME_ENV)
    os.environ[paths.HOME_ENV] = str(here / "home")
    try:
        importlib.reload(paths)
        check("state starts empty", paths.read_state() == {})
        paths.write_state(last_workspace=str(here))
        check("state round-trips", paths.read_state().get("last_workspace") == str(here))
        paths.write_state(other="kept")
        check("updates merge, not replace",
              paths.read_state().get("last_workspace") == str(here))
        (here / "home" / "state.json").write_text("{corrupt", encoding="utf-8")
        check("corrupt state reads as empty", paths.read_state() == {})
    finally:
        if saved is None:
            os.environ.pop(paths.HOME_ENV, None)
        else:
            os.environ[paths.HOME_ENV] = saved
        importlib.reload(paths)


def test_step_cap_and_unlimited() -> None:
    """The per-turn step cap must bind at its limit and vanish at 0.

    A fake client emits a fresh tool call every round (arguments varied so
    the repeated-call guard never triggers), then finishes. With the default
    cap the turn must stop at exactly max_steps rounds; with max_steps=0 it
    must run through all rounds and finish naturally. Also pins the thinking
    checkpoint to *periodic*: an earlier version thought on every step >= 8,
    which quietly turned a long turn into always-on reasoning.
    """
    from agent import Agent, ThinkingPolicy
    from tools import Workspace

    print("\n6. Step cap and unlimited mode")

    # The periodicity pin, first: pure policy, no machinery.
    pol = ThinkingPolicy("auto")
    thinking_steps = [
        s for s in range(2, 41)
        if pol.decide(step=s, last_tool_failed=False, is_first_step=False)[0]
    ]
    check("checkpoints are periodic, not permanent",
          thinking_steps == [8, 16, 24, 32, 40], str(thinking_steps[:6]))

    class FakeResult:
        def __init__(self, tool_calls):
            self.content = "" if tool_calls else "done"
            self.thinking = ""
            self.tool_calls = tool_calls
            self.done_reason = ""
            self.prompt_tokens = 100
            self.gen_tokens = 5
            self.gen_tps = 1.0
            self.total_s = 0.1

    class FakeClient:
        def __init__(self, rounds):
            self.rounds = rounds
            self.calls = 0

        async def chat_stream(self, model, messages, **kw):
            self.calls += 1
            if self.calls <= self.rounds:
                return FakeResult([{"function": {
                    "name": "read_file",
                    "arguments": {"path": "f.txt", "start_line": self.calls},
                }}])
            return FakeResult([])

    root = Path(tempfile.mkdtemp(prefix="otw-steps-"))
    (root / "f.txt").write_text("line\n" * 50)

    def run(max_steps, rounds):
        client = FakeClient(rounds)
        agent = Agent(client, "m", Workspace(root), interactive=False,
                      auto_approve=True, max_steps=max_steps)
        asyncio.run(agent.run_turn("go"))
        return client.calls

    check("explicit cap binds at its limit", run(25, rounds=30) == 25)
    check("small cap binds too", run(5, rounds=30) == 5)
    check("unlimited runs to natural completion", run(0, rounds=30) == 31,
          "30 tool rounds + the finishing reply")

    agent = Agent(FakeClient(0), "m", Workspace(root), interactive=False,
                  auto_approve=True)
    check("the default is unlimited", agent.max_steps == 0,
          f"max_steps={agent.max_steps}")


def test_context_limit_recovery() -> None:
    """A reply cut off by the context filling triggers compact-and-continue.

    The old behavior told the user to run /maxtokens by hand; the software
    already knew the fix. The recovery must run at most once per turn (a
    model that hits the limit twice gets the honest warning instead of a
    loop), and must not fire at all when there is nothing to compact.
    """
    from agent import Agent
    from tools import Workspace

    print("\n7. Context-limit recovery")

    class FakeResult:
        def __init__(self, content, done_reason):
            self.content = content
            self.thinking = ""
            self.tool_calls = []
            self.done_reason = done_reason
            self.prompt_tokens = 100
            self.gen_tokens = 5
            self.gen_tps = 1.0
            self.total_s = 0.1

    class FakeClient:
        def __init__(self, truncate_times):
            self.truncate_times = truncate_times
            self.chats = 0
            self.summaries = 0

        async def chat_stream(self, model, messages, **kw):
            self.chats += 1
            if self.chats <= self.truncate_times:
                return FakeResult("partial thought that got cut", "length")
            return FakeResult("…and here is the rest.", "stop")

        async def generate(self, model, prompt, **kw):  # compaction pass
            self.summaries += 1

            class R:
                content = "summary of earlier work"
                gen_tokens = 10
                total_s = 0.1

            return R()

    root = Path(tempfile.mkdtemp(prefix="otw-lenrec-"))

    def run(truncate_times, seed_messages):
        client = FakeClient(truncate_times)
        agent = Agent(client, "m", Workspace(root), interactive=False,
                      auto_approve=True)
        for i in range(seed_messages):
            agent.session.messages.append(
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"old turn {i} " * 30})
        asyncio.run(agent.run_turn("go"))
        return client, agent

    client, agent = run(truncate_times=1, seed_messages=10)
    check("recovery compacts", client.summaries >= 1, f"{client.summaries} passes")
    check("model is asked to continue", client.chats == 2, f"{client.chats} calls")
    check("continuation note in transcript",
          any("Continue exactly" in str(m.get("content", ""))
              for m in agent.session.messages))
    check("the finished thought arrives",
          agent.session.messages[-1]["content"].endswith("the rest."))

    client, _ = run(truncate_times=99, seed_messages=10)
    check("recovery runs at most once per turn", client.chats == 2,
          f"{client.chats} calls")

    client, _ = run(truncate_times=1, seed_messages=0)
    check("no recovery when nothing to compact",
          client.chats == 1 and client.summaries == 0,
          f"chats={client.chats} summaries={client.summaries}")


def test_midturn_compaction_uncapped() -> None:
    """A long turn compacts as often as the context fills.

    The old 2-per-turn cap made sense at 25 steps and starves an unlimited
    turn: step 60 of a long refactor deserves the same rescue as step 3. The
    loop protection is futility, not a count -- a compaction that ends still
    over budget is not retried until enough new messages have aged past the
    keep-window to give it fresh material.
    """
    from agent import Agent
    from session import KEEP_RECENT_MESSAGES

    print("\n8. Mid-turn compaction, uncapped")

    class FakeResult:
        def __init__(self, tool_calls, prompt_tokens):
            self.content = "" if tool_calls else "done"
            self.thinking = ""
            self.tool_calls = tool_calls
            self.done_reason = ""
            self.prompt_tokens = prompt_tokens
            self.gen_tokens = 5
            self.gen_tps = 1.0
            self.total_s = 0.1

    class FakeClient:
        """Reports a nearly-full window after every reply, so the pre-step
        check sees needs_compaction() before each subsequent step."""

        def __init__(self, rounds, prompt_tokens=30_000):
            self.rounds = rounds
            self.prompt_tokens = prompt_tokens
            self.chats = 0
            self.summaries = 0

        async def chat_stream(self, model, messages, **kw):
            self.chats += 1
            calls = (
                [{"function": {"name": "read_file",
                               "arguments": {"path": "f.txt",
                                             "start_line": self.chats}}}]
                if self.chats <= self.rounds else []
            )
            return FakeResult(calls, self.prompt_tokens)

        async def generate(self, model, prompt, **kw):  # compaction pass
            self.summaries += 1

            class R:
                content = "summary of earlier work"
                gen_tokens = 10
                total_s = 0.1

            return R()

    root = Path(tempfile.mkdtemp(prefix="otw-midcompact-"))
    (root / "f.txt").write_text("line\n" * 50)

    def run(rounds, seed_chars=200):
        client = FakeClient(rounds)
        agent = Agent(client, "m", Workspace(root), interactive=False,
                      auto_approve=True)
        for i in range(10):
            agent.session.messages.append(
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": "x" * seed_chars})
        asyncio.run(agent.run_turn("go"))
        return client, agent

    # Every reply reports ~91% of the window, so every pre-step check wants
    # a compaction; each one succeeds (the fake summary is tiny). Five tool
    # rounds must therefore compact five times -- more than double the old
    # per-turn cap of two.
    client, agent = run(rounds=5)
    check("compaction is not capped per turn", agent.session.compactions >= 3,
          f"{agent.session.compactions} compactions across 5 rounds")
    check("the turn still completes", client.chats == 6, f"{client.chats} calls")

    # Futility: recent messages so large that even after folding the older
    # ones the estimate stays over budget. The first compaction runs and
    # fails to free space; two tool rounds add only four messages -- fewer
    # than the keep-window -- so it must NOT be retried within this turn.
    assert 2 * 2 < KEEP_RECENT_MESSAGES, "rounds chosen to stay below the retry bar"
    client, agent = run(rounds=2, seed_chars=200_000)
    check("futile compaction is not retried immediately",
          agent.session.compactions == 1,
          f"{agent.session.compactions} compactions across 2 rounds")
    check("the turn is never blocked by futility", client.chats == 3,
          f"{client.chats} calls")


def main() -> int:
    print("=" * 68)
    print("WORKSPACE AND COMMAND TESTS")
    print("=" * 68)
    test_confinement()
    test_run_command_basics()
    test_cancellation_kills_the_tree()
    test_timeout_kills_the_tree()
    test_workspace_memory()
    test_step_cap_and_unlimited()
    test_context_limit_recovery()
    test_midturn_compaction_uncapped()
    print("\n" + "=" * 68)
    print("All workspace/command tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
