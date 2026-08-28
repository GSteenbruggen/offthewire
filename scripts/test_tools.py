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


def main() -> int:
    print("=" * 68)
    print("WORKSPACE AND COMMAND TESTS")
    print("=" * 68)
    test_confinement()
    test_run_command_basics()
    test_cancellation_kills_the_tree()
    test_timeout_kills_the_tree()
    test_workspace_memory()
    print("\n" + "=" * 68)
    print("All workspace/command tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
