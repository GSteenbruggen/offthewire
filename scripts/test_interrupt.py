"""Tests for stopping a turn with Ctrl+C. No network, no model.

What matters here is that a stop cancels the in-flight call rather than
raising through it, that the signal handler is always restored, and that a
second press still exits -- if the first one appears not to have worked,
people press again and expect out.

    .venv\\Scripts\\python.exe scripts\\test_interrupt.py
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent import Interrupter  # noqa: E402

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


async def _sleep_forever() -> str:
    await asyncio.sleep(30)
    return "finished"


async def case_cancels_in_flight_call() -> None:
    print("\n1. A stop cancels the in-flight call")
    it = Interrupter()
    task = asyncio.create_task(_sleep_forever())
    await asyncio.sleep(0)  # let it start
    it.arm(task)

    it._handle(signal.SIGINT, None)  # simulate Ctrl+C

    cancelled = False
    try:
        await task
    except asyncio.CancelledError:
        cancelled = True
    it.disarm()

    check("task was cancelled, not left running", cancelled)
    check("interrupter recorded the stop", it.fired)
    check("heartbeat cleaned up", it._heartbeat is None)


async def case_handler_is_restored() -> None:
    print("\n2. The SIGINT handler is always restored")
    original = signal.getsignal(signal.SIGINT)

    it = Interrupter()
    task = asyncio.create_task(_sleep_forever())
    await asyncio.sleep(0)
    it.arm(task)
    check("handler replaced while armed", signal.getsignal(signal.SIGINT) is not original)

    it._handle(signal.SIGINT, None)
    try:
        await task
    except asyncio.CancelledError:
        pass
    it.disarm()

    check("handler restored after disarm", signal.getsignal(signal.SIGINT) is original)


async def case_second_press_exits() -> None:
    print("\n3. A second press within the window exits")
    it = Interrupter()
    task = asyncio.create_task(_sleep_forever())
    await asyncio.sleep(0)
    it.arm(task)

    it._handle(signal.SIGINT, None)  # first: cancels
    raised = False
    try:
        it._handle(signal.SIGINT, None)  # second: should propagate
    except KeyboardInterrupt:
        raised = True

    try:
        await task
    except asyncio.CancelledError:
        pass
    it.disarm()

    check("second press raises KeyboardInterrupt", raised)


async def case_completed_call_is_untouched() -> None:
    print("\n4. A stop after completion does no harm")
    it = Interrupter()

    async def quick() -> str:
        return "done"

    task = asyncio.create_task(quick())
    result = await task
    it.arm(task)
    it._handle(signal.SIGINT, None)  # arrives too late
    it.disarm()

    check("result survived", result == "done")
    check("finished task not marked cancelled", not task.cancelled())


async def case_disarm_without_signal() -> None:
    print("\n5. Normal completion disarms cleanly")
    original = signal.getsignal(signal.SIGINT)
    it = Interrupter()

    async def quick() -> str:
        return "ok"

    task = asyncio.create_task(quick())
    it.arm(task)
    out = await task
    it.disarm()

    check("call returned normally", out == "ok")
    check("never fired", not it.fired)
    check("handler restored", signal.getsignal(signal.SIGINT) is original)


async def main() -> int:
    print("=" * 68)
    print("INTERRUPT TESTS")
    print("=" * 68)
    await case_cancels_in_flight_call()
    await case_handler_is_restored()
    await case_second_press_exits()
    await case_completed_call_is_untouched()
    await case_disarm_without_signal()
    print("\n" + "=" * 68)
    print("All interrupt tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
