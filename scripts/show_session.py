"""Read back a saved session in human form.

Every turn is already written to sessions/<id>.jsonl as it happens -- this does
not add logging, it renders what is there.

    show_session.py                 list every session
    show_session.py --last          render the most recent
    show_session.py 20260810_16     render one (id prefix is enough)
    show_session.py --last --full   include tool calls and their results
    show_session.py --last --md     markdown, for pasting elsewhere

Rendering lives in src/transcript.py so this and the agent's /history command
cannot drift apart. No network, no model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import transcript as T  # noqa: E402

SESSIONS = ROOT / "sessions"
DIM, BOLD, RESET, YELLOW, RED = "\033[2m", "\033[1m", "\033[0m", "\033[33m", "\033[31m"


def list_sessions() -> int:
    files = sorted(SESSIONS.glob("*.jsonl"), reverse=True)
    if not files:
        print(f"No sessions in {SESSIONS}")
        return 1

    print(f"{BOLD}{'id':<20} {'when':<18} {'turns':>5} {'size':>8}  first message{RESET}")
    for p in files:
        records = T.load(p)
        if not records:
            continue
        first = " ".join(T.first_user_message(records).split())[:46]
        print(
            f"{p.stem:<20} {T.stamp(records[0].get('ts')):<18} "
            f"{T.count_turns(records):>5} {p.stat().st_size / 1024:>7.1f}K  "
            f"{DIM}{first}{RESET}"
        )
    print(f"\n{DIM}show_session.py <id>   |   --last   |   --full for tool calls{RESET}")
    return 0


def resolve(session_id: str | None, use_last: bool) -> Path | None:
    if not SESSIONS.is_dir():
        return None
    files = [p for p in SESSIONS.glob("*.jsonl") if p.stat().st_size > 0]
    if not files:
        return None
    if use_last:
        return max(files, key=lambda p: p.stat().st_mtime)

    exact = SESSIONS / f"{session_id}.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(p for p in files if p.stem.startswith(session_id or ""))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"{YELLOW}{session_id!r} matches {len(matches)} sessions:{RESET}")
        for m in matches:
            print(f"  {m.stem}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Read a saved agent session.")
    ap.add_argument("session", nargs="?", help="Session id, or a unique prefix.")
    ap.add_argument("--last", action="store_true", help="Most recent session.")
    ap.add_argument("--full", action="store_true", help="Include tool calls and results.")
    ap.add_argument("--md", action="store_true", help="Markdown output.")
    args = ap.parse_args()

    if not args.session and not args.last:
        return list_sessions()

    path = resolve(args.session, args.last)
    if path is None:
        target = repr(args.session) if args.session else "any session"
        print(f"{RED}No session matching {target} in {SESSIONS}.{RESET}")
        return 1

    records = T.load(path)
    if not records:
        print(f"{path.stem} is empty.")
        return 1

    print(T.header(path, records, markdown=args.md))
    print(T.render(records, full=args.full, markdown=args.md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
