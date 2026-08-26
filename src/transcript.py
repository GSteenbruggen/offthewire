"""Render a conversation for humans.

Sessions are stored as JSONL -- append-only, so a crash mid-turn cannot corrupt
what came before. That is the right storage format and the wrong reading
format, hence this.

Shared by ``scripts/show_session.py`` (reads a file) and the agent's
``/history`` command (renders what is in memory), so the two cannot drift.

No network, no model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# The one home for ANSI codes; agent.py imports these rather than keeping a
# copy that can drift.
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
BLUE, GREEN, CYAN, RED = "\033[34m", "\033[32m", "\033[36m", "\033[31m"
YELLOW = "\033[33m"

# Grounding we injected, not something the user typed. Hidden unless --full.
ENV_PREFIXES = (
    "Today is ",
    "[environment update]",
    "[Earlier conversation, compacted",
    # the note _record_interruption leaves after Ctrl+C -- injected, not typed
    "[I stopped you mid-response",
)

MAX_TOOL_RESULT = 2000


def load(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def is_injected(rec: dict[str, Any]) -> bool:
    return rec.get("role") == "user" and str(rec.get("content") or "").startswith(
        ENV_PREFIXES
    )


def first_user_message(records: list[dict[str, Any]]) -> str:
    return next(
        (
            str(r.get("content") or "")
            for r in records
            if r.get("role") == "user" and not is_injected(r)
        ),
        "",
    )


def count_turns(records: list[dict[str, Any]]) -> int:
    return sum(1 for r in records if r.get("role") == "user" and not is_injected(r))


def relative(ts: float | None, *, now: float | None = None) -> str:
    """Human-scale age: '4m', '2h', 'yesterday', '3d', then a date.

    Session ids are already timestamps, but '20260812_093606' is not something
    you read at a glance while scanning a list.
    """
    if not ts:
        return ""
    now = now if now is not None else time.time()
    delta = max(0, now - ts)
    mins = delta / 60
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{int(mins)}m ago"
    hours = mins / 60
    if hours < 24:
        return f"{int(hours)}h ago"
    days = hours / 24
    if days < 2:
        return "yesterday"
    if days < 7:
        return f"{int(days)}d ago"
    return time.strftime("%d %b", time.localtime(ts))


def ellipsize(text: str, limit: int) -> str:
    """Truncate on a word boundary where possible, with a visible marker."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut[limit // 2 :]:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def clock(ts: float | None) -> str:
    return time.strftime("%H:%M", time.localtime(ts)) if ts else ""


def stamp(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "?"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln for ln in (text or "").rstrip().splitlines())


class _Style:
    """Colour codes, or empty strings when rendering plain text/markdown."""

    def __init__(self, colour: bool):
        self.on = colour

    def __getattr__(self, name: str) -> str:
        if not self.on:
            return ""
        return {
            "dim": DIM, "bold": BOLD, "reset": RESET,
            "blue": BLUE, "green": GREEN, "cyan": CYAN, "red": RED,
        }.get(name, "")


def render(
    records: list[dict[str, Any]],
    *,
    full: bool = False,
    markdown: bool = False,
    colour: bool = True,
) -> str:
    """Return the conversation as text.

    ``full`` includes injected grounding, tool arguments in full, and tool
    results. Without it you get just what was said, which is usually what you
    want when scanning back through a session.
    """
    s = _Style(colour and not markdown)
    out: list[str] = []

    for rec in records:
        role = rec.get("role")
        content = str(rec.get("content") or "").rstrip()
        ts = rec.get("ts")

        if role == "_system_note":
            if full:
                note = f"--- {rec.get('content')} (kept {rec.get('kept')}) ---"
                out.append(f"> {note}\n" if markdown else f"{s.dim}{note}{s.reset}\n")
            continue

        if is_injected(rec) and not full:
            continue

        if role == "user":
            label = "environment" if is_injected(rec) else "you"
            # Only the paths are logged, never the base64, so this is all there
            # is to show -- and without it a turn about a screenshot replays as
            # a bare "what is wrong here?" with no sign an image was attached.
            # ASCII like the -> and <- markers below: this string is handed to
            # a plain print(), and a console still running cp1252 raises rather
            # than degrading on a character it cannot encode.
            shots = [Path(p).name for p in (rec.get("image_paths") or [])]
            if markdown:
                extra = "".join(f"\n\n_attached: {n}_" for n in shots)
                out.append(
                    f"### {label}{(' · ' + clock(ts)) if ts else ''}\n\n{content}{extra}\n"
                )
            else:
                head = s.dim if is_injected(rec) else s.blue
                out.append(f"{head}{s.bold}{label}{s.reset} {s.dim}{clock(ts)}{s.reset}")
                out.append(f"{_indent(content)}")
                for n in shots:
                    out.append(f"{s.cyan}    attached: {n}{s.reset}")
                out.append("")

        elif role == "assistant":
            if content:
                if markdown:
                    out.append(f"### assistant{(' · ' + clock(ts)) if ts else ''}\n\n{content}\n")
                else:
                    out.append(f"{s.green}{s.bold}assistant{s.reset} {s.dim}{clock(ts)}{s.reset}")
                    out.append(f"{_indent(content)}\n")

            for call in rec.get("tool_calls") or []:
                fn = call.get("function") or {}
                name = fn.get("name", "?")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}

                if not full:
                    # One line saying what it did. The file body is deliberately
                    # excluded -- a write_file argument can be 10k characters.
                    brief = ", ".join(
                        f"{k}={str(v)[:60]}"
                        for k, v in args.items()
                        if k not in ("content", "new_text")
                    )
                    line = f"-> {name}({brief})"
                    out.append(f"- `{line}`" if markdown else f"{s.cyan}  {line}{s.reset}")
                else:
                    out.append(f"**-> {name}**" if markdown else f"{s.cyan}  -> {name}{s.reset}")
                    for k, v in args.items():
                        body = str(v)
                        if "\n" in body:
                            out.append(f"{s.dim}    {k}:{s.reset}")
                            out.append(_indent(body, "      "))
                        else:
                            out.append(f"{s.dim}    {k}: {s.reset}{body[:200]}")
            if (rec.get("tool_calls") and not full) or (rec.get("tool_calls") and markdown):
                out.append("")

        elif role == "tool":
            if not full:
                continue
            head = f"<- {rec.get('tool_name', '?')}"
            body = content[:MAX_TOOL_RESULT]
            if markdown:
                out.append(f"```\n{head}\n{body}\n```\n")
            else:
                tone = s.red if content.startswith("ERROR") else s.dim
                out.append(f"{tone}  {head}{s.reset}")
                out.append(f"{s.dim}{_indent(body)}{s.reset}\n")

    return "\n".join(out)


def header(path: Path, records: list[dict[str, Any]], *, markdown: bool = False) -> str:
    when = stamp(records[0].get("ts") if records else None)
    size = path.stat().st_size / 1024 if path.exists() else 0
    if markdown:
        return f"# Session {path.stem}\n\n_{when} — {len(records)} records_\n"
    return (
        f"{BOLD}session {path.stem}{RESET}  "
        f"{DIM}{when}  ·  {len(records)} records  ·  {size:.1f}K{RESET}\n"
    )
