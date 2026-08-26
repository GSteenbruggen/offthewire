"""Terminal presentation.

One place for colour, spacing and formatting so the agent loop stays about
agent logic.

The shape follows the conventions of modern agent CLIs, because they solve the
problem this program has: you spend a long time watching output arrive, and
you need to be able to tell at a glance what is the agent's *reasoning*, what
is an *action*, and what is the *answer*. That is done with three devices and
almost nothing else:

  * **One accent colour**, blue. It marks the
    things you look for -- the bullet on an action, the prompt, the spinner --
    and nothing else. A palette where five things are coloured is a palette
    where nothing stands out.
  * **A bullet and an elbow.** ``⏺`` opens an action, ``⎿`` returns its result
    underneath. Two glyphs carry the entire call/result structure without
    boxes, rules or indentation games.
  * **Dim for everything mechanical.** Arguments, counts, timings and paths
    recede; the eye lands on the bullet line and the answer.

Two things earn their keep beyond that:

  * **Syntax highlighting on file writes.** Since Ollama delivers tool calls
    atomically, the preview is the only chance to read generated code before it
    lands; unhighlighted it is a wall of grey.
  * **Markdown rendering for answers.** The model emits ``**bold**`` and ``-``
    bullets, which previously printed literally.

Everything degrades to plain text when stdout is not a terminal, so piped
output and the test suite stay readable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# The accent, and its dark counterpart for borders and rules. Everything else
# is either dim or one of the three semantic colours -- green for a result that
# worked, yellow for a decision you have to make, red for a failure. Those stay
# conventional on purpose: recolouring "error" would cost more than it buys.
ACCENT = "#5B9DFF"
DEEP = "#2C5A9E"

# The same two, as raw escape sequences. The spinner repaints with \r and
# cannot go through rich, which owns whole lines.
A_ACCENT = "\033[38;2;91;157;255m"
A_DEEP = "\033[38;2;44;90;158m"
A_DIM = "\033[2m"
A_RESET = "\033[0m"

THEME = Theme(
    {
        "accent": ACCENT,
        "border": DEEP,
        "bullet": f"bold {ACCENT}",
        "step": "bold",
        "meta": "dim",
        "reason": "dim italic",
        "tool": f"bold {ACCENT}",
        "ok": "green",
        "warn": "yellow",
        "err": "bold red",
        "path": ACCENT,
        "prompt": f"bold {ACCENT}",
        "head": "bold",
    }
)

# Piped output is not necessarily UTF-8. Redirected to a file, or through a
# shell that reports cp1252, rich falls back to the legacy Windows renderer and
# every glyph below -- the bullets, elbows and box corners -- raises
# UnicodeEncodeError rather than printing badly. That kills the run mid-answer,
# which is a poor trade for a character. Ask for UTF-8, and if the stream
# refuses, for unencodable characters to degrade to "?".
#
# Only when stdout is redirected: forcing an encoding on a real Windows console
# bypasses the wide-character path Python uses there and would introduce
# mojibake into the one case that currently works.
for _stream in (sys.stdout, sys.stderr):
    try:
        if not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

console = Console(theme=THEME, soft_wrap=False, highlight=False)

# Code theme that reads on both light and dark terminals without assuming a
# background colour of its own.
CODE_THEME = "ansi_dark"
PREVIEW_LINES = 40

# The glyph vocabulary, in one place. A terminal that cannot encode these is
# handled by the reconfigure above; a font that lacks them shows a box, which
# is legible enough that a lowest-common-denominator fallback is not worth the
# branch.
BULLET = "⏺"  # an action is starting
ELBOW = "⎿"  # its result, returning underneath
SPARK = "✻"  # the agent is busy or greeting you
CLIP = "⧉"  # an attachment


def rule_char() -> str:
    return "─" if console.is_terminal else "-"


# ------------------------------------------------------------------- session


def banner(model: str, workspace: Path, ctx: int, tools: list[str], session: str,
           input_note: str, web: str | None) -> None:
    """The welcome box: what you are talking to, and where it can reach.

    Boxed rather than listed because it is the one thing on screen that is not
    part of the conversation. Everything after this point is a scrolling log.
    """
    rows = [
        ("cwd", str(workspace)),
        ("context", f"{ctx:,} tokens"),
        ("session", session),
        ("tools", ", ".join(tools)),
    ]
    if web:
        rows.insert(2, ("web", web))
    width = max(len(k) for k, _ in rows)

    body = Text()
    body.append(f"{SPARK} ", style="accent")
    body.append("offthewire", style="head")
    body.append("  ")
    body.append(model, style="accent")
    body.append("\n\n")
    body.append("  /help for commands", style="meta")
    # Only present when input is degraded -- the key hints normally live on the
    # line under the input box, not up here.
    body.append(f"{('  ·  ' + input_note) if input_note else ''}\n", style="meta")
    for k, v in rows:
        body.append(f"  {k.ljust(width)}  {v}\n", style="meta")
    body.rstrip()  # the last row's newline would draw an empty line in the box

    console.print()
    console.print(
        Panel(body, box=ROUNDED, border_style="border", padding=(0, 1), expand=False)
    )
    console.print()


def user_line(text: str) -> None:
    """Echo a submitted turn.

    The input box erases itself on submit, so without this the conversation
    scrollback would contain the agent's half and not yours. Printed as a
    plain record: the prompt character, then what you said, and nothing
    else -- it is a record, not an invitation.
    """
    first, *rest = text.splitlines() or [""]
    console.print(f"[accent]>[/accent] {escape(first)}")
    for line in rest:
        console.print(f"  {escape(line)}")
    console.print()


# ---------------------------------------------------------------------- turn


def step_header(n: int, mode: str, why: str, used: int, limit: int) -> None:
    """One aligned line per step: what number, how (direct / think:level), why,
    and how full the context is.

    Kept dim and kept at all: the thinking policy is the thing this project is
    actually about, and a turn that silently decided not to reason is a turn
    you cannot account for later.
    """
    pct = 100 * used / max(1, limit)
    # 12 fits the longest mode, "think:medium", without breaking alignment.
    console.print(
        f"  [meta]{n:>2}[/meta] [border]·[/border] [meta]{mode:<12} {why:<24}"
        f"{used:>7,}/{limit:,} tok  {pct:>3.0f}%[/meta]"
    )


def stats_line(gen_tokens: int, gen_tps: float, total_s: float) -> None:
    """Condensed: the three numbers that matter, not the full telemetry dump."""
    console.print(
        f"  [meta]{gen_tokens:,} tok · {gen_tps:.1f} tok/s · {total_s:.1f}s[/meta]"
    )
    console.print()


def _line(markup: str) -> None:
    """Print exactly one terminal line, cropping rather than wrapping.

    A long argument -- an absolute path to the venv interpreter, say -- is
    wider than the terminal, and rich's default is to wrap it. Wrapping breaks
    at whatever character lands on the edge, which put the bullet alone on one
    line and the tool name on the next, destroying the only visual structure
    the log has. The tail of a long path is never worth that.
    """
    console.print(markup, no_wrap=True, overflow="ellipsis", crop=True)


def tool_call(name: str, summary: str) -> None:
    """Open an action. The bullet is the thing you scan the log for."""
    # name and summary are model-generated; escape() keeps a stray "[/tag]" in
    # them from being read as markup and crashing the print.
    _line(
        f"[bullet]{BULLET}[/bullet] [tool]{escape(name)}[/tool]"
        f"[meta]({escape(summary)})[/meta]"
    )


def file_write(tool_name: str, path: str, body: str, span: str = "") -> None:
    """Show generated code, highlighted, before it is written.

    ``span`` is plain text (e.g. "lines 3-7"); styling is applied here.
    """
    lines = body.splitlines()
    count = f"{len(lines)} line{'' if len(lines) == 1 else 's'}, {len(body)} chars"
    # The span belongs after the closing paren with the counts, not inside the
    # argument list: "edit_lines(calc.py) lines 2-2 · 1 line" reads as a call
    # with a note, which is what it is.
    detail = f"{span} · {count}" if span else count
    _line(
        f"[bullet]{BULLET}[/bullet] [tool]{escape(tool_name)}[/tool]"
        f"[meta]([/meta][path]{escape(path)}[/path][meta])"
        f"  {escape(detail)}[/meta]"
    )
    shown = "\n".join(lines[:PREVIEW_LINES])
    try:
        lexer = Syntax.guess_lexer(path, code=shown)
    except Exception:
        lexer = "text"
    # No elbow here: the elbow marks a *result*, and this is the argument shown
    # before approval. The result's own elbow follows once it has run.
    console.print(
        Syntax(
            shown,
            lexer,
            theme=CODE_THEME,
            line_numbers=True,
            word_wrap=False,
            background_color="default",
            padding=(0, 0, 0, 5),
        )
    )
    if len(lines) > PREVIEW_LINES:
        console.print(f"     [meta]… {len(lines) - PREVIEW_LINES} more lines[/meta]")


def tool_result(text: str) -> None:
    """Close an action, underneath the bullet that opened it.

    The trailing blank line is what separates one call from the next. A step
    that makes four calls is otherwise sixteen packed lines with no seam, and
    the bullets stop being scannable -- which was the whole point of them.
    """
    # Tool output is arbitrary text -- shell output, file contents, exception
    # messages -- and must never be interpreted as markup.
    first = escape((text or "").split("\n")[0][:150])
    if text.startswith("ERROR"):
        _line(f"  [border]{ELBOW}[/border]  [err]{first}[/err]")
    else:
        _line(f"  [border]{ELBOW}[/border]  [meta]{first}[/meta]")
    console.print()


def denied(name: str) -> None:
    _line(f"  [border]{ELBOW}[/border]  [err]denied[/err] [meta]{escape(name)}[/meta]")
    console.print()


def attachments(described: list[str]) -> None:
    """Confirm what is being sent with a turn.

    A terminal cannot show the picture, so the description is the only evidence
    the right file was picked up -- worth printing even for a single image,
    since a drag-and-drop can easily land the wrong one.
    """
    for line in described:
        _line(f"[accent]{CLIP}[/accent] [meta]{escape(line)}[/meta]")


def note(text: str) -> None:
    console.print(f"  [meta]{escape(text)}[/meta]")


def warn(text: str) -> None:
    console.print(f"  [warn]{escape(text)}[/warn]")


def error(text: str) -> None:
    console.print(f"  [err]{escape(text)}[/err]")


# ------------------------------------------------------------------ deciding


def _choice_box(title: str, options: list[str], hint: str = "") -> None:
    """The one piece of chrome that is not a scrolling log.

    A question that stops the run is worth interrupting the layout for -- it is
    the only moment the program is waiting on you rather than the other way
    round, and it should not look like another line of output.
    """
    body = Text()
    body.append(title, style="head")
    body.append("\n\n")
    for i, opt in enumerate(options, 1):
        marker = "❯" if i == 1 else " "
        body.append(f" {marker} {i}. ", style="accent" if i == 1 else "meta")
        body.append(f"{opt}\n")
    if hint:
        body.append(f"\n   {hint}", style="meta")
    console.print(
        Panel(body, box=ROUNDED, border_style="border", padding=(0, 1), expand=False)
    )


def question(text: str, options: list[str]) -> str:
    """Render a clarifying question and read the answer.

    Numbered rather than arrow-driven: this has to work when stdin is a pipe,
    in a plain console, and alongside prompt_toolkit's own input handling.
    Typing anything that is not a number is taken as a free-text answer, so the
    options are suggestions rather than a cage.
    """
    console.print()
    _choice_box(
        f"{SPARK} {text}",
        options,
        "or type your own answer" if options else "",
    )
    try:
        raw = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    if raw.isdigit() and options and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return raw


def approval(title: str) -> str:
    """Approval prompt for a mutating tool call.

    Uses input() rather than rich's own so it composes with the REPL's input
    handling. Numbered *and* lettered: the numbers match the box, and y/n/a
    still work for anyone with the old muscle memory or a piped 'y'.
    """
    console.print()
    _choice_box(
        title,
        ["yes", "yes, and stop asking this session", "no"],
        "1/2/3, or y/n/a",
    )
    return input("  > ")


# ------------------------------------------------------------------ spinner


# The spinner cycles a glyph while it waits, which is the difference between
# "working" and "hung" on a model that can go 60s without emitting a token.
SPINNER_FRAMES = ("·", "✢", "✳", "✻", "✽", "✻", "✳", "✢")


def spinner_line(elapsed: float, label: str = "thinking") -> str:
    """One frame of the waiting indicator, as a raw ANSI string.

    Not a rich renderable: this repaints in place with \\r, and rich owns whole
    lines. Returned rather than printed so the caller keeps control of when the
    line is erased.
    """
    frame = SPINNER_FRAMES[int(elapsed * 3) % len(SPINNER_FRAMES)]
    return (
        f"{A_ACCENT}{frame}{A_RESET} {A_DIM}{label}… "
        f"({elapsed:.0f}s · ctrl+c to interrupt){A_RESET}"
    )


SPINNER_WIDTH = 52  # enough to erase the longest frame of the above


# ------------------------------------------------------------------ streaming


class ReasoningStream:
    """Dim, raw, unformatted -- reasoning is scratch work, not output."""

    def __init__(self, show: bool):
        self.show = show
        self.started = False
        self.chars = 0
        # Raw escape codes are junk when stdout is a pipe or a file; degrade to
        # plain text like everything else in this module.
        term = console.is_terminal
        self._dim = A_DIM if term else ""
        self._accent = A_ACCENT if term else ""
        self._reset = A_RESET if term else ""

    def _open(self) -> None:
        # Same shape as a tool call -- a bullet, then what it is doing -- so
        # reasoning reads as one more step in the log rather than stray text.
        sys.stdout.write(f"{self._accent}{SPARK}{self._reset} {self._dim}thinking")
        self.started = True

    def push(self, piece: str) -> None:
        self.chars += len(piece)
        if not self.show:
            if not self.started:
                self._open()
            sys.stdout.write(".")
            sys.stdout.flush()
            return
        if not self.started:
            self._open()
            sys.stdout.write("\n  ")
        sys.stdout.write(piece.replace("\n", "\n  "))
        sys.stdout.flush()

    def close(self) -> None:
        if self.started:
            sys.stdout.write(f"{self._reset}\n\n")
            sys.stdout.flush()
            self.started = False


class AnswerStream:
    """Renders the answer as markdown while it streams, under a bullet.

    Re-rendering on each chunk sounds expensive but the model produces ~20
    tokens a second, so this repaints a handful of times per second at most.
    ``vertical_overflow='visible'`` lets long answers scroll normally instead of
    being clipped to the live region.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self._live: Live | None = None
        # Live repaints by moving the cursor. Redirected to a file or a pipe
        # there is no cursor, so every refresh appends a fresh copy of the whole
        # answer -- one observed run produced 37KB of duplicated frames. Fall
        # back to plain incremental writes whenever this is not a terminal.
        self._live_ok = console.is_terminal

    def push(self, piece: str) -> None:
        self.buffer += piece
        if not self._live_ok:
            sys.stdout.write(piece)
            sys.stdout.flush()
            return
        if self._live is None:
            self._live = Live(
                console=console,
                refresh_per_second=8,
                vertical_overflow="visible",
                transient=False,
            )
            self._live.__enter__()
        self._live.update(self._render())

    def _body(self):
        try:
            return Markdown(self.buffer)
        except Exception:
            # Half-written markdown can be malformed mid-stream; plain text is
            # always renderable.
            return Text(self.buffer)

    def _render(self):
        """The answer, hung off a bullet like everything else.

        A two-column grid rather than a text prefix, so the bullet sits beside
        the first line and wrapped lines stay aligned under it instead of
        drifting back to column zero.
        """
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1, no_wrap=True, style="bullet")
        grid.add_column(overflow="fold")
        grid.add_row(BULLET, self._body())
        return grid

    def close(self) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.__exit__(None, None, None)
            self._live = None
            console.print()
        elif self.buffer and not self._live_ok:
            sys.stdout.write("\n")
            sys.stdout.flush()
        # buffer is deliberately NOT cleared: after an interrupt the caller
        # reads it back to record whatever was said before the stop.


# --------------------------------------------------------------------- tables


def table(title: str, columns: list[tuple[str, str]], rows: list[list[str]]) -> None:
    """columns is [(header, justify)]."""
    t = Table(
        title=None,
        header_style="head",
        border_style="border",
        show_edge=False,
        pad_edge=False,
        box=None,
    )
    for header, justify in columns:
        t.add_column(header, justify=justify, overflow="fold")
    for row in rows:
        t.add_row(*row)
    if title:
        console.print(f"[head]{title}[/head]")
    console.print(t)


def session_table(rows: list[dict[str, Any]], current: str = "") -> None:
    """Sessions, newest first.

    Metadata is dimmed so the eye lands on the opening message, which is the
    only column you actually scan when looking for a past conversation.
    """
    t = Table(header_style="head", border_style="border", show_edge=False,
              pad_edge=False, box=None)
    # min_width, not width. With fixed widths (or a ratio column) rich shrinks
    # everything proportionally when the table does not fit, which clipped ids
    # to "20260…" and squeezed "when" and "turns" out entirely at 80 columns.
    # With min_width it degrades in priority order instead: at 80 it drops
    # "turns" and keeps the id and timestamp intact.
    t.add_column("id", style="dim", min_width=17, no_wrap=True)
    t.add_column("when", style="dim", justify="right", min_width=9, no_wrap=True)
    t.add_column("turns", style="dim", justify="right", min_width=5, no_wrap=True)
    t.add_column("opening message", no_wrap=True, overflow="ellipsis")

    for s in rows:
        # The marker rides inside the id column rather than having its own.
        # A standalone one-character column is the first thing rich drops when
        # space is tight, which is exactly when you still want to know which
        # session you are in.
        marker = "[accent]●[/accent] " if s["id"] == current else "  "
        # The opening message is user text; escape it so brackets in what they
        # typed cannot be parsed as markup.
        first = " ".join((s["first"] or "").split())
        t.add_row(
            f"{marker}{s['id']}",
            s.get("when", ""),
            str(s["messages"]) if s["messages"] else "-",
            escape(first) if first else "[meta](nothing yet)[/meta]",
        )
    console.print(t)


def key_values(pairs: list[tuple[str, Any]]) -> None:
    width = max((len(str(k)) for k, _ in pairs), default=0)
    for k, v in pairs:
        console.print(f"  [meta]{str(k).ljust(width)}[/meta]  {escape(str(v))}")
