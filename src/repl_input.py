"""Terminal input for the REPL, with working multi-line paste and image paste.

``input()`` returns at the first newline, so pasting a five-line block submits
line one as a prompt and then runs the remaining four as four more prompts.
That is unusable for pasting a stack trace, a config file, or a function.

The fix is bracketed paste: the terminal wraps pasted text in escape sequences
so the application can tell a paste from typing and insert it whole. This is
what prompt_toolkit does natively, so the key bindings here only need to invert
its multiline defaults:

    Enter        submit
    Alt+Enter    insert a newline by hand
    paste        inserted verbatim, newlines and all, never submits
    Alt+V        pull an image off the clipboard and reference it in the line

Alt+V rather than Ctrl+V because Windows Terminal and VS Code both bind Ctrl+V
themselves and never forward it; with an image on the clipboard their paste
produces nothing at all, which is precisely the case we need to handle. Ctrl+V
is bound too, for terminals that do pass it through.

Falling back to ``input()`` when prompt_toolkit is missing or stdin is not a
terminal keeps piped and ``--prompt`` usage working, at the cost of the paste
behaviour.

Nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import images as IM
import ui

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import run_in_terminal
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    HAVE_PROMPT_TOOLKIT = True
except ImportError:  # pragma: no cover - exercised only on a bare install
    HAVE_PROMPT_TOOLKIT = False

# The input is drawn as a rounded box that erases itself on submit, and the
# REPL then echoes what you said. A box is a *live* control: while it is on
# screen it is where your attention goes, and once the turn is sent it would
# be a stale widget sitting in the scrollback pretending to still be editable.
HINTS = "alt+enter newline · alt+v image · /help"


def _term_width(default: int = 80) -> int:
    """How wide to draw the border.

    Asked of prompt_toolkit rather than the OS, because prompt_toolkit is what
    lays the line out: if the two disagree by even one column the border either
    stops short of the edge or wraps onto a row of its own. ``shutil`` is the
    fallback for when this is called outside a running prompt.

    Recomputed per prompt rather than cached, so a window resized between turns
    redraws correctly.
    """
    try:
        from prompt_toolkit.application.current import get_app

        columns = get_app().output.get_size().columns
        if columns:
            return max(24, columns)
    except Exception:
        pass
    try:
        return max(24, shutil.get_terminal_size((default, 24)).columns)
    except OSError:
        return default


def clean_input(text: str) -> str:
    """Strip invisible junk that would stop a line matching what it looks like.

    Windows PowerShell prepends a UTF-8 BOM when piping a string into a native
    command, so the first line arrives as '\\ufeff/help'. ``strip()`` does not
    remove it -- U+FEFF is not whitespace -- so the leading slash is no longer
    at position 0 and the command is sent to the model as a prompt instead.
    The same applies to a piped 'y' at an approval prompt, which silently reads
    as a denial. Only ever bites the first line of piped input, which is
    exactly the kind of thing that survives to production.
    """
    return text.lstrip("\ufeff\ufffe").replace("\x00", "")


def _resolve_dir(image_dir) -> Path | None:
    """Accept a Path or a zero-arg callable returning one.

    The callable form exists for /savesession: where pastes belong depends on
    whether the conversation is being saved, and that can now change
    mid-session. A Path captured at startup would keep writing to the temp
    directory after the user opted in to persistence -- and those files would
    then be missing on resume.
    """
    return image_dir() if callable(image_dir) else image_dir


def _build_bindings(image_dir=None) -> "KeyBindings":
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:
        """Accept the buffer.

        multiline=True is required so the buffer can hold pasted newlines and
        render them, but its default binding makes Enter insert a newline and
        Alt+Enter submit -- backwards for a chat prompt. Swap them.
        """
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event) -> None:
        """Alt+Enter (sent as ESC then Enter) inserts a literal newline."""
        event.current_buffer.insert_text("\n")

    @kb.add("escape", "v")
    @kb.add("c-v")
    def _paste_image(event) -> None:
        """Save the clipboard image and insert a reference to it.

        A marker rather than the raw bytes: the line stays readable and
        editable, the file is on disk where it can be looked at, and the
        submit-time parser turns it back into an attachment. Feedback goes
        through run_in_terminal so printing does not fight the prompt's own
        redraw and leave the line mangled.
        """
        target = _resolve_dir(image_dir)
        if target is None:
            return
        try:
            path = IM.grab_clipboard(target)
        except IM.ImageError as e:
            run_in_terminal(lambda: print(f"  {e}"))
            return
        except Exception as e:  # a clipboard API that failed in a new way
            run_in_terminal(lambda: print(f"  could not read the clipboard: {e}"))
            return

        if path is None:
            run_in_terminal(
                lambda: print("  no image on the clipboard (copy one, then Alt+V)")
            )
            return

        buf = event.current_buffer
        if buf.text and not buf.text[: buf.cursor_position].endswith((" ", "\n")):
            buf.insert_text(" ")
        buf.insert_text(IM.marker(path) + " ")
        run_in_terminal(lambda: print(f"  attached {path.name}"))

    return kb


class InputReader:
    """Reads a user turn, preserving pasted multi-line text."""

    def __init__(
        self,
        history_path: Path | None = None,
        *,
        image_dir=None,
        _input: object | None = None,
        _output: object | None = None,
    ):
        """``_input``/``_output`` exist so tests can drive a real PromptSession
        through a pipe; passing them also bypasses the isatty() check.

        ``image_dir`` is where clipboard images are written -- a Path, or a
        zero-arg callable returning one so the destination can change
        mid-session (/savesession). Without it the paste-image key is not
        bound at all.
        """
        self.session = None
        self.using_fallback = True
        self.image_dir = image_dir
        # Right-hand side of the hint line under the box. Set per turn by the
        # REPL, read by the toolbar callback when prompt_toolkit repaints.
        self.status = ""

        if not HAVE_PROMPT_TOOLKIT:
            return
        if _input is None and not sys.stdin.isatty():
            return

        try:
            history = InMemoryHistory()
            if history_path is not None:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                history = FileHistory(str(history_path))

            extra = {}
            if _input is not None:
                extra["input"] = _input
            if _output is not None:
                extra["output"] = _output

            self.session = PromptSession(
                message=self._box_head,
                multiline=True,
                key_bindings=_build_bindings(image_dir),
                history=history,
                enable_history_search=True,
                # The 2nd..nth line of a pasted block continues the box's left
                # edge, indented to line up under the first line's text.
                prompt_continuation=self._box_continuation,
                rprompt=self._box_right_edge,
                bottom_toolbar=self._box_foot,
                # The default toolbar style is reverse video, which would put a
                # solid bar where the bottom border should be.
                style=Style.from_dict({"bottom-toolbar": "noreverse"}),
                # The box is a live control; on submit it goes away and the
                # REPL echoes the turn as plain text. See HINTS above.
                erase_when_done=True,
                wrap_lines=True,
                **extra,
            )
            self.using_fallback = False
        except Exception:
            # A hostile or unsupported terminal should degrade, not crash.
            self.session = None
            self.using_fallback = True

    # ----------------------------------------------------------------- the box
    #
    # prompt_toolkit has no frame widget for a plain prompt, so the box is
    # assembled from the four things it *does* let you draw around an input:
    # a multi-line message above it, a continuation prefix on wrapped lines, an
    # rprompt at the right edge, and a bottom toolbar underneath. Each callback
    # re-measures the terminal, so a resize between turns redraws correctly.

    def _box_head(self) -> "ANSI":
        """Top border, then the prompt line's left edge and caret."""
        width = _term_width()
        return ANSI(
            f"{ui.A_DEEP}╭{'─' * (width - 2)}╮{ui.A_RESET}\n"
            f"{ui.A_DEEP}│{ui.A_RESET} {ui.A_ACCENT}>{ui.A_RESET} "
        )

    def _box_continuation(self, width: int, line_no: int, wrap_count: int) -> "ANSI":
        return ANSI(f"{ui.A_DEEP}│{ui.A_RESET}   ")

    def _box_right_edge(self) -> "ANSI":
        """The box's right edge, on the first line of the input.

        prompt_toolkit only offers an rprompt for that one line, so a turn that
        grows past it closes on three sides. Worth having anyway: nearly every
        turn is one line, and the alternative is no right edge ever.
        """
        return ANSI(f"{ui.A_DEEP}│{ui.A_RESET}")

    def _box_foot(self) -> "ANSI":
        """Bottom border, and the hint line beneath the box."""
        width = _term_width()
        left = f"  {HINTS}"
        gap = width - len(left) - len(self.status) - 2
        line = left + (" " * gap + self.status if self.status and gap > 2 else "")
        return ANSI(
            f"{ui.A_DEEP}╰{'─' * (width - 2)}╯{ui.A_RESET}\n"
            f"{ui.A_DIM}{line}{ui.A_RESET}"
        )

    @property
    def can_paste_images(self) -> bool:
        return (
            self.session is not None
            and self.image_dir is not None
            and IM.clipboard_available()
        )

    @property
    def capability_note(self) -> str:
        """Only says anything when input is degraded.

        The keys used to be listed in the banner; they now live on the hint
        line under the box, where they are visible at the moment you need them
        rather than once at startup.
        """
        if self.using_fallback:
            return (
                "plain input -- multi-line paste will be split into separate "
                "prompts (install prompt_toolkit to fix)"
            )
        return ""

    def _fallback(self, prompt: str, default: str) -> str:
        """Plain input(), with any pre-filled text prepended.

        input() cannot seed an editable buffer, so the text is shown and then
        joined to whatever is typed. Less pleasant than a real prefill, but it
        keeps /paste working when prompt_toolkit is unavailable rather than
        silently dropping the image.
        """
        if default:
            print(f"{prompt}{default}")
        typed = clean_input(input(prompt if not default else ""))
        return (default + typed) if default else typed

    async def read_async(self, prompt: str, default: str = "", status: str = "") -> str:
        """Return one user turn. Use this from inside an event loop.

        ``PromptSession.prompt()`` is synchronous and calls ``asyncio.run()``
        internally, which raises "cannot be called from a running event loop"
        when the REPL is already inside one. ``prompt_async`` is the coroutine
        form and is the correct entry point here.

        ``default`` pre-fills the editable buffer -- used by /paste to put an
        image reference in front of the message about to be typed. ``status``
        is shown at the right of the hint line. ``prompt`` is only used by the
        plain-input fallback; the box draws its own caret.
        """
        self.status = status
        if self.session is None:
            # Blocks the loop, which is harmless: nothing else runs while we
            # are waiting for the user to type.
            return self._fallback(prompt, default)
        return clean_input(await self.session.prompt_async(default=default))

    def read(self, prompt: str, default: str = "", status: str = "") -> str:
        """Synchronous read, for callers that are not inside an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "InputReader.read() was called from a running event loop, where "
                "prompt_toolkit cannot start its own. Use read_async() instead."
            )
        self.status = status
        if self.session is None:
            return self._fallback(prompt, default)
        return clean_input(self.session.prompt(default=default))


def summarize_input(text: str, max_line: int = 96) -> str:
    """Describe a multi-line turn compactly for echoing back to the terminal."""
    lines = text.splitlines()
    if len(lines) <= 1:
        return ""
    first = lines[0][:max_line]
    return f"({len(lines)} lines, {len(text)} chars) {first}..."
