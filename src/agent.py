"""A fully offline coding agent driven by a local Ollama model.

    .venv\\Scripts\\python.exe src\\agent.py [workspace_dir]

Nothing here talks to anything but Ollama on this machine; the loopback guard
in ollama_client makes that structural rather than a promise.

The design choice that matters most is the thinking policy. Reasoning models
emit a deliberation block before answering -- measured on this hardware at 263
tokens / 23.5s versus 10 tokens / 1.8s for the same trivial prompt. Paid on
every turn of an agent loop, that is the difference between usable and
unbearable. But turning it off globally makes the model plan badly. So it is
decided per turn: on when the model is deciding what to do, off when it is
carrying out something already decided.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.getLogger("httpx").setLevel(logging.WARNING)

import images as IM  # noqa: E402
import models as M  # noqa: E402
import paths  # noqa: E402
from ollama_client import GenResult, NonLocalHostError, OllamaClient  # noqa: E402
from repl_input import InputReader, clean_input, summarize_input  # noqa: E402
from environment import (  # noqa: E402
    describe_delta, probe_static, probe_volatile, static_block, volatile_block,
)
from session import (  # noqa: E402
    COMPACT_AT, Session, estimate_tokens, find_session, latest_session,
    list_sessions,
)
import transcript as T  # noqa: E402
import ui  # noqa: E402
from tools import ToolRegistry, Workspace  # noqa: E402
from websearch import DEFAULT_SEARXNG, WebSearch  # noqa: E402

DEFAULT_CONTEXT = 32768
MAX_STEPS_PER_TURN = 25

# ANSI codes live in transcript.py; Windows Terminal and VS Code handle these.
from transcript import BLUE, BOLD, DIM, GREEN, RED, RESET, YELLOW  # noqa: E402


SYSTEM_PROMPT = """You are a coding agent working in {workspace}.

{environment}

You have tools. Use them instead of guessing:
- Never claim a file's contents without read_file first.
- Never edit a file without read_file immediately before, so line numbers are current.
- Use search_text and find_files to locate things rather than assuming paths.
- Run tests and commands with run_command to check your work.

Rules:
- Take one step at a time. Call a tool, look at the actual result, then decide.
- If a tool returns ERROR, read why and fix your call. Do not repeat it unchanged.
- Make the smallest edit that does the job. Do not rewrite files wholesale.
- When the task is done, say so plainly and stop calling tools.
- If a request is ambiguous in a way that changes what you would write, call
  ask_user rather than guessing. Ask early, before writing code on a guess --
  a wrong assumption discovered late costs far more than a question. Ask about
  requirements, scope, and choices only the user can make. Do not ask about
  anything you could find out with read_file, search_text or run_command, and
  do not ask about details you could reasonably decide yourself.
{web_guidance}
Be concise. No preamble, no summaries of what you are about to do."""

WEB_GUIDANCE = """
You can search the internet with search_web. Use it rather than guessing:
- If you are unsure of an API signature, a version number, a config key or a
  command, look it up instead of writing something plausible.
- If your knowledge of a library may be out of date, check.
- Say when an answer came from a search, and cite the URL.

Treat search results as evidence, not fact. Results carry the quote they came
from, and are labelled when the source is low-authority or when better sources
could not be read. Before repeating a figure or an API detail, check that it is
plausible and consistent with the quote. If a value looks wrong for its context,
or the only sources are flagged low-quality, say so rather than reporting it
confidently. Reporting an uncertain answer as certain is worse than saying you
could not verify it.

A search takes 30-60 seconds, so ask one specific question rather than several
vague ones. Do not search for things you already know.
"""

NO_WEB_GUIDANCE = """
You have no internet access. If you genuinely do not know something and cannot
determine it from the files or by running a command, say so plainly rather than
guessing at an API or a version.
"""


# Effort levels Ollama's `think` field accepts alongside booleans. A bare
# `true` inherits the model template's default, which on qwen3.8 is its
# MAXIMUM ("xhigh") -- measured on this machine, "low" cut a planning turn's
# reasoning by ~58% and wall time by ~43% with no visible quality loss.
# "default" is the escape hatch: send plain `true` for models whose templates
# reject level strings.
THINK_LEVELS = ("low", "medium", "high", "max")
DEFAULT_THINK_LEVEL = "low"


class ThinkingPolicy:
    """Decides whether to spend a reasoning block on this specific turn,
    and how hard to think when it does.

    Reasoning is worth its cost when choosing what to do next, and wasted when
    executing something already decided. The heuristics below approximate that
    distinction from the shape of the conversation. ``level`` is orthogonal:
    it sets the effort Ollama requests whenever a turn does think.
    """

    def __init__(self, mode: str = "auto", level: str = DEFAULT_THINK_LEVEL):
        self.mode = mode  # auto | always | never
        self.level = level  # low | medium | high | max | default

    def wire_value(self, think: bool) -> bool | str:
        """What actually goes in the request's ``think`` field."""
        if not think:
            return False
        return True if self.level == "default" else self.level

    def decide(self, step: int, last_tool_failed: bool, is_first_step: bool) -> tuple[bool, str]:
        if self.mode == "always":
            return True, "forced on"
        if self.mode == "never":
            return False, "forced off"

        if is_first_step:
            # planning the approach -- the one place it reliably pays
            return True, "planning"
        if last_tool_failed:
            # something went wrong; the model needs to actually reason about why
            return True, "recovering from tool error"
        if step >= 8:
            # long chains tend to drift; a checkpoint helps it re-orient
            return True, "long chain checkpoint"
        return False, "mechanical step"


class StreamPrinter:
    """Writes tokens to the terminal as they arrive.

    At ~20 tok/s a long paragraph still takes many seconds, so batching output
    until the turn completes leaves the terminal looking hung. Reasoning and
    answer are styled differently because they are different things --
    reasoning is progress, the answer is the result.
    """

    def __init__(self, show_reasoning: bool = True):
        self.reasoning = ui.ReasoningStream(show_reasoning)
        self.answer = ui.AnswerStream()
        self._emitted = False

    def _first_output(self) -> None:
        """Clear the elapsed-time ticker before anything real is printed.

        Doing it here rather than in the ticker's own cleanup means the ticker
        can never erase characters we have already written.
        """
        if not self._emitted:
            if ui.console.is_terminal:
                sys.stdout.write("\r" + " " * ui.SPINNER_WIDTH + "\r")
                sys.stdout.flush()
            self._emitted = True

    def on_thinking(self, piece: str) -> None:
        self._first_output()
        self.reasoning.push(piece)

    def on_content(self, piece: str) -> None:
        self._first_output()
        # Reasoning always precedes the answer; close it so the markdown
        # renderer owns the terminal from here.
        self.reasoning.close()
        self.answer.push(piece)

    def finish(self) -> None:
        self.reasoning.close()
        self.answer.close()

    @property
    def silent(self) -> bool:
        """True while nothing has been emitted for this turn.

        Ollama's local daemon buffers tool calls and delivers them in a single
        chunk when generation completes -- measured at 60s of total silence for
        a 238-token write_file. Nothing can be rendered during that window, so
        the caller shows elapsed time instead of a frozen terminal.
        """
        return not self._emitted


async def tick_while_silent(printer: StreamPrinter, label: str = "working") -> None:
    """Animate a spinner with elapsed time until real output starts.

    Cancelled by the caller once the model call returns. The glyph cycles
    because a frozen counter and a hung process look identical, and this model
    can go a minute between the request and its first token.
    """
    # \r overwrites in place only on a terminal. Piped to a file it just appends,
    # turning a one-line counter into hundreds of lines of noise.
    if not ui.console.is_terminal:
        return

    started = time.monotonic()
    try:
        while True:
            # Fast enough that the glyph reads as motion rather than a blink.
            await asyncio.sleep(0.12)
            if not printer.silent:
                return
            sys.stdout.write("\r" + ui.spinner_line(time.monotonic() - started, label))
            sys.stdout.flush()
    except asyncio.CancelledError:
        raise
    finally:
        # Only erase if nothing has been printed over us. Once real output has
        # started, StreamPrinter has already cleared the line and written to
        # it -- erasing again would blank its first characters.
        if printer.silent:
            sys.stdout.write("\r" + " " * ui.SPINNER_WIDTH + "\r")
            sys.stdout.flush()


class Interrupter:
    """Ctrl+C stops the current turn instead of killing the program.

    Two things make this harder than installing a handler:

    * Python only runs a signal handler between bytecodes, and while awaiting a
      model call the interpreter is parked in the event loop. Ollama delivers
      tool calls atomically -- 60s with no chunks at all -- so there can be a
      long window with nothing to interrupt. A heartbeat task keeps the loop
      turning so the signal is seen promptly.
    * Raising KeyboardInterrupt into the middle of a streaming read leaves the
      HTTP connection half-consumed. Cancelling the task instead lets httpx
      unwind properly.

    A second Ctrl+C within two seconds exits, matching what people expect when
    the first one appears not to have worked.
    """

    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.fired = False
        self._previous: dict = {}
        self._last_signal = 0.0
        self._heartbeat: asyncio.Task | None = None

    def _handle(self, signum, frame) -> None:
        now = time.monotonic()
        if self.fired and (now - self._last_signal) < 2.0:
            raise KeyboardInterrupt  # second press: let it propagate and exit
        self._last_signal = now
        self.fired = True
        if self.task is not None and not self.task.done():
            self.task.get_loop().call_soon_threadsafe(self.task.cancel)

    async def _tick(self) -> None:
        # Nothing to do but exist: this keeps the loop scheduling so a pending
        # SIGINT is delivered without waiting for the model call to finish.
        while True:
            await asyncio.sleep(0.15)

    # Ctrl+Break is a separate signal on Windows and is what people reach for
    # when Ctrl+C seems stuck, so catch both.
    _SIGNALS = [signal.SIGINT] + (
        [signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else []
    )

    def arm(self, task: asyncio.Task) -> None:
        self.task = task
        self.fired = False
        self._previous = {}
        for sig in self._SIGNALS:
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass  # not the main thread, or unsupported here
        self._heartbeat = asyncio.create_task(self._tick())

    def disarm(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            self._heartbeat = None
        for sig, handler in (self._previous or {}).items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        self._previous = {}
        self.task = None


def parse_tokens(text: str) -> int | None:
    """Accept 65536, 64k, 128K, 1m -- whichever way someone types a window."""
    text = text.strip().lower().replace(",", "").replace("_", "")
    if not text:
        return None
    mult = 1
    if text.endswith("k"):
        mult, text = 1024, text[:-1]
    elif text.endswith("m"):
        mult, text = 1024 * 1024, text[:-1]
    try:
        value = int(float(text) * mult)
    except ValueError:
        return None
    return value if value > 0 else None


def fmt_args(args: dict[str, Any], limit: int = 90) -> str:
    parts = []
    for k, v in (args or {}).items():
        s = str(v).replace("\n", "\\n")
        if len(s) > limit:
            s = s[:limit] + f"...+{len(str(v)) - limit}ch"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        workspace: Workspace,
        *,
        context_length: int = DEFAULT_CONTEXT,
        thinking: str = "auto",
        think_level: str = DEFAULT_THINK_LEVEL,
        auto_approve: bool = False,
        show_reasoning: bool = True,
        web: Any = None,
        max_steps: int = MAX_STEPS_PER_TURN,
        interactive: bool = True,
        supports_vision: bool = False,
        save: bool = False,
    ):
        self.max_steps = max_steps
        self.client = client
        self.model = model
        self.workspace = workspace
        self.context_length = context_length
        self.tools = ToolRegistry(workspace)
        self.policy = ThinkingPolicy(thinking, think_level)
        self.auto_approve = auto_approve
        self.show_reasoning = show_reasoning
        self.web = web
        self.supports_vision = supports_vision
        self.save = save
        self.interrupter = Interrupter()

        # Only offer to ask questions when somebody is there to answer. In a
        # --prompt or piped run the tool still exists but returns immediately
        # telling the model to assume and continue, rather than blocking on an
        # input() that will never be satisfied.
        self.tools.add_ask_tool(interactive=interactive)

        if web is not None and web.enabled:
            self.tools.add_web_tools(web)

        # Session-lifetime facts go in the system prompt: they never change, so
        # they never invalidate the prompt cache. Volatile facts are injected at
        # the tail of the conversation instead -- see _inject_environment.
        self.static_facts = probe_static(workspace.root)
        self._env_snapshot: dict[str, Any] = {}
        self._env_injected = False

        self.session = Session(
            model=model,
            context_limit=context_length,
            system_prompt=SYSTEM_PROMPT.format(
                workspace=workspace.root,
                environment=static_block(self.static_facts),
                web_guidance=(
                    WEB_GUIDANCE if (web is not None and web.enabled) else NO_WEB_GUIDANCE
                ),
            ),
            # Not derived from __file__: an installed build runs from Program
            # Files, which is read-only. See paths.py.
            sessions_dir=paths.sessions_dir(),
            persist=save,
        )
        # Tool schemas are resent on every request and count toward the prompt,
        # but they live outside the message list -- so the budget has to be told
        # about them explicitly or it under-counts before the first response.
        self.session.tool_overhead_tokens = estimate_tokens(
            json.dumps(self.tools.ollama_schemas())
        )

    # ------------------------------------------------------------- approvals

    def render_call(self, tool_name: str, args: dict[str, Any]) -> None:
        """Show what is about to run, before it runs.

        Always rendered -- including under --yes -- so there is a record of what
        was written. For a file write the content IS the decision, and since
        Ollama delivers tool calls atomically this is the first moment it can be
        shown at all; a one-line truncation would mean approving a 300-line
        overwrite sight unseen.
        """
        body = args.get("content") or args.get("new_text") or ""
        if tool_name in ("write_file", "edit_lines") and body:
            span = (
                f"lines {args.get('start_line')}-{args.get('end_line')}"
                if tool_name == "edit_lines"
                else ""
            )
            ui.file_write(tool_name, args.get("path", "?"), body, span)
        else:
            ui.tool_call(tool_name, fmt_args(args))

    def approve(self, tool_name: str, args: dict[str, Any]) -> bool:
        if self.auto_approve:
            return True
        tool = self.tools.get(tool_name)
        if tool is None or not tool.mutating:
            return True

        target = args.get("path") or args.get("command") or ""
        title = f"allow {tool_name}?" + (f"  {str(target)[:60]}" if target else "")
        try:
            answer = clean_input(ui.approval(title)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        # Numbers match the box; the letters are what anyone who has used this
        # before will type, and what a piped 'y' sends.
        if answer in ("2", "a"):
            self.auto_approve = True
            ui.note("auto-approving for the rest of this session")
            return True
        return answer in ("1", "y", "yes")

    # ------------------------------------------------------------- main loop

    async def set_context(self, requested: int) -> None:
        """Change the context window mid-session.

        Two values have to move together: the budget compaction watches, and the
        num_ctx sent to Ollama. Changing num_ctx makes Ollama reload the model,
        and the KV cache grows with the window out of the same VRAM as the
        weights -- on a card that cannot already hold the model, a larger window
        pushes more of it onto the CPU. So this reloads immediately and reports
        the resulting GPU split rather than letting the cost show up later as
        unexplained slowness.
        """
        caps = await M.model_capabilities(self.client, self.model)
        ceiling = caps.get("max_context") or requested
        target = min(requested, ceiling)
        if target != requested:
            ui.warn(f"capped at the model's maximum of {ceiling:,}")

        before = self.context_length
        if target == before:
            ui.note(f"already {before:,} tokens")
            return

        used = self.session.used_tokens()
        if target < used:
            ui.warn(
                f"{target:,} is below current usage ({used:,}); "
                f"the next turn will compact immediately"
            )

        self.context_length = target
        self.session.context_limit = target
        if self.web is not None:
            self.web.context_length = target

        ui.note(f"context {before:,} → {target:,} tokens · reloading the model…")
        try:
            await self.client.load(self.model, context_length=target, keep_alive="30m")
            loaded = await M.loaded_models(self.client)
        except Exception as e:
            ui.error(f"reload failed: {type(e).__name__}: {e}")
            return

        for entry in loaded:
            if entry["name"].startswith(self.model.split(":")[0]):
                pct = entry.get("gpu_percent", 0)
                line = (
                    f"{entry['in_vram']} of {entry['total']} in VRAM "
                    f"({pct}% GPU), context {entry.get('context')}"
                )
                if pct < 100:
                    ui.warn(f"{line} — {100 - pct}% on CPU, generation will be slower")
                else:
                    ui.note(line)

    def load_session(self, path: Path) -> str:
        """Replace the current conversation with a saved one and continue it."""
        self.session = Session.resume(
            path,
            model=self.model,
            context_limit=self.context_length,
            # current system prompt, not the one from the old run -- the
            # environment block in it describes today, not that day
            system_prompt=self.session.system_prompt,
            persist=self.save,
        )
        self.session.tool_overhead_tokens = estimate_tokens(
            json.dumps(self.tools.ollama_schemas())
        )
        # Force a fresh environment block: the date, branch and working tree
        # have almost certainly moved since this conversation was written.
        self._env_injected = False
        self._env_snapshot = {}

        # Attachments are stored by path, so a resumed session re-reads them.
        # Say when one has gone: the conversation still refers to it.
        if missing := self.session.missing_images:
            ui.warn(f"{len(missing)} attached image(s) could not be re-read:")
            for p in missing[:3]:
                ui.note(f"  {p}")

        where = (
            "Continuing in the same file."
            if self.save
            else "Loaded read-only -- this conversation is not being saved, so "
            "nothing further is written to it (--save to change that)."
        )
        # Compaction itself is deliberately left until the next turn actually
        # needs it -- resuming a session just to read it with /history should
        # not spend a minute of local inference summarising it. But say now
        # that it is coming, so the wait is expected rather than mysterious.
        if self.session.needs_compaction():
            ui.warn(
                f"this conversation is {self.session.budget_line()} — it will be "
                f"compacted automatically when you send your next message"
            )

        return (
            f"Resumed {path.stem} -- {len(self.session.messages)} messages, "
            f"{self.session.budget_line()}. {where}"
        )

    def change_workspace(self, raw: str) -> str:
        """Move the agent to a different working directory, mid-conversation.

        Three things are bound to the workspace and must move together: the
        Workspace the tools resolve against, the static environment block in
        the system prompt (which names the root and its project facts), and
        the volatile snapshot. Changing the system prompt does invalidate
        Ollama's cached prefix -- a one-time reprocess of the conversation on
        the next turn, which is the honest price of not lying to the model
        about where it is working.

        A relative path is taken relative to the current workspace, so
        ``/folder ..`` and ``/folder sibling`` behave the way a shell user
        expects.
        """
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = self.workspace.root / target
        new_root = self.workspace.change_root(target)  # ValueError if invalid

        self.static_facts = probe_static(new_root)
        self.session.system_prompt = SYSTEM_PROMPT.format(
            workspace=new_root,
            environment=static_block(self.static_facts),
            web_guidance=(
                WEB_GUIDANCE
                if (self.web is not None and self.web.enabled)
                else NO_WEB_GUIDANCE
            ),
        )
        # The old snapshot described the old folder; force the full volatile
        # block for the new one onto the next turn.
        self._env_injected = False
        self._env_snapshot = {}

        return f"Successfully moved to {new_root}"

    def reset_conversation(self) -> str:
        """Start a genuinely fresh conversation.

        A new session file, fresh token accounting, and the environment block
        re-injected on the next turn. Clearing only the message list left the
        old measurement driving compaction, the old file collecting new turns,
        and the model never re-told what machine it is on.
        """
        self.session = Session(
            model=self.model,
            context_limit=self.context_length,
            system_prompt=self.session.system_prompt,
            sessions_dir=self.session.sessions_dir,
            persist=self.save,
        )
        self.session.tool_overhead_tokens = estimate_tokens(
            json.dumps(self.tools.ollama_schemas())
        )
        self._env_injected = False
        self._env_snapshot = {}
        if self.save:
            return f"conversation cleared — continuing in {self.session.path.name}"
        return "conversation cleared"

    def _record_interruption(self, printer: "StreamPrinter") -> None:
        """Leave the conversation coherent after a stop.

        Whatever streamed is kept as the assistant's turn, followed by a note
        that the user cut it off. Without the note the model sees a truncated
        reply and tends to resume mid-sentence, which is precisely what you
        interrupted it to prevent.
        """
        partial = printer.answer.buffer
        if partial.strip():
            self.session.append(
                {"role": "assistant", "content": partial.rstrip() + " …[cut off]"}
            )
        self.session.append(
            {
                "role": "user",
                "content": (
                    "[I stopped you mid-response. Do not continue that answer or "
                    "repeat it. Wait for my next instruction.]"
                ),
            }
        )

    def _answer_dangling_calls(self, note: str) -> None:
        """Append synthetic results for any unanswered tool calls at the tail.

        The wire protocol pairs every assistant tool_call with a tool message.
        An interrupt can leave that pair half-written in the *live* list, and
        trim_dangling_tool_calls only repairs the on-disk copy at resume time;
        without this the model is sent its own unanswered request and either
        repeats the call or invents the result.
        """
        msgs = self.session.messages
        i = len(msgs) - 1
        answered = 0
        while i >= 0 and msgs[i].get("role") == "tool":
            answered += 1
            i -= 1
        if i < 0 or msgs[i].get("role") != "assistant":
            return
        for tc in (msgs[i].get("tool_calls") or [])[answered:]:
            fn = tc.get("function") or {}
            self.session.append(
                {"role": "tool", "tool_name": fn.get("name", ""), "content": note}
            )

    def _inject_environment(self) -> None:
        """Put volatile environment facts at the tail of the conversation.

        The first turn gets the full block. After that only a one-line delta,
        and only when something actually moved -- most turns produce nothing and
        cost nothing. Placing this at the tail rather than rebuilding the system
        prompt is what keeps the cached prefix intact; a changed system prompt
        would force a full reprocess of the conversation every single turn.
        """
        snap = probe_volatile(self.workspace.root)

        if not self._env_injected:
            self.session.append(
                {"role": "user", "content": volatile_block(snap, self.workspace.root)}
            )
            self._env_injected = True
        else:
            delta = describe_delta(self._env_snapshot, snap)
            if delta:
                self.session.append({"role": "user", "content": delta})
                print(f"{DIM}  {delta}{RESET}")

        self._env_snapshot = snap

    # ----------------------------------------------------------- attachments

    def status_line(self) -> str:
        """The right-hand side of the hint line under the input box.

        Context pressure is the number that decides whether the next turn is
        going to compact, and it is the one thing you want in front of you
        while deciding what to type -- not buried behind /tokens.
        """
        pct = 100 * self.session.used_tokens() / max(1, self.session.context_limit)
        return f"{self.model}  ·  {pct:.0f}% context"

    @property
    def images_dir(self) -> Path:
        """Where clipboard pastes are written.

        Beside the sessions they belong to when saving, because a pasted
        screenshot is part of the conversation and a resumed session has to be
        able to read it back. When not saving there is no conversation to read
        back, so it goes to the temp directory instead -- the image still has
        to exist as a file for the moment it is encoded, but it has no business
        outliving the run in a folder the user did not choose.
        """
        if self.save:
            return self.session.sessions_dir / "pasted"
        return Path(tempfile.gettempdir()) / "OffTheWire" / "pasted"

    def _attach_images(self, user_input: str) -> tuple[str, list[IM.Attachment]]:
        """Turn image references in a typed line into attachments.

        Returns the text to send and the images to send with it. A file that
        will not load, or a model that cannot see, degrades to leaving the path
        in the text and saying so -- never to silently dropping the reference,
        which would leave the model answering about a picture it was never
        shown and had no way to know about.
        """
        found = IM.extract(user_input, base=self.workspace.root)
        if not found.has_images:
            return user_input, []

        attachments, errors = IM.load_all(found.refs)
        for message in errors:
            ui.warn(message)

        if attachments and not self.supports_vision:
            ui.warn(
                f"{self.model} has no vision capability, so the image cannot be "
                f"sent — only its path. Try a vision model: "
                f"ollama pull qwen2.5vl:7b"
            )
            attachments = []

        if attachments:
            ui.attachments([a.describe() for a in attachments])
            big = sum(len(a.data) for a in attachments)
            if big > IM.LARGE_IMAGE_BYTES:
                ui.note(
                    f"{IM.human_bytes(big)} of image is re-sent with every turn "
                    f"it stays in context; /clear when you are done with it"
                )

        return found.render([a.path for a in attachments]), attachments

    # ---------------------------------------------------------------- a turn

    async def run_turn(self, user_input: str) -> None:
        text, attachments = self._attach_images(user_input)

        self._inject_environment()
        message: dict[str, Any] = {"role": "user", "content": text}
        if attachments:
            message["images"] = [a.b64 for a in attachments]
            message["image_paths"] = [str(a.path) for a in attachments]
        self.session.append(message)

        last_tool_failed = False
        compactions_this_turn = 0
        # Signatures of tool calls made this turn, to catch the model repeating
        # itself. Hitting the step cap is usually the symptom; a loop is the
        # disease, and 25 identical calls is 25 wasted minutes on a local model.
        # Each signature maps to (repeats, mutations-count when last seen): an
        # identical call only counts as a repeat if nothing changed state in
        # between, so edit → test → edit → test loops are never blocked.
        call_counts: dict[str, tuple[int, int]] = {}
        mutations = 0
        for step in range(1, self.max_steps + 1):
            # Before building the request, never after sending it.
            #
            # This check used to sit at the foot of the loop, which had two
            # holes. A resumed conversation is over budget from its very first
            # message -- one saved session here measured 19x the window -- so
            # the opening request went out oversized and Ollama silently
            # truncated it, which is the failure this whole module exists to
            # prevent. And a reply with no tool calls breaks out of the loop
            # before reaching the foot, so a session could stay over budget
            # indefinitely and only /compact by hand would rescue it.
            #
            # At the top it covers both, and still covers the original case:
            # checking before step N+1 is the same moment as checking after
            # step N.
            if self.session.needs_compaction() and compactions_this_turn < 2:
                compactions_this_turn += 1
                print(f"{YELLOW}  context {self.session.budget_line()} -- compacting{RESET}")
                print(f"{DIM}  {await self.session.compact(self.client)}{RESET}")
                # Compaction summarizes older turns away, which can take the
                # environment block with it. Re-inject on the next turn.
                self._env_injected = False

            think, why = self.policy.decide(
                step=step, last_tool_failed=last_tool_failed, is_first_step=(step == 1)
            )
            think_wire = self.policy.wire_value(think)

            mode = (
                f"think:{self.policy.level}" if think and self.policy.level != "default"
                else ("thinking" if think else "direct")
            )
            ui.step_header(
                step, mode, why,
                self.session.used_tokens(), self.session.context_limit,
            )

            printer = StreamPrinter(show_reasoning=self.show_reasoning)
            # The spinner says which kind of turn you are waiting on, since a
            # reasoning turn is legitimately several times longer and knowing
            # that is the difference between patience and reaching for Ctrl+C.
            ticker = asyncio.create_task(
                tick_while_silent(printer, "thinking" if think else "working")
            )
            call = asyncio.create_task(
                self.client.chat_stream(
                    self.model,
                    self.session.wire_messages(),
                    tools=self.tools.ollama_schemas(),
                    think=think_wire,
                    context_length=self.context_length,
                    on_content=printer.on_content,
                    on_thinking=printer.on_thinking,
                )
            )
            self.interrupter.arm(call)
            try:
                result: GenResult = await call
            except asyncio.CancelledError:
                printer.finish()
                self._record_interruption(printer)
                ui.warn("stopped. The conversation is kept — tell it what to do instead.")
                return
            except Exception as e:
                printer.finish()
                ui.error(f"model call failed: {type(e).__name__}: {e}")
                return
            finally:
                self.interrupter.disarm()
                ticker.cancel()
                try:
                    await ticker
                except asyncio.CancelledError:
                    pass
            printer.finish()

            self.session.note_actual(result.prompt_tokens)
            ui.stats_line(result.gen_tokens, result.gen_tps, result.total_s)
            if result.done_reason == "length":
                ui.warn(
                    "the reply hit the token/context limit and is cut off "
                    "mid-thought — /maxtokens to raise the window"
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content,
            }
            if result.tool_calls:
                assistant_msg["tool_calls"] = result.tool_calls
            self.session.append(assistant_msg)

            if not result.tool_calls:
                # the answer already streamed to the terminal; don't reprint it
                if not result.content.strip():
                    ui.note("(no output)")
                break

            last_tool_failed = False
            interrupted = False
            for call in result.tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or {}
                # some builds hand back a JSON string rather than an object
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}
                # valid JSON is not necessarily an object ("[]", "\"x\"");
                # anything but a dict would crash the first args.get() below
                if not isinstance(raw_args, dict):
                    raw_args = {}

                signature = f"{name}:{json.dumps(raw_args, sort_keys=True, default=str)}"
                prev_repeats, prev_state = call_counts.get(signature, (0, -1))
                repeats = prev_repeats + 1 if prev_state == mutations else 1
                call_counts[signature] = (repeats, mutations)

                self.render_call(name, raw_args)
                if repeats >= 3:
                    # Nothing has changed state since the identical earlier
                    # calls, so running it again would return the same thing.
                    # Say so plainly instead, which is information the model
                    # does not currently have.
                    output = (
                        f"ERROR: you have called {name} with identical arguments "
                        f"{repeats} times with nothing changing in between, so it "
                        f"would return the same result again. Either try a materially "
                        f"different approach, or stop and explain what is blocking you."
                    )
                    ui.warn(f"repeated call #{repeats} — not re-running")
                    last_tool_failed = True
                elif not self.approve(name, raw_args):
                    output = "ERROR: the user declined this action. Try a different approach or ask them."
                    ui.denied(name)
                else:
                    try:
                        output = await self.tools.call(name, raw_args)
                    except KeyboardInterrupt:
                        # Ctrl+C landed while the tool was running -- the
                        # interrupter only guards the model call. The call
                        # still gets an answer below, because an assistant
                        # message whose tool_calls have no results is exactly
                        # the dangling state that derails the next turn.
                        output = (
                            "ERROR: the user interrupted this call before it "
                            "finished. Do not assume it ran. Wait for their "
                            "next instruction."
                        )
                        interrupted = True
                    ui.tool_result(output)
                    if not interrupted:
                        tool = self.tools.get(name)
                        if tool is not None and tool.mutating:
                            mutations += 1

                if output.startswith("ERROR"):
                    last_tool_failed = True

                self.session.append(
                    {"role": "tool", "tool_name": name, "content": output}
                )
                if interrupted:
                    break

            if interrupted:
                self._answer_dangling_calls(
                    "ERROR: skipped — the user interrupted this turn."
                )
                ui.warn(
                    "stopped during a tool call. The conversation is kept — "
                    "tell it what to do instead."
                )
                return

            # The compaction check now happens at the top of the next
            # iteration, which is the same moment as here but also catches the
            # cases this position could not. `compactions_this_turn` still caps
            # it per turn: the stale-measurement fix in Session is what
            # actually prevents a compaction loop, but a runaway here would
            # burn minutes of local inference before anyone noticed.
        else:
            stuck = [s for s, (n, _) in call_counts.items() if n >= 3]
            ui.warn(f"stopped after {self.max_steps} steps — the task is unfinished.")
            if stuck:
                ui.note(f"it repeated {len(stuck)} call(s) without making progress:")
                for s in stuck[:3]:
                    ui.note(f"  {s.split(':', 1)[0]}  ×{call_counts[s][0]}")
            ui.note(
                "The conversation is intact — say 'continue' to pick up from here, "
                f"or /maxsteps {self.max_steps * 2} to allow more."
            )


# ------------------------------------------------------------------ startup


async def pick_model(client: OllamaClient) -> str | None:
    rec = await M.recommend_agent_model(client)
    candidates = rec.get("candidates") or []
    if not candidates:
        print(f"{RED}{rec.get('reason')}{RESET}")
        return None

    print(f"\n{BOLD}Tool-capable models on this machine:{RESET}")
    for i, c in enumerate(candidates, 1):
        marker = "*" if c["name"] == rec["recommended"] else " "
        # max_context is None when the model card reports no context length;
        # feeding None to a numeric format spec is a crash, not a display.
        ctx = f"{c['max_context']:,}" if c.get("max_context") else "?"
        print(f"  {marker} {i}. {c['name']:28} {c['parameters']:>8}  ctx {ctx}")
    print(f"{DIM}  * = recommended: {rec.get('reason')}{RESET}")

    if len(candidates) == 1:
        return candidates[0]["name"]
    try:
        choice = input(f"\nmodel [1-{len(candidates)}, enter for recommended]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice:
        return rec["recommended"]
    try:
        return candidates[int(choice) - 1]["name"]
    except (ValueError, IndexError):
        return rec["recommended"]


HELP = f"""{BOLD}commands{RESET}
  /help              this
  /save              is this conversation being written to disk?
  /model             show current model and its capabilities
  /think <arg>       when: auto | always | never · effort: low | medium | high | max | default
  /reasoning         toggle streaming the model's reasoning to the terminal
  /approve           toggle auto-approval of file writes and shell commands
  /env               what the model has been told about this machine
  /paste             attach the image on the clipboard (or press Alt+V)
  /folder [path]     show or change the directory the agent works in
  /web               internet lookup status and queries made this session
  /maxtokens [n]     show or change the context window, e.g. /maxtokens 64k
  /maxsteps [n]      show or change the tool-call limit per turn
  /tokens            context usage right now
  /compact           summarize older turns to free context
  /history [full]    replay this conversation ("full" adds tool calls/results)
  /sessions          list saved sessions
  /resume [id]       continue a saved session (no id = the most recent)
  /clear             start a fresh conversation
  /quit              exit

Anything else is sent to the model as a request.

{BOLD}images{RESET}
  Alt+V              paste a screenshot straight into the line
  drag a file in     the path it types is recognised as an attachment
  or type a path     any .png/.jpg/.gif/.webp that exists is attached"""


async def repl(agent: Agent) -> None:
    # The input history is a file of every prompt you have typed, which is
    # the same material as the transcript itself -- so it follows the same rule
    # rather than quietly persisting when the conversation does not.
    reader = InputReader(
        agent.session.sessions_dir / ".input_history" if agent.save else None,
        image_dir=agent.images_dir,
    )
    ui.banner(
        model=agent.model,
        workspace=agent.workspace.root,
        ctx=agent.context_length,
        tools=agent.tools.names,
        session=(
            agent.session.path.name if agent.save
            else "not saved  (--save to keep it)"
        ),
        input_note=reader.capability_note,
        web=(agent.web.searxng_url if (agent.web and agent.web.enabled) else None),
    )

    # A colored prompt is escape-code junk when stdout is a pipe or a file.
    # Only reached by the plain-input fallback; the box draws its own caret.
    prompt = f"{BLUE}> {RESET}" if ui.console.is_terminal else "> "
    # Text /paste leaves in the buffer for the next read, so the image
    # reference and the message about it are composed as one line.
    prefill = ""
    while True:
        try:
            line = (
                await reader.read_async(prompt, default=prefill, status=agent.status_line())
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        finally:
            prefill = ""
        if os.environ.get("AGENT_DEBUG_INPUT"):
            print(f"{DIM}  [debug] read {line!r}{RESET}")
        if not line:
            continue

        # The box erases itself on submit, so the turn has to be echoed or it
        # leaves no trace in the scrollback. Not in fallback mode, where the
        # terminal has already echoed what was typed.
        if not reader.using_fallback:
            ui.user_line(line)

        # A pasted block scrolls past; confirm what was actually captured.
        if note := summarize_input(line):
            print(f"{DIM}  {note}{RESET}")

        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            cmd, arg = cmd.lower(), arg.strip()

            if cmd in ("quit", "exit", "q"):
                return
            if cmd == "help":
                print(HELP)
            elif cmd == "model":
                print(json.dumps(await M.model_capabilities(agent.client, agent.model), indent=2))
            elif cmd == "think":
                if arg in ("auto", "always", "never"):
                    agent.policy.mode = arg
                    print(f"{DIM}thinking: {agent.policy.mode}, effort {agent.policy.level}{RESET}")
                elif arg in THINK_LEVELS or arg == "default":
                    agent.policy.level = arg
                    if arg == "default":
                        print(f"{DIM}thinking effort: the model template's default "
                              f"(on qwen3.8 that is its maximum){RESET}")
                    else:
                        print(f"{DIM}thinking: {agent.policy.mode}, effort {agent.policy.level}{RESET}")
                elif not arg:
                    print(f"{DIM}thinking: {agent.policy.mode}, effort {agent.policy.level}\n"
                          f"when:   /think auto|always|never\n"
                          f"effort: /think {'|'.join(THINK_LEVELS)}|default{RESET}")
                else:
                    print(f"{DIM}unknown: {arg!r}. Use auto|always|never to set when, "
                          f"{'|'.join(THINK_LEVELS)}|default to set effort.{RESET}")
            elif cmd == "reasoning":
                agent.show_reasoning = not agent.show_reasoning
                state = "shown" if agent.show_reasoning else "hidden (dots only)"
                print(f"{DIM}reasoning: {state}{RESET}")
            elif cmd == "approve":
                agent.auto_approve = not agent.auto_approve
                state = "ON (no prompts)" if agent.auto_approve else "OFF (will ask)"
                print(f"{DIM}auto-approve: {state}{RESET}")
            elif cmd == "env":
                print(f"{DIM}--- static (system prompt, cached) ---{RESET}")
                print(static_block(agent.static_facts))
                print(f"{DIM}--- volatile (re-probed each turn) ---{RESET}")
                print(volatile_block(probe_volatile(agent.workspace.root), agent.workspace.root))
            elif cmd == "save":
                # Read-only on purpose. Flipping it mid-conversation would
                # write out turns the user typed while it was off, which is the
                # opposite of what someone checking this would want.
                if agent.save:
                    ui.key_values([
                        ("saving", "yes"),
                        ("file", str(agent.session.path)),
                        ("images", str(agent.images_dir)),
                    ])
                else:
                    ui.note("this conversation is not being saved")
                    ui.note("nothing is written to disk: no transcript, no input")
                    ui.note("history, no pasted images kept after the run")
                    ui.note("restart with --save to keep it")
            elif cmd == "paste":
                # The same job Alt+V does, for terminals that swallow the key
                # or a build without prompt_toolkit. Both end at the same
                # place: a marker in the line, parsed when the turn is sent.
                try:
                    path = IM.grab_clipboard(agent.images_dir)
                except IM.ImageError as e:
                    ui.error(str(e))
                    path = None
                if path is None:
                    ui.note("no image on the clipboard — copy one and try again")
                else:
                    prefill = IM.marker(path) + " "
                    ui.note(f"attached {path.name} — now type your message")
            elif cmd == "web":
                if agent.web is None:
                    print(f"{DIM}no web backend configured{RESET}")
                else:
                    print(json.dumps(await agent.web.status(), indent=2))
                    if agent.web.queries_made:
                        print(f"{DIM}queries this session:{RESET}")
                        for q in agent.web.queries_made:
                            print(f"  {q}")
            elif cmd in ("maxtokens", "context", "ctx"):
                if not arg:
                    caps = await M.model_capabilities(agent.client, agent.model)
                    used = agent.session.used_tokens()
                    limit = agent.session.context_limit
                    max_ctx = caps.get("max_context")
                    ui.key_values([
                        ("window", f"{limit:,} tokens"),
                        ("model max", f"{max_ctx:,}" if max_ctx else "?"),
                        ("in use", f"{used:,}  ({100 * used / max(1, limit):.0f}%)"),
                        ("compacts at", f"{int(limit * COMPACT_AT):,}  ({COMPACT_AT:.0%})"),
                    ])
                    ui.note("/maxtokens 64k   to change it")
                else:
                    target = parse_tokens(arg)
                    if target is None:
                        ui.error(f"could not read {arg!r} as a token count (try 64k)")
                    else:
                        await agent.set_context(target)
            elif cmd == "maxsteps":
                if not arg:
                    ui.note(f"{agent.max_steps} tool calls per turn")
                elif arg.isdigit() and int(arg) > 0:
                    agent.max_steps = int(arg)
                    ui.note(f"max steps per turn: {agent.max_steps}")
                else:
                    ui.error(f"expected a positive number, got {arg!r}")
            elif cmd == "tokens":
                print(f"{DIM}{agent.session.budget_line()}, {agent.session.compactions} compaction(s){RESET}")
            elif cmd == "compact":
                print(f"{DIM}{await agent.session.compact(agent.client)}{RESET}")
            elif cmd == "history":
                if not agent.session.messages:
                    print(f"{DIM}nothing yet in this session{RESET}")
                else:
                    print(T.render(agent.session.messages, full=arg.startswith("f")))
                    if agent.save:
                        print(f"{DIM}saved at {agent.session.path}{RESET}")
                    else:
                        print(f"{DIM}this conversation is not being saved{RESET}")
            elif cmd == "sessions":
                rows = list_sessions(agent.session.sessions_dir)[:15]
                # A session with no turns yet has never been written to disk,
                # so it cannot appear in a listing of files. Synthesize it
                # rather than touching an empty file on every launch and
                # littering the directory.
                if agent.save and not any(
                    r["id"] == agent.session.session_id for r in rows
                ):
                    rows.insert(0, {
                        "id": agent.session.session_id,
                        "when": "just now",
                        # same counting rule as the saved rows: user turns,
                        # excluding injected environment blocks and notes
                        "messages": T.count_turns(agent.session.messages),
                        "first": "",
                    })
                if not rows:
                    ui.note("no saved sessions yet")
                else:
                    ui.session_table(
                        rows, current=agent.session.session_id
                    )
                    ui.note("/resume <id> to continue one (prefix is enough)")
                if not agent.save:
                    # Otherwise the listing reads as though this conversation
                    # is in it somewhere, just scrolled off.
                    ui.note("this conversation is not being saved (--save to keep it)")
            elif cmd == "resume":
                sdir = agent.session.sessions_dir
                path = latest_session(sdir) if not arg else find_session(sdir, arg)
                if path is None:
                    print(f"{RED}no session matching {arg!r}. /sessions to list.{RESET}")
                elif path.stem == agent.session.session_id:
                    print(f"{DIM}already in {path.stem}{RESET}")
                else:
                    print(f"{DIM}{agent.load_session(path)}{RESET}")
            elif cmd == "folder":
                if not arg:
                    ui.note(f"working in {agent.workspace.root} — /folder <path> to change")
                else:
                    try:
                        print(f"{GREEN}{agent.change_workspace(arg)}{RESET}")
                    except ValueError as e:
                        ui.error(str(e))
            elif cmd == "clear":
                print(f"{DIM}{agent.reset_conversation()}{RESET}")
            else:
                print(f"{DIM}unknown command. /help{RESET}")
            continue

        try:
            await agent.run_turn(line)
        except KeyboardInterrupt:
            # Safety net for an interrupt that escaped run_turn's own handling:
            # make sure no tool call is left unanswered before the next prompt.
            agent._answer_dangling_calls(
                "ERROR: the user interrupted this turn before the call completed."
            )
            print(f"\n{YELLOW}  interrupted{RESET}\n")


async def main() -> int:
    ap = argparse.ArgumentParser(description="Offline coding agent on local Ollama models.")
    ap.add_argument("workspace", nargs="?", default=".", help="Directory the agent may work in.")
    ap.add_argument("--model", help="Skip the picker and use this model.")
    ap.add_argument("--context", type=int, default=DEFAULT_CONTEXT, help="Context window in tokens.")
    ap.add_argument("--think", choices=["auto", "always", "never"], default="auto")
    ap.add_argument(
        "--think-level",
        choices=[*THINK_LEVELS, "default"],
        default=DEFAULT_THINK_LEVEL,
        help="Reasoning effort when a turn does think. qwen3.8's own default "
        "is its maximum, which overthinks badly on agent work; 'default' "
        "sends a plain boolean for models that reject effort levels. "
        f"Default {DEFAULT_THINK_LEVEL}.",
    )
    ap.add_argument("--yes", action="store_true", help="Auto-approve writes and commands.")
    ap.add_argument(
        "--save",
        action="store_true",
        help="Write this conversation to disk so it can be resumed later. Off "
        "by default: nothing about the conversation, including the input "
        "history and pasted images, touches the disk unless you ask.",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_PER_TURN,
        help=f"Tool calls allowed per turn, default {MAX_STEPS_PER_TURN}.",
    )
    ap.add_argument(
        "--hide-reasoning",
        action="store_true",
        help="Show progress dots instead of streaming the reasoning block.",
    )
    ap.add_argument("--prompt", help="Run one prompt non-interactively and exit.")
    ap.add_argument(
        "--resume",
        nargs="?",
        const="__latest__",
        metavar="ID",
        help="Continue a saved session. Bare --resume takes the most recent; "
        "otherwise pass a session id from /sessions (a prefix works).",
    )
    ap.add_argument(
        "--web",
        action="store_true",
        help="Enable internet lookup via SearXNG. This is the ONLY thing that "
        "causes outbound traffic; off by default.",
    )
    ap.add_argument(
        "--searxng",
        default=DEFAULT_SEARXNG,
        help=f"SearXNG base URL, default {DEFAULT_SEARXNG}.",
    )
    args = ap.parse_args()

    try:
        client = OllamaClient()
    except NonLocalHostError as e:
        print(f"{RED}{e}{RESET}")
        return 2

    if not await client.ping():
        print(f"{RED}Ollama is not reachable at {client.host}. Start it and retry.{RESET}")
        return 2

    model = args.model or await pick_model(client)
    if not model:
        return 2

    # A typo'd --model used to surface as a raw httpx 404 traceback. Tolerable
    # when you have the source open; not when the program arrived as an
    # installed .exe and the user has no idea what httpx is.
    try:
        caps = await M.model_capabilities(client, model)
    except Exception as e:
        print(f"{RED}Could not read the model {model!r}: {type(e).__name__}{RESET}")
        installed = [m["name"] for m in await M.list_models(client)]
        if installed:
            print(f"{DIM}Installed models:{RESET}")
            for name in installed[:15]:
                print(f"{DIM}  {name}{RESET}")
        else:
            print(f"{DIM}No models are installed. Try: ollama pull qwen3.8:27b{RESET}")
        return 2

    if not caps["supports_tools"]:
        print(f"{RED}{model} does not support tool calling; it cannot drive an agent loop.{RESET}")
        return 2

    ctx = min(args.context, caps["max_context"] or args.context)

    try:
        workspace = Workspace(args.workspace)
    except ValueError as e:
        print(f"{RED}{e}{RESET}")
        return 2

    web = WebSearch(
        client, model,
        searxng_url=args.searxng,
        enabled=args.web,
        context_length=ctx,
    )

    agent = Agent(
        client, model, workspace,
        context_length=ctx, thinking=args.think, think_level=args.think_level,
        auto_approve=args.yes,
        show_reasoning=not args.hide_reasoning, web=web,
        max_steps=args.max_steps,
        interactive=sys.stdin.isatty() and not args.prompt,
        supports_vision=caps["supports_vision"],
        save=args.save,
    )

    if args.web:
        status = await web.status()
        if not status.get("searxng_reachable"):
            # Name the copy of the script this user actually has. From an
            # installed build "scripts\setup_searxng.ps1" is a path to nothing,
            # so the one instruction the message gave could not be followed.
            script = "setup_searxng.ps1" if os.name == "nt" else "setup_searxng.sh"
            setup = paths.install_root() / (
                script if paths.is_frozen() else str(Path("scripts") / script)
            )
            if not setup.is_file():
                how = f"scripts/{script} from the project source"
            elif os.name == "nt":
                how = f'powershell -ExecutionPolicy Bypass -File "{setup}"'
            else:
                how = f'bash "{setup}"'
            print(
                f"{YELLOW}Web access requested but SearXNG is not reachable at "
                f"{args.searxng}.{RESET}\n"
                f"{DIM}  {status.get('error', '')}{RESET}\n"
                f"{DIM}  Web lookup runs SearXNG in a local Docker container. "
                f"With Docker Desktop running:{RESET}\n"
                f"{DIM}    {how}{RESET}\n"
                f"{DIM}  Continuing with web tools registered but failing.{RESET}"
            )
        else:
            print(f"{GREEN}Web lookup enabled via {args.searxng}{RESET}")

    if args.resume:
        sdir = agent.session.sessions_dir
        path = (
            latest_session(sdir)
            if args.resume == "__latest__"
            else find_session(sdir, args.resume)
        )
        if path is None:
            target = "any saved session" if args.resume == "__latest__" else repr(args.resume)
            print(f"{RED}Could not find {target} in {sdir}.{RESET}")
            for s in list_sessions(sdir)[:10]:
                print(f"{DIM}  {s['id']}  {s['messages']:>4} msgs  {s['first']}{RESET}")
            return 2
        print(f"{GREEN}{agent.load_session(path)}{RESET}")

    if args.prompt:
        await agent.run_turn(args.prompt)
        return 0

    await repl(agent)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
