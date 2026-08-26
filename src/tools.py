"""Built-in tools for the local agent.

Design notes, all of which are concessions to driving a ~27B local model rather
than a frontier one:

  * **Few tools, flat arguments.** Small models pick the wrong tool as the menu
    grows, and they fumble nested object arguments. Everything here takes flat
    strings and ints.
  * **Line-range edits, not exact-string replacement.** Exact-match editing is
    where local models fail most often -- they reproduce whitespace slightly
    wrong and the edit silently no-ops or errors. Addressing lines by number is
    far more reliable, and ``read_file`` numbers its output so the model has the
    numbers in front of it.
  * **Truncation everywhere.** A 30k-token file dumped into a 32k window ends
    the session. Every tool caps its output and says so.
  * **Confinement + approval.** Paths resolve inside the workspace root; the
    mutating tools are marked so the agent can prompt before running them.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import ui as _ui

IS_WINDOWS = os.name == "nt"

MAX_OUTPUT_CHARS = 12_000
MAX_GREP_HITS = 60
MAX_GLOB_HITS = 200
SHELL_TIMEOUT = 120

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".next", ".idea", ".vscode",
}


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut + f"\n\n[... truncated, {len(text) - limit} more chars. Narrow your request.]"


class ToolError(Exception):
    """Raised for conditions the model can recover from; text goes back as the
    tool result so it can retry rather than the run dying."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    mutating: bool = False


class Workspace:
    """Filesystem access confined to one root directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace root is not a directory: {self.root}")

    def change_root(self, root: str | Path) -> Path:
        """Re-point the workspace at a different directory, mid-session.

        Every tool reads ``self.root`` at call time, so mutating it here
        retargets file access, search and run_command's cwd in one place --
        no registry rebuild. Validation matches the constructor: the new root
        must exist, or ValueError and the old root stays in effect.
        """
        new_root = Path(root).resolve()
        if not new_root.is_dir():
            raise ValueError(f"not a directory: {new_root}")
        self.root = new_root
        return self.root

    def resolve(self, path: str) -> Path:
        p = Path(path)
        p = p if p.is_absolute() else self.root / p
        p = p.resolve()
        # containment check; note Path.is_relative_to is 3.9+
        if not p.is_relative_to(self.root):
            raise ToolError(
                f"Path escapes the workspace root ({self.root}). Refused: {path}"
            )
        return p

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)


# --------------------------------------------------------------------- tools


def build_tools(ws: Workspace) -> list[Tool]:
    def read_file(path: str, start_line: int = 1, num_lines: int = 400) -> str:
        """Numbered read. The numbers matter -- edit_lines addresses by them."""
        p = ws.resolve(path)
        if not p.is_file():
            raise ToolError(f"Not a file: {ws.rel(p)}")
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            raise ToolError(f"Could not read {ws.rel(p)}: {e}")

        start = max(1, start_line)
        chunk = lines[start - 1 : start - 1 + max(1, num_lines)]
        if not chunk:
            return f"{ws.rel(p)} has {len(lines)} lines; start_line {start} is past the end."

        body = "\n".join(f"{start + i:>5}| {ln}" for i, ln in enumerate(chunk))
        end = start + len(chunk) - 1
        header = f"{ws.rel(p)} lines {start}-{end} of {len(lines)}"
        more = "" if end >= len(lines) else f"\n[{len(lines) - end} more lines below]"
        return _truncate(f"{header}\n{body}{more}")

    def write_file(path: str, content: str) -> str:
        p = ws.resolve(path)
        existed = p.is_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        # newline="" so the content lands byte-for-byte; the default would
        # rewrite every \n as \r\n on Windows.
        p.write_text(content, encoding="utf-8", newline="")
        n = len(content.splitlines())
        return f"{'Overwrote' if existed else 'Created'} {ws.rel(p)} ({n} lines)."

    def edit_lines(path: str, start_line: int, end_line: int, new_text: str) -> str:
        """Replace an inclusive line range. Chosen over exact-string matching
        because local models get whitespace subtly wrong far too often.

        Reads and writes with newline="" and strict UTF-8: a round-trip through
        this tool must not rewrite LF as CRLF, invent a trailing newline, or
        bake decode-replacement chars into a file that was never UTF-8."""
        p = ws.resolve(path)
        if not p.is_file():
            raise ToolError(f"Not a file: {ws.rel(p)}")
        try:
            # read_bytes + strict decode: no newline translation on the way in,
            # and no read_text(newline=...) -- Path only grew that in 3.13
            original = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{ws.rel(p)} is not valid UTF-8; refusing to edit it in place")
        lines = original.splitlines()

        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            raise ToolError(
                f"Bad range {start_line}-{end_line}; {ws.rel(p)} has {len(lines)} lines. "
                f"Call read_file first to get current numbers."
            )

        eol = "\r\n" if "\r\n" in original else "\n"
        replacement = new_text.splitlines()
        updated = lines[: start_line - 1] + replacement + lines[end_line:]
        trailing = eol if original.endswith(("\n", "\r")) else ""
        p.write_text(eol.join(updated) + trailing, encoding="utf-8", newline="")

        removed, added = end_line - start_line + 1, len(replacement)
        # echo the neighbourhood back so the model sees the real result and
        # does not have to guess whether it worked
        lo = max(1, start_line - 3)
        hi = min(len(updated), start_line - 1 + added + 3)
        ctx = "\n".join(f"{lo + i:>5}| {ln}" for i, ln in enumerate(updated[lo - 1 : hi]))
        return _truncate(
            f"Edited {ws.rel(p)}: replaced {removed} line(s) with {added}.\n"
            f"Now reads:\n{ctx}"
        )

    def list_dir(path: str = ".") -> str:
        p = ws.resolve(path)
        if not p.is_dir():
            raise ToolError(f"Not a directory: {ws.rel(p)}")
        entries = []
        for e in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
            if e.name in SKIP_DIRS:
                continue
            try:
                entries.append(f"{e.name}/" if e.is_dir() else f"{e.name}  ({e.stat().st_size}b)")
            except OSError:
                # OneDrive placeholders, broken links and permission walls all
                # refuse stat(); one bad entry must not sink the whole listing
                entries.append(f"{e.name}  (?b)")
        if not entries:
            return f"{ws.rel(p)} is empty."
        return _truncate(f"{ws.rel(p)}:\n" + "\n".join(f"  {e}" for e in entries))

    def find_files(pattern: str) -> str:
        """Glob by name, e.g. '*.py' or 'src/**/*.ts'."""
        hits = []
        for dirpath, dirnames, filenames in os.walk(ws.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                full = Path(dirpath) / fn
                rel = str(full.relative_to(ws.root)).replace("\\", "/")
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fn, pattern):
                    hits.append(rel)
                    if len(hits) >= MAX_GLOB_HITS:
                        break
            if len(hits) >= MAX_GLOB_HITS:
                break
        if not hits:
            return f"No files match {pattern!r}."
        return _truncate(f"{len(hits)} match(es) for {pattern!r}:\n" + "\n".join(hits))

    def search_text(pattern: str, file_pattern: str = "*") -> str:
        """Regex search across the workspace."""
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Bad regex {pattern!r}: {e}")

        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(ws.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fnmatch.fnmatch(fn, file_pattern):
                    continue
                full = Path(dirpath) / fn
                try:
                    text = full.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        rel = str(full.relative_to(ws.root)).replace("\\", "/")
                        hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                        if len(hits) >= MAX_GREP_HITS:
                            break
                if len(hits) >= MAX_GREP_HITS:
                    break
            if len(hits) >= MAX_GREP_HITS:
                break

        if not hits:
            return f"No matches for {pattern!r} in files matching {file_pattern!r}."
        capped = " (capped)" if len(hits) >= MAX_GREP_HITS else ""
        return _truncate(f"{len(hits)} match(es){capped}:\n" + "\n".join(hits))

    async def run_command(command: str) -> str:
        """Run a shell command in the workspace root, without blocking the loop.

        On Windows, a bare shell would be cmd.exe, which has none of the POSIX
        utilities models reach for by default -- an observed run died on
        ``tail -5``. PowerShell at least provides equivalents, so we route
        there and tell the model so in the tool description.

        Async, not ``subprocess.run``: the blocking version parked the entire
        event loop for up to SHELL_TIMEOUT seconds, during which Ctrl+C could
        not be processed at all -- a hung install froze the program with no
        exit but killing it. As a task this can be cancelled, and cancellation
        (or timeout) kills the whole process *tree*, because the direct child
        is a shell whose children would otherwise keep running orphaned.
        """

        async def kill_tree(proc: "asyncio.subprocess.Process") -> None:
            if proc.returncode is not None:
                return
            try:
                if IS_WINDOWS:
                    # taskkill /T is the only reliable tree kill on Windows;
                    # proc.kill() would take down powershell and leave its
                    # children running.
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill", "/F", "/T", "/PID", str(proc.pid),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await killer.wait()
                else:
                    # start_new_session below made the shell a process-group
                    # leader, so the group id is its pid.
                    os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            try:
                await proc.wait()
            except Exception:
                pass

        try:
            if IS_WINDOWS:
                # Two Windows-specific corrections, both observed breaking real
                # runs:
                #
                # 1. A command beginning with a quoted path is parsed by
                #    PowerShell as a *string expression*, not a command --
                #    `"C:\...\python.exe" script.py` fails with "Unexpected
                #    token". The call operator `&` makes it a command again.
                #    The model is told to use a full interpreter path, so it
                #    quotes one often.
                # 2. The default locale here is cp1252, so decoded output would
                #    mangle any non-ASCII into mojibake -- which the model then
                #    cannot read, and starts flailing. Force UTF-8 on both sides.
                cmd = command.lstrip()
                if cmd.startswith(('"', "'")):
                    cmd = "& " + cmd
                prelude = (
                    "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                    "$OutputEncoding=[Text.Encoding]::UTF8; "
                )
                proc = await asyncio.create_subprocess_exec(
                    "powershell", "-NoProfile", "-NonInteractive",
                    "-Command", prelude + cmd,
                    cwd=str(ws.root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(ws.root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
        except FileNotFoundError as e:
            raise ToolError(f"Shell not available: {e}")

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=SHELL_TIMEOUT
            )
        except asyncio.TimeoutError:
            await kill_tree(proc)
            raise ToolError(f"Command timed out after {SHELL_TIMEOUT}s: {command}")
        except asyncio.CancelledError:
            # The user interrupted the turn. The command must die with it --
            # a cancelled await that leaves the process running turns Ctrl+C
            # into a lie.
            await kill_tree(proc)
            raise

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        out = stdout + (f"\n[stderr]\n{stderr}" if stderr.strip() else "")
        status = "" if proc.returncode == 0 else f"[exit {proc.returncode}]\n"
        return _truncate(status + (out.strip() or "(no output)"))

    return [
        Tool(
            name="read_file",
            description=(
                "Read a text file with line numbers. Use the line numbers it "
                "returns when calling edit_lines."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "start_line": {"type": "integer", "description": "First line to read. Default 1."},
                    "num_lines": {"type": "integer", "description": "How many lines. Default 400."},
                },
                "required": ["path"],
            },
            handler=read_file,
        ),
        Tool(
            name="write_file",
            description="Create a file, or completely overwrite an existing one.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
            handler=write_file,
            mutating=True,
        ),
        Tool(
            name="edit_lines",
            description=(
                "Replace an inclusive range of lines in a file with new text. "
                "Always read_file first so the line numbers are current."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "start_line": {"type": "integer", "description": "First line to replace (1-based, inclusive)."},
                    "end_line": {"type": "integer", "description": "Last line to replace (inclusive)."},
                    "new_text": {"type": "string", "description": "Replacement text. May be multiple lines, or empty to delete."},
                },
                "required": ["path", "start_line", "end_line", "new_text"],
            },
            handler=edit_lines,
            mutating=True,
        ),
        Tool(
            name="list_dir",
            description="List files and folders in a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Default '.' for workspace root."}
                },
                "required": [],
            },
            handler=list_dir,
        ),
        Tool(
            name="find_files",
            description="Find files by name pattern, e.g. '*.py' or 'src/**/*.ts'.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern."}
                },
                "required": ["pattern"],
            },
            handler=find_files,
        ),
        Tool(
            name="search_text",
            description="Search file contents with a regular expression.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression."},
                    "file_pattern": {"type": "string", "description": "Limit to files matching this glob, e.g. '*.py'. Default '*'."},
                },
                "required": ["pattern"],
            },
            handler=search_text,
        ),
        Tool(
            name="run_command",
            description=(
                "Run a shell command in the workspace root and return its output. "
                "Use for tests, builds, git, and installs."
                + (
                    " This machine runs PowerShell, NOT bash. Unix tools like "
                    "tail, head, wc and which do not exist. Use: "
                    "'Get-Content f -Tail 5' not 'tail -5 f'; "
                    "'Get-Content f -TotalCount 5' not 'head -5 f'; "
                    "'Get-ChildItem' not 'ls -la'. Chain with ';' not '&&'."
                    if IS_WINDOWS
                    else ""
                )
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run."}
                },
                "required": ["command"],
            },
            handler=run_command,
            mutating=True,
        ),
    ]


class ToolRegistry:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._tools = {t.name: t for t in build_tools(workspace)}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        """Deregister a tool so it stops appearing in the wire schema.

        Exists for /web off: a tool left registered but failing wastes whole
        turns on a slow local model, because the model has no way to know the
        calls are pointless until each one returns an error.
        """
        self._tools.pop(name, None)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def ollama_schemas(self) -> list[dict[str, Any]]:
        """Tool definitions in the shape Ollama's /api/chat expects."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def add_ask_tool(self, interactive: bool) -> None:
        """Let the model ask the user a question instead of guessing.

        Without this the only way to ask is to end the turn with a question,
        which costs a full round trip -- expensive when a turn is measured in
        tens of seconds. As a tool the model can ask mid-task and carry on with
        the answer in the same turn.

        Options are a pipe-separated string rather than an array because small
        models are markedly more reliable with flat arguments, and the user can
        always type something else anyway.
        """

        def ask_user(question: str, options: str = "") -> str:
            choices = [o.strip() for o in options.split("|") if o.strip()]
            if not interactive:
                # --prompt and piped runs have nobody to answer. Blocking here
                # would hang forever, so say so and let the model proceed on a
                # stated assumption rather than waiting.
                return (
                    "ERROR: no user is available to answer in this run "
                    "(non-interactive). Choose the most reasonable "
                    "interpretation, state the assumption you made, and continue."
                )
            answer = _ui.question(question, choices)
            if not answer:
                return (
                    "The user skipped the question. Make a reasonable assumption, "
                    "say what you assumed, and carry on."
                )
            return f"The user answered: {answer}"

        self._tools["ask_user"] = Tool(
            name="ask_user",
            description=(
                "Ask the user a clarifying question when the answer would change "
                "what you build. Use it instead of guessing at requirements, file "
                "locations, library choices or scope. Do NOT use it for things you "
                "can find out yourself with read_file, search_text or run_command, "
                "and do not ask about trivial details you could reasonably decide."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "One specific question, in plain language.",
                    },
                    "options": {
                        "type": "string",
                        "description": (
                            "Optional suggested answers separated by | , e.g. "
                            "'Postgres|SQLite|MySQL'. The user may type their own."
                        ),
                    },
                },
                "required": ["question"],
            },
            handler=ask_user,
        )

    def add_web_tools(self, web: Any) -> None:
        """Register internet lookup, backed by the WebSearch instance.

        Kept separate from build_tools because these are the only tools that
        cause outbound traffic; they exist solely when the agent was started
        with web access enabled.
        """

        async def search_web(query: str, pages: int = 3) -> str:
            return await web.answer(query, pages=pages)

        async def fetch_url(url: str) -> str:
            text = await web.fetch_text(url)
            return f"{url}\n\n{text}"

        self._tools["search_web"] = Tool(
            name="search_web",
            description=(
                "Look something up on the internet when you do not know the answer "
                "or your knowledge may be out of date. Returns short summaries "
                "with the supporting quote from each source, not raw pages. Use "
                "this instead of guessing at an API, a version number, a command, "
                "or a library's behaviour. "
                "Results are labelled [LOW-QUALITY SOURCE] for tutorial farms and "
                "aggregators, and unreadable sources are listed explicitly -- read "
                "those labels before trusting a figure. Each call takes 30-60 "
                "seconds, so make the query specific."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific question, e.g. 'httpx AsyncClient stream response Python'.",
                    },
                    "pages": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "How many results to read, 1-5. Default 3. More is slower.",
                    },
                },
                "required": ["query"],
            },
            handler=search_web,
            mutating=True,  # outbound traffic: gate it behind approval too
        )

        self._tools["fetch_url"] = Tool(
            name="fetch_url",
            description=(
                "Fetch one specific URL and return its article text. Use only when "
                "you already have the exact URL; prefer search_web otherwise."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL including https://."}
                },
                "required": ["url"],
            },
            handler=fetch_url,
            mutating=True,
        )

    async def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return (
                f"ERROR: no tool named {name!r}. "
                f"Available: {', '.join(self._tools)}"
            )
        # Validate the binding before calling: a bare `except TypeError` around
        # the handler would also swallow TypeErrors raised inside it, reporting
        # real bugs back to the model as "bad arguments".
        try:
            inspect.signature(tool.handler).bind(**(args or {}))
        except TypeError as e:
            # wrong/missing arguments -- tell the model exactly what it must send
            required = tool.parameters.get("required", [])
            return (
                f"ERROR: bad arguments for {name}: {e}. "
                f"Required: {required}. Got: {list((args or {}).keys())}"
            )
        try:
            result = tool.handler(**(args or {}))
            if inspect.isawaitable(result):
                result = await result
            return result
        except ToolError as e:
            return f"ERROR: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
