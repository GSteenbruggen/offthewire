"""Environment grounding: tell the model what it cannot know on its own.

A local model has no idea what day it is, which shell it is driving, or which
Python interpreter it should be calling. Left to guess it writes `tail -5` at a
cmd.exe prompt and `pip install` against whatever interpreter happens to be on
PATH -- both observed on this machine.

The design constraint here is prompt caching, not token count. Ollama reuses the
KV cache for whatever prefix of the request is unchanged, which is why prompt
ingestion measures 2400-3900 tok/s on warm turns versus ~1200 cold. Rebuilding
the system prompt every turn would change the prefix and force a full
reprocess of the conversation on every step.

So facts are split by volatility:

  * :func:`static_block` -- OS, shell, interpreter, tool versions. Computed
    once, lives in the system prompt, never invalidates the cache.
  * :func:`volatile_block` -- date, git state, project files. Injected at the
    *tail* of the conversation, so a change only reprocesses from that point.
    Re-emitted only when something actually differs from the last turn.

Every probe is local. Nothing here touches the network.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import paths

PROBE_TIMEOUT = 4.0

IS_WINDOWS = os.name == "nt"

# Files that, if present, carry project-specific instructions the model should
# read before touching anything.
INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", ".cursorrules")

# Marker -> (label, how to run its tests)
PROJECT_MARKERS: tuple[tuple[str, str, str], ...] = (
    ("pyproject.toml", "Python", "pytest"),
    ("requirements.txt", "Python", "pytest"),
    ("package.json", "Node", "npm test"),
    ("Cargo.toml", "Rust", "cargo test"),
    ("go.mod", "Go", "go test ./..."),
    ("pom.xml", "Java/Maven", "mvn test"),
    ("Gemfile", "Ruby", "bundle exec rspec"),
)


def _run(argv: list[str], cwd: str | None = None) -> str:
    """Run a probe command, returning stripped stdout or '' on any failure."""
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
        )
    except Exception:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text else ""


def _version_of(text: str) -> str:
    """Pull a version number out of a tool's --version banner.

    Naively taking the last token gets this wrong in both directions:
    ``docker --version`` prints "Docker version 28.3.0, build 38b7060" (last
    token is the build hash) and ``git --version`` prints
    "git version 2.51.0.windows.1".

    The lookbehind rather than ``\\b`` matters: node prints "v22.13.1", and
    there is no word boundary between the "v" and the digits, so ``\\b`` skips
    the real version and matches "13.1".
    """
    import re

    m = re.search(r"(?<![\d.])(\d+\.\d+\.\d+)", text) or re.search(
        r"(?<![\d.])(\d+\.\d+)", text
    )
    return m.group(1) if m else text.strip()


def _os_name() -> str:
    """Human-readable OS, working around Windows 11 reporting itself as 10.

    ``platform.release()`` returns "10" on Windows 11 -- the major version was
    never bumped. The build number is the only reliable discriminator, and 11
    starts at 22000.
    """
    if sys.platform.startswith("linux"):
        # "Ubuntu 24.04" tells the model which package manager and paths to
        # expect; "Linux 6.8.0-49-generic" tells it almost nothing.
        try:
            text = Path("/etc/os-release").read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return f"Linux {platform.release()}"
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0] or platform.release()}"
    if not IS_WINDOWS:
        return f"{platform.system()} {platform.release()}"
    try:
        build = int(platform.version().split(".")[2])
    except (IndexError, ValueError):
        build = 0
    return f"Windows {'11' if build >= 22000 else '10'}"


# --------------------------------------------------------------- static facts


def find_interpreter(workspace: Path) -> tuple[str, str]:
    """Locate the interpreter the model should actually use.

    Prefers a virtualenv inside the workspace. Getting this wrong is not
    cosmetic: an observed run executed `pip install httpx`, which installed into
    the global interpreter on PATH rather than the project's venv.
    """
    candidates = [
        workspace / ".venv" / ("Scripts" if IS_WINDOWS else "bin") / ("python.exe" if IS_WINDOWS else "python"),
        workspace / "venv" / ("Scripts" if IS_WINDOWS else "bin") / ("python.exe" if IS_WINDOWS else "python"),
    ]
    for c in candidates:
        if c.is_file():
            version = _first_line(_run([str(c), "--version"])) or "unknown version"
            return str(c), version

    if not paths.is_frozen():
        return sys.executable, f"Python {platform.python_version()}"

    # In an installed build sys.executable is OffTheWire.exe, and
    # platform.python_version() reports the Python baked inside it. Returning
    # those told the model "Python: use OffTheWire.exe (Python 3.11.9)",
    # which is not merely cosmetic -- the agent will cheerfully try to run a
    # test suite with the LLM launcher. The interpreter it should use is
    # whatever the user has on PATH, which may be nothing at all.
    for name in ("python", "py", "python3"):
        if found := shutil.which(name):
            version = _first_line(_run([found, "--version"])) or "unknown version"
            # The Windows Store stub named python.exe is on PATH by default and
            # does nothing but open the Store. It reports no version, which is
            # how we tell it apart from a real interpreter.
            if version.lower().startswith("python"):
                return found, version
    return "", ""


@dataclass
class StaticFacts:
    os_name: str
    shell: str
    interpreter: str
    interpreter_version: str
    in_venv: bool
    tools: dict[str, str] = field(default_factory=dict)


def probe_static(workspace: Path) -> StaticFacts:
    """Session-lifetime facts. Runs a handful of subprocesses; call once."""
    interpreter, version = find_interpreter(workspace)
    in_venv = ".venv" in interpreter or "venv" in Path(interpreter).parts

    tools: dict[str, str] = {}
    for name, argv in (
        ("git", ["git", "--version"]),
        ("node", ["node", "--version"]),
        ("docker", ["docker", "--version"]),
    ):
        out = _first_line(_run(argv))
        if out:
            tools[name] = _version_of(out)

    return StaticFacts(
        os_name=_os_name(),
        shell="PowerShell" if IS_WINDOWS else os.environ.get("SHELL", "sh").split("/")[-1],
        interpreter=interpreter,
        interpreter_version=version,
        in_venv=in_venv,
        tools=tools,
    )


def static_block(facts: StaticFacts) -> str:
    lines = [
        "ENVIRONMENT",
        f"OS: {facts.os_name}. Shell: {facts.shell}"
        + (", NOT bash. Unix tools like tail/head/wc do not exist." if IS_WINDOWS else "."),
    ]
    if not facts.interpreter:
        # Saying nothing would leave the model to assume `python` works and
        # then puzzle over the error; saying this makes the next move obvious.
        lines.append(
            "Python: none found on PATH. Do not try to run Python; say so "
            "instead if a task needs it."
        )
    elif facts.in_venv:
        lines.append(
            f"Python: use `{facts.interpreter}` ({facts.interpreter_version}). "
            "Do NOT use bare `python` or `pip` -- they resolve to a different "
            "interpreter and will install into the wrong place."
        )
    else:
        lines.append(f"Python: {facts.interpreter} ({facts.interpreter_version}).")
    if facts.tools:
        lines.append("Available: " + ", ".join(f"{k} {v}" for k, v in facts.tools.items()) + ".")
    return "\n".join(lines)


# ------------------------------------------------------------- volatile facts


def probe_volatile(workspace: Path) -> dict[str, Any]:
    """Facts that can change during a session. Cheap enough to run every turn.

    Only local filesystem checks and one `git status`, so this costs a few tens
    of milliseconds -- far less than a single generated token at 8 tok/s.
    """
    now = time.localtime()
    snap: dict[str, Any] = {
        "date": time.strftime("%A, %d %B %Y", now),
        "time": time.strftime("%H:%M %Z", now),
    }

    # git: branch and whether the tree is dirty
    porcelain = _run(["git", "status", "--porcelain", "-b"], cwd=str(workspace))
    if porcelain:
        lines = porcelain.splitlines()
        branch = lines[0].removeprefix("## ").split("...")[0] if lines else "?"
        changed = len([l for l in lines[1:] if l.strip()])
        snap["git"] = f"branch {branch}, " + (f"{changed} uncommitted change(s)" if changed else "clean")
    else:
        snap["git"] = None

    # project type and test command
    kinds, test_cmd = [], None
    for marker, label, cmd in PROJECT_MARKERS:
        if (workspace / marker).exists():
            if label not in kinds:
                kinds.append(label)
            test_cmd = test_cmd or cmd
    snap["project"] = ", ".join(kinds) if kinds else None
    snap["test_cmd"] = test_cmd

    # project-specific instructions
    snap["instructions"] = [f for f in INSTRUCTION_FILES if (workspace / f).exists()]

    return snap


def volatile_block(snap: dict[str, Any], workspace: Path) -> str:
    lines = [
        f"Today is {snap['date']}, {snap['time']}.",
        "Your training data predates this. Treat anything you believe about "
        "library versions, releases or current events as possibly stale -- "
        "verify it rather than asserting it.",
        "",
        f"WORKSPACE {workspace}",
    ]
    if snap.get("git"):
        lines.append(f"git: {snap['git']}")
    else:
        lines.append("git: not a repository")
    if snap.get("project"):
        lines.append(f"type: {snap['project']}" + (f" -- tests: {snap['test_cmd']}" if snap.get("test_cmd") else ""))
    if snap.get("instructions"):
        found = ", ".join(snap["instructions"])
        lines.append(f"{found} present -- read before making changes.")
    return "\n".join(lines)


# ------------------------------------------------------------------- deltas

# Keys worth telling the model about mid-session. Time-of-day changes every
# turn and is pure noise, so it is deliberately excluded.
DELTA_KEYS = ("date", "git", "project", "instructions")


def describe_delta(previous: dict[str, Any], current: dict[str, Any]) -> str | None:
    """One line per meaningful change, or None if nothing moved.

    Re-emitting the whole environment block every turn would invalidate the
    prompt cache from that point onward for no benefit. Almost every turn
    produces no delta at all and costs nothing.
    """
    if not previous:
        return None

    changes = []
    for key in DELTA_KEYS:
        before, after = previous.get(key), current.get(key)
        if before == after:
            continue
        if key == "date":
            changes.append(f"the date is now {after}")
        elif key == "git":
            changes.append(f"git is now {after}")
        elif key == "project":
            changes.append(f"project type is now {after}")
        elif key == "instructions":
            gained = set(after or []) - set(before or [])
            if gained:
                changes.append(f"new instruction file(s): {', '.join(sorted(gained))}")

    if not changes:
        return None
    return "[environment update] " + "; ".join(changes) + "."
