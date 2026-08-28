"""Conversation state, token accounting, and compaction.

The failure mode this exists to prevent: Ollama silently truncates anything
past the loaded context window. No error, no warning -- the model just stops
seeing the beginning of the conversation and starts behaving erratically. On a
local agent that reads files into context, you hit it fast.

So we track the budget ourselves and compact deliberately when it gets tight,
which degrades visibly instead of mysteriously.

Sessions persist as JSON Lines next to the project, and never leave the machine.

Attached images are the one thing deliberately *not* stored inline: a message
carries its base64 in ``images`` for the wire and its file paths in
``image_paths`` for the log, and only the latter is written. A screenshot is a
couple of megabytes, which as base64 would be roughly 2.7MB of a single JSONL
line, per image, forever. Resuming re-reads the files.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import images as IM

# Rough chars-per-token. Real tokenizers vary; we refine this at runtime from
# the prompt_eval_count Ollama reports, so the initial value only has to be in
# the right neighbourhood.
CHARS_PER_TOKEN = 3.6

# Start compacting here rather than at 100% -- the model still needs room to
# generate a reply, and tool results can arrive in large chunks.
COMPACT_AT = 0.75

# Never compact away the most recent exchanges; that is the working set.
KEEP_RECENT_MESSAGES = 6

# How much of the window one summarisation pass may spend on transcript. The
# rest is the instruction, the running summary, and room to generate.
COMPACT_PROMPT_BUDGET = 0.55

# Per-message caps when building the summarisation prompt. Tool results get a
# short one because they are the bulk of a long session and the least useful
# thing in it to summarise: measured on a 1627-message conversation, tool
# output was 86% of the prompt -- file contents being fed back to be told, at
# length, that a file was read. What matters on resuming is what was asked,
# what was decided and which files changed.
TEXT_CAP = 1500
TOOL_CAP = 200
# ...except when they failed. An error is why the work changed direction, so it
# earns more room than the file listing next to it.
TOOL_ERROR_CAP = 600

# Successively harsher caps, used only if the conversation is so large that it
# still will not fold in a sane number of passes. tool_cap 0 drops non-error
# tool results entirely.
COMPACT_LADDER = ((TEXT_CAP, TOOL_CAP), (600, 120), (300, 0))

# Each pass is a model call. On a local model that is tens of seconds, so this
# bounds how long a compaction can take before it stops feeling like a hang.
MAX_COMPACT_PASSES = 5


def estimate_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    return int(len(text) / chars_per_token) + 1


def message_tokens(msg: dict[str, Any], chars_per_token: float = CHARS_PER_TOKEN) -> int:
    total = estimate_tokens(str(msg.get("content") or ""), chars_per_token)
    for call in msg.get("tool_calls") or []:
        total += estimate_tokens(json.dumps(call), chars_per_token)
    # An image costs a block of vision tokens bearing no relation to the length
    # of its base64, so it is counted per image rather than by character. Wrong
    # in both directions by some hundreds of tokens; corrected by the exact
    # prompt_eval_count on the next response.
    total += IM.TOKENS_PER_IMAGE * len(msg.get("images") or msg.get("image_paths") or [])
    return total + 4  # per-message role/framing overhead


@dataclass
class Session:
    model: str
    context_limit: int
    system_prompt: str
    sessions_dir: Path
    messages: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    chars_per_token: float = CHARS_PER_TOKEN
    compactions: int = 0
    # exact prompt size from the model's last response; 0 until the first reply
    last_prompt_tokens: int = 0
    # estimated size of messages appended since that measurement
    tokens_since_measure: int = 0
    # tool schemas ride along with every request but are not in `messages`
    tool_overhead_tokens: int = 0
    # attachments a resumed session could no longer find on disk
    missing_images: list[str] = field(default_factory=list)
    # Whether this conversation is written to disk at all. Off by default: a
    # transcript of everything you have ever asked, sitting in a folder you did
    # not choose, is not a reasonable thing to opt people into silently. --save
    # turns it on, and then the whole resume/compact/history machinery below
    # works exactly as it always did.
    persist: bool = False

    def __post_init__(self) -> None:
        if self.persist:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if not self.session_id:
            # Second granularity is not unique: two agents launched in the same
            # second would otherwise share a file and interleave their turns.
            base = time.strftime("%Y%m%d_%H%M%S")
            session_id, n = base, 1
            while (self.sessions_dir / f"{session_id}.jsonl").exists():
                n += 1
                session_id = f"{base}-{n}"
            self.session_id = session_id

    # ------------------------------------------------------------------ state

    @property
    def path(self) -> Path:
        return self.sessions_dir / f"{self.session_id}.jsonl"

    def wire_messages(self) -> list[dict[str, Any]]:
        """What actually gets sent: system prompt plus the live history.

        ``image_paths`` is bookkeeping for the log and means nothing to Ollama,
        so it is dropped here rather than shipped as an unknown field.
        """
        history = [
            {k: v for k, v in m.items() if k != "image_paths"} if "image_paths" in m else m
            for m in self.messages
        ]
        return [{"role": "system", "content": self.system_prompt}] + history

    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.tokens_since_measure += message_tokens(message, self.chars_per_token)
        self._write_line(message)

    def begin_persisting(self, pasted_dir: Path) -> int:
        """Start writing this conversation to disk, including its backlog.

        Exists for /savesession: the user launched without --save, and ten
        turns in the conversation turned out to matter. Every message
        accumulated so far is backfilled to the file, because a session that
        starts at the moment of the command is missing exactly the part worth
        keeping.

        Images need special care: without --save, a pasted image's temp file
        was deliberately deleted once its bytes were in memory (the privacy
        rule), so ``image_paths`` may point at files that no longer exist.
        Resuming would then rehydrate nothing. The bytes are still in each
        message's ``images`` list, so they are written back out under the
        session's own pasted directory and the paths updated to match.

        Timestamps on backfilled lines are the time of saving, not of the
        original turns -- the in-memory messages never carried one.
        """
        if self.persist:
            return len(self.messages)

        for msg in self.messages:
            paths = msg.get("image_paths") or []
            blobs = msg.get("images") or []
            if blobs and paths:
                import base64

                pasted_dir.mkdir(parents=True, exist_ok=True)
                rewritten = []
                for raw_path, b64 in zip(paths, blobs):
                    p = Path(raw_path)
                    if not p.is_file():
                        p = pasted_dir / Path(raw_path).name
                        try:
                            p.write_bytes(base64.b64decode(b64))
                        except (OSError, ValueError):
                            pass
                    rewritten.append(str(p))
                msg["image_paths"] = rewritten

        self.persist = True
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        for msg in self.messages:
            self._write_line(msg)
        return len(self.messages)

    def _write_line(self, message: dict[str, Any]) -> None:
        # The single point where anything reaches the disk, so the persist
        # check belongs here rather than at each of its callers.
        if not self.persist:
            return
        # The base64 stays in memory. See the module docstring.
        record = {k: v for k, v in message.items() if k != "images"}
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), **record}, default=str) + "\n")

    # ------------------------------------------------------------- accounting

    def note_actual(self, prompt_tokens: int) -> None:
        """Record the exact prompt size Ollama reported for the last request.

        This is ground truth and replaces estimating the whole conversation.
        An earlier version tried to calibrate a chars-per-token ratio from this
        number, but the numerator only summed message *content* -- excluding
        tool_call arguments (which carry entire file bodies for write_file) and
        the tool schemas resent every request. The ratio therefore collapsed and
        every estimate inflated with it, triggering compaction at roughly 2k
        real tokens instead of 24k. Measuring beats estimating.
        """
        if prompt_tokens > 0:
            self.last_prompt_tokens = prompt_tokens
            self.tokens_since_measure = 0

    def used_tokens(self) -> int:
        """Exact count from the last request, plus an estimate of what has been
        appended since. Only the small delta is ever estimated."""
        if self.last_prompt_tokens:
            return self.last_prompt_tokens + self.tokens_since_measure
        # before the first response, estimate -- and include the tool schemas,
        # which are part of every request but live outside the message list
        return (
            sum(message_tokens(m, self.chars_per_token) for m in self.wire_messages())
            + self.tool_overhead_tokens
        )

    def usage_ratio(self) -> float:
        return self.used_tokens() / max(1, self.context_limit)

    def needs_compaction(self) -> bool:
        return self.usage_ratio() >= COMPACT_AT

    def budget_line(self) -> str:
        used = self.used_tokens()
        pct = 100 * used / max(1, self.context_limit)
        return f"{used:,}/{self.context_limit:,} tok ({pct:.0f}%)"

    # ------------------------------------------------------------- compaction

    def can_compact(self) -> bool:
        """Whether there is anything old enough to fold away."""
        return len(self.messages) > KEEP_RECENT_MESSAGES

    def _split_for_compaction(self) -> tuple[list[dict], list[dict]]:
        if len(self.messages) <= KEEP_RECENT_MESSAGES:
            return [], self.messages

        cut = len(self.messages) - KEEP_RECENT_MESSAGES
        # Do not split a tool call from its result -- an assistant message with
        # tool_calls must stay adjacent to the tool messages answering it, or
        # the model sees a dangling call and gets confused.
        while cut < len(self.messages) and self.messages[cut].get("role") == "tool":
            cut += 1
        return self.messages[:cut], self.messages[cut:]

    def _transcript_lines(
        self, messages: list[dict[str, Any]], *, text_cap: int, tool_cap: int
    ) -> list[str]:
        """Flatten messages into summarisable lines, budgeting by role."""
        lines = []
        for m in messages:
            role = m.get("role", "?")
            content = str(m.get("content") or "").strip()
            if calls := m.get("tool_calls"):
                names = ", ".join(
                    c.get("function", {}).get("name", "?") for c in calls
                )
                content = (content + f" [called: {names}]").strip()
            if not content:
                continue
            if role == "tool":
                failed = content.startswith("ERROR")
                if not failed and tool_cap == 0:
                    continue
                cap = TOOL_ERROR_CAP if failed else tool_cap
            else:
                cap = text_cap
            lines.append(f"{role}: {content[:cap]}")
        return lines

    def _chunk(self, lines: list[str], budget: int) -> list[list[str]]:
        """Split lines into groups that each fit one pass's token budget."""
        chunks: list[list[str]] = []
        current: list[str] = []
        used = 0
        for line in lines:
            cost = estimate_tokens(line, self.chars_per_token)
            if current and used + cost > budget:
                chunks.append(current)
                current, used = [], 0
            current.append(line)
            used += cost
        if current:
            chunks.append(current)
        return chunks

    async def compact(self, client: Any, on_progress: Any = None) -> str:
        """Summarize older turns using the local model, in place.

        Thinking is off: this is a mechanical summarization, and on a local
        model a reasoning block here can cost more time than the entire rest of
        the turn.

        The summarisation prompt is itself bounded and folded in passes, which
        is not a refinement -- it is the difference between compaction working
        and appearing to. A resumed 1627-message session built a 217,900-token
        prompt and handed it to a 32,768-token window, so Ollama truncated it
        silently and "the summary" described only the fragment that survived.
        That is the exact failure this module exists to prevent, reproduced
        inside the mechanism meant to prevent it.
        """
        older, recent = self._split_for_compaction()
        if not older:
            return "Nothing old enough to compact."

        budget = max(1000, int(self.context_limit * COMPACT_PROMPT_BUDGET))
        # Tighten the per-message caps until the backlog folds in a bearable
        # number of passes, rather than letting a huge session cost minutes.
        for text_cap, tool_cap in COMPACT_LADDER:
            lines = self._transcript_lines(older, text_cap=text_cap, tool_cap=tool_cap)
            chunks = self._chunk(lines, budget)
            if len(chunks) <= MAX_COMPACT_PASSES:
                break
        dropped = ""
        if len(chunks) > MAX_COMPACT_PASSES:
            # Still too big even at the harshest setting. Summarise the most
            # recent portion and say so, rather than quietly losing the rest.
            kept = chunks[-MAX_COMPACT_PASSES:]
            dropped = (
                f" The oldest {len(chunks) - MAX_COMPACT_PASSES} block(s) were too "
                f"large to summarise and were dropped."
            )
            chunks = kept

        summary = ""
        gen_tokens = 0
        elapsed = 0.0
        for i, chunk in enumerate(chunks, 1):
            # Each pass is a full model call -- tens of seconds locally. Five
            # of them with nothing on screen is indistinguishable from a hang,
            # which is exactly the failure mode this module exists to prevent
            # showing up elsewhere.
            if on_progress and len(chunks) > 1:
                on_progress(f"compaction pass {i}/{len(chunks)}…")
            if summary:
                head = (
                    "Here are notes on the earlier part of a coding session:\n\n"
                    f"{summary}\n\n"
                    "Extend those notes with this next portion. Return the "
                    "combined notes, not a commentary on them.\n\n"
                )
            else:
                head = "Summarize this portion of a coding session.\n\n"
            prompt = (
                head
                + "Preserve: what the user asked for, which files were read or "
                "modified and what they contain, decisions made, and anything "
                "still outstanding. Drop redundant tool output. Be specific "
                "about file paths and function names. Write it as notes for "
                "resuming the work.\n\n" + "\n\n".join(chunk)
            )
            result = await client.generate(
                self.model,
                prompt,
                think=False,
                max_tokens=800,
                context_length=self.context_limit,
            )
            summary = result.content.strip()
            gen_tokens += result.gen_tokens
            elapsed += result.total_s

        summary_msg = {
            "role": "user",
            "content": "[Earlier conversation, compacted to save context]\n" + summary,
        }
        self.messages = [summary_msg] + recent
        self.compactions += 1
        passes = len(chunks)

        # The ground-truth measurement described the pre-compaction prompt and
        # is now stale and far too large. Leaving it in place would make
        # needs_compaction() immediately true again and compact every single
        # turn. Fall back to estimating until the next response gives a real
        # count for the new, smaller conversation.
        self.last_prompt_tokens = 0
        self.tokens_since_measure = 0

        self._write_line({"role": "_system_note", "content": "compacted", "kept": len(recent)})

        note = ""
        if self.needs_compaction():
            # The retained tail alone exceeds the threshold -- usually a few
            # very large tool results. Compacting again will not help.
            note = (
                " WARNING: still above the compaction threshold; the recent "
                "messages are individually large. Consider a bigger --context."
            )
        how = f" in {passes} passes" if passes > 1 else ""
        return (
            f"Compacted {len(older)} messages into a summary{how} "
            f"({gen_tokens} tok, {elapsed:.1f}s). "
            f"Now at {self.budget_line()}.{dropped}{note}"
        )

    # --------------------------------------------------------------- resuming

    @classmethod
    def resume(
        cls,
        path: Path,
        *,
        model: str,
        context_limit: int,
        system_prompt: str,
        persist: bool = False,
    ) -> "Session":
        """Reload a saved conversation and continue appending to the same file.

        The system prompt is deliberately taken from the *current* run rather
        than the log, so a resumed session picks up today's date and this
        machine's environment instead of whatever was true when it started.
        """
        messages: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec.pop("ts", None)
            if rec.get("role") in {"user", "assistant", "tool"}:
                messages.append(rec)

        messages = trim_dangling_tool_calls(messages)
        missing = rehydrate_images(messages)

        s = cls(
            model=model,
            context_limit=context_limit,
            system_prompt=system_prompt,
            sessions_dir=path.parent,
            session_id=path.stem,
            persist=persist,
        )
        s.messages = messages
        s.missing_images = missing
        # No measurement applies to this reassembled conversation yet; fall back
        # to estimating until the first response reports a real prompt size.
        s.last_prompt_tokens = 0
        s.tokens_since_measure = 0
        return s


def rehydrate_images(messages: list[dict[str, Any]]) -> list[str]:
    """Re-read attached images from disk when resuming, in place.

    Only the paths were logged, so a resumed conversation would otherwise
    reference "[image 1: chart.png]" with no image behind it -- and a model
    asked about a picture it cannot see will confidently describe one. A file
    that has since been moved or deleted is therefore reported, and its label
    is amended so the text no longer claims an image is attached.
    """
    missing: list[str] = []
    for msg in messages:
        paths = msg.get("image_paths") or []
        if not paths:
            continue
        data, kept, lost = [], [], 0
        for raw in paths:
            try:
                data.append(IM.load(raw).b64)
                kept.append(raw)
            except IM.ImageError:
                missing.append(str(raw))
                lost += 1
        if data:
            msg["images"] = data
        else:
            msg.pop("images", None)
        msg["image_paths"] = kept
        if lost:
            msg["content"] = (
                str(msg.get("content") or "")
                + f"\n[{lost} image(s) referenced above are no longer available]"
            )
    return missing


def trim_dangling_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop trailing tool calls that never received their full results.

    A session killed mid-turn -- Ctrl+C, a crash, closing the window while the
    model was working -- ends in one of two malformed shapes: an assistant
    message requesting tools with no tool messages answering it, or (killed
    between result writes) an assistant requesting N tools with fewer than N
    results behind it. Replaying either leaves the model staring at its own
    unanswered request, and it typically either repeats the call or invents
    the result. A partial exchange is trimmed whole -- a tool result without
    its sibling results is just as malformed as a call without any -- and the
    trim repeats until the tail is a complete exchange, the safe resume point.
    """
    end = len(messages)
    while end > 0:
        msg = messages[end - 1]
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            end -= 1  # unanswered call: drop it and look further back
            continue
        if role == "tool":
            # Walk back over the run of trailing tool results and compare it
            # with what the assistant message before them actually requested.
            start = end
            while start > 0 and messages[start - 1].get("role") == "tool":
                start -= 1
            asst = messages[start - 1] if start > 0 else None
            if asst and asst.get("role") == "assistant" and asst.get("tool_calls"):
                if len(asst["tool_calls"]) <= end - start:
                    break  # every requested call has its result: complete
                end = start - 1  # partial: drop the results AND their request
            else:
                end = start  # results with no request at all: drop them
            continue
        break
    return messages[:end]


def latest_session(sessions_dir: Path) -> Path | None:
    """Most recently modified non-empty session file."""
    candidates = [
        p for p in sessions_dir.glob("*.jsonl") if p.is_file() and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_session(sessions_dir: Path, session_id: str) -> Path | None:
    """Resolve a session id, allowing a unique prefix."""
    exact = sessions_dir / f"{session_id}.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(sessions_dir.glob(f"{session_id}*.jsonl"))
    return matches[0] if len(matches) == 1 else None


def list_sessions(sessions_dir: Path) -> list[dict[str, Any]]:
    """Summarize saved sessions, newest first.

    Delegates to transcript so that "first message" and the turn count mean the
    same thing here as in scripts/show_session.py. Without it this reported the
    injected environment block as the opening message, since that is literally
    the first user-role record in the file.
    """
    import transcript as T

    if not sessions_dir.is_dir():
        return []
    out = []
    # Sort by mtime, not filename: the "when" column shows mtime, so filename
    # order put a just-resumed old session saying "just now" at the bottom of
    # the list. Filename order also mis-sorts collision suffixes ("-10" lands
    # before "-2" lexically).
    for p in sorted(
        sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            records = T.load(p)
        except Exception:
            continue
        # Empty files are kept, not skipped: the session you are currently in
        # has no records yet, and dropping it means it never appears in its own
        # listing and can never be marked as current.
        out.append(
            {
                "id": p.stem,
                "path": p,
                "messages": T.count_turns(records),
                "records": len(records),
                "when": T.relative(p.stat().st_mtime),
                "first": T.first_user_message(records),
            }
        )
    return out
