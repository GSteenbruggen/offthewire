"""Thin async client for the local Ollama HTTP API.

Shared by the MCP server (phase 1) and the agent (phase 2). Nothing in here
talks to anything but localhost.

Two things this wraps that the raw API makes easy to get wrong:

  * ``think`` -- the qwen3.x family emits a reasoning block by default. On a
    27B running partly on CPU that measured 263 tokens / 23.5s for "hello
    world" versus 10 tokens / 1.8s with thinking off. Every call here takes an
    explicit ``think`` argument so the caller decides per turn. Ollama also
    accepts an effort level ("low" | "medium" | "high" | "max") instead of a
    boolean; this matters on qwen3.8, whose chat template defaults a bare
    ``true`` to its maximum ("xhigh") effort.
  * ``num_ctx`` -- Ollama loads models at 4096 context regardless of what the
    model supports, and silently truncates past it. Pass ``context_length``
    and it is set per request.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

DEFAULT_TIMEOUT = 600.0

# This project is offline-only by design. Every outbound request must terminate
# on this machine, so the host is checked against a loopback allowlist and a
# non-local address is a hard error rather than a warning. Enforcing it here --
# in the one place that constructs the HTTP client -- means no call site can
# opt out, including by setting OLLAMA_HOST.
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


class NonLocalHostError(RuntimeError):
    """Raised when configuration points anywhere but this machine."""


def normalize_host(host: str | None = None) -> str:
    """Turn any of the shapes people put in OLLAMA_HOST into a base URL.

    Ollama's own convention is a bare ``localhost`` or ``host:port``, which is
    not a valid URL. Goose's config had exactly this problem.

    Raises:
        NonLocalHostError: if the resulting host is not loopback.
    """
    host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    host = host.strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host

    scheme, _, rest = host.partition("//")
    authority = rest.split("/", 1)[0]
    # rpartition(":") would mangle IPv6 here -- "::1" is an address, not a
    # host:port pair -- so split the port off only where one can actually be.
    if authority.startswith("["):
        # bracketed IPv6, optionally with a port: [::1] or [::1]:11434
        bracket, _, port = authority.partition("]")
        hostname = bracket + "]"
        port = port.lstrip(":")
    elif authority.count(":") > 1:
        # bare IPv6 -- every colon belongs to the address. Bracket it so the
        # rebuilt base URL is valid.
        hostname, port = f"[{authority}]", ""
    else:
        hostname, _, port = authority.partition(":")
        hostname = hostname or authority

    if hostname.lower() not in LOOPBACK_HOSTS:
        raise NonLocalHostError(
            f"Refusing to connect to non-local host {hostname!r}. This project "
            f"only talks to Ollama on this machine. Allowed: "
            f"{', '.join(sorted(LOOPBACK_HOSTS))}."
        )

    return f"{scheme}//{hostname}:{port or '11434'}"


@dataclass
class GenResult:
    """A completion plus the timing counters Ollama returns alongside it."""

    content: str
    thinking: str = ""
    model: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    done_reason: str = ""

    prompt_tokens: int = 0
    gen_tokens: int = 0
    prompt_ns: int = 0
    gen_ns: int = 0
    load_ns: int = 0
    total_ns: int = 0

    @staticmethod
    def _rate(tokens: int, ns: int) -> float:
        return (tokens / (ns / 1e9)) if tokens and ns else 0.0

    @property
    def gen_tps(self) -> float:
        """Generation speed. The number people quote."""
        return self._rate(self.gen_tokens, self.gen_ns)

    @property
    def prompt_tps(self) -> float:
        """Prompt ingestion speed. The number that decides if an agent loop is
        bearable, since every turn re-reads the whole conversation."""
        return self._rate(self.prompt_tokens, self.prompt_ns)

    @property
    def total_s(self) -> float:
        return self.total_ns / 1e9

    def stats_line(self) -> str:
        return (
            f"{self.gen_tokens} tok @ {self.gen_tps:.1f} tok/s gen | "
            f"{self.prompt_tokens} prompt @ {self.prompt_tps:.0f} tok/s | "
            f"{self.total_s:.2f}s total"
        )


class OllamaClient:
    """Async wrapper over the endpoints we actually need."""

    def __init__(self, host: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.host = normalize_host(host)
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.host, timeout=self._timeout)

    # ---------------------------------------------------------------- queries

    async def ping(self) -> bool:
        try:
            async with self._client() as c:
                r = await c.get("/api/tags", timeout=5.0)
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        async with self._client() as c:
            r = await c.get("/api/tags")
            r.raise_for_status()
            return r.json().get("models", [])

    async def show(self, model: str) -> dict[str, Any]:
        """Model card: architecture, parameter count, context length, quant,
        and the capability list (``tools``, ``thinking``, ``vision``)."""
        async with self._client() as c:
            r = await c.post("/api/show", json={"model": model})
            r.raise_for_status()
            return r.json()

    async def ps(self) -> list[dict[str, Any]]:
        """Currently resident models, with the VRAM/RAM split."""
        async with self._client() as c:
            r = await c.get("/api/ps")
            r.raise_for_status()
            return r.json().get("models", [])

    # ------------------------------------------------------------- generation

    def _payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        think: bool | str,
        tools: list[dict[str, Any]] | None,
        context_length: int | None,
        max_tokens: int | None,
        temperature: float | None,
        keep_alive: str | None,
        extra_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if context_length is not None:
            options["num_ctx"] = context_length
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if extra_options:
            options.update(extra_options)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = tools
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        return payload

    @staticmethod
    def _result_from(data: dict[str, Any], model: str, msg: dict[str, Any]) -> GenResult:
        return GenResult(
            content=msg.get("content", "") or "",
            thinking=msg.get("thinking", "") or "",
            model=data.get("model", model),
            tool_calls=msg.get("tool_calls") or [],
            done_reason=data.get("done_reason", "") or "",
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            gen_tokens=data.get("eval_count", 0) or 0,
            prompt_ns=data.get("prompt_eval_duration", 0) or 0,
            gen_ns=data.get("eval_duration", 0) or 0,
            load_ns=data.get("load_duration", 0) or 0,
            total_ns=data.get("total_duration", 0) or 0,
        )

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        think: bool | str = False,
        tools: list[dict[str, Any]] | None = None,
        context_length: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        keep_alive: str | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> GenResult:
        payload = self._payload(
            model, messages, stream=False, think=think, tools=tools,
            context_length=context_length, max_tokens=max_tokens,
            temperature=temperature, keep_alive=keep_alive,
            extra_options=extra_options,
        )
        async with self._client() as c:
            r = await c.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        if err := data.get("error"):
            raise RuntimeError(f"Ollama returned an error: {err}")
        return self._result_from(data, model, data.get("message") or {})

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        on_content: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        think: bool | str = False,
        tools: list[dict[str, Any]] | None = None,
        context_length: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        keep_alive: str | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> GenResult:
        """Same as :meth:`chat`, but invokes callbacks as tokens arrive.

        Returns the identical GenResult once the stream completes, so callers
        get the full text and timing counters either way -- the callbacks are
        purely for display.

        Ollama streams newline-delimited JSON. Content and reasoning arrive as
        incremental fragments and are concatenated; tool calls arrive whole
        (usually in one late chunk) and are collected. The final object, flagged
        ``done``, carries the timing counters.
        """
        payload = self._payload(
            model, messages, stream=True, think=think, tools=tools,
            context_length=context_length, max_tokens=max_tokens,
            temperature=temperature, keep_alive=keep_alive,
            extra_options=extra_options,
        )

        content: list[str] = []
        thinking: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        final: dict[str, Any] = {}

        async with self._client() as c:
            async with c.stream("POST", "/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # A mid-stream {"error": ...} object (runner crash, OOM) has
                    # no "message" and no "done" -- skipping it would end the
                    # stream as a silently empty result.
                    if err := chunk.get("error"):
                        raise RuntimeError(f"Ollama returned an error mid-stream: {err}")

                    msg = chunk.get("message") or {}
                    if piece := msg.get("thinking"):
                        thinking.append(piece)
                        if on_thinking:
                            on_thinking(piece)
                    if piece := msg.get("content"):
                        content.append(piece)
                        if on_content:
                            on_content(piece)
                    if calls := msg.get("tool_calls"):
                        tool_calls.extend(calls)

                    if chunk.get("done"):
                        final = chunk

        return self._result_from(
            final,
            model,
            {
                "content": "".join(content),
                "thinking": "".join(thinking),
                "tool_calls": tool_calls,
            },
        )

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> GenResult:
        """Single-shot convenience wrapper over :meth:`chat`."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(model, messages, **kwargs)

    # -------------------------------------------------------------- lifecycle

    async def load(
        self,
        model: str,
        *,
        context_length: int | None = None,
        keep_alive: str = "30m",
    ) -> dict[str, Any]:
        """Warm a model into memory. An empty message list is Ollama's documented
        way to load without generating."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [],
            "keep_alive": keep_alive,
        }
        if context_length is not None:
            payload["options"] = {"num_ctx": context_length}
        async with self._client() as c:
            r = await c.post("/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    async def unload(self, model: str) -> dict[str, Any]:
        """Evict a model. ``keep_alive: 0`` frees the VRAM immediately, which
        matters on a 16GB card holding a 17GB model."""
        async with self._client() as c:
            r = await c.post(
                "/api/chat", json={"model": model, "messages": [], "keep_alive": 0}
            )
            r.raise_for_status()
            return r.json()
