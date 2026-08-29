"""Async client for OpenAI-compatible local servers: llama.cpp and LM Studio.

Both expose ``/v1/chat/completions`` with the same request shape, so one
client serves both; ``backend`` only selects the default port and the
startup hints. The agent talks to this class through the same duck-typed
surface as ``OllamaClient`` and receives the same ``GenResult``, so nothing
above the client knows which runtime produced a reply.

Session files stay in the storage format the rest of the program uses --
Ollama-shaped messages. Translation to the OpenAI wire format happens here,
per request, in ``to_openai_messages``: images become data-URI content
parts, tool calls gain synthesized ids and JSON-string arguments, and tool
results are paired with those ids by position. Replies are translated back
before anyone sees them. A conversation started against one backend
resumes against another.

The same offline rule applies as everywhere else: the host is validated at
construction against the loopback allowlist, and a non-local address is a
hard error no configuration can bypass.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

from ollama_client import DEFAULT_TIMEOUT, GenResult, normalize_host

BACKENDS: dict[str, dict[str, str]] = {
    "llamacpp": {
        "label": "llama.cpp (llama-server)",
        "port": "8080",
        "hint": "start it with:  llama-server -m model.gguf --jinja -c 32768",
    },
    "lmstudio": {
        "label": "LM Studio",
        "port": "1234",
        "hint": "enable the local server in LM Studio (Developer tab) and load a model",
    },
}


def _image_mime(b64: str) -> str:
    """Sniff the format from the base64 prefix; the bytes already tell us."""
    if b64.startswith("/9j/"):
        return "image/jpeg"
    if b64.startswith("R0lGO"):
        return "image/gif"
    if b64.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate stored (Ollama-shaped) messages to the OpenAI wire format.

    Tool-call ids do not exist in the stored format, so they are synthesized
    deterministically from message position -- ``call_<msg index>_<call
    index>`` -- and the ``role: tool`` results that follow an assistant
    message consume those ids in order. Determinism matters: the same
    conversation must serialize identically on every request, or the model
    sees phantom edits in its own history.
    """
    out: list[dict[str, Any]] = []
    pending_ids: list[str] = []

    for i, m in enumerate(messages):
        role = m.get("role", "user")

        if role == "assistant" and m.get("tool_calls"):
            ids = [f"call_{i}_{j}" for j in range(len(m["tool_calls"]))]
            pending_ids = list(ids)
            out.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": [
                    {
                        "id": id_,
                        "type": "function",
                        "function": {
                            "name": (tc.get("function") or {}).get("name", ""),
                            "arguments": json.dumps(
                                (tc.get("function") or {}).get("arguments") or {}
                            ),
                        },
                    }
                    for id_, tc in zip(ids, m["tool_calls"])
                ],
            })
        elif role == "tool":
            # An orphaned result (its call compacted away) still needs *an*
            # id -- servers reject the role without one.
            call_id = pending_ids.pop(0) if pending_ids else f"call_{i}_orphan"
            out.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(m.get("content") or ""),
            })
        elif m.get("images"):
            parts: list[dict[str, Any]] = [
                {"type": "text", "text": m.get("content") or ""}
            ]
            for b64 in m["images"]:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{_image_mime(b64)};base64,{b64}"},
                })
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": m.get("content") or ""})

    return out


def from_openai_tool_calls(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI tool calls (JSON-string arguments) back to the stored shape."""
    out = []
    for c in calls or []:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                # Hand the malformed string to the tool layer, whose error
                # message teaches the model to correct itself -- silently
                # dropping the call would leave a dangling assistant turn.
                args = {"_malformed_arguments": args}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args or {}}})
    return out


class OpenAICompatClient:
    """Speaks ``/v1/chat/completions`` to a loopback llama.cpp or LM Studio.

    Duck-type-compatible with ``OllamaClient`` for everything the agent
    uses: ``ping``, ``chat``, ``chat_stream``, ``generate``, ``list_models``,
    ``capabilities``, ``ps``, ``load``/``unload`` (no-ops -- the server owns
    the model's lifecycle).
    """

    # The server's context size is set at launch (-c) and cannot move per
    # request; /maxtokens reads this to explain instead of pretending.
    context_is_fixed = True

    # Tests inject an httpx.MockTransport here; None means the real network
    # stack (which the loopback-validated base_url confines to this machine).
    transport: Any = None

    def __init__(
        self,
        backend: str = "llamacpp",
        host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(BACKENDS)}")
        self.backend = backend
        port = BACKENDS[backend]["port"]
        # An explicit host string means OLLAMA_HOST is never consulted for
        # these backends; the loopback check still applies to every shape.
        self.host = normalize_host(host or f"http://localhost:{port}", default_port=port)
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.host, timeout=self._timeout, transport=self.transport
        )

    # ---------------------------------------------------------------- queries

    async def ping(self) -> bool:
        try:
            async with self._client() as c:
                r = await c.get("/v1/models", timeout=5.0)
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """The served model list, in the dict shape models.py expects."""
        async with self._client() as c:
            r = await c.get("/v1/models")
            r.raise_for_status()
            data = r.json().get("data") or []
        return [{"name": m.get("id", ""), "model": m.get("id", "")} for m in data]

    async def props(self) -> dict[str, Any]:
        """llama.cpp's /props: the launch-time context size and model path.
        Returns {} on servers that do not expose it (LM Studio)."""
        try:
            async with self._client() as c:
                r = await c.get("/props", timeout=5.0)
                if r.status_code != 200:
                    return {}
                return r.json()
        except Exception:
            return {}

    async def capabilities(self, model: str) -> dict[str, Any]:
        """Same shape as models.model_capabilities, from what this API offers.

        There is no capability manifest here. Tool support is verified by an
        actual probe at startup (``probe_tools``); the context ceiling comes
        from /props on llama.cpp and stays None on LM Studio, which lets the
        user's --context stand.
        """
        props = await self.props()
        ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        return {
            "name": model,
            "architecture": "",
            "parameters": "",
            "quantization": "",
            "max_context": ctx,
            "capabilities": ["tools"],
            "supports_tools": True,
            "supports_thinking": True,
            "supports_vision": False,
            "backend": BACKENDS[self.backend]["label"],
        }

    async def probe_tools(self, model: str) -> str | None:
        """Verify the server accepts a tools request; the error string if not.

        llama-server launched without ``--jinja`` rejects tools with a 500
        mid-conversation -- the worst possible moment. One tiny request at
        startup turns that into a clear sentence before any work begins.
        """
        probe_tool = {
            "type": "function",
            "function": {
                "name": "noop",
                "description": "no-op",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        try:
            async with self._client() as c:
                r = await c.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "tools": [probe_tool],
                        "max_tokens": 1,
                    },
                    timeout=120.0,
                )
            if r.status_code == 200:
                return None
            detail = ""
            try:
                detail = (r.json().get("error") or {}).get("message", "")
            except Exception:
                detail = r.text[:200]
            hint = (
                " (llama-server must be launched with --jinja for tool calling)"
                if self.backend == "llamacpp"
                else ""
            )
            return f"the server rejected a tool-calling request: {detail}{hint}"
        except Exception as e:
            return f"the tool-calling probe failed: {type(e).__name__}: {e}"

    async def ps(self) -> list[dict[str, Any]]:
        """No resident-model introspection on this API."""
        return []

    # ------------------------------------------------------------- generation

    def _payload(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        think: bool | str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": to_openai_messages(messages),
            "stream": stream,
        }
        if stream:
            # Ask for the final usage chunk; the measured token budget
            # (compaction, truncation recovery) depends on prompt_tokens.
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if self.backend == "llamacpp":
            # Honored by qwen3-style chat templates, ignored harmlessly by
            # the rest. Effort strings do not map to this API; any truthy
            # think enables the block.
            payload["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        return payload

    @staticmethod
    def _finish_to_done(finish_reason: str | None) -> str:
        # "length" is the one value with behavior attached upstream
        # (truncation recovery); everything else collapses to "stop".
        return "length" if finish_reason == "length" else "stop"

    @staticmethod
    def _result_from(
        *,
        model: str,
        content: str,
        thinking: str,
        tool_calls: list[dict[str, Any]],
        finish_reason: str | None,
        usage: dict[str, Any],
        timings: dict[str, Any],
        wall_s: float,
    ) -> GenResult:
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        gen_tokens = usage.get("completion_tokens", 0) or 0
        # llama.cpp reports real phase timings; without them (LM Studio),
        # the wall clock is attributed to generation so tok/s stays roughly
        # honest instead of reading zero.
        prompt_ns = int(timings.get("prompt_ms", 0) * 1e6)
        gen_ns = int(timings.get("predicted_ms", 0) * 1e6)
        if not gen_ns:
            gen_ns = int(wall_s * 1e9)
        return GenResult(
            content=content,
            thinking=thinking,
            model=model,
            tool_calls=from_openai_tool_calls(tool_calls),
            done_reason=OpenAICompatClient._finish_to_done(finish_reason),
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            prompt_ns=prompt_ns,
            gen_ns=gen_ns,
            total_ns=int(wall_s * 1e9),
        )

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        think: bool | str = False,
        tools: list[dict[str, Any]] | None = None,
        context_length: int | None = None,  # fixed at server launch; accepted for interface parity
        max_tokens: int | None = None,
        temperature: float | None = None,
        keep_alive: str | None = None,  # no equivalent on this API
        extra_options: dict[str, Any] | None = None,
    ) -> GenResult:
        payload = self._payload(
            model, messages, stream=False, think=think, tools=tools,
            max_tokens=max_tokens, temperature=temperature,
        )
        started = time.monotonic()
        async with self._client() as c:
            r = await c.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        if err := data.get("error"):
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise RuntimeError(f"{BACKENDS[self.backend]['label']} returned an error: {msg}")

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return self._result_from(
            model=data.get("model", model),
            content=msg.get("content") or "",
            thinking=msg.get("reasoning_content") or "",
            tool_calls=msg.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
            timings=data.get("timings") or {},
            wall_s=time.monotonic() - started,
        )

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
        """SSE streaming: ``data: {...}`` lines terminated by ``[DONE]``.

        Content and reasoning arrive as deltas. Tool calls arrive as indexed
        fragments whose argument strings concatenate per index; they are
        assembled whole once the stream ends, because a half-received JSON
        string is not a call.
        """
        payload = self._payload(
            model, messages, stream=True, think=think, tools=tools,
            max_tokens=max_tokens, temperature=temperature,
        )

        content: list[str] = []
        thinking: list[str] = []
        # index -> {"name": str, "arguments": [fragments]}
        partial_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        model_name = model
        started = time.monotonic()

        async with self._client() as c:
            async with c.stream("POST", "/v1/chat/completions", json=payload) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "replace")
                    detail = body[:300]
                    try:
                        detail = (json.loads(body).get("error") or {}).get("message", detail)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"{BACKENDS[self.backend]['label']} returned "
                        f"HTTP {r.status_code}: {detail}"
                    )
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue  # comments and keep-alives
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    if err := chunk.get("error"):
                        msg = err.get("message", err) if isinstance(err, dict) else err
                        raise RuntimeError(
                            f"{BACKENDS[self.backend]['label']} returned an "
                            f"error mid-stream: {msg}"
                        )

                    model_name = chunk.get("model", model_name)
                    if u := chunk.get("usage"):
                        usage = u
                    if t := chunk.get("timings"):
                        timings = t

                    for choice in chunk.get("choices") or []:
                        if fr := choice.get("finish_reason"):
                            finish_reason = fr
                        delta = choice.get("delta") or {}
                        if piece := delta.get("reasoning_content"):
                            thinking.append(piece)
                            if on_thinking:
                                on_thinking(piece)
                        if piece := delta.get("content"):
                            content.append(piece)
                            if on_content:
                                on_content(piece)
                        for frag in delta.get("tool_calls") or []:
                            idx = frag.get("index", 0)
                            slot = partial_calls.setdefault(
                                idx, {"name": "", "arguments": []}
                            )
                            fn = frag.get("function") or {}
                            if name := fn.get("name"):
                                slot["name"] = name
                            if args := fn.get("arguments"):
                                slot["arguments"].append(args)

        tool_calls = [
            {
                "function": {
                    "name": slot["name"],
                    "arguments": "".join(slot["arguments"]),
                }
            }
            for _, slot in sorted(partial_calls.items())
        ]
        return self._result_from(
            model=model_name,
            content="".join(content),
            thinking="".join(thinking),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            timings=timings,
            wall_s=time.monotonic() - started,
        )

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        **kwargs: Any,
    ) -> GenResult:
        """Single-shot convenience wrapper over :meth:`chat` (compaction uses this)."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(model, messages, **kwargs)

    # -------------------------------------------------------------- lifecycle

    async def load(self, model: str, **kwargs: Any) -> dict[str, Any]:
        """The server owns the model's lifecycle; nothing to do."""
        return {}

    async def unload(self, model: str) -> dict[str, Any]:
        return {}
