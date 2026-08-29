"""Tests for the OpenAI-compat backend client (llama.cpp / LM Studio).

Everything runs offline against httpx.MockTransport. The three things that
must never regress: the loopback guard applies to these backends exactly as
it does to Ollama; the stored session format survives a round-trip through
the OpenAI wire format unchanged; and the SSE stream parser reassembles
content, reasoning, and fragmented tool calls into the same GenResult the
Ollama client would have produced.

    .venv\\Scripts\\python.exe scripts\\test_backends.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from ollama_client import NonLocalHostError  # noqa: E402
from openai_compat import (  # noqa: E402
    OpenAICompatClient, from_openai_tool_calls, to_openai_messages,
)

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def sse(*chunks: dict) -> bytes:
    """Encode chunks the way llama-server streams them."""
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def client_with(handler) -> OpenAICompatClient:
    c = OpenAICompatClient("llamacpp")
    c.transport = httpx.MockTransport(handler)
    return c


def test_loopback_guard() -> None:
    print("\n1. Loopback guard covers the new backends")

    for bad in (
        "http://evil.example.com:8080",  # offline-fixture
        "192.168.1.50",  # offline-fixture
    ):
        try:
            OpenAICompatClient("llamacpp", bad)
            check(f"refused {bad!r}", False, "was accepted")
        except NonLocalHostError:
            check(f"refused {bad!r}", True)

    c = OpenAICompatClient("llamacpp")
    check("llama.cpp default is loopback:8080", c.host == "http://localhost:8080", c.host)
    c = OpenAICompatClient("lmstudio")
    check("LM Studio default is loopback:1234", c.host == "http://localhost:1234", c.host)
    c = OpenAICompatClient("llamacpp", "localhost")
    check("bare host gets the backend port", c.host.endswith(":8080"), c.host)

    try:
        OpenAICompatClient("nonsense")
        check("unknown backend refused", False, "was accepted")
    except ValueError:
        check("unknown backend refused", True)

    # A local model sends nothing while it processes a long prompt, and on
    # spilled-to-CPU hardware that silence exceeds any finite limit -- a
    # 600s read timeout was observed killing an honest 27B turn. The read
    # timeout must stay off for every backend.
    from ollama_client import DEFAULT_TIMEOUT

    check("no read timeout on local inference", DEFAULT_TIMEOUT.read is None)
    check("connecting still fails fast",
          (DEFAULT_TIMEOUT.connect or 999) <= 30, str(DEFAULT_TIMEOUT.connect))


def test_message_translation() -> None:
    print("\n2. Message translation")

    stored = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "look", "images": ["iVBORfake"],
         "image_paths": ["/tmp/x.png"]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": {"path": "a.py"}}},
            {"function": {"name": "run_command", "arguments": {"command": "ls"}}},
        ]},
        {"role": "tool", "tool_name": "read_file", "content": "contents"},
        {"role": "tool", "tool_name": "run_command", "content": "a.py"},
        {"role": "assistant", "content": "done"},
    ]
    wire = to_openai_messages(stored)

    check("system/plain messages pass through",
          wire[0] == {"role": "system", "content": "be brief"}
          and wire[5] == {"role": "assistant", "content": "done"})

    parts = wire[1]["content"]
    check("image becomes a data-URI content part",
          isinstance(parts, list) and parts[0]["text"] == "look"
          and parts[1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR"))
    check("image_paths never reach the wire", "image_paths" not in wire[1])

    calls = wire[2]["tool_calls"]
    check("tool calls gain ids and JSON-string arguments",
          calls[0]["id"] == "call_2_0" and calls[1]["id"] == "call_2_1"
          and json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"})
    check("tool results pair with their call ids in order",
          wire[3]["tool_call_id"] == "call_2_0"
          and wire[4]["tool_call_id"] == "call_2_1")
    check("tool_name never reaches the wire", "tool_name" not in wire[3])

    orphan = to_openai_messages([{"role": "tool", "content": "late"}])
    check("orphaned tool result still gets an id",
          orphan[0].get("tool_call_id", "").endswith("orphan"))

    check("translation is deterministic",
          to_openai_messages(stored) == wire)

    back = from_openai_tool_calls([
        {"function": {"name": "f", "arguments": '{"x": 1}'}},
        {"function": {"name": "g", "arguments": "not json {"}},
        {"function": {"name": "h", "arguments": ""}},
    ])
    check("arguments parse back to dicts", back[0]["function"]["arguments"] == {"x": 1})
    check("malformed arguments surface, not vanish",
          "_malformed_arguments" in back[1]["function"]["arguments"])
    check("empty arguments become an empty dict",
          back[2]["function"]["arguments"] == {})


def test_streaming() -> None:
    print("\n3. SSE stream parsing")

    body = sse(
        {"model": "srv-model", "choices": [{"delta": {"reasoning_content": "hmm "}}]},
        {"choices": [{"delta": {"reasoning_content": "ok."}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "read_file", "arguments": '{"pa'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": 'th": "a.py"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 321, "completion_tokens": 7},
         "timings": {"prompt_ms": 100.0, "predicted_ms": 500.0}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})

    got_content, got_thinking = [], []
    r = asyncio.run(client_with(handler).chat_stream(
        "m", [{"role": "user", "content": "hi"}],
        on_content=got_content.append, on_thinking=got_thinking.append,
    ))

    check("content assembles in order", r.content == "Hello", r.content)
    check("reasoning streams separately", r.thinking == "hmm ok."
          and got_thinking == ["hmm ", "ok."])
    check("content callback fired per delta", got_content == ["Hel", "lo"])
    check("fragmented tool call reassembles",
          r.tool_calls == [{"function": {"name": "read_file",
                                         "arguments": {"path": "a.py"}}}],
          str(r.tool_calls))
    check("usage feeds the token budget", r.prompt_tokens == 321 and r.gen_tokens == 7)
    check("timings feed the rate display", r.gen_tps > 0, f"{r.gen_tps:.1f} tok/s")
    check("non-length finish collapses to stop", r.done_reason == "stop")
    check("server model name is reported", r.model == "srv-model")

    body2 = sse(
        {"choices": [{"delta": {"content": "cut off"},
                      "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    )
    r = asyncio.run(client_with(
        lambda req: httpx.Response(200, content=body2)
    ).chat_stream("m", [{"role": "user", "content": "hi"}]))
    check("length maps to done_reason length", r.done_reason == "length")

    def err_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"content": "par"}}]},
            {"error": {"message": "slot exhausted"}},
        ))
    try:
        asyncio.run(client_with(err_handler).chat_stream(
            "m", [{"role": "user", "content": "hi"}]))
        check("mid-stream error raises", False, "no exception")
    except RuntimeError as e:
        check("mid-stream error raises", "slot exhausted" in str(e), str(e))

    def http_err(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "no --jinja"}})
    try:
        asyncio.run(client_with(http_err).chat_stream(
            "m", [{"role": "user", "content": "hi"}]))
        check("HTTP error surfaces its message", False, "no exception")
    except RuntimeError as e:
        check("HTTP error surfaces its message", "no --jinja" in str(e), str(e))


def test_non_streaming_and_generate() -> None:
    print("\n4. Non-streaming chat and generate (compaction path)")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["max_tokens"] == 800
        return httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {"content": "a summary"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 3},
        })

    r = asyncio.run(client_with(handler).generate(
        "m", "summarize this", max_tokens=800, think=False, context_length=32768))
    check("generate returns the completion", r.content == "a summary")
    check("usage mapped", r.prompt_tokens == 50 and r.gen_tokens == 3)
    check("wall clock stands in for missing timings", r.total_s >= 0)


def test_capabilities_and_probe() -> None:
    print("\n5. Capabilities and the tools probe")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={
                "default_generation_settings": {"n_ctx": 16384},
            })
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen-served"}]})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    c = client_with(handler)
    caps = asyncio.run(c.capabilities("qwen-served"))
    check("n_ctx becomes max_context", caps["max_context"] == 16384)
    check("tools assumed until probed", caps["supports_tools"] is True)

    models = asyncio.run(c.list_models())
    check("served models listed", models == [{"name": "qwen-served",
                                              "model": "qwen-served"}])
    check("probe passes on a working server",
          asyncio.run(c.probe_tools("qwen-served")) is None)

    def no_jinja(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(500, json={
                "error": {"message": "tools require a jinja template"}})
        return httpx.Response(200, json={"data": []})

    msg = asyncio.run(client_with(no_jinja).probe_tools("m"))
    check("missing --jinja produces a clear startup error",
          msg is not None and "--jinja" in msg, str(msg))

    def lmstudio_noprops(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    c = OpenAICompatClient("lmstudio")
    c.transport = httpx.MockTransport(lmstudio_noprops)
    caps = asyncio.run(c.capabilities("m"))
    check("no /props leaves max_context unset (user --context stands)",
          caps["max_context"] is None)


def main() -> int:
    print("=" * 68)
    print("OPENAI-COMPAT BACKEND TESTS")
    print("=" * 68)
    test_loopback_guard()
    test_message_translation()
    test_streaming()
    test_non_streaming_and_generate()
    test_capabilities_and_probe()
    print("\n" + "=" * 68)
    print("All backend tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
