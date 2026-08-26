"""End-to-end check of the ollama-mcp server against live Ollama.

Two passes:
  1. In-memory transport (mcp 2.x lets a Client wrap an MCPServer directly) --
     exercises every tool and its generated schema, fast and without a subprocess.
  2. Real stdio subprocess -- proves the server actually boots the way MCP
     Code will launch it. Passing pass 1 but failing pass 2 usually means an
     import error that only shows up outside the test process.

    .venv\\Scripts\\python.exe scripts\\smoke_test.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# httpx logs every request at INFO and drowns the report
logging.getLogger("httpx").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp import Client, ClientSession, StdioServerParameters, stdio_client  # noqa: E402

SERVER_PATH = ROOT / "src" / "server.py"


def pick_model() -> str:
    """Whichever tool-capable model is installed, largest first.

    Hardcoding a tag means the suite breaks the day you pull a new model and
    remove the old one -- which is exactly what happened going from qwen3.6 to
    qwen3.8.
    """
    import asyncio as _asyncio

    import models as _M
    from ollama_client import OllamaClient as _Client

    async def _pick() -> str:
        rec = await _M.recommend_agent_model(_Client())
        return rec.get("recommended") or ""

    return _asyncio.run(_pick())


MODEL = pick_model()

PASS, FAIL = "  [PASS]", "  [FAIL]"

CHECKS: list[tuple[str, dict]] = [
    ("ollama_status", {}),
    ("list_models", {}),
    ("model_info", {"model": MODEL}),
    ("recommend_agent_model", {}),
    ("loaded_models", {}),
    (
        "generate",
        {
            "prompt": "Reply with exactly one word: pong",
            "model": MODEL,
            "max_tokens": 16,
            "think": False,
        },
    ),
]


def text_of(result) -> str:
    parts = []
    for block in result.content:
        if getattr(block, "text", None):
            parts.append(block.text)
    return "".join(parts)


def preview(text: str, n: int = 240) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("..." if len(text) > n else "")


async def in_memory_pass() -> int:
    from server import server as mcp_server

    failures = 0
    print("=" * 70)
    print("PASS 1: in-memory transport")
    print("=" * 70)

    async with Client(mcp_server) as client:
        tools = (await client.list_tools()).tools
        print(f"\n{len(tools)} tools advertised:")
        for t in tools:
            first_line = (t.description or "").strip().split("\n")[0]
            print(f"  - {t.name:24} {preview(first_line, 70)}")

        print("\nCalling tools:")
        for name, args in CHECKS:
            try:
                result = await client.call_tool(name, args)
                body = text_of(result)
                # mcp 2.x exposes snake_case on the model; `isError` is only
                # the wire alias.
                if result.is_error:
                    print(f"{FAIL} {name}: {preview(body)}")
                    failures += 1
                else:
                    print(f"{PASS} {name}: {preview(body)}")
            except Exception as e:
                print(f"{FAIL} {name}: {type(e).__name__}: {e}")
                failures += 1
    return failures


async def stdio_pass() -> int:
    print("\n" + "=" * 70)
    print("PASS 2: real stdio subprocess (how MCP clients launch it)")
    print("=" * 70)
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                print(f"{PASS} server booted over stdio, {len(tools)} tools listed")
                result = await session.call_tool("ollama_status", {})
                print(f"{PASS} ollama_status: {preview(text_of(result), 160)}")
        return 0
    except Exception as e:
        print(f"{FAIL} stdio boot failed: {type(e).__name__}: {e}")
        return 1


async def main() -> int:
    failures = await in_memory_pass()
    failures += await stdio_pass()
    print("\n" + "=" * 70)
    print("All checks passed." if not failures else f"{failures} check(s) FAILED.")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
