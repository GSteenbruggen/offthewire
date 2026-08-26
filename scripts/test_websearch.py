"""Pure-logic tests for web lookup. No network, no model.

Covers the silent-failure class that produced a confidently wrong answer:
sources that contribute nothing while appearing to contribute.

    .venv\\Scripts\\python.exe scripts\\test_websearch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from websearch import (  # noqa: E402
    SearchHit, is_low_quality, is_non_answer, is_preferred, rank_hits,
)

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL} {label}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def test_non_answer_detection() -> None:
    print("\n1. Non-answers are not counted as sources")

    # The literal contract
    check("literal NOT ANSWERED", is_non_answer("NOT ANSWERED"))

    # The observed real failure: a courteous refusal scored as a source
    observed = (
        "ANSWER: The current weather in Boston cannot be determined from the "
        "provided text as all specific data points are displayed as placeholders.\n"
        "QUOTE: Feels Like -- HHigh -- LLow Chance of Rain 0%\n"
        "CONCWER: none"
    )
    check("observed placeholder-page refusal", is_non_answer(observed))

    for phrase in (
        "ANSWER: The source does not provide the version number.",
        "ANSWER: No information about this API is available in the text.",
        "ANSWER: Unable to determine the correct signature.",
        "",
        "   ",
    ):
        check(f"refusal: {phrase.strip()[:48] or '(empty)'}", is_non_answer(phrase))

    # Real answers must survive, including ones whose CONCERN mentions a gap
    real = (
        "ANSWER: Use AsyncClient.stream(method, url) as an async context manager, "
        "then iterate with aiter_bytes().\n"
        "QUOTE: To stream a response, use client.stream(...)\n"
        "CONCERN: none"
    )
    check("genuine answer is kept", not is_non_answer(real))

    concern_mentions_gap = (
        "ANSWER: The default timeout is 5 seconds.\n"
        "QUOTE: Timeout defaults to 5s.\n"
        "CONCERN: the page does not provide a version, so this may be outdated"
    )
    check(
        "CONCERN mentioning a gap does not void the answer",
        not is_non_answer(concern_mentions_gap),
    )


def test_source_ranking() -> None:
    # The URLs below are classification fixtures -- strings fed to a pure
    # function, never fetched. They are tagged so verify_offline's source scan
    # does not read them as outbound destinations.
    print("\n2. Source ranking")

    check("docs subdomain is preferred", is_preferred("https://docs.python.org/3/library/asyncio.html"))  # offline-fixture
    check("MDN is preferred", is_preferred("https://developer.mozilla.org/en-US/docs/Web/API"))  # offline-fixture
    check("tutorial farm is low quality", is_low_quality("https://www.w3schools.com/python/"))  # offline-fixture
    check("the offending weather farm is low quality", is_low_quality("https://www.easeweather.com/x"))  # offline-fixture
    check("a neutral site is neither", not is_preferred("https://example.org/a") and not is_low_quality("https://example.org/a"))  # offline-fixture

    hits = [
        SearchHit("farm", "https://www.geeksforgeeks.org/x", ""),  # offline-fixture
        SearchHit("neutral", "https://someblog.dev/x", ""),  # offline-fixture
        SearchHit("official", "https://docs.python.org/3/x", ""),  # offline-fixture
    ]
    order = [h.title for h in rank_hits(hits)]
    check("official first, farm last", order == ["official", "neutral", "farm"], str(order))

    # ranking must not drop anything
    check("nothing is filtered out", len(rank_hits(hits)) == len(hits))


def test_private_network_refusal() -> None:
    """The web channel must never reach back into local or private networks.

    A fetched URL is attacker-influenced data -- it arrives from search
    results, or from a model that just read a page an attacker wrote. Without
    this guard the agent is a proxy into everything this machine can reach and
    the internet cannot: Ollama's own unauthenticated API on 11434, the
    router's admin page, anything on the LAN that trusts private callers.

    Offline-safe: literal IPs and 'localhost' resolve without any network.
    """
    import asyncio

    from websearch import WebSearch, WebSearchError

    print("\n3. Private-network refusal (SSRF guard)")

    ws = WebSearch(None, "m", enabled=True)

    refused = [
        ("Ollama's own API", "http://127.0.0.1:11434/api/tags"),
        ("localhost by name", "http://localhost:8080/admin"),
        ("RFC1918 router", "http://192.168.1.1/"),  # offline-fixture
        ("RFC1918 ten-net", "http://10.0.0.5/"),  # offline-fixture
        ("link-local metadata", "http://169.254.169.254/"),  # offline-fixture
        ("IPv6 loopback", "http://[::1]/"),
        ("non-HTTP scheme", "ftp://example.com/x"),  # offline-fixture
        ("file scheme", "file:///C:/Windows/win.ini"),
        ("no host at all", "http:///path"),
    ]

    async def run() -> None:
        for label, url in refused:
            try:
                await ws._refuse_non_public(url)
                check(f"refuses {label}", False, url)
            except WebSearchError:
                check(f"refuses {label}", True)
        # The one address class that must PASS: the public internet. A guard
        # that refuses everything would also pass every refusal test above.
        try:
            await ws._refuse_non_public("https://example.com/")  # offline-fixture
            check("allows a public address", True)
        except WebSearchError as e:
            # No DNS available is not a guard failure -- this suite must stay
            # runnable fully offline.
            check("allows a public address", "resolve" in str(e).lower(), str(e))

    asyncio.run(run())


def test_runtime_toggle() -> None:
    """/web on and /web off must move three things together: the enabled
    flag, the tool registry, and the system prompt's guidance block. Moving
    fewer than all three produces a model that can call tools it was never
    told about, or is told about tools that no longer exist."""
    import asyncio
    import tempfile

    from agent import Agent
    from ollama_client import OllamaClient
    from tools import Workspace
    from websearch import WebSearch

    print("\n4. Runtime web toggle")

    ws = Workspace(tempfile.mkdtemp(prefix="otw-toggle-"))
    client = OllamaClient()  # loopback-only construction; no traffic here
    web = WebSearch(client, "m", enabled=False, searxng_url="http://localhost:1")
    agent = Agent(client, "m", ws, web=web, interactive=False)

    check("starts with web off", "search_web" not in agent.tools.names)
    check("prompt says no internet", "no internet access" in agent.session.system_prompt)
    overhead_off = agent.session.tool_overhead_tokens

    async def run() -> None:
        msg = await agent.set_web(True)
        check("reports the unreachable backend", "not reachable" in msg, msg)
        check("tools registered", "search_web" in agent.tools.names
              and "fetch_url" in agent.tools.names)
        check("prompt gains web guidance", "search_web" in agent.session.system_prompt)
        check("flag is on", web.enabled is True)
        check("tool overhead grew", agent.session.tool_overhead_tokens > overhead_off)

        check("double-enable is a no-op",
              "already" in await agent.set_web(True))

        msg = await agent.set_web(False)
        check("off deregisters", "search_web" not in agent.tools.names)
        check("prompt loses web guidance",
              "no internet access" in agent.session.system_prompt)
        check("flag is off", web.enabled is False)
        check("overhead shrank back", agent.session.tool_overhead_tokens == overhead_off)

    asyncio.run(run())


def main() -> int:
    print("=" * 68)
    print("WEB LOOKUP LOGIC TESTS")
    print("=" * 68)
    test_non_answer_detection()
    test_source_ranking()
    test_private_network_refusal()
    test_runtime_toggle()
    print("\n" + "=" * 68)
    print("All web lookup tests passed." if not failures else f"{failures} FAILED.")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
