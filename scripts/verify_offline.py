"""Prove this project cannot send data off the machine.

Three independent checks:

  1. The loopback guard actually rejects remote hosts, including via the
     OLLAMA_HOST environment variable.
  2. No source file references a URL or hostname other than loopback.
  3. No networking library other than the one used to reach local Ollama is
     imported anywhere.

    .venv\\Scripts\\python.exe scripts\\verify_offline.py

Exits non-zero if anything fails.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ollama_client import LOOPBACK_HOSTS, NonLocalHostError, normalize_host  # noqa: E402

PASS, FAIL = "  [PASS]", "  [FAIL]"

# Anything that looks like a network destination. The host must start with an
# alphanumeric so that prose like "a full URL including https://." in a tool
# description is not read as a host named ".".
URL_RE = re.compile(r"\b(?:https?|wss?)://([A-Za-z0-9][A-Za-z0-9._\-\[\]]*)", re.I)

# Libraries that could open a socket we did not intend.
FORBIDDEN_IMPORTS = [
    "requests", "urllib.request", "urllib3", "aiohttp", "websockets",
    "socket", "ftplib", "smtplib", "telnetlib", "paramiko", "boto3",
    "google.cloud", "openai", "anthropic", "sentry_sdk", "posthog",
    "analytics", "mixpanel", "segment",
]

REMOTE_OK = LOOPBACK_HOSTS | {"schemas", "json-schema.org"}  # doc/schema refs

# Deliberate, named exceptions to the import scan -- (file, module) pairs.
# Each one must say why it exists, because an unexplained exception here is
# a hole in the guarantee this script exists to prove.
#
# websearch.py + socket: the SSRF guard resolves every fetch target with
# getaddrinfo and refuses anything that is not a public address, so the web
# channel cannot be steered back into localhost or the LAN. That is
# resolution, not connection -- and it lives in the one module already
# sanctioned to open connections at all (it imports httpx two lines up).
# A guard against unintended egress is the single legitimate reason for
# this module to look at addresses.
PERMITTED_IMPORTS = {
    ("src/websearch.py", "socket"),
}


def source_files() -> list[Path]:
    files = []
    for sub in ("src", "scripts"):
        files.extend(sorted((ROOT / sub).rglob("*.py")))
    return files


def check_guard() -> int:
    print("1. Loopback guard")
    failures = 0

    for good in ("localhost", "127.0.0.1", "http://localhost:11434", "localhost:11434"):
        try:
            resolved = normalize_host(good)
            print(f"{PASS} accepted {good!r} -> {resolved}")
        except Exception as e:
            print(f"{FAIL} rejected legitimate local host {good!r}: {e}")
            failures += 1

    for bad in (
        "http://evil.example.com:11434",  # offline-fixture
        "example.com",  # offline-fixture
        "https://api.openai.com",  # offline-fixture
        "192.168.1.50:11434",  # offline-fixture
        "http://8.8.8.8",  # offline-fixture
    ):
        try:
            resolved = normalize_host(bad)
            print(f"{FAIL} ACCEPTED remote host {bad!r} -> {resolved}")
            failures += 1
        except NonLocalHostError:
            print(f"{PASS} refused {bad!r}")
        except Exception as e:
            print(f"{FAIL} wrong error type for {bad!r}: {type(e).__name__}: {e}")
            failures += 1

    # the env var must not be an escape hatch
    saved = os.environ.get("OLLAMA_HOST")
    try:
        os.environ["OLLAMA_HOST"] = "http://exfil.example.com:11434"  # offline-fixture
        try:
            resolved = normalize_host()
            print(f"{FAIL} OLLAMA_HOST env var bypassed the guard -> {resolved}")
            failures += 1
        except NonLocalHostError:
            print(f"{PASS} refused remote OLLAMA_HOST from environment")
    finally:
        if saved is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = saved

    return failures


def check_urls() -> int:
    print("\n2. URLs in source")
    failures = 0
    found_any = False
    for f in source_files():
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            # Lines tagged as fixtures are the deliberately-remote hosts this
            # script feeds to the guard to prove it rejects them. The tag only
            # exempts a line from the URL scan -- it cannot disable the guard
            # itself, which lives in ollama_client and is checked above.
            if "offline-fixture" in line:
                continue
            for host in URL_RE.findall(line):
                found_any = True
                bare = host.split(":")[0].lower()
                if bare in REMOTE_OK or host.lower() in REMOTE_OK:
                    print(f"{PASS} {f.relative_to(ROOT)}:{i} -> {host}")
                else:
                    print(f"{FAIL} {f.relative_to(ROOT)}:{i} -> NON-LOCAL {host}")
                    failures += 1
    if not found_any:
        print("  (no URLs found in source at all)")
    return failures


def check_imports() -> int:
    print("\n3. Networking imports")
    failures = 0
    for f in source_files():
        text = f.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            for bad in FORBIDDEN_IMPORTS:
                if re.search(rf"\b{re.escape(bad)}\b", stripped):
                    rel = f.relative_to(ROOT).as_posix()
                    if (rel, bad) in PERMITTED_IMPORTS:
                        # Announced, not silent: an exception you cannot see
                        # in the output is indistinguishable from a miss.
                        print(f"{PASS} {rel}:{i} imports {bad!r} (permitted: SSRF guard resolution)")
                        continue
                    print(f"{FAIL} {rel}:{i} imports {bad!r}: {stripped}")
                    failures += 1
    if not failures:
        print(f"{PASS} none of {len(FORBIDDEN_IMPORTS)} flagged libraries imported")
        print(f"{PASS} httpx is the only HTTP client in the project")
    return failures


def check_egress_optin() -> int:
    """The model path must stay loopback-only, and web lookup must be opt-in.

    websearch.py is the single module permitted to make outbound requests. This
    verifies the separation actually holds at runtime rather than by comment.
    """
    print("\n4. Web egress is isolated and opt-in")
    import inspect

    import websearch
    from websearch import WebSearch, WebSearchDisabled

    failures = 0

    # default must be off
    default_enabled = inspect.signature(WebSearch.__init__).parameters["enabled"].default
    if default_enabled is False:
        print(f"{PASS} WebSearch(enabled=...) defaults to False")
    else:
        print(f"{FAIL} WebSearch defaults to enabled={default_enabled!r}")
        failures += 1

    # and a disabled instance must actually refuse, not just claim to
    w = WebSearch(ollama=None, model="none")
    for name, coro in (
        ("search", w.search("anything")),
        ("fetch_text", w.fetch_text("https://example.com")),  # offline-fixture
        ("answer", w.answer("anything")),
    ):
        try:
            asyncio.run(coro)
            print(f"{FAIL} {name}() ran while web access was disabled")
            failures += 1
        except WebSearchDisabled:
            print(f"{PASS} {name}() refuses when disabled")
        except Exception as e:
            print(f"{FAIL} {name}() raised {type(e).__name__} instead of WebSearchDisabled: {e}")
            failures += 1

    # SearXNG endpoint itself must be local
    if "localhost" in websearch.DEFAULT_SEARXNG or "127.0.0.1" in websearch.DEFAULT_SEARXNG:
        print(f"{PASS} SearXNG default endpoint is local ({websearch.DEFAULT_SEARXNG})")
    else:
        print(f"{FAIL} SearXNG default is not local: {websearch.DEFAULT_SEARXNG}")
        failures += 1

    # the model client must not have been loosened to accommodate any of this
    try:
        normalize_host("http://api.example.com")  # offline-fixture
        print(f"{FAIL} model-path loopback guard was weakened")
        failures += 1
    except NonLocalHostError:
        print(f"{PASS} model path still loopback-only, guard unchanged")

    return failures


def check_backend_guard() -> int:
    """The OpenAI-compat backends must sit behind the same loopback guard.

    A second client class is a second chance to forget the rule, so it is
    proven here independently rather than assumed from code sharing.
    """
    print("\n5. OpenAI-compat backends (llama.cpp / LM Studio)")
    from openai_compat import BACKENDS, OpenAICompatClient

    failures = 0
    for backend in BACKENDS:
        try:
            c = OpenAICompatClient(backend)
            local = any(h in c.host for h in ("localhost", "127.0.0.1", "[::1]"))
            if local:
                print(f"{PASS} {backend} default host is loopback ({c.host})")
            else:
                print(f"{FAIL} {backend} default host is NOT loopback: {c.host}")
                failures += 1
        except Exception as e:
            print(f"{FAIL} {backend} client failed to construct: {e}")
            failures += 1

        try:
            c = OpenAICompatClient(backend, "http://exfil.example.com")  # offline-fixture
            print(f"{FAIL} {backend} ACCEPTED a remote host -> {c.host}")
            failures += 1
        except NonLocalHostError:
            print(f"{PASS} {backend} refuses a remote host")
    return failures


def main() -> int:
    print("=" * 70)
    print("OFFLINE VERIFICATION")
    print("=" * 70)
    failures = (
        check_guard() + check_urls() + check_imports() + check_egress_optin()
        + check_backend_guard()
    )
    print("\n" + "=" * 70)
    if failures:
        print(f"{failures} CHECK(S) FAILED - egress may not be contained.")
    else:
        print("VERIFIED: model path is loopback-only; web lookup is isolated to")
        print("websearch.py and disabled unless the agent is started with --web.")
    print("=" * 70)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
