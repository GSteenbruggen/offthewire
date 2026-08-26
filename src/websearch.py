"""Web lookup so the model can check facts instead of inventing them.

THIS MODULE IS THE ONLY PART OF THIS PROJECT THAT TALKS TO THE INTERNET.

Everything else -- the agent, the MCP server, the Ollama client -- is confined
to loopback by a hard guard in ``ollama_client.normalize_host``. That guard is
untouched. Web access lives here, in one file, and is **disabled by default**;
the agent enables it only when started with ``--web``.

The egress split:

  * SearXNG runs on *your* machine (localhost:8080). Queries reach search
    engines through it, so there is no API key, no account, and no vendor
    holding a log tied to you.
  * Fetching a result page is a direct request to that page's host. This is
    real outbound traffic and the reason ``--web`` is opt-in.

Why results get condensed before the agent ever sees them: a single web page is
5,000-15,000 tokens. On a 32k window driven at ~8 tok/s, dropping two raw pages
into context ends the session. So each page is stripped to article text, then
the *local* model reduces it to a short answer to the actual question. A lookup
costs ~600 tokens instead of ~20,000. The condensation runs locally, so the page
content never goes anywhere either.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_SEARXNG = "http://localhost:8080"

# Refuse to download something enormous before we have a chance to strip it.
MAX_PAGE_BYTES = 2_000_000
FETCH_TIMEOUT = 20.0
SEARCH_TIMEOUT = 20.0

# How many redirects fetch_text will follow by hand. Redirects are followed
# manually, not by httpx, because each hop has to pass the private-address
# check below -- follow_redirects=True would let a public URL 302 straight to
# localhost and be fetched without ever being inspected.
MAX_REDIRECTS = 5

# Chars of article text handed to the condenser per page. Beyond this the local
# model spends minutes reading boilerplate for no gain.
MAX_EXTRACT_CHARS = 6_000

# Target size of each condensed result, in tokens.
CONDENSE_TOKENS = 220

# Hard ceiling on pages fetched per answer() call. The tool schema advertises
# the same maximum, but schemas are advice to a model, not enforcement: a model
# that passes pages=50 would otherwise trigger 50 concurrent outbound fetches.
MAX_ANSWER_PAGES = 5

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


# Domains whose content is generally authoritative for technical questions:
# first-party documentation, standards bodies, and source repositories.
PREFERRED_DOMAIN_MARKERS = (
    "docs.", "developer.", ".readthedocs.io", "developer.mozilla.org",
    "github.com", "stackoverflow.com", "python.org", "pypi.org",
    "nodejs.org", "rust-lang.org", "go.dev", "kernel.org", "w3.org",
    "ietf.org", "microsoft.com/en-us/docs", "learn.microsoft.com",
)

# Aggregators and tutorial farms. These rank well and are frequently outdated
# or simply wrong -- the incident that motivated this list was easeweather.com
# printing Celsius values labelled as Fahrenheit, which the condenser then
# reported verbatim and with total confidence. They are demoted and labelled,
# not excluded, because occasionally they are the only thing that covers a
# topic.
LOW_QUALITY_DOMAIN_MARKERS = (
    "w3schools.com", "geeksforgeeks.org", "tutorialspoint.com",
    "javatpoint.com", "programcreek.com", "codegrepper.com",
    "delftstack.com", "askpython.com", "easeweather.com",
    "sitepoint.com/tutorials", "educba.com", "codecademy.com/resources",
)


# A model asked to reply "NOT ANSWERED" will often write a courteous refusal
# instead. Counting that as a contributing source is the same silent-failure
# class as dropping an unreadable page: an observed run scored a page of empty
# JavaScript placeholders as one of its two corroborating sources.
NON_ANSWER_MARKERS = (
    "not answered",
    "cannot be determined",
    "cannot determine",
    "could not be determined",
    "does not answer",
    "does not provide",
    "does not contain",
    "no information",
    "not available in",
    "not specified in",
    "unable to determine",
    "no specific data",
    "placeholder",
    "no relevant",
)


def is_non_answer(summary: str) -> bool:
    """True when a condensed result does not actually answer the question."""
    text = (summary or "").strip()
    if not text:
        return True
    if text.upper().startswith("NOT ANSWERED"):
        return True

    # Prefer judging the ANSWER field alone; a CONCERN or QUOTE line may
    # legitimately mention that something was unavailable. The label is matched
    # loosely because small models mistype it (an observed run wrote "CONCWER").
    for line in text.splitlines():
        stripped = line.strip()
        if stripped[:7].upper().startswith("ANSWER"):
            body = stripped.split(":", 1)[-1].lower()
            return any(m in body for m in NON_ANSWER_MARKERS)

    return any(m in text.lower() for m in NON_ANSWER_MARKERS)


def domain_of(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return (urlparse(url).netloc or "").lower().removeprefix("www.")
    except Exception:
        return ""


def is_preferred(url: str) -> bool:
    u = url.lower()
    return any(marker in u for marker in PREFERRED_DOMAIN_MARKERS)


def is_low_quality(url: str) -> bool:
    u = url.lower()
    return any(marker in u for marker in LOW_QUALITY_DOMAIN_MARKERS)


def rank_hits(hits: list["SearchHit"]) -> list["SearchHit"]:
    """Authoritative sources first, tutorial farms last, search order otherwise.

    Search engines rank by popularity, which for technical questions correlates
    poorly with correctness. This does not filter -- it only reorders, so the
    pages actually fetched skew toward first-party documentation.
    """

    def score(h: "SearchHit") -> int:
        if is_preferred(h.url):
            return 0
        if is_low_quality(h.url):
            return 2
        return 1

    return sorted(hits, key=score)


class WebSearchError(RuntimeError):
    """Recoverable: the message is handed back to the model as a tool result."""


class WebSearchDisabled(WebSearchError):
    """Raised when a web tool is called but egress was never enabled."""


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str


class WebSearch:
    """SearXNG search + page extraction + local condensation."""

    def __init__(
        self,
        ollama: Any,
        model: str,
        *,
        searxng_url: str = DEFAULT_SEARXNG,
        enabled: bool = False,
        context_length: int | None = None,
    ):
        self.ollama = ollama
        self.model = model
        self.searxng_url = searxng_url.rstrip("/")
        self.enabled = enabled
        self.context_length = context_length
        self.queries_made: list[str] = []

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise WebSearchDisabled(
                "Web access is disabled. The agent must be started with --web "
                "to allow outbound requests."
            )

    # ------------------------------------------------------------------ search

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        self._require_enabled()
        self.queries_made.append(query)

        params = {
            "q": query,
            "format": "json",
            "safesearch": "0",
            "categories": "general",
        }
        try:
            async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as c:
                r = await c.get(f"{self.searxng_url}/search", params=params)
                r.raise_for_status()
                data = r.json()
        except httpx.ConnectError:
            script = "setup_searxng.ps1" if os.name == "nt" else "setup_searxng.sh"
            raise WebSearchError(
                f"Cannot reach SearXNG at {self.searxng_url}. Start it with "
                f"scripts/{script} (Docker must be running)."
            )
        except httpx.TimeoutException:
            # Must come before the catch-all: an earlier version funneled
            # timeouts into the settings.yml diagnosis below, sending the user
            # off to edit a config file when the container was simply warming
            # up. TimeoutException covers connect, read, write and pool waits.
            raise WebSearchError(
                f"SearXNG did not answer within {SEARCH_TIMEOUT:.0f}s -- it may "
                "still be starting. Try again in a moment."
            )
        except httpx.HTTPStatusError as e:
            # 403 (and 400 on some builds) is how SearXNG refuses a format it
            # was not configured to serve -- the one failure that genuinely
            # points at settings.yml.
            if e.response.status_code in (400, 403):
                raise WebSearchError(
                    f"SearXNG returned {e.response.status_code} for a JSON "
                    "request. Its settings.yml needs 'json' added to "
                    "search.formats, then a restart."
                )
            raise WebSearchError(f"SearXNG error: {e}")
        except Exception as e:
            raise WebSearchError(
                f"SearXNG request failed ({type(e).__name__}: {e})."
            )

        hits = []
        for item in (data.get("results") or [])[:limit]:
            url = item.get("url") or ""
            if not url.startswith(("http://", "https://")):
                continue
            hits.append(
                SearchHit(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                )
            )
        if not hits:
            raise WebSearchError(f"No results for {query!r}.")
        return hits

    # ----------------------------------------------------------------- fetch

    async def _refuse_non_public(self, url: str) -> None:
        """Refuse URLs that resolve anywhere but the public internet.

        The fetch channel exists to read web pages, but a URL is
        attacker-influenced data: it arrives from search results, or from a
        model that just read a page an attacker wrote. Unchecked, that makes
        the agent a proxy into everything this machine can reach and the
        internet cannot -- Ollama's own unauthenticated API on 11434, the
        router's admin page, anything on the LAN that trusts private callers.
        The rest of this project promises the model path never leaves
        localhost; this check is the mirror image, the web path never coming
        *back* in.

        Every resolved address must be public: a hostname with one public and
        one private A record is refused, because httpx would be free to pick
        the private one.
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise WebSearchError(f"Refusing non-HTTP URL: {url}")
        host = parsed.hostname or ""
        if not host:
            raise WebSearchError(f"Refusing URL with no host: {url}")

        try:
            # getaddrinfo blocks; run it off the loop. It also handles the
            # host being a literal IP, so there is one code path, not two.
            infos = await asyncio.get_running_loop().run_in_executor(
                None, lambda: socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            )
        except socket.gaierror as e:
            raise WebSearchError(f"Could not resolve {host}: {e}")

        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise WebSearchError(
                    f"Refusing to fetch {url}: {host} resolves to {ip}, which "
                    f"is not a public internet address. The web channel does "
                    f"not reach local or private networks."
                )

    async def fetch_text(self, url: str) -> str:
        """Download a page and strip it to article text."""
        self._require_enabled()
        try:
            async with httpx.AsyncClient(
                timeout=FETCH_TIMEOUT,
                # Redirects are followed manually below so each hop is
                # checked; see _refuse_non_public.
                follow_redirects=False,
                headers={"User-Agent": USER_AGENT},
            ) as c:
                for _hop in range(MAX_REDIRECTS + 1):
                    await self._refuse_non_public(url)
                    # Stream rather than get(): a plain get() buffers the
                    # entire body before any size check runs, so the cap never
                    # actually prevented a download -- a 500 MB response would
                    # be pulled in full and only *then* rejected. Streaming
                    # lets us refuse as soon as the cap is crossed, and a
                    # Content-Length header lets us refuse before reading a
                    # single body byte.
                    async with c.stream("GET", url) as r:
                        if r.is_redirect:
                            target = r.headers.get("location", "")
                            if not target:
                                raise WebSearchError(f"Redirect with no target from {url}")
                            url = str(httpx.URL(url).join(target))
                            continue
                        r.raise_for_status()
                        declared = r.headers.get("content-length", "")
                        if declared.isdigit() and int(declared) > MAX_PAGE_BYTES:
                            raise WebSearchError(
                                f"Page too large ({declared} bytes): {url}"
                            )
                        received = 0
                        chunks: list[bytes] = []
                        async for chunk in r.aiter_bytes():
                            received += len(chunk)
                            if received > MAX_PAGE_BYTES:
                                raise WebSearchError(
                                    f"Page too large (over {MAX_PAGE_BYTES} bytes): {url}"
                                )
                            chunks.append(chunk)
                        # The stream was consumed manually, so r.text is not
                        # populated; decode with the header charset if one was
                        # declared, and never let a bad byte kill the fetch.
                        html = b"".join(chunks).decode(
                            r.encoding or "utf-8", errors="replace"
                        )
                        break
                else:
                    raise WebSearchError(
                        f"Gave up after {MAX_REDIRECTS} redirects: {url}"
                    )
        except WebSearchError:
            raise
        except Exception as e:
            raise WebSearchError(f"Could not fetch {url}: {type(e).__name__}: {e}")

        try:
            import trafilatura

            text = trafilatura.extract(
                html, include_comments=False, include_tables=True, no_fallback=False
            )
        except ImportError:
            raise WebSearchError("trafilatura is not installed; run pip install -r requirements.txt")

        if not text or len(text.strip()) < 80:
            raise WebSearchError(f"No readable article text extracted from {url}")
        return text.strip()[:MAX_EXTRACT_CHARS]

    # -------------------------------------------------------------- condense

    async def condense(self, query: str, source: str, text: str) -> str:
        """Reduce one page to a short answer, with the supporting quote.

        The quote is not decoration. Condensation is a lossy rewrite performed
        by a weak model, and it launders a bad source into confident prose --
        an observed run reported "33 F" for Boston in August because the page
        itself mislabelled Celsius values as Fahrenheit. Carrying the original
        sentence through means a wrong answer can be spotted instead of merely
        believed, and it costs about forty tokens.
        """
        result = await self.ollama.generate(
            self.model,
            (
                f"Question: {query}\n\n"
                f"Source text:\n{text}\n\n"
                "Using ONLY the source text above, reply in exactly this format:\n"
                "ANSWER: <at most three sentences. Include exact API names, "
                "signatures, version numbers or commands if present.>\n"
                "QUOTE: <the single sentence from the source that supports the "
                "answer, copied verbatim>\n"
                "CONCERN: <none, OR a note if a figure or claim contradicts the "
                "surrounding text, uses ambiguous units, or is implausible for "
                "the context>\n\n"
                "If the source does not answer the question, reply with exactly "
                "the two words: NOT ANSWERED"
            ),
            think=False,
            max_tokens=CONDENSE_TOKENS,
            context_length=self.context_length,
        )
        return result.content.strip()

    # -------------------------------------------------------------- pipeline

    async def answer(self, query: str, pages: int = 3) -> str:
        """Full lookup: search, fetch, condense, and report what happened.

        Failures are reported rather than silently dropped. An earlier version
        skipped unreadable pages without a word, so when weather.com returned
        nothing extractable (JS-rendered) the digest quietly fell through to a
        content farm and presented its bad data as the answer. If the best
        source dropped out, the caller needs to know that before trusting what
        is left.
        """
        self._require_enabled()
        # Defense in depth alongside the schema's maximum: never let a model's
        # argument choose how many concurrent outbound fetches happen.
        pages = max(1, min(int(pages), MAX_ANSWER_PAGES))
        ranked = rank_hits(await self.search(query, limit=pages + 3))
        hits, reserve = ranked[:pages], ranked[pages:]

        async def one(hit: SearchHit) -> dict[str, Any]:
            entry: dict[str, Any] = {
                "hit": hit,
                "domain": domain_of(hit.url),
                "low_quality": is_low_quality(hit.url),
            }
            try:
                text = await self.fetch_text(hit.url)
            except WebSearchError as e:
                entry["status"] = "unreadable"
                entry["detail"] = str(e)
                return entry
            try:
                entry["summary"] = await self.condense(query, hit.url, text)
                entry["status"] = (
                    "not_answered" if is_non_answer(entry["summary"]) else "ok"
                )
            except Exception as e:
                entry["status"] = "unreadable"
                entry["detail"] = f"condensation failed: {type(e).__name__}: {e}"
            return entry

        entries = list(await asyncio.gather(*(one(h) for h in hits)))

        def answered_of(es: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [e for e in es if e["status"] == "ok"]

        # Escalate rather than merely warning. Pages fail often -- 403s, paywalls
        # and JavaScript rendering knocked out three of five sources in testing --
        # and landing on a single readable source is exactly the situation that
        # produced a confident wrong answer. Spend the extra fetches to get a
        # second opinion instead of making the user ask twice.
        if len(answered_of(entries)) < 2 and reserve:
            extra = await asyncio.gather(*(one(h) for h in reserve[:2]))
            entries.extend(extra)

        answered = answered_of(entries)
        problems = [e for e in entries if e["status"] != "ok"]

        lines: list[str] = [f"Results for {query!r}:"]

        for e in answered:
            hit: SearchHit = e["hit"]
            flag = "  [LOW-QUALITY SOURCE]" if e["low_quality"] else ""
            lines.append(f"\n- {hit.title or hit.url}{flag}\n  {hit.url}\n  {e['summary']}")

        if problems:
            lines.append("\nSources that did NOT contribute:")
            for e in problems:
                hit = e["hit"]
                why = (
                    "no readable text extracted (often a JavaScript-rendered page)"
                    if e["status"] == "unreadable" and "readable" in e.get("detail", "")
                    else e.get("detail", "did not answer the question")
                )
                lines.append(f"  - {hit.url}\n    {why}")

        if not answered:
            lines.append(
                "\nNo page answered the question. Raw search snippets, which are "
                "NOT verified:"
            )
            for e in entries:
                hit = e["hit"]
                lines.append(f"  - {hit.url}\n    {hit.snippet}")
            return "\n".join(lines)

        # Corroboration. A single source is unfalsifiable: with nothing to
        # contradict it, a wrong answer is indistinguishable from a right one.
        # The observed bad run had exactly one readable source and reported its
        # mislabelled units as fact; the run that got it right had two and
        # noticed they disagreed. Say which situation the caller is in.
        if len(answered) == 1:
            lines.append(
                "\nUNCORROBORATED: only one source could be read, so nothing "
                "cross-checks this. Do not present it as established fact. If it "
                "matters, search again with different wording to find a second "
                "source."
            )
        elif len(answered) >= 2:
            lines.append(
                f"\nCROSS-CHECK: {len(answered)} independent sources above. "
                "Compare their figures and quotes before answering. If they "
                "disagree, report the disagreement rather than picking one."
            )

        if all(e["low_quality"] for e in answered):
            lines.append(
                "WARNING: every source that answered is a low-authority site. "
                "Treat these figures and API details as unverified."
            )
        if problems:
            lines.append(
                "Note: some sources could not be read, so this rests on fewer "
                "sources than requested."
            )

        return "\n".join(lines)

    # ----------------------------------------------------------------- status

    async def status(self) -> dict[str, Any]:
        """Health snapshot, shown at --web startup and by /web.

        The probe hits SearXNG's /config endpoint, which SearXNG answers
        entirely by itself. An earlier version issued a real search (q=test),
        which fanned out to upstream engines on every startup -- outbound
        traffic the user never asked for -- and never appeared in
        queries_made, so the egress accounting understated reality. /config
        also reports the enabled result formats on builds that expose them,
        which verifies the 'json' requirement without a search. A real search
        remains only as a fallback for builds without /config, and it is
        recorded in queries_made like any other outbound query.
        """
        info: dict[str, Any] = {
            "enabled": self.enabled,
            "searxng_url": self.searxng_url,
            "queries_this_session": len(self.queries_made),
        }
        if not self.enabled:
            info["note"] = "Web access is off. Start the agent with --web to enable."
            return info
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{self.searxng_url}/config")
                if r.status_code == 200:
                    info["searxng_reachable"] = True
                    try:
                        formats = (r.json().get("search") or {}).get("formats")
                    except Exception:
                        formats = None
                    if isinstance(formats, list):
                        info["json_format_enabled"] = "json" in formats
                    return info
                # No /config on this build: fall back to one real search so
                # reachability can still be judged, and account for it -- this
                # one does fan out to upstream engines.
                self.queries_made.append("status probe: test")
                info["queries_this_session"] = len(self.queries_made)
                r = await c.get(
                    f"{self.searxng_url}/search",
                    params={"q": "test", "format": "json"},
                )
                info["searxng_reachable"] = r.status_code == 200
                if r.status_code != 200:
                    info["searxng_status"] = r.status_code
        except Exception as e:
            info["searxng_reachable"] = False
            info["error"] = f"{type(e).__name__}: {e}"
        return info
