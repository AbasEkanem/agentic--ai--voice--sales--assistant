"""Web search tool for the voice sales agent, powered by Tavily.

One LangChain tool (`web_search`) that wraps `langchain_tavily.TavilySearch` and
tunes it for a realtime voice pipeline:

- Async end to end, so a lookup never blocks the audio event loop.
- Never raises into the graph. A missing key, a timeout, a rate limit or a bad
  response all come back as a short, speakable message instead of an exception,
  so one failed search can never crash a live call.
- Output is shaped for speech: a direct answer first, then a few short sources
  with their domain. No raw HTML, no giant content blobs for the LLM to wade
  through before it speaks.
- The key is read from TAVILY_API_KEY, lazily on first use, so importing this
  module is cheap and never fails because of a missing key or load order.

Built on langchain_tavily (already a dependency) rather than the raw `tavily`
SDK, which is not installed in this project. Design mirrors
paystack_api.build_paystack_tools: a configurable factory plus a ready-to-use
default instance.

Env: TAVILY_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import os

from langchain_core.tools import tool

log = logging.getLogger("tavily_search")

# Import defensively so a packaging problem degrades the tool gracefully instead
# of taking down the whole agent at import time (agent_tools -> graph -> main all
# import this transitively).
try:
    from langchain_tavily import TavilySearch

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - only hit on a broken install
    TavilySearch = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = exc

# ---- defaults, tuned for a low-latency voice agent -----------------------------------

# "basic" (1 API credit) is the right default for voice: noticeably faster than
# "advanced" (2 credits), and latency matters more than exhaustive depth when the
# customer is waiting on a spoken reply. Raise it via the factory for research use.
# Allowed: basic | advanced | fast | ultra-fast.
DEFAULT_SEARCH_DEPTH = "basic"
DEFAULT_TOPIC = "general"  # general | news | finance
DEFAULT_MAX_RESULTS = 5  # fetched (feeds the synthesized answer)
DEFAULT_TIMEOUT = 12.0  # seconds allowed for the search call
DEFAULT_RETRIES = 1  # extra attempts after the first, for transient failures

MAX_QUERY_LEN = 400
MAX_SNIPPET_LEN = 300
MAX_SOURCES_IN_REPLY = 3  # how many sources to surface (kept short for speech)

# Speakable, non-technical messages. The model reads these and relays the gist.
_MSG_NOT_CONFIGURED = (
    "Web search is not available right now because it has not been set up. "
    "Please continue without it."
)
_MSG_EMPTY_QUERY = "I need something specific to search for before I can look it up."
_MSG_BAD_KEY = (
    "Web search is unavailable because the search service rejected its credentials."
)
_MSG_RATE_LIMITED = "Web search has hit its usage limit for now, so I can't look that up."
_MSG_BAD_REQUEST = "I couldn't run that search the way it was phrased. Try rewording it."
_MSG_UNREACHABLE = "I couldn't reach web search just now. Please try again in a moment."


class _NoApiKey(Exception):
    """Raised internally when no TAVILY_API_KEY is available."""


def _domain(url: str) -> str:
    """Bare hostname for a source, e.g. https://www.bbc.com/news -> bbc.com."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _clean(text: str) -> str:
    """Collapse whitespace so multi-line scraped content reads cleanly aloud."""
    return " ".join((text or "").split())


def _classify_failure(text: str) -> str:
    """Map an error message to a speakable message (and, by identity, its severity).

    Returns one of the module-level message constants. A result that is NOT
    `_MSG_UNREACHABLE` is treated as permanent (no point retrying).
    """
    t = (text or "").lower()
    if any(k in t for k in ("401", "403", "unauthor", "forbidden", "invalid api key", "api key")):
        return _MSG_BAD_KEY
    if any(k in t for k in ("429", "432", "usage limit", "rate limit", "too many requests", "quota")):
        return _MSG_RATE_LIMITED
    if any(k in t for k in ("400", "422", "bad request", "invalid request")):
        return _MSG_BAD_REQUEST
    return _MSG_UNREACHABLE


def _format(resp: dict, query: str) -> str:
    """Turn a Tavily response into a compact, speech-friendly string."""
    results = resp.get("results") or []
    answer = _clean(resp.get("answer") or "")

    if not answer and not results:
        return f'I searched the web for "{query}" but didn\'t find anything useful.'

    lines: list[str] = []
    if answer:
        lines.append(answer)

    shown = results[:MAX_SOURCES_IN_REPLY]
    if shown:
        if lines:
            lines.append("")
        lines.append("Sources:")
        for r in shown:
            title = _clean(r.get("title") or "") or "Untitled"
            snippet = _clean(r.get("content") or "")
            if len(snippet) > MAX_SNIPPET_LEN:
                snippet = snippet[:MAX_SNIPPET_LEN].rstrip() + "..."
            dom = _domain(r.get("url") or "")
            label = f"{title} ({dom})" if dom else title
            lines.append(f"- {label}: {snippet}" if snippet else f"- {label}")

    return "\n".join(lines).strip()


def build_web_search_tool(
    *,
    api_key: str | None = None,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    topic: str = DEFAULT_TOPIC,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    include_answer: bool | str = True,
):
    """Build the `web_search` LangChain tool.

    The underlying TavilySearch searcher is created lazily on first use, with the
    key resolved then (argument first, then TAVILY_API_KEY) — so importing this
    module is cheap and tolerant of load order (agent_tools imports this before it
    runs load_dotenv()). Pass `include_domains`/`exclude_domains` to scope searches
    (e.g. to a company's own site), or raise `search_depth` to "advanced" for
    deeper research at the cost of latency and credits.
    """
    attempts = max(1, 1 + retries)
    box: dict = {}

    def _get_searcher():
        if TavilySearch is None:
            raise RuntimeError(f"langchain_tavily is not importable: {_IMPORT_ERROR}")
        if "searcher" not in box:
            resolved = api_key or os.environ.get("TAVILY_API_KEY")
            if not resolved:
                raise _NoApiKey
            kwargs: dict = {
                "tavily_api_key": resolved,
                "max_results": max_results,
                "search_depth": search_depth,
                "topic": topic,
                "include_answer": include_answer,
                "include_raw_content": False,
            }
            if include_domains:
                kwargs["include_domains"] = include_domains
            if exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
            box["searcher"] = TavilySearch(**kwargs)
        return box["searcher"]

    @tool
    async def web_search(query: str) -> str:
        """Search the public web for up-to-date, external information.

        Use this when the customer asks about something current, factual, or
        outside our own knowledge base — recent news, figures or prices that
        change over time, public details about another company, or to verify a
        claim. Do NOT use it for questions about our own products, services or
        pricing (use the knowledge-base retriever for those), and never use it to
        take or confirm payments.

        Pass a short, specific query. Returns a brief answer followed by a few
        sources, or a short message when nothing useful is found.
        """
        try:
            searcher = _get_searcher()
        except _NoApiKey:
            log.warning("web_search called but TAVILY_API_KEY is not set")
            return _MSG_NOT_CONFIGURED
        except RuntimeError as exc:
            log.error("web_search unavailable: %s", exc)
            return _MSG_NOT_CONFIGURED
        except Exception as exc:  # noqa: BLE001 - construction (e.g. bad config/key)
            log.error("web_search init failed: %s", exc)
            return _classify_failure(str(exc))

        q = _clean(query)[:MAX_QUERY_LEN]
        if not q:
            return _MSG_EMPTY_QUERY

        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = await asyncio.wait_for(
                    searcher.ainvoke({"query": q}), timeout=timeout + 3.0
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
                log.warning("web_search timeout %d/%d for %r", attempt + 1, attempts, q)
            except Exception as exc:  # noqa: BLE001 - never let a search crash the call
                last_error = exc
                msg = _classify_failure(str(exc))
                if msg is not _MSG_UNREACHABLE:  # permanent — don't retry
                    log.error("web_search permanent error for %r: %s", q, exc)
                    return msg
                log.warning(
                    "web_search error %d/%d for %r: %s", attempt + 1, attempts, q, exc
                )
            else:
                if isinstance(resp, dict) and resp.get("error"):
                    err = str(resp.get("error"))
                    msg = _classify_failure(err)
                    if msg is not _MSG_UNREACHABLE:
                        log.error("web_search api error for %r: %s", q, err)
                        return msg
                    last_error = RuntimeError(err)
                    log.warning(
                        "web_search api error %d/%d for %r: %s",
                        attempt + 1,
                        attempts,
                        q,
                        err,
                    )
                elif isinstance(resp, dict):
                    return _format(resp, q)
                else:  # unexpected shape (string/other) — coerce defensively
                    return _clean(str(resp)) or _MSG_UNREACHABLE

            if attempt + 1 < attempts:
                await asyncio.sleep(0.4 * (attempt + 1))

        log.error("web_search exhausted retries for %r: %s", q, last_error)
        return _MSG_UNREACHABLE

    return web_search


# Ready-to-use default instance for the common case (reads TAVILY_API_KEY from env).
web_search = build_web_search_tool()
