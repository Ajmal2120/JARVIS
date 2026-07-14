"""
Web search via DuckDuckGo (ddgs library, free, no API key).

We try `ddgs` first (the maintained package), then fall back to the legacy
`duckduckgo_search` for older installs.
"""
from __future__ import annotations

from typing import Any


def _ddgs_text(query: str, max_results: int) -> list[dict[str, Any]]:
    """Try the maintained `ddgs` package."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _legacy_ddg_text(query: str, max_results: int) -> list[dict[str, Any]]:
    """Fall back to the renamed `duckduckgo_search` package."""
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Run a web search and return a list of result dicts.
    Each result has: title, href, body.
    """
    if not query or not query.strip():
        return []
    for impl in (_ddgs_text, _legacy_ddg_text):
        try:
            return impl(query, max_results)
        except Exception:
            continue
    return [{"title": "Search error", "href": "", "body": "Search failed: no working DuckDuckGo client"}]


def format_results(results: list[dict[str, Any]]) -> str:
    """Format search results into a text block for Claude to consume."""
    if not results:
        return "No search results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}\n   {href}\n   {body}")
    return "\n\n".join(lines)
