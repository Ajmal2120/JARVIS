"""
Free external APIs: weather (Open-Meteo) and news (RSS).
No keys required.

Uses httpx.AsyncClient so calls don't block the event loop, and respects
HTTP_PROXY/HTTPS_PROXY env vars if set (which your network may require).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional
from urllib.parse import quote

import httpx


logger = logging.getLogger("jarvis.apis")


def _proxy_url() -> Optional[str]:
    return os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None


# -----------------------------------------------------------------------------
# Weather — Open-Meteo
# -----------------------------------------------------------------------------

async def get_weather(lat: float, lon: float) -> dict[str, Any]:
    """
    Return current weather and a short forecast for the given coordinates.
    Falls back to a structured error dict on failure.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
        f"&timezone=auto"
        f"&forecast_days=3"
    )
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            proxy=_proxy_url(),
            follow_redirects=True,
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        return {
            "ok": True,
            "location": {"lat": lat, "lon": lon},
            "current": data.get("current_weather", {}),
            "daily": data.get("daily", {}),
        }
    except Exception as e:
        logger.warning("get_weather failed: %s", e)
        return {"ok": False, "error": str(e)}


def format_weather(w: dict[str, Any]) -> str:
    if not w.get("ok"):
        return f"Weather lookup failed: {w.get('error', 'unknown error')}"
    cur = w.get("current", {})
    daily = w.get("daily", {})
    parts = []
    if cur:
        parts.append(
            f"Currently: {cur.get('temperature', '?')}°C, "
            f"wind {cur.get('windspeed', '?')} km/h, "
            f"weather code {cur.get('weathercode', '?')}."
        )
    if daily:
        days = daily.get("time", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        prcp = daily.get("precipitation_sum", [])
        for i, day in enumerate(days[:3]):
            try:
                parts.append(
                    f"{day}: high {tmax[i]}°C, low {tmin[i]}°C, "
                    f"precipitation {prcp[i]} mm."
                )
            except (IndexError, TypeError):
                pass
    return "\n".join(parts) if parts else "Weather data unavailable."


# -----------------------------------------------------------------------------
# News — RSS feeds (no key required)
# -----------------------------------------------------------------------------

_NEWS_FEEDS = [
    ("BBC World",            "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Tech",             "http://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("Reuters Top",          "https://feeds.reuters.com/Reuters/worldNews"),
    ("Hacker News (Front)",  "https://hnrss.org/frontpage"),
    ("Ars Technica",         "https://feeds.arstechnica.com/arstechnica/index"),
]


async def get_news(topic: Optional[str] = None, max_per_feed: int = 3) -> list[dict[str, str]]:
    """
    Pull headlines from free RSS feeds. If topic is provided, prefer feeds
    that mention it in their top entries. Falls back to general headlines.
    """
    out: list[dict[str, str]] = []
    try:
        import feedparser  # type: ignore
    except ImportError:
        return [{"source": "system", "title": "news unavailable",
                 "link": "", "summary": "feedparser not installed"}]

    for source, url in _NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = getattr(entry, "title", "")
                link = getattr(entry, "link", "")
                summary = getattr(entry, "summary", "")
                if topic and topic.lower() not in (title + summary).lower():
                    continue
                out.append({
                    "source": source,
                    "title": title,
                    "link": link,
                    "summary": summary[:300],
                })
        except Exception as e:
            logger.debug("news feed %s failed: %s", source, e)
            continue
    if not out and topic:
        # If we filtered too aggressively, return general headlines
        return await get_news(topic=None, max_per_feed=max_per_feed)
    return out[:15]


def format_news(items: list[dict[str, str]]) -> str:
    if not items:
        return "No news headlines available right now."
    lines = []
    for it in items:
        lines.append(f"- [{it['source']}] {it['title']}\n  {it['link']}")
    return "\n".join(lines)
