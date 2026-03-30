"""
modules/web_research.py
-----------------------
Fetches concise factual topic text from web sources for web-assisted quiz mode.
Primary source: Wikipedia/MediaWiki APIs.
"""

from __future__ import annotations

from urllib.parse import quote

import requests

USER_AGENT = "quiz-app/1.0 (educational project)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def search_topic(query: str) -> str:
    """Resolve a user query to the most likely Wikipedia page title."""
    if not query or not query.strip():
        raise RuntimeError("Please enter a topic for web-based quiz generation.")

    session = _session()
    response = session.get(
        WIKI_API,
        params={
            "action": "opensearch",
            "search": query.strip(),
            "limit": 1,
            "namespace": 0,
            "format": "json",
        },
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    titles = payload[1] if len(payload) > 1 else []
    if not titles:
        raise RuntimeError(f"No web result found for '{query}'.")
    return titles[0]


def fetch_topic_context(query: str) -> dict:
    """
    Fetch summary and plain-text extract for a topic from Wikipedia.
    Returns a dict with title, text, summary, and source_url.
    """
    title = search_topic(query)
    session = _session()

    summary_resp = session.get(f"{WIKI_SUMMARY}/{quote(title)}", timeout=10)
    summary_resp.raise_for_status()
    summary_data = summary_resp.json()

    extract_resp = session.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "titles": title,
            "format": "json",
        },
        timeout=10,
    )
    extract_resp.raise_for_status()
    extract_data = extract_resp.json()
    pages = extract_data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})

    summary = summary_data.get("extract", "") or ""
    extract = page.get("extract", "") or summary
    if not extract.strip():
        raise RuntimeError(f"Could not fetch usable web text for '{title}'.")

    source_url = (
        summary_data.get("content_urls", {})
        .get("desktop", {})
        .get("page", f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}")
    )

    return {
        "title": summary_data.get("title", title),
        "summary": summary,
        "text": extract,
        "source_url": source_url,
    }
