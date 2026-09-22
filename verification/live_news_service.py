"""
verification/live_news_service.py
Live News and External Web Evidence Retrieval Service.
Fetches recent news articles and corroborated reporting from live news APIs
(NewsAPI, GNews, DuckDuckGo News, and Wikipedia OpenSearch) with zero-key fallback.
"""

import os
import json
import logging
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)

# Request timeout in seconds
DEFAULT_TIMEOUT = 3.5


class LiveNewsService:
    """
    Manages connections to external live news APIs and search engines.
    """

    def __init__(self):
        self.news_api_key = os.environ.get("NEWS_API_KEY", "").strip()
        self.gnews_api_key = os.environ.get("GNEWS_API_KEY", "").strip()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TruthPulse-FakeNewsDetector/2.0 (Academic Research; contact@truthpulse.org)"
        })

    def search_live_news(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Searches recent news across configured APIs and public news search engines.
        Returns normalized list of articles with publication dates and sources.
        """
        if not query or not query.strip():
            return {
                "success": False,
                "service_mode": "Empty Query",
                "articles": [],
                "articles_found": 0,
                "error": "No query provided for evidence retrieval."
            }

        articles = []
        service_mode = "Zero-Key Fallback (Open Search)"
        errors = []

        # 1. Primary Attempt: NewsAPI.org if API key is present
        if self.news_api_key:
            try:
                newsapi_results = self._query_newsapi(query, max_results)
                if newsapi_results:
                    articles.extend(newsapi_results)
                    service_mode = "NewsAPI.org (Live Commercial Feed)"
            except Exception as e:
                logger.warning(f"NewsAPI error: {e}")
                errors.append(f"NewsAPI: {str(e)}")

        # 2. Secondary Attempt: GNews API if key present and articles needed
        if len(articles) < 3 and self.gnews_api_key:
            try:
                gnews_results = self._query_gnews(query, max_results - len(articles))
                if gnews_results:
                    articles.extend(gnews_results)
                    service_mode = "GNews API (Live Global Feed)"
            except Exception as e:
                logger.warning(f"GNews error: {e}")
                errors.append(f"GNews: {str(e)}")

        # 3. Tertiary Attempt: DuckDuckGo News / Web Instant Search (No API Key Required)
        if len(articles) < 2:
            try:
                ddg_results = self._query_duckduckgo(query, max_results - len(articles))
                if ddg_results:
                    articles.extend(ddg_results)
            except Exception as e:
                logger.warning(f"DuckDuckGo search error: {e}")
                errors.append(f"DuckDuckGo: {str(e)}")

        # 4. Quaternary Attempt: Wikipedia OpenSearch API (Knowledge Verification)
        if len(articles) < 2:
            try:
                wiki_results = self._query_wikipedia(query, max_results - len(articles))
                if wiki_results:
                    articles.extend(wiki_results)
            except Exception as e:
                logger.warning(f"Wikipedia search error: {e}")
                errors.append(f"Wikipedia: {str(e)}")

        # Deduplicate articles by title and URL
        unique_articles = self._deduplicate_articles(articles)

        return {
            "success": True,
            "query_used": query,
            "service_mode": service_mode,
            "articles": unique_articles[:max_results],
            "articles_found": len(unique_articles[:max_results]),
            "errors": errors if errors else None
        }

    def _query_newsapi(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Queries NewsAPI.org /v2/everything endpoint sorted by recency."""
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": max_results,
            "language": "en",
            "apiKey": self.news_api_key
        }
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            return []

        data = resp.json()
        articles_raw = data.get("articles", [])
        results = []

        for item in articles_raw:
            pub_date = item.get("publishedAt", "")
            formatted_date = self._format_date(pub_date)
            
            results.append({
                "source": item.get("source", {}).get("name", "News Source"),
                "title": item.get("title", "").strip(),
                "snippet": item.get("description") or item.get("content") or "No excerpt available.",
                "url": item.get("url", ""),
                "published_date": formatted_date,
                "provider": "NewsAPI"
            })
        return results

    def _query_gnews(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Queries GNews API."""
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "lang": "en",
            "max": max_results,
            "apikey": self.gnews_api_key
        }
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            return []

        data = resp.json()
        articles_raw = data.get("articles", [])
        results = []

        for item in articles_raw:
            pub_date = item.get("publishedAt", "")
            formatted_date = self._format_date(pub_date)

            results.append({
                "source": item.get("source", {}).get("name", "GNews Source"),
                "title": item.get("title", "").strip(),
                "snippet": item.get("description", "No excerpt available."),
                "url": item.get("url", ""),
                "published_date": formatted_date,
                "provider": "GNews"
            })
        return results

    def _query_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Queries DuckDuckGo Instant Answer and Related Topics."""
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            return []

        data = resp.json()
        results = []

        # Check Abstract
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append({
                "source": data.get("AbstractSource", "DuckDuckGo Knowledge Index"),
                "title": data.get("Heading", query.title()),
                "snippet": data.get("AbstractText"),
                "url": data.get("AbstractURL"),
                "published_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "provider": "DuckDuckGo"
            })

        # Check Related Topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if "Text" in topic and "FirstURL" in topic:
                text_content = topic["Text"]
                title_part = text_content.split(" - ")[0] if " - " in text_content else text_content[:60]
                results.append({
                    "source": "Open Web Reference",
                    "title": title_part,
                    "snippet": text_content,
                    "url": topic["FirstURL"],
                    "published_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "provider": "DuckDuckGo"
                })

        return results

    def _query_wikipedia(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Queries Wikipedia OpenSearch API."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": max_results,
            "namespace": 0,
            "format": "json"
        }
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            return []

        data = resp.json()
        if len(data) < 4:
            return []

        titles = data[1]
        descriptions = data[2]
        urls = data[3]
        results = []

        for i in range(len(titles)):
            if descriptions[i]:
                results.append({
                    "source": "Wikipedia Verified Encyclopedia",
                    "title": titles[i],
                    "snippet": descriptions[i],
                    "url": urls[i],
                    "published_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "provider": "Wikipedia"
                })

        return results

    def _format_date(self, date_str: str) -> str:
        """Formats ISO date strings into readable YYYY-MM-DD."""
        if not date_str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            # Handle ISO format like 2026-08-31T04:25:00Z
            clean_date = date_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_date)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return date_str[:10] if len(date_str) >= 10 else date_str

    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates articles by title and URL."""
        seen_titles = set()
        seen_urls = set()
        unique = []

        for art in articles:
            t_key = art.get("title", "").strip().lower()
            u_key = art.get("url", "").strip().lower()

            if t_key and t_key not in seen_titles and (not u_key or u_key not in seen_urls):
                seen_titles.add(t_key)
                if u_key:
                    seen_urls.add(u_key)
                unique.append(art)

        return unique


# Singleton helper instance
_live_news_service = LiveNewsService()

def retrieve_live_news(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Helper function to fetch live news for a query."""
    return _live_news_service.search_live_news(query=query, max_results=max_results)
