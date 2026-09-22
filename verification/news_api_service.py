"""
Modular News and Evidence Retrieval Service (Component B).
Queries external news sources and fact-checking indexes to verify claims,
with graceful fallback and resilience against network failures or missing API keys.
"""

import os
import re
import urllib.parse
from typing import Dict, Any, List, Optional
import requests

from ml_model.preprocess import extract_keywords_for_search
from verification.stance_analyzer import analyze_evidence_stance, synthesize_final_status


class EvidenceRetrievalService:
    """
    Evidence retrieval manager that checks external news APIs, public search indexes,
    and fact-checking sources.
    """

    def __init__(self):
        self.news_api_key = os.environ.get("NEWS_API_KEY", "").strip()
        self.gnews_api_key = os.environ.get("GNEWS_API_KEY", "").strip()
        self.timeout = 3.5  # Max seconds to wait for external API

    def retrieve_evidence(self, news_text: str) -> Dict[str, Any]:
        """
        Main entry point for verifying evidence for an input headline or article text.

        Returns:
            dict: {
                'status': 'SUPPORTING' | 'CONFLICTING' | 'INSUFFICIENT' | 'UNAVAILABLE',
                'summary': str,
                'sources': list of { 'title', 'url', 'source', 'snippet' },
                'query_used': str,
                'api_available': bool
            }
        """
        if not news_text or len(news_text.strip()) == 0:
            return {
                "status": "INSUFFICIENT",
                "summary": "No text provided for evidence search.",
                "sources": [],
                "query_used": "",
                "api_available": True
            }

        # Extract search query terms
        query = extract_keywords_for_search(news_text, max_keywords=5)
        if not query or len(query.split()) < 1:
            # Fallback to first 6 words
            words = news_text.split()[:6]
            query = ' '.join([w for w in words if len(w) > 2])

        articles = []
        api_error_occurred = False

        # Attempt 1: NewsAPI if configured
        if self.news_api_key:
            try:
                articles = self._query_newsapi(query)
            except Exception as e:
                api_error_occurred = True

        # Attempt 2: GNews if configured and no articles found
        if not articles and self.gnews_api_key:
            try:
                articles = self._query_gnews(query)
            except Exception as e:
                api_error_occurred = True

        # Attempt 3: Public Knowledge & News Fallback (DuckDuckGo / Wikipedia API)
        if not articles:
            try:
                articles = self._query_public_news_search(query)
            except Exception as e:
                api_error_occurred = True

        # Handle complete failure or network offline
        if not articles and api_error_occurred and not self._is_online():
            return {
                "status": "UNAVAILABLE",
                "summary": "Evidence verification is currently unavailable. The result below is based on the trained machine learning model.",
                "sources": [],
                "query_used": query,
                "api_available": False
            }

        if not articles:
            return {
                "status": "INSUFFICIENT",
                "summary": f"No recent news reports were found matching the key query terms: '{query}'.",
                "sources": [],
                "query_used": query,
                "api_available": True
            }

        # Analyze stance of retrieved articles against submitted news
        evidence_status, evidence_summary = analyze_evidence_stance(news_text, articles)

        return {
            "status": evidence_status,
            "summary": evidence_summary,
            "sources": articles[:5],
            "query_used": query,
            "api_available": True
        }

    def _query_newsapi(self, query: str) -> List[Dict[str, Any]]:
        """Queries NewsAPI.org."""
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": self.news_api_key
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 200:
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("name", "News Source"),
                    "snippet": item.get("description", "") or item.get("content", "")
                })
            return articles
        return []

    def _query_gnews(self, query: str) -> List[Dict[str, Any]]:
        """Queries GNews.io."""
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "lang": "en",
            "max": 5,
            "apikey": self.gnews_api_key
        }
        resp = requests.get(url, params=params, timeout=self.timeout)
        if resp.status_code == 200:
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("name", "GNews Source"),
                    "snippet": item.get("description", "")
                })
            return articles
        return []

    def _query_public_news_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Queries public search API (DuckDuckGo Instant / Wikipedia Search)
        as a keyless, zero-configuration evidence fallback.
        """
        results = []
        
        # 1. DuckDuckGo Instant Search API
        try:
            ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
            headers = {"User-Agent": "FakeNewsDetectionSDG/1.0"}
            resp = requests.get(ddg_url, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("AbstractText", "")
                heading = data.get("Heading", "")
                abstract_url = data.get("AbstractURL", "")
                source = data.get("AbstractSource", "Reference Index")
                
                if abstract and heading:
                    results.append({
                        "title": heading,
                        "url": abstract_url or "#",
                        "source": source,
                        "snippet": abstract
                    })

                # Related Topics
                for topic in data.get("RelatedTopics", [])[:3]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({
                            "title": topic.get("Text", "")[:60] + "...",
                            "url": topic.get("FirstURL", "#"),
                            "source": "Related Public Source",
                            "snippet": topic.get("Text", "")
                        })
        except Exception:
            pass

        # 2. Wikipedia Search API as additional public reference source
        if not results:
            try:
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&utf8=1"
                headers = {"User-Agent": "FakeNewsDetectionSDG/1.0"}
                resp = requests.get(wiki_url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    search_items = data.get("query", {}).get("search", [])
                    for item in search_items[:3]:
                        title = item.get("title", "")
                        snippet = re.sub(r'<.*?>', '', item.get("snippet", ""))
                        page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                        results.append({
                            "title": f"Encyclopedia Reference: {title}",
                            "url": page_url,
                            "source": "Wikipedia Public Knowledge",
                            "snippet": snippet
                        })
            except Exception:
                pass

        return results

    def _is_online(self) -> bool:
        """Checks if internet connectivity is active."""
        try:
            requests.get("https://1.1.1.1", timeout=1.5)
            return True
        except Exception:
            return False


# Singleton service instance
_evidence_service = EvidenceRetrievalService()


def get_evidence_for_news(news_text: str) -> Dict[str, Any]:
    """Convenience helper to retrieve evidence for a given news text."""
    return _evidence_service.retrieve_evidence(news_text)
