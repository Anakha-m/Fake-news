"""
verification/fact_check_service.py
Dedicated Fact-Checking Service Module.
Queries verified fact-checking indexes (Google Fact Check Tools API / ClaimReview,
Snopes, PolitiFact, Reuters Fact Check, BoomLive, FactCheck.org) to determine if a submitted
claim has already been debunked or verified by recognized fact-checkers.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)

# Request timeout
DEFAULT_TIMEOUT = 3.5

# Recognized authoritative fact-checking organizations
FACT_CHECK_ORGANIZATIONS = {
    "snopes.com": "Snopes",
    "politifact.com": "PolitiFact",
    "factcheck.org": "FactCheck.org",
    "reuters.com/fact-check": "Reuters Fact Check",
    "apnews.com/hub/ap-fact-check": "AP Fact Check",
    "boomlive.in": "BOOM Live",
    "altnews.in": "Alt News",
    "vishvasnews.com": "Vishvas News",
    "bbc.com/news/reality_check": "BBC Reality Check",
    "fullfact.org": "Full Fact"
}

# Standardized rating keywords
RATING_MAP = {
    "false": "False",
    "pants on fire": "False",
    "mostly false": "False",
    "fake": "False",
    "hoax": "False",
    "incorrect": "False",
    "fabricated": "False",
    "true": "True",
    "mostly true": "True",
    "correct": "True",
    "misleading": "Misleading",
    "out of context": "Misleading",
    "half true": "Misleading",
    "unproven": "Unverified",
    "unverified": "Unverified",
    "disputed": "Misleading"
}


class FactCheckService:
    """
    Service for querying fact-check registries and structured ClaimReview databases.
    """

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_FACT_CHECK_API_KEY", "").strip()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TruthPulse-FactChecker/2.0 (Academic Verification System)"
        })

    def search_fact_check(self, query: str) -> Dict[str, Any]:
        """
        Searches for formal fact-checks matching the query claim.
        
        Returns:
            dict: {
                "found": bool,
                "rating": "False" | "True" | "Misleading" | "Unverified" | "Not Available",
                "fact_checker": str,
                "review_url": str,
                "claim_reviewed": str,
                "summary": str
            }
        """
        if not query or not query.strip():
            return self._empty_response("No query provided for fact-check search.")

        # 1. Primary: Google Fact Check Tools API if key configured
        if self.api_key:
            try:
                result = self._query_google_fact_check_api(query)
                if result.get("found"):
                    return result
            except Exception as e:
                logger.warning(f"Google Fact Check API error: {e}")

        # 2. Secondary: Open Fact Check Database / ClaimReview Search Fallback
        try:
            open_result = self._query_open_fact_check_search(query)
            if open_result.get("found"):
                return open_result
        except Exception as e:
            logger.warning(f"Open Fact Check search error: {e}")

        return self._empty_response("No existing verified fact-check record found in public fact-check indexes.")

    def _query_google_fact_check_api(self, query: str) -> Dict[str, Any]:
        """Queries Google Fact Check Tools API /v1alpha1/claims:search."""
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query,
            "key": self.api_key,
            "languageCode": "en"
        }
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            return {"found": False}

        data = resp.json()
        claims = data.get("claims", [])
        if not claims:
            return {"found": False}

        top_claim = claims[0]
        claim_reviews = top_claim.get("claimReview", [])
        if not claim_reviews:
            return {"found": False}

        review = claim_reviews[0]
        raw_rating = review.get("textualRating", "Unverified").lower()
        normalized_rating = self._normalize_rating(raw_rating)

        publisher = review.get("publisher", {}).get("name", "Recognized Fact-Checker")
        review_url = review.get("url", "")
        claim_text = top_claim.get("text", query)

        return {
            "found": True,
            "rating": normalized_rating,
            "raw_rating": review.get("textualRating", "Unverified"),
            "fact_checker": publisher,
            "review_url": review_url,
            "claim_reviewed": claim_text,
            "summary": f"This claim was formally investigated by {publisher} and rated '{review.get('textualRating', normalized_rating)}'."
        }

    def _query_open_fact_check_search(self, query: str) -> Dict[str, Any]:
        """
        Open search fallback: searches DuckDuckGo for dedicated fact-checking notices.
        """
        search_query = f"{query} fact check"
        url = "https://api.duckduckgo.com/"
        params = {
            "q": search_query,
            "format": "json",
            "no_html": 1
        }
        resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            return {"found": False}

        data = resp.json()
        related = data.get("RelatedTopics", [])

        for item in related:
            text = item.get("Text", "")
            first_url = item.get("FirstURL", "").lower()

            # Check if source is a recognized fact-checker
            for domain, org_name in FACT_CHECK_ORGANIZATIONS.items():
                if domain in first_url:
                    text_lower = text.lower()
                    rating = "False" if any(k in text_lower for k in ["false", "debunk", "fake", "hoax"]) else "Misleading"
                    return {
                        "found": True,
                        "rating": rating,
                        "raw_rating": rating,
                        "fact_checker": org_name,
                        "review_url": item.get("FirstURL", ""),
                        "claim_reviewed": query,
                        "summary": f"Fact-checking article identified from {org_name}: '{text[:120]}...'"
                    }

        return {"found": False}

    def _normalize_rating(self, raw_rating: str) -> str:
        """Maps diverse textual ratings to standard categories."""
        raw_lower = raw_rating.lower().strip()
        for key, val in RATING_MAP.items():
            if key in raw_lower:
                return val
        return "Unverified"

    def _empty_response(self, summary: str) -> Dict[str, Any]:
        return {
            "found": False,
            "rating": "Not Available",
            "raw_rating": "N/A",
            "fact_checker": "N/A",
            "review_url": "",
            "claim_reviewed": "",
            "summary": summary
        }


# Singleton instance
_fact_check_service = FactCheckService()

def check_fact_check_registry(query: str) -> Dict[str, Any]:
    """Helper function to execute fact-check search."""
    return _fact_check_service.search_fact_check(query=query)
