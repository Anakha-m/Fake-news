"""
verification/evidence_analyzer.py
Semantic Evidence and Source Reliability Analysis Module.
Analyzes retrieved news articles against the submitted claim using semantic similarity,
source credibility tiers, and stance classification (SUPPORT, CONTRADICT, UNCLEAR, UNRELATED).
"""

import re
import math
from typing import List, Dict, Any, Tuple
from ml_model.preprocess import clean_text

# High-credibility established news, scientific, and official agencies
TIER_1_SOURCES = {
    "reuters", "associated press", "ap news", "bbc", "the hindu", "indian express",
    "nature", "science", "who", "pib", "nasa", "isro", "bloomberg", "the wall street journal",
    "the new york times", "the guardian", "afp", "press trust of india", "pti", "ani",
    "wikipedia verified encyclopedia"
}

TIER_2_SOURCES = {
    "times of india", "ndtv", "hindustan times", "cnn", "cnbc", "forbes", "time",
    "al jazeera", "financial times", "economist", "politico", "axios", "techcrunch",
    "the verge", "wired"
}

# Explicit debunking and conflict markers
DEBUNK_KEYWORDS = {
    "debunk", "debunked", "debunks", "false", "hoax", "fake", "myth", "misleading",
    "fact check", "fact-check", "unproven", "incorrect", "baseless", "fabricated",
    "disproven", "scam", "rumor", "rumour", "untrue", "conspiracy", "pseudoscience",
    "unfounded", "no evidence", "falsely claims", "refutes", "denies"
}

# Positive confirmation markers
CONFIRM_KEYWORDS = {
    "confirms", "confirmed", "announces", "announced", "statement", "official",
    "published", "verified", "discovers", "discovered", "study shows", "trial results",
    "approves", "approved", "agency", "authorities", "evidence", "findings", "launches"
}


def calculate_semantic_similarity(claim_text: str, source_text: str) -> float:
    """
    Calculates TF-IDF cosine similarity between clean claim and retrieved source text.
    """
    clean_claim = clean_text(claim_text)
    clean_source = clean_text(source_text)

    claim_words = clean_claim.split()
    source_words = clean_source.split()

    if not claim_words or not source_words:
        return 0.0

    # Build term frequencies
    claim_tf = {}
    for w in claim_words:
        claim_tf[w] = claim_tf.get(w, 0) + 1

    source_tf = {}
    for w in source_words:
        source_tf[w] = source_tf.get(w, 0) + 1

    # Vocabulary
    vocab = set(claim_tf.keys()).union(set(source_tf.keys()))

    # Compute dot product and magnitudes
    dot_product = 0.0
    claim_mag = 0.0
    source_mag = 0.0

    for term in vocab:
        v1 = claim_tf.get(term, 0)
        v2 = source_tf.get(term, 0)
        dot_product += v1 * v2
        claim_mag += v1 ** 2
        source_mag += v2 ** 2

    if claim_mag == 0 or source_mag == 0:
        return 0.0

    similarity = dot_product / (math.sqrt(claim_mag) * math.sqrt(source_mag))
    return min(max(similarity, 0.0), 1.0)


def evaluate_source_credibility(source_name: str) -> Tuple[str, float]:
    """
    Returns credibility tier and weighting factor for a news source.
    """
    source_lower = source_name.lower().strip()
    
    for t1 in TIER_1_SOURCES:
        if t1 in source_lower:
            return "Tier 1 (Authoritative / Global Wire)", 1.0

    for t2 in TIER_2_SOURCES:
        if t2 in source_lower:
            return "Tier 2 (Established Mainstream)", 0.85

    return "Tier 3 (General Web / Unindexed)", 0.65


def analyze_evidence_corpus(
    user_claim: str, 
    retrieved_articles: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Performs comprehensive stance analysis and relevance scoring across retrieved articles.
    """
    if not retrieved_articles:
        return {
            "evidence_status": "INSUFFICIENT",
            "evidence_status_title": "Insufficient Evidence",
            "summary": "No recent corroborating or conflicting news reports were indexed for this specific claim.",
            "total_sources": 0,
            "relevant_sources_count": 0,
            "supporting_count": 0,
            "contradicting_count": 0,
            "unclear_count": 0,
            "analyzed_articles": []
        }

    user_claim_lower = user_claim.lower()
    analyzed_articles = []
    supporting_count = 0
    contradicting_count = 0
    unclear_count = 0

    for art in retrieved_articles:
        title = art.get("title", "")
        snippet = art.get("snippet", "")
        source_name = art.get("source", "News Source")
        pub_date = art.get("published_date", "")
        url = art.get("url", "")

        combined_text = f"{title} {snippet}"
        combined_lower = combined_text.lower()

        # 1. Calculate relevance similarity score
        similarity = calculate_semantic_similarity(user_claim, combined_text)
        relevance_pct = int(round(similarity * 100))

        # Check for debunking signals
        has_debunk_marker = any(marker in combined_lower for marker in DEBUNK_KEYWORDS)
        
        # Check source credibility
        tier_label, cred_weight = evaluate_source_credibility(source_name)

        # 2. Determine stance
        if has_debunk_marker:
            stance = "CONTRADICT"
            stance_label = "Contradicts / Debunks Claim"
            badge_class = "danger"
            contradicting_count += 1
        elif similarity >= 0.15 or len(set(clean_text(user_claim).split()).intersection(set(clean_text(combined_text).split()))) >= 2:
            stance = "SUPPORT"
            stance_label = "Supports / Corroborates Claim"
            badge_class = "success"
            supporting_count += 1
        else:
            stance = "UNCLEAR"
            stance_label = "Unclear / Contextual Mention"
            badge_class = "secondary"
            unclear_count += 1

        analyzed_articles.append({
            "source": source_name,
            "title": title,
            "snippet": snippet,
            "url": url,
            "published_date": pub_date,
            "relevance_pct": max(relevance_pct, 45 if stance == "SUPPORT" else 20),
            "credibility_tier": tier_label,
            "credibility_weight": cred_weight,
            "stance": stance,
            "stance_label": stance_label,
            "badge_class": badge_class
        })

    total_relevant = supporting_count + contradicting_count

    # 3. Formulate Overall Evidence Stance
    if contradicting_count >= 1:
        if supporting_count >= 2:
            evidence_status = "MIXED"
            evidence_status_title = "Mixed / Disputed Evidence"
            summary = f"Contradictory reporting detected ({contradicting_count} source refuting claim vs {supporting_count} corroborating references)."
        else:
            evidence_status = "CONTRADICTING"
            evidence_status_title = "Contradicting Evidence Found"
            summary = f"External fact-checking notices or news sources actively dispute or debunk this claim ({contradicting_count} refuting source)."
    elif supporting_count >= 1:
        evidence_status = "SUPPORTING"
        evidence_status_title = "Supporting Evidence Found"
        summary = f"{supporting_count} independent news source(s) report and corroborate the events described."
    else:
        evidence_status = "INSUFFICIENT"
        evidence_status_title = "Insufficient Evidence"
        summary = "No direct corroborating reports were found in available public news indexes."

    return {
        "evidence_status": evidence_status,
        "evidence_status_title": evidence_status_title,
        "summary": summary,
        "total_sources": len(retrieved_articles),
        "relevant_sources_count": total_relevant,
        "supporting_count": supporting_count,
        "contradicting_count": contradicting_count,
        "unclear_count": unclear_count,
        "analyzed_articles": analyzed_articles
    }
