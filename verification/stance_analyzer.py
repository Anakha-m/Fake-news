"""
Stance and Evidence Alignment Analyzer.
Analyzes retrieved search results and news articles against the user's submitted claim
to determine if external reporting supports, conflicts with, or provides insufficient evidence.
"""

import re
from typing import List, Dict, Any, Tuple
from ml_model.preprocess import clean_text

DEBUNK_KEYWORDS = {
    "debunk", "debunked", "debunks", "false", "hoax", "fake", "myth", "misleading", 
    "fact check", "fact-check", "unproven", "incorrect", "baseless", "fabricated",
    "disproven", "scam", "rumor", "rumour", "untrue", "conspiracy", "pseudoscience"
}

CONFIRM_KEYWORDS = {
    "confirms", "confirmed", "reports", "announced", "statement", "official", 
    "published", "verified", "discover", "discovered", "study shows", "trial results",
    "approves", "approved", "agency", "authorities", "evidence", "findings"
}


def analyze_evidence_stance(
    user_text: str, 
    retrieved_articles: List[Dict[str, Any]]
) -> Tuple[str, str]:
    """
    Evaluates retrieved news articles to deduce evidence stance.

    Returns:
        tuple: (evidence_status, evidence_summary)
            evidence_status: 'SUPPORTING' | 'CONFLICTING' | 'INSUFFICIENT' | 'UNAVAILABLE'
            evidence_summary: Human-readable summary of evidence analysis
    """
    if not retrieved_articles:
        return (
            "INSUFFICIENT",
            "No direct corroborating or conflicting news reports were found in available public news indexes for this specific claim."
        )

    user_clean = clean_text(user_text)
    user_words = set(user_clean.split())

    debunk_hits = 0
    corroboration_hits = 0
    relevant_sources = 0

    for item in retrieved_articles:
        title = item.get("title", "").lower()
        snippet = item.get("snippet", "").lower()
        combined_source_text = f"{title} {snippet}"

        # Check for debunking signals in headlines and snippets
        has_debunk = any(keyword in combined_source_text for keyword in DEBUNK_KEYWORDS)
        
        # Check for keyword overlap with user claim
        source_clean = clean_text(combined_source_text)
        source_words = set(source_clean.split())
        overlap = user_words.intersection(source_words)
        overlap_ratio = len(overlap) / max(len(user_words), 1)

        if overlap_ratio >= 0.15 or len(overlap) >= 2:
            relevant_sources += 1
            if has_debunk:
                debunk_hits += 1
            else:
                corroboration_hits += 1

    # Stance Decision Logic
    if debunk_hits >= 1:
        evidence_status = "CONFLICTING"
        evidence_summary = (
            "External fact-checking records or news reports indicate active debunking notices, "
            "contradictory evidence, or refute this specific claim."
        )
    elif corroboration_hits >= 1 and relevant_sources >= 1:
        evidence_status = "SUPPORTING"
        evidence_summary = (
            "Matching news reports or public reference records corroborate "
            "the core subject matter described in this submission."
        )
    else:
        evidence_status = "INSUFFICIENT"
        evidence_summary = (
            "Retrieved public records did not contain direct verification for this exact claim."
        )

    return evidence_status, evidence_summary


def synthesize_final_status(
    ml_prediction: str, 
    ml_confidence: float, 
    evidence_status: str
) -> str:
    """
    Synthesizes ML prediction with evidence status to generate the final status.
    Delivers clear, unambiguous verdicts.
    """
    if evidence_status == "CONFLICTING":
        return "Likely Fake / Disputed (Refuted by External Evidence)"

    if evidence_status == "SUPPORTING":
        if ml_prediction == "REAL":
            return "Likely Real (Corroborated by External News)"
        else:
            return "Mixed / Disputed (ML flags stylistic anomalies, but topic found in news)"

    # If evidence is INSUFFICIENT or UNAVAILABLE:
    if ml_prediction == "REAL":
        return "Likely Real (Based on Linguistic Analysis)"
    else:
        return "Likely Fake (Based on Linguistic Analysis)"
