"""
verification/claim_extractor.py
Intelligent Claim Extraction and Search Query Generation Module.
Extracts factual claims, entities (people, orgs, locations, dates), removes clickbait markers,
and detects temporal recency cues to identify breaking/live news.
"""

import re
import string
from typing import Dict, Any, List, Set

# Sensationalist & Clickbait noise terms to filter out from search queries
CLICKBAIT_TERMS = {
    "shocking", "proof", "exposed", "bombshell", "must see", "leaked", 
    "viral", "breaking", "urgent", "unbelievable", "share this", "secret",
    "big pharma", "doctors hate", "miracle", "cure", "conspiracy", "warning",
    "before it gets deleted", "you wont believe", "top secret", "insider reveals",
    "exclusive", "mind-blowing", "scandal", "100%", "guaranteed", "truth revealed"
}

# Temporal recency cues indicating current/live news events
RECENCY_TERMS = {
    "today", "yesterday", "tonight", "this morning", "this afternoon", "this week",
    "currently", "latest", "just in", "breaking", "announced today", "unveiled today",
    "minutes ago", "hours ago", "live update", "developing story", "recent",
    "2026", "2025"
}

# Standard English stopwords and query filler words
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "stated", "plans", "planned", "related", "reported", "connected", "according",
    "spokesperson", "administration"
}


def normalize_acronyms(text: str) -> str:
    """Normalizes dotted acronyms like U.S. -> US, U.K. -> UK."""
    text = re.sub(r'\bU\.S\.A\.\b', 'USA', text, flags=re.IGNORECASE)
    text = re.sub(r'\bU\.S\.\b', 'US', text, flags=re.IGNORECASE)
    text = re.sub(r'\bU\.K\.\b', 'UK', text, flags=re.IGNORECASE)
    text = re.sub(r'\bE\.U\.\b', 'EU', text, flags=re.IGNORECASE)
    text = re.sub(r'\bU\.N\.\b', 'UN', text, flags=re.IGNORECASE)
    return text


def extract_entities_and_keywords(text: str) -> Dict[str, Any]:
    """
    Extracts proper nouns, capitalized entities, organizations, acronyms,
    and key factual nouns from input text.
    """
    if not text or not text.strip():
        return {"entities": [], "keywords": [], "acronyms": []}

    normalized = normalize_acronyms(text)

    # Extract Acronyms (2 to 6 uppercase letters like NASA, ISRO, WHO, RBI, ECB, MIT, US, UK, EU)
    acronyms = re.findall(r'\b[A-Z]{2,6}\b', normalized)

    # Extract Multi-Word Capitalized Proper Nouns (e.g., "James Webb", "European Union", "Associated Press")
    multi_caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', normalized)

    # Extract Single Capitalized Nouns
    single_caps = re.findall(r'\b[A-Z][a-z]+\b', normalized)

    # Combine unique entities
    all_entities = []
    seen = set()
    for ent in acronyms + multi_caps + single_caps:
        ent_lower = ent.lower()
        if ent_lower not in CLICKBAIT_TERMS and ent_lower not in STANDARD_STOPWORDS and ent_lower not in seen:
            all_entities.append(ent)
            seen.add(ent_lower)

    # Extract remaining informative words (length >= 3, not in stopwords/clickbait)
    clean_words = re.findall(r'\b[a-zA-Z]{3,}\b', normalized)
    informative_keywords = [
        w for w in clean_words 
        if w.lower() not in STANDARD_STOPWORDS and w.lower() not in CLICKBAIT_TERMS
    ]

    return {
        "entities": all_entities,
        "keywords": list(dict.fromkeys(informative_keywords)),
        "acronyms": list(set(acronyms))
    }


# Debunk & Fact-Check detection patterns in submitted text
DEBUNK_PATTERNS = [
    (r'\b(?:fake|false|hoax|untrue|fabricated|debunked|misleading|incorrect)\b.*?\b(?:reuters|snopes|politifact|ap\s+fact|fact\s*check|boomlive|altnews|bbc\s+reality|full\s*fact)\b', 'False'),
    (r'\b(?:reuters|snopes|politifact|ap\s+fact|fact\s*check|boomlive|altnews|full\s*fact)\b.*?\b(?:debunk|rated\s+(?:it\s+)?false|found\s+(?:it\s+)?false|untrue|fake|no\s+evidence|hoax|not\s+true|fabricated)\b', 'False'),
    (r'^\s*(?:fake|false|hoax|debunked|misleading)[\.:\-\s]', 'False'),
    (r'\b(?:verdict|rating|fact\s*check\s*verdict)\s*:\s*(?:fake|false|hoax|debunked|pants\s*on\s*fire)\b', 'False'),
]


def detect_embedded_fact_check(text: str) -> Dict[str, Any]:
    """
    Detects if the user pasted a fact-checking debunk summary or a prompt that contains
    an explicit fact-check verdict (e.g., 'Fake. Reuters Fact Check reported...').
    """
    if not text or not text.strip():
        return {"has_debunk_note": False, "detected_verdict": None, "debunk_source": None}

    text_lower = text.lower()
    for pattern, verdict in DEBUNK_PATTERNS:
        match = re.search(pattern, text_lower, flags=re.DOTALL)
        if match:
            # Extract source if present
            source = "Fact-Checking Report"
            if "reuters" in text_lower:
                source = "Reuters Fact Check"
            elif "snopes" in text_lower:
                source = "Snopes"
            elif "politifact" in text_lower:
                source = "PolitiFact"
            elif "ap" in text_lower:
                source = "AP Fact Check"
            elif "boom" in text_lower:
                source = "BOOM Live"

            return {
                "has_debunk_note": True,
                "detected_verdict": verdict,
                "debunk_source": source,
                "matched_text": match.group(0)
            }

    return {"has_debunk_note": False, "detected_verdict": None, "debunk_source": None}


def detect_recency_and_current_news(text: str) -> Dict[str, Any]:
    """
    Analyzes whether the input text refers to an ongoing, breaking, or current news event.
    """
    text_lower = text.lower()
    found_cues = []

    for cue in RECENCY_TERMS:
        pattern = r'\b' + re.escape(cue) + r'\b'
        if re.search(pattern, text_lower):
            found_cues.append(cue)

    # Check for recent calendar dates (e.g. "Aug 31", "2026", "Monday", etc.)
    has_date_pattern = bool(re.search(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b', text_lower))
    if has_date_pattern:
        found_cues.append("specific date reference")

    is_current = len(found_cues) > 0

    return {
        "is_current_news": is_current,
        "recency_cues": found_cues,
        "note": "Flagged as Current / Live News event. Prioritizing Real-Time Evidence Verification." if is_current else "Historical or general claim."
    }


def extract_search_query(title: str = "", text: str = "", max_terms: int = 5) -> Dict[str, Any]:
    """
    Generates a clean, concise, high-precision search query from a news title or article body.
    Strips sensationalism, clickbait, and embedded debunk metadata, prioritizing high-value entities.
    """
    primary_source = title.strip() if title and len(title.strip()) > 5 else text.strip()
    
    if not primary_source:
        return {
            "query": "",
            "entities": [],
            "is_current_news": False,
            "recency_info": {},
            "has_debunk_note": False,
            "detected_verdict": None,
            "debunk_source": None
        }

    # Detect embedded fact checks in the entire input
    debunk_info = detect_embedded_fact_check(f"{title} {text}")

    # If text starts with or contains quoted claim (e.g. "Japan has banned..."), prioritize the quoted claim
    quote_match = re.search(r'["\u201c\u2018\']([^"\u201d\u2019\']{10,})["\u201d\u2019\']', primary_source)
    if quote_match:
        claim_candidate = quote_match.group(1)
    else:
        # If there's an explicit "Fake." or "False." separator, take the text before it
        split_parts = re.split(r'\b(?:fake|false|debunked|hoax)[\.:\-\s]', primary_source, flags=re.IGNORECASE)
        if len(split_parts) > 1 and len(split_parts[0].strip()) > 10:
            claim_candidate = split_parts[0].strip()
        else:
            claim_candidate = primary_source

    # Normalize acronyms
    clean_primary = normalize_acronyms(claim_candidate)

    # Detect recency
    recency_info = detect_recency_and_current_news(f"{title} {text}")

    # Extract entities and keywords
    extracted = extract_entities_and_keywords(clean_primary)
    entities = extracted["entities"]
    keywords = extracted["keywords"]

    # Construct query following headline token flow while keeping entities
    words = re.findall(r'\b[a-zA-Z0-9]{2,}\b', clean_primary)
    query_tokens = []
    seen = set()

    for w in words:
        w_lower = w.lower()
        if w_lower not in seen and w_lower not in STANDARD_STOPWORDS and w_lower not in CLICKBAIT_TERMS:
            query_tokens.append(w)
            seen.add(w_lower)
        if len(query_tokens) >= max_terms:
            break

    # If still short, supplement with entities/keywords
    if len(query_tokens) < 3:
        for ent in entities:
            for w in ent.split():
                w_lower = w.lower()
                if w_lower not in seen and w_lower not in STANDARD_STOPWORDS and w_lower not in CLICKBAIT_TERMS:
                    query_tokens.append(w)
                    seen.add(w_lower)
                if len(query_tokens) >= max_terms:
                    break

    final_query = " ".join(query_tokens)

    return {
        "query": final_query,
        "entities": entities[:5],
        "is_current_news": recency_info["is_current_news"],
        "recency_cues": recency_info["recency_cues"],
        "recency_note": recency_info["note"],
        "has_debunk_note": debunk_info["has_debunk_note"],
        "detected_verdict": debunk_info["detected_verdict"],
        "debunk_source": debunk_info["debunk_source"]
    }
