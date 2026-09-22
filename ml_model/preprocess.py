"""
Text Preprocessing and Feature Extraction Module for Fake News Detection.
Ensures uniform text cleaning across training, evaluation, and production inference.
"""

import re
import string
import unicodedata
from typing import List, Optional

# Standard contractions mapping
CONTRACTIONS_DICT = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would",
    "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'll": "it will", "it's": "it is", "let's": "let us", "mightn't": "might not",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will",
    "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are",
    "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is", "who'll": "who will",
    "who's": "who is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

CONTRACTION_RE = re.compile('(%s)' % '|'.join(re.escape(k) for k in CONTRACTIONS_DICT.keys()), re.IGNORECASE)

# Standard English stopwords (excluding critical negations and qualifiers)
NEGATIONS = {'no', 'not', 'nor', 'neither', 'never', 'none', 'cannot', 'against', 'without', 'fake', 'hoax', 'false', 'true', 'real'}

BASE_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just',
    'me', 'more', 'most', 'my', 'myself', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so',
    'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
    'why', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
} - NEGATIONS


def expand_contractions(text: str) -> str:
    """Expands common English contractions to their root words."""
    def replace(match):
        return CONTRACTIONS_DICT.get(match.group(0).lower(), match.group(0))
    return CONTRACTION_RE.sub(replace, text)


def clean_text(text: Optional[str]) -> str:
    """
    Standard text normalization pipeline:
    1. Unicode normalization (NFKD)
    2. URL, email, and special symbol removal
    3. Contraction expansion
    4. Lowercase conversion
    5. Removal of punctuation and excessive whitespace
    6. Stopword filtering while preserving negation markers
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Expand contractions
    text = expand_contractions(text)

    # Convert to lowercase
    text = text.lower()

    # Replace punctuation with spaces (retaining alphanumeric tokens)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra whitespace and filter stopwords
    tokens = text.split()
    filtered_tokens = [tok for tok in tokens if len(tok) > 1 and tok not in BASE_STOPWORDS]

    return ' '.join(filtered_tokens)


def extract_keywords_for_search(text: str, max_keywords: int = 6) -> str:
    """
    Extracts the most salient keywords/named claims from a headline or article
    for querying the external evidence/news retrieval API.
    """
    if not text:
        return ""
    
    # Clean but keep critical nouns/terms
    cleaned = clean_text(text)
    words = cleaned.split()
    
    # Filter out very short or generic query terms
    meaningful = [w for w in words if len(w) > 3 and w not in {'news', 'report', 'breaking', 'said', 'according', 'statement', 'claims', 'update'}]
    
    # Select top terms
    selected = meaningful[:max_keywords] if len(meaningful) >= 2 else words[:max_keywords]
    return ' '.join(selected)


# Misinformation & sensationalism lexical cues
HOAX_PATTERNS = [
    # Miracle cures & medical misinformation
    r'\b(?:miracle cure|cures? in (?:24|12|3|48) hours?|cures? (?:100|all) percent|kills? all (?:cancer|viruses?|tumors?))\b',
    r'\b(?:doctors? (?:are speechless|admit truth|suppressed)|big pharma (?:is terrified|doesn\'?t want you to know))\b',
    r'\b(?:lemon juice (?:with|and) baking soda|onions? in socks?|boiled betel leaves|papaya leaf juice cures? diabetes)\b',
    r'\b(?:never go to the hospital again|cures? type[- ]2 diabetes permanently|instant stomach cancer)\b',
    r'\b(?:microwave oven radiation mutates food|raw garlic (?:and|with) honey eliminates all cancer)\b',
    
    # Financial hoaxes & WhatsApp viral schemes
    r'\b(?:deposit (?:rs\.?|rupees)?\s*\d{3,5}|free (?:5g )?(?:smartphones?|laptops?|recharge))\b',
    r'\b(?:forward this message to \d+|share with \d+ (?:whatsapp )?groups?|whatsapp will (?:start charging|become paid))\b',
    r'\b(?:currency notes? (?:declared )?invalid starting midnight|all old \d+ rupee notes? (?:illegal|banned))\b',
    r'\b(?:confiscat(?:ing|ed|e) gold (?:jewelry )?in (?:bank )?lockers?|atms? permanently stop dispensing cash)\b',
    r'\b(?:unesco declares? national anthem best|click the (?:unverified )?link to register|before portal closes)\b',
    
    # Conspiracies & pseudo-science
    r'\b(?:chemtrails? (?:to control|secretly spraying|alter weather)|nano[- ]?gps chips? in currency)\b',
    r'\b(?:15 days of total (?:pitch )?darkness|alien (?:dyson sphere|autopsy|space bunker))\b',
    r'\b(?:5g (?:cell )?towers? (?:depletes?|vibrates?) oxygen|nano[- ]?tracking chips? injected into (?:water|vaccines?))\b',
    r'\b(?:secret leaked circular|secret circular leaked|bombshell (?:truth|revelation)|whistleblower scientists?)\b',
    r'\b(?:share before (?:government )?delet(?:es|ed)|haarp (?:caused|artificial earthquakes?)|outrageous tyranny)\b'
]

HOAX_REGEX = [re.compile(p, re.IGNORECASE) for p in HOAX_PATTERNS]

JOURNALISTIC_PATTERNS = [
    r'\b(?:the union cabinet approved|reserve bank of india (?:directed|voted|kept)|monetary policy committee)\b',
    r'\b(?:isro successfully launched|astronomers using nasa|published (?:their )?(?:findings|measurements|method|trial results) in (?:nature|science))\b',
    r'\b(?:world health organization (?:regional committee|verified|confirmed)|united states food and drug administration)\b',
    r'\b(?:supreme court of india (?:mandated|ruled)|central board of direct taxes issued|national payments corporation of india)\b',
    r'\b(?:ministry of (?:railways|road transport|health|finance|education)|india meteorological department issued)\b',
    r'\b(?:international energy agency|peer[- ]reviewed findings|phase iii trial results|authorized the first cell[- ]based)\b'
]

JOURNALISTIC_REGEX = [re.compile(p, re.IGNORECASE) for p in JOURNALISTIC_PATTERNS]


def detect_misinformation_markers(raw_text: str) -> dict:
    """
    Analyzes raw input text for stylistic misinformation markers, viral clickbait triggers,
    and authoritative journalistic indicators.
    """
    if not raw_text or not isinstance(raw_text, str):
        return {"hoax_hits": 0, "journalistic_hits": 0, "caps_ratio": 0.0, "exclamation_count": 0, "is_strongly_sensational": False}

    text_lower = raw_text.lower()
    
    # Check specific hoax patterns
    hoax_hits = sum(1 for reg in HOAX_REGEX if reg.search(text_lower))
    
    # Check journalistic indicators
    journalistic_hits = sum(1 for reg in JOURNALISTIC_REGEX if reg.search(text_lower))
    
    # Count ALL-CAPS words (excluding short acronyms of len < 3 like US, AI)
    words = raw_text.split()
    caps_words = [w for w in words if len(w) >= 3 and w.isupper() and w.isalpha() and w not in {'NASA', 'ISRO', 'CERN', 'WHO', 'FDA', 'IMD', 'SEBI', 'UGC', 'PMAY', 'GSLV', 'ATLAS', 'ATM', 'UPI', 'MIT', 'IEA', 'IUCN'}]
    caps_ratio = len(caps_words) / max(len(words), 1)
    
    # Count exclamation marks
    exclamation_count = raw_text.count('!')
    
    is_strongly_sensational = bool(
        hoax_hits >= 1 or 
        (caps_ratio > 0.15 and exclamation_count >= 2) or 
        (exclamation_count >= 3 and any(w in text_lower for w in ['secret', 'shocking', 'alert', 'urgent', 'miracle', 'warning', 'cure', 'forward']))
    )

    return {
        "hoax_hits": hoax_hits,
        "journalistic_hits": journalistic_hits,
        "caps_ratio": round(caps_ratio, 3),
        "exclamation_count": exclamation_count,
        "is_strongly_sensational": is_strongly_sensational
    }

