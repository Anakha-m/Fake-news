"""
Comprehensive Lexicon and Feature Extraction Helpers for Fake News Detection.
Combines subword n-grams, journalistic credibility cues, and sensationalism markers.
"""

import re
import string
import numpy as np

# Sensationalist, conspiracy, and misinformation lexical cues
SENSATIONALISM_CUES = {
    "shocking", "bombshell", "unbelievable", "mind blowing", "miracle cure", "big pharma",
    "secretly control", "illuminati", "shadow elite", "hidden truth", "they do not want you to know",
    "wake up", "leaked documents prove", "banned by doctors", "destroy tumors overnight",
    "100 percent cure", "mind control", "chemtrails", "weather control", "reptilian",
    "5g tracking", "injected chips", "microchips in water", "synthetic clone", "deep state",
    "hoax exposed", "dystopian trap", "alien pyramids", "total darkness", "hollow earth",
    "dinosaur underground", "arrest billionaire treason", "share before deleted",
    "doctors are terrified", "doctors are speechless", "secret midnight vote", "ban all gas stoves",
    "atmospheric respiration fee", "tax for breathing", "cash machines disabled permanently",
    "emergency broadcast tonight", "classified defense leak", "secret orbital mirror"
}

# Journalistic and empirical credibility cues
CREDIBILITY_CUES = {
    "according to", "spokesperson", "published in", "peer reviewed", "clinical trial",
    "world health organization", "food and drug administration", "reuters", "associated press",
    "supreme court ruled", "passed resolution", "official statement", "ministry of health",
    "department of justice", "federal reserve", "quarterly monetary policy", "statistical data",
    "spectroscopic observations", "randomized trial", "electoral commission certified",
    "international energy agency", "geological survey", "environmental protection agency",
    "materials scientists at", "astronomers confirmed", "unanimous decision", "monetary policy report"
}

def extract_stylistic_features(text: str) -> dict:
    """Extracts high-level stylistic indicators from text."""
    if not text:
        return {"exclamation_ratio": 0.0, "caps_ratio": 0.0, "sensational_hits": 0, "credibility_hits": 0}
    
    clean_lower = text.lower()
    
    # Exclamation mark frequency
    exclamations = text.count('!')
    
    # Capital letters ratio
    words = text.split()
    caps_words = [w for w in words if len(w) > 2 and w.isupper()]
    caps_ratio = len(caps_words) / max(len(words), 1)
    
    # Lexical hits
    sensational_hits = sum(1 for cue in SENSATIONALISM_CUES if cue in clean_lower)
    credibility_hits = sum(1 for cue in CREDIBILITY_CUES if cue in clean_lower)
    
    return {
        "exclamation_count": exclamations,
        "caps_ratio": caps_ratio,
        "sensational_hits": sensational_hits,
        "credibility_hits": credibility_hits
    }
