"""
verification/aggregator.py
Master Coordinator for the Dual-Engine Fake News Detection Pipeline.
Connects the ML Stylistic Classifier with Claim Extraction, Live News Retrieval,
Fact-Check Search, Semantic Evidence Analysis, and Decision Synthesis.
"""

from typing import Dict, Any, Optional
from ml_model.predict import predict_news
from verification.claim_extractor import extract_search_query
from verification.live_news_service import retrieve_live_news
from verification.fact_check_service import check_fact_check_registry
from verification.evidence_analyzer import analyze_evidence_corpus
from verification.decision_engine import synthesize_decision


def verify_news_claim(
    title: str = "", 
    text: str = "", 
    skip_live_verification: bool = False
) -> Dict[str, Any]:
    """
    Executes the full dual-engine verification pipeline for an input headline and/or article.
    """
    combined_input = f"{title}\n{text}".strip() if title and text else (title.strip() or text.strip())
    
    if not combined_input or len(combined_input.split()) < 3:
        return {
            "success": False,
            "error": "Please provide a valid news claim or article (at least 3 words required)."
        }

    # -------------------------------------------------------------
    # 1. Machine Learning Stylistic Prediction
    # -------------------------------------------------------------
    ml_result = predict_news(title=title, text=text) if 'title' in predict_news.__code__.co_varnames else predict_news(combined_input)
    
    # -------------------------------------------------------------
    # 2. Claim Extraction & Search Query Generation
    # -------------------------------------------------------------
    claim_info = extract_search_query(title=title, text=text)
    query = claim_info.get("query", "")

    # -------------------------------------------------------------
    # 3. Live News Retrieval (if not skipped)
    # -------------------------------------------------------------
    if skip_live_verification:
        live_news_raw = {
            "success": True,
            "service_mode": "Skipped (ML-Only Mode)",
            "articles": [],
            "articles_found": 0
        }
        fact_check_res = {
            "found": False,
            "rating": "Not Checked",
            "fact_checker": "N/A",
            "review_url": "",
            "summary": "Live fact-check search skipped by user selection."
        }
    else:
        live_news_raw = retrieve_live_news(query=query, max_results=5)
        fact_check_res = check_fact_check_registry(query=query)

    # -------------------------------------------------------------
    # 4. Semantic Evidence & Source Reliability Analysis
    # -------------------------------------------------------------
    evidence_res = analyze_evidence_corpus(
        user_claim=combined_input,
        retrieved_articles=live_news_raw.get("articles", [])
    )

    # -------------------------------------------------------------
    # 5. Combined Decision Synthesis
    # -------------------------------------------------------------
    final_assessment = synthesize_decision(
        ml_analysis=ml_result,
        evidence_analysis=evidence_res,
        fact_check_result=fact_check_res,
        claim_info=claim_info
    )

    return {
        "success": True,
        "input_title": title,
        "input_text": text,
        "combined_input": combined_input,
        "claim_info": claim_info,
        "ml_analysis": ml_result,
        "live_news_raw": live_news_raw,
        "fact_check_result": fact_check_res,
        "evidence_analysis": evidence_res,
        "final_assessment": final_assessment
    }
