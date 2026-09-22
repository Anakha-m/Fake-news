"""
verification/decision_engine.py
Transparent Dual-Engine Synthesis and Decision System.
Combines historical ML stylistic classification, live external news evidence,
fact-checking registries, and temporal recency indicators into an explainable assessment.
"""

from typing import Dict, Any

STANDARD_DISCLAIMER = (
    "This system combines machine learning predictions with currently available online evidence. "
    "The result is an automated assessment and should not be considered absolute proof."
)


def synthesize_decision(
    ml_analysis: Dict[str, Any],
    evidence_analysis: Dict[str, Any],
    fact_check_result: Dict[str, Any],
    claim_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesizes ML model prediction, live news evidence, fact-check registry,
    and recency flags into a final explainable verdict.

    Returns:
        dict: {
            "final_assessment": "LIKELY REAL" | "LIKELY FAKE" | "MISLEADING" | "UNVERIFIED" | "INSUFFICIENT EVIDENCE",
            "badge_class": "success" | "danger" | "warning" | "secondary" | "info",
            "reliability_score": float,
            "explanation": str,
            "decision_factors": list of str,
            "disclaimer": str
        }
    """
    ml_pred = ml_analysis.get("prediction", "REAL")
    ml_conf = ml_analysis.get("confidence_score", ml_analysis.get("confidence", 50.0))
    if isinstance(ml_conf, float) and ml_conf <= 1.0:
        ml_conf = round(ml_conf * 100, 1)
    
    ev_status = evidence_analysis.get("evidence_status", "INSUFFICIENT")
    sup_count = evidence_analysis.get("supporting_count", 0)
    contra_count = evidence_analysis.get("contradicting_count", 0)
    
    fc_found = fact_check_result.get("found", False)
    fc_rating = fact_check_result.get("rating", "Not Available")
    fc_checker = fact_check_result.get("fact_checker", "")
    
    is_current = claim_info.get("is_current_news", False)
    has_debunk = claim_info.get("has_debunk_note", False)
    detected_verdict = claim_info.get("detected_verdict", "")
    debunk_source = claim_info.get("debunk_source", "Independent Fact-Checkers")

    factors = []

    # -------------------------------------------------------------
    # CASE 0: Explicit Fact-Check / Debunk Note Detected in Input (Highest Priority)
    # -------------------------------------------------------------
    if has_debunk:
        factors.append(f"Embedded Fact-Check Reference: Contains debunking statement linked to {debunk_source}.")
        return {
            "final_assessment": "LIKELY FAKE",
            "badge_class": "danger",
            "reliability_score": 95.0,
            "explanation": (
                f"The submitted text includes an explicit fact-check review from {debunk_source} indicating "
                f"the claim is fabricated, false, or misleading."
            ),
            "decision_factors": factors,
            "disclaimer": STANDARD_DISCLAIMER
        }

    # -------------------------------------------------------------
    # CASE 1: Explicit Fact-Check Registry Record Found
    # -------------------------------------------------------------
    if fc_found:
        factors.append(f"Formal Fact-Check Found: Reviewed by {fc_checker} with verdict '{fc_rating}'.")
        
        if fc_rating in ("False", "Fake"):
            return {
                "final_assessment": "LIKELY FAKE",
                "badge_class": "danger",
                "reliability_score": 92.0,
                "explanation": (
                    f"A formal fact-checking investigation by {fc_checker} identified this claim and rated it '{fc_rating}'. "
                    "Independent fact-checkers concluded the statement is fabricated or incorrect."
                ),
                "decision_factors": factors,
                "disclaimer": STANDARD_DISCLAIMER
            }
        elif fc_rating in ("True", "Mostly True"):
            return {
                "final_assessment": "LIKELY REAL",
                "badge_class": "success",
                "reliability_score": 90.0,
                "explanation": (
                    f"A formal fact-checking review by {fc_checker} verified this statement and rated it '{fc_rating}'."
                ),
                "decision_factors": factors,
                "disclaimer": STANDARD_DISCLAIMER
            }
        elif fc_rating in ("Misleading", "Half True"):
            return {
                "final_assessment": "MISLEADING",
                "badge_class": "warning",
                "reliability_score": 85.0,
                "explanation": (
                    f"Fact-checker {fc_checker} evaluated this claim as '{fc_rating}'. "
                    "The statement contains partial facts stripped of essential context or exaggerated."
                ),
                "decision_factors": factors,
                "disclaimer": STANDARD_DISCLAIMER
            }

    # -------------------------------------------------------------
    # CASE 2: Live News Contradiction / Debunking Notices
    # -------------------------------------------------------------
    if ev_status == "CONTRADICTING" or contra_count >= 1:
        factors.append(f"External News Debunk: {contra_count} external news/reference source(s) refute or dispute this claim.")
        return {
            "final_assessment": "LIKELY FAKE",
            "badge_class": "danger",
            "reliability_score": 88.0,
            "explanation": (
                "External news reporting and investigative notices actively dispute or debunk this claim. "
                "Reliable media sources have published clarifications or refutations regarding this statement."
            ),
            "decision_factors": factors,
            "disclaimer": STANDARD_DISCLAIMER
        }

    # -------------------------------------------------------------
    # CASE 3: Mixed / Conflicting External Reporting
    # -------------------------------------------------------------
    if ev_status == "MIXED":
        factors.append(f"Mixed Evidence: Conflicting reports detected ({sup_count} supporting vs {contra_count} refuting).")
        return {
            "final_assessment": "MISLEADING",
            "badge_class": "warning",
            "reliability_score": 75.0,
            "explanation": (
                "External news coverage is divided with contradictory claims. "
                "Some outlets report the event while others publish caveats or refutations."
            ),
            "decision_factors": factors,
            "disclaimer": STANDARD_DISCLAIMER
        }

    # -------------------------------------------------------------
    # CASE 4: External News Corroboration Found (Supporting >= 1)
    # -------------------------------------------------------------
    if ev_status == "SUPPORTING" and sup_count >= 1:
        factors.append(f"Live News Corroboration: {sup_count} independent news source(s) confirm the story.")
        
        if ml_pred == "REAL":
            return {
                "final_assessment": "LIKELY REAL",
                "badge_class": "success",
                "reliability_score": min(max(ml_conf, 85.0), 96.0),
                "explanation": (
                    f"The claim is corroborated by {sup_count} independent news reporting source(s) and exhibits "
                    "an objective, journalistic writing style consistent with verified news wire reporting."
                ),
                "decision_factors": factors,
                "disclaimer": STANDARD_DISCLAIMER
            }
        else:
            # ML flagged sensationalism, but external news confirmed the event
            factors.append("Linguistic Flag: ML classifier detected sensationalist or emotional phrasing.")
            return {
                "final_assessment": "MISLEADING",
                "badge_class": "warning",
                "reliability_score": 70.0,
                "explanation": (
                    f"While the underlying event is corroborated by {sup_count} news source(s), the submitted text "
                    "uses sensationalist or emotionally charged writing patterns commonly associated with clickbait."
                ),
                "decision_factors": factors,
                "disclaimer": STANDARD_DISCLAIMER
            }

    # -------------------------------------------------------------
    # CASE 5: External Evidence Not Available -> Rely on Machine Learning Model
    # -------------------------------------------------------------
    factors.append("No live news corroboration or fact-check found in public indexes.")

    if is_current and ml_pred == "FAKE" and ml_conf >= 60.0:
        factors.append("Current event cue with high stylistic sensationalism / clickbait markers.")
        return {
            "final_assessment": "LIKELY FAKE",
            "badge_class": "danger",
            "reliability_score": ml_conf,
            "explanation": (
                f"This submission uses recent/breaking event phrasing but exhibits strong sensationalist or clickbait patterns "
                f"({ml_conf}% ML confidence) with zero corroborating reporting found in reputable news indexes."
            ),
            "decision_factors": factors,
            "disclaimer": STANDARD_DISCLAIMER
        }

    # Evaluate directly using calibrated ML Model Prediction with uncertainty guards
    is_uncertain = ml_analysis.get("is_uncertain", False) or (50.0 <= ml_conf <= 58.0)

    if is_uncertain:
        factors.append(f"Model Statistical Ambiguity: ML confidence ({ml_conf}%) is in the uncertainty margin.")
        return {
            "final_assessment": "UNVERIFIED",
            "badge_class": "secondary",
            "reliability_score": 50.0,
            "explanation": (
                f"Statistical ML confidence ({ml_conf}%) is within the borderline uncertainty threshold, and no corroborating "
                "reports were found in verified news indexes. This claim cannot be confirmed without primary sources."
            ),
            "decision_factors": factors,
            "disclaimer": STANDARD_DISCLAIMER
        }

    if ml_pred == "REAL" and ml_conf > 58.0:
        return {
            "final_assessment": "LIKELY REAL",
            "badge_class": "success",
            "reliability_score": ml_conf,
            "explanation": (
                f"The text exhibits objective, journalistic vocabulary and framing with {ml_conf}% statistical ML confidence. "
                "No conflicting external reports were found."
            ),
            "decision_factors": factors,
            "disclaimer": STANDARD_DISCLAIMER
        }
    elif ml_pred == "FAKE":
        return {
            "final_assessment": "LIKELY FAKE",
            "badge_class": "danger",
            "reliability_score": ml_conf,
            "explanation": (
                f"The text exhibits high sensationalism, exaggerated claims, or conspiracy patterns ({ml_conf}% ML confidence) "
                "with zero corroborating coverage in legitimate news indexes."
            ),
            "decision_factors": factors,
            "disclaimer": STANDARD_DISCLAIMER
        }

    # Default fallback only if prediction is genuinely undetermined
    return {
        "final_assessment": "UNVERIFIED",
        "badge_class": "secondary",
        "reliability_score": 50.0,
        "explanation": (
            "Insufficient context available for high-confidence classification."
        ),
        "decision_factors": factors,
        "disclaimer": STANDARD_DISCLAIMER
    }
