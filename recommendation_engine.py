"""
recommendation_engine.py
--------------------------
Security Recommendation Engine + Alert Engine.

Given a detected attack type, returns:
  * a curated list of mitigation recommendations (config.RECOMMENDATIONS)
  * the alert severity tier (delegates to risk_engine / config)

   # FUTURE SCOPE: personalize recommendations using asset criticality /
   # CMDB context (e.g. "isolate host" only for non-production assets).
"""

import config
import risk_engine


def get_recommendations(attack_type: str) -> list:
    return config.RECOMMENDATIONS.get(attack_type, [
        "Review the flagged traffic manually.",
        "Escalate to the on-call security analyst if suspicious.",
    ])


def build_alert(attack_type: str, confidence: float, risk_score: float, risk_category: str) -> dict:
    """Construct an alert-card payload for the Alert Engine UI."""
    severity = risk_engine.get_severity_for_attack(attack_type)
    return {
        "attack_type": attack_type,
        "severity": severity,
        "confidence": round(confidence * 100, 1),
        "risk_score": risk_score,
        "risk_category": risk_category,
        "recommendations": get_recommendations(attack_type),
    }
