"""
risk_engine.py
----------------
Risk Scoring Engine.

Risk Score = Confidence x Severity Weight x 100   (scaled to a 0-1000 band)

Severity weights (default, admin-editable via Settings):
    Critical = 10   High = 7   Medium = 5   Low = 2

Risk categories (default, admin-editable):
    0-200   Low
    201-500 Medium
    501-800 High
    801-1000 Critical
"""

import database
import config


def get_severity_for_attack(attack_type: str) -> str:
    """Look up severity tier for an attack type (admin-configurable)."""
    severity_map = database.get_setting("attack_severity") or config.DEFAULT_ATTACK_SEVERITY
    return severity_map.get(attack_type, "Medium")


def compute_risk_score(confidence: float, severity: str) -> float:
    """
    confidence: 0.0 - 1.0
    severity: one of "Critical" | "High" | "Medium" | "Low"
    """
    weight = config.SEVERITY_WEIGHTS.get(severity, 5)
    score = confidence * weight * 100  # max = 1.0 * 10 * 100 = 1000
    return round(min(max(score, 0), 1000), 1)


def categorize_risk(score: float) -> dict:
    """Return the matching risk band dict {label, min, max, color}."""
    bands = database.get_setting("risk_bands") or config.DEFAULT_RISK_BANDS
    for band in bands:
        if band["min"] <= score <= band["max"]:
            return band
    return bands[-1]  # fall back to highest band if score exceeds bounds


def evaluate(attack_type: str, confidence: float) -> dict:
    """One-stop helper: severity, score, category for a given prediction."""
    severity = get_severity_for_attack(attack_type)
    score = compute_risk_score(confidence, severity)
    band = categorize_risk(score)
    return {
        "severity": severity,
        "risk_score": score,
        "risk_category": band["label"],
        "risk_color": band["color"],
    }
