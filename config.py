"""
config.py
----------
Central place for every constant the platform shares across modules:
attack taxonomy, severity weights, risk bands, the recommendation
knowledge base, and the human-readable feature dictionary used by the
explainability engine.

Keeping these in one file means the Admin Settings page can expose them
for editing (risk thresholds, alert rules, etc.) without hunting through
business logic scattered across the codebase.
"""

import os

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "shadowsec.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
FALLBACK_MODEL_PATH = os.path.join(MODELS_DIR, "fallback_rf_model.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.pkl")

for _d in (DATABASE_DIR, MODELS_DIR, REPORTS_DIR, ASSETS_DIR, UPLOADS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------
APP_NAME = "ShadowSec"
APP_TAGLINE = "AI Powered Intrusion Detection Platform"
APP_SUBTITLE = "Detect, Explain, and Respond to Cyber Threats"

# ---------------------------------------------------------------------
# Attack taxonomy
# ---------------------------------------------------------------------
ATTACK_CATEGORIES = [
    "Benign",
    "DDoS",
    "DoS",
    "Port Scan",
    "Botnet",
    "Brute Force",
    "Web Attack",
]

# Severity weight per attack type (used by the Risk Scoring Engine)
SEVERITY_WEIGHTS = {
    "Critical": 10,
    "High": 7,
    "Medium": 5,
    "Low": 2,
}

# Default mapping of attack -> severity tier. Admins can override these
# via the Admin Settings page (stored in the `settings` table).
DEFAULT_ATTACK_SEVERITY = {
    "DDoS": "Critical",
    "Botnet": "Critical",
    "Brute Force": "High",
    "DoS": "High",
    "Port Scan": "Medium",
    "Web Attack": "Medium",
    "Benign": "Low",
}

# Risk score bands (0 - 1000 scale)
DEFAULT_RISK_BANDS = [
    {"label": "Low", "min": 0, "max": 200, "color": "#2ecc71"},
    {"label": "Medium", "min": 201, "max": 500, "color": "#f1c40f"},
    {"label": "High", "min": 501, "max": 800, "color": "#e67e22"},
    {"label": "Critical", "min": 801, "max": 1000, "color": "#e74c3c"},
]

# ---------------------------------------------------------------------
# Security recommendation knowledge base
# ---------------------------------------------------------------------
RECOMMENDATIONS = {
    "DDoS": [
        "Enable rate limiting at the edge / load balancer.",
        "Block or null-route the offending source IP ranges.",
        "Activate upstream DDoS protection / scrubbing service.",
        "Scale out behind a CDN to absorb volumetric traffic.",
    ],
    "DoS": [
        "Throttle requests from the offending source.",
        "Apply connection-rate limits on the affected service.",
        "Patch the targeted service if a resource-exhaustion bug is suspected.",
    ],
    "Brute Force": [
        "Enable Multi-Factor Authentication (MFA) on the targeted account(s).",
        "Temporarily lock the affected account(s).",
        "Review authentication logs for credential-stuffing patterns.",
        "Introduce progressive login delays / CAPTCHA after failed attempts.",
    ],
    "Port Scan": [
        "Restrict open ports to only what is operationally required.",
        "Update firewall / security-group rules to deny reconnaissance traffic.",
        "Enable intrusion-prevention alerts for repeated scan signatures.",
    ],
    "Botnet": [
        "Isolate the affected host from the network immediately.",
        "Run a full endpoint malware scan.",
        "Investigate command-and-control (C2) traffic and block known C2 domains/IPs.",
        "Rotate credentials that may have been exposed on the host.",
    ],
    "Web Attack": [
        "Deploy / tune a Web Application Firewall (WAF) rule for this pattern.",
        "Patch the vulnerable web component (e.g. SQLi / XSS sink).",
        "Review application input-validation and output-encoding logic.",
    ],
    "Benign": [
        "No action required — traffic matches expected baseline behaviour.",
        "Continue routine monitoring.",
    ],
}

# ---------------------------------------------------------------------
# Human-readable feature dictionary for the Explainable AI module.
# Maps raw / engineered feature-name substrings to a plain-English
# description used when building "why this was flagged" explanations.
# ---------------------------------------------------------------------
FEATURE_EXPLANATION_MAP = [
    (("syn_flag", "syn_count"), "High SYN Flag Count"),
    (("rst_flag",), "Elevated RST Flag Count"),
    (("fin_flag",), "Elevated FIN Flag Count"),
    (("psh_flag",), "Elevated PSH Flag Count"),
    (("ack_flag",), "Abnormal ACK Flag Count"),
    (("urg_flag",), "Elevated URG Flag Count"),
    (("packet_rate", "packets_per_sec", "packets/s"), "High Packet Rate"),
    (("byte_rate", "bytes_per_sec", "bytes/s"), "High Byte Rate"),
    (("flow_duration",), "Unusual Flow Duration"),
    (("duration",), "Abnormal Connection Duration"),
    (("fwd_packet_length", "bwd_packet_length", "packet_length"), "Abnormal Packet Size"),
    (("total_fwd_packets", "total_bwd_packets", "total_packets"), "High Packet Volume"),
    (("active_mean", "active_time"), "Irregular Active-Time Pattern"),
    (("idle_mean", "idle_time"), "Irregular Idle-Time Pattern"),
    (("protocol",), "Unusual Protocol Usage"),
    (("dst_port", "destination_port"), "Suspicious Destination Port"),
    (("src_port", "source_port"), "Suspicious Source Port"),
    (("iat",), "Irregular Inter-Arrival Time"),
    (("window_size", "win_bytes"), "Abnormal TCP Window Size"),
    (("login_attempt", "failed_login"), "Repeated Failed Login Attempts"),
]


def describe_feature(feature_name: str) -> str:
    """Translate a raw feature name into a human-readable phrase."""
    name = feature_name.lower()
    for keys, description in FEATURE_EXPLANATION_MAP:
        if any(k in name for k in keys):
            return description
    # Fallback: prettify the raw name
    return feature_name.replace("_", " ").strip().title()


# ---------------------------------------------------------------------
# Password policy defaults (editable by Admin)
# ---------------------------------------------------------------------
DEFAULT_PASSWORD_POLICY = {
    "min_length": 8,
    "require_upper": True,
    "require_lower": True,
    "require_number": True,
    "require_special": True,
}

# ---------------------------------------------------------------------
# OTP defaults (editable by Admin)
# ---------------------------------------------------------------------
DEFAULT_OTP_SETTINGS = {
    "validity_minutes": 5,
    "max_attempts": 3,
    "otp_length": 6,
}

# ---------------------------------------------------------------------
# Demo / seed credentials note (shown on first run only)
# ---------------------------------------------------------------------
SEED_ADMIN_USERNAME = "admin"
SEED_ADMIN_EMAIL = "admin@shadowsec.local"
SEED_ADMIN_PASSWORD = "Admin@12345"
