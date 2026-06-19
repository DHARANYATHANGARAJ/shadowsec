"""
app.py
--------
ShadowSec — Explainable Intrusion Detection System Using Deep Learning,
Risk Scoring, and Security Recommendations Dashboard.

Main entry point. Run with:

    streamlit run app.py

Responsibilities:
  * One-time DB initialization (creates tables + seeds the default admin).
  * Injects assets/style.css for the light glassmorphism SOC theme.
  * Renders the public landing page when nobody is logged in.
  * Hands off to login.py for register / login / OTP / forgot-password.
  * Once authenticated, renders a role-based sidebar and routes to the
    appropriate page module (user_dashboard.py or admin_dashboard.py).

Session-state keys owned by this module:
    route       -> "landing" | "auth" | "app"
    user        -> the logged-in user dict, or None
    nav_page    -> currently selected sidebar page label
"""

import streamlit as st

import config
import database
import login
import upload_module
import user_dashboard
import admin_dashboard


# ---------------------------------------------------------------------
# Page config + one-time setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title=f"{config.APP_NAME} — AI Powered Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

database.init_db()

with open(config.ASSETS_DIR + "/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.session_state.setdefault("route", "landing")
st.session_state.setdefault("user", None)
st.session_state.setdefault("auth_view", "login")


# ---------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------
FEATURES = [
    ("🧠", "Deep Learning Detection",
     "An LSTM-based classifier (with an automatic scikit-learn fallback) "
     "identifies DDoS, DoS, Port Scan, Botnet, Brute Force, and Web attacks in real time."),
    ("🔍", "Explainable AI",
     "SHAP-powered explanations translate model internals into plain-English "
     "reasons — \"High Packet Rate\", \"Unusual Flow Duration\", and more."),
    ("🎯", "Risk Scoring",
     "Every detection is scored 0–1000 by blending model confidence with "
     "attack-severity weighting, then banded into Low / Medium / High / Critical."),
    ("🛠️", "Security Recommendations",
     "A curated mitigation playbook per attack type — rate limiting, MFA, "
     "host isolation, WAF tuning — so analysts know what to do next."),
    ("📄", "Incident Reporting",
     "One click generates a polished, downloadable PDF incident report with "
     "the full detection, explanation, and recommendation trail."),
    ("📊", "Threat Analytics",
     "An admin analytics dashboard tracks attack distribution, severity trends, "
     "and platform-wide risk over time."),
]


def _render_hero():
    st.markdown(
        f"""
        <div class="ss-hero">
            <div class="ss-hero-badge">AI-Powered SOC Platform</div>
            <h1>{config.APP_NAME} — {config.APP_TAGLINE}</h1>
            <p class="subtitle">{config.APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns([2.5, 1, 1, 2.5])
    with c2:
        if st.button("Login", type="primary", use_container_width=True):
            st.session_state.route = "auth"
            st.session_state.auth_view = "login"
            st.rerun()
    with c3:
        if st.button("Register", use_container_width=True):
            st.session_state.route = "auth"
            st.session_state.auth_view = "register"
            st.rerun()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 2, 3])
    with c2:
        if st.button("🔎 Explore Dashboard", use_container_width=True):
            st.session_state.route = "auth"
            st.session_state.auth_view = "login"
            st.rerun()


def _render_features():
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ss-section-title' style='justify-content:center; text-align:center;'>"
        "<span class='bar'></span> Platform Features</div>",
        unsafe_allow_html=True,
    )
    rows = [FEATURES[i:i + 3] for i in range(0, len(FEATURES), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="ss-feature-card">
                        <div class="ss-feature-icon">{icon}</div>
                        <div class="ss-feature-title">{title}</div>
                        <div class="ss-feature-desc">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_stats():
    stats = database.get_platform_stats()
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    cols = st.columns(5)
    items = [
        ("Registered Analysts", stats["total_users"]),
        ("Datasets Analyzed", stats["total_datasets"]),
        ("Threats Detected", stats["total_attacks"]),
        ("Critical Alerts", stats["critical_alerts"]),
        ("Incident Reports", stats["total_reports"]),
    ]
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='ss-card ss-stat'><div class='value'>{value}</div>"
                f"<div class='label'>{label}</div></div>",
                unsafe_allow_html=True,
            )


def render_landing_page():
    _render_hero()
    _render_features()
    _render_stats()
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.caption(
        f"Demo admin login — username **{config.SEED_ADMIN_USERNAME}**, "
        f"password **{config.SEED_ADMIN_PASSWORD}**"
    )


# ---------------------------------------------------------------------
# Authenticated app shell
# ---------------------------------------------------------------------
USER_PAGES = {
    "Overview": ("👋", user_dashboard.render_overview),
    "Upload Dataset": ("📂", upload_module.render_upload_page),
    "Threat Detection": ("🧠", user_dashboard.render_threat_detection),
    "Real-Time Monitor": ("📡", user_dashboard.render_realtime_monitor),
    "My Reports": ("🗂️", user_dashboard.render_my_reports),
}

ADMIN_PAGES = {
    "Overview": ("🛡️", admin_dashboard.render_admin_overview),
    "User Management": ("👥", admin_dashboard.render_user_management),
    "All Platform Data": ("🗄️", admin_dashboard.render_all_data),
    "Analytics": ("📊", admin_dashboard.render_analytics),
    "Model Performance": ("📈", admin_dashboard.render_model_performance),
    "Upload Dataset": ("📂", upload_module.render_upload_page),
    "Threat Detection": ("🧠", user_dashboard.render_threat_detection),
    "Real-Time Monitor": ("📡", user_dashboard.render_realtime_monitor),
    "Settings": ("⚙️", admin_dashboard.render_admin_settings),
}


def render_app_shell():
    user = st.session_state.user
    pages = ADMIN_PAGES if user["role"] == "admin" else USER_PAGES

    with st.sidebar:
        st.markdown(
            f"<div style='text-align:center; padding:10px 0 6px;'>"
            f"<div style='font-size:1.4rem; font-weight:800;"
            f" background:linear-gradient(95deg,#0369A1,#0EA5E9 45%,#22D3EE);"
            f" -webkit-background-clip:text; background-clip:text; color:transparent;'>"
            f"🛡️ {config.APP_NAME}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='text-align:center; margin-bottom:14px;'>"
            f"<span class='ss-badge {'critical' if user['role'] == 'admin' else 'low'}'>"
            f"{user['role'].upper()}</span><br/>"
            f"<span style='font-size:0.85rem; color:#64748B;'>{user['full_name']}</span></div>",
            unsafe_allow_html=True,
        )

        nav_page = st.radio(
            "Navigate",
            list(pages.keys()),
            format_func=lambda p: f"{pages[p][0]}  {p}",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            database.log_action(user["id"], "logout")
            for key in ("user", "active_df", "detection_result", "explanation_cache", "live_feed"):
                st.session_state.pop(key, None)
            st.session_state.route = "landing"
            st.session_state.auth_view = "login"
            st.rerun()

    _, render_fn = pages[nav_page]
    render_fn(user)


# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------
if st.session_state.user is not None:
    st.session_state.route = "app"

if st.session_state.route == "landing":
    render_landing_page()
elif st.session_state.route == "auth":
    if st.session_state.user is not None:
        st.session_state.route = "app"
        st.rerun()
    login.render_auth_page()
else:
    render_app_shell()
