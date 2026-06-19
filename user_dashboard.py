"""
user_dashboard.py
--------------------
The analyst-facing workspace. Three render functions, wired up by app.py:

    render_overview(user)            -> personal KPI snapshot + quick-start
    render_threat_detection(user)    -> the core pipeline:
                                         Feature Extraction -> Deep Learning IDS
                                         -> Explainable AI -> Risk Scoring
                                         -> Recommendations -> Incident Report
    render_realtime_monitor(user)    -> simulated live SOC feed
    render_my_reports(user)          -> history of past detections + PDF downloads

Session-state contract used on top of login.py's:
    active_df / active_label_col / active_dataset_id / active_dataset_name
        -> set by upload_module.py, consumed here
    detection_result / detection_feature_df / detection_dataset_id
        -> cached output of the last "Run Detection" click
    explanation_cache: dict[int -> ExplanationResult]
    live_feed: list[dict]  (simulated real-time monitor events)
"""

import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import charts
import config
import data_utils
import database
import explainability
import model_utils
import preprocessing
import recommendation_engine
import report_generator
import risk_engine


def _section_title(text, icon="🛰️"):
    st.markdown(
        f"<div class='ss-section-title'><span class='bar'></span>{icon} {text}</div>",
        unsafe_allow_html=True,
    )


def _badge(text, severity):
    css_class = severity.lower()
    return f"<span class='ss-badge {css_class}'>{text}</span>"


@st.cache_resource(show_spinner=False)
def _get_model():
    return model_utils.load_active_model()


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
def render_overview(user):
    _section_title(f"Welcome back, {user['full_name'].split(' ')[0]}", "👋")

    datasets = database.list_datasets(user_id=user["id"])
    predictions = database.list_predictions(user_id=user["id"])
    reports = database.list_reports(user_id=user["id"])
    critical = sum(1 for p in predictions if p["risk_category"] == "Critical")
    avg_risk = round(np.mean([p["risk_score"] for p in predictions]), 1) if predictions else 0.0

    cols = st.columns(5)
    stats = [
        ("Datasets Uploaded", len(datasets)),
        ("Detections Run", len(predictions)),
        ("Critical Alerts", critical),
        ("Avg Risk Score", avg_risk),
        ("Reports Generated", len(reports)),
    ]
    for col, (label, value) in zip(cols, stats):
        with col:
            st.markdown(
                f"<div class='ss-card ss-stat'><div class='value'>{value}</div>"
                f"<div class='label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("##### 🚀 Quick Start")
        st.markdown(
            "1. **Upload Dataset** — bring your own CICIDS2017/NSL-KDD CSV, or load a synthetic sample.\n"
            "2. **Threat Detection** — run the Deep Learning IDS, review the SHAP explanation and risk score.\n"
            "3. **Generate an Incident Report** — download a polished PDF for your SOC records.\n"
            "4. **Real-Time Monitor** — watch a simulated live feed of incoming traffic."
        )

    if predictions:
        st.markdown("##### Recent Detections")
        recent = pd.DataFrame(predictions[:8])[
            ["created_at", "attack_type", "confidence", "risk_score", "risk_category"]
        ]
        recent.columns = ["When", "Attack Type", "Confidence", "Risk Score", "Risk Category"]
        recent["Confidence"] = (recent["Confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(recent, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Threat Detection (the core pipeline)
# ---------------------------------------------------------------------
def _build_batch_risk_table(labels, confidence):
    rows = []
    for i, (label, conf) in enumerate(zip(labels, confidence)):
        ev = risk_engine.evaluate(label, float(conf))
        rows.append({
            "row": i,
            "attack_type": label,
            "confidence": float(conf),
            "severity": ev["severity"],
            "risk_score": ev["risk_score"],
            "risk_category": ev["risk_category"],
            "risk_color": ev["risk_color"],
        })
    return pd.DataFrame(rows)


def render_threat_detection(user):
    _section_title("Threat Detection — Deep Learning IDS", "🧠")

    if "active_df" not in st.session_state or st.session_state.active_df is None:
        st.info("No active dataset yet. Head to **Upload Dataset** to bring in a CSV or generate a sample.")
        return

    df = st.session_state.active_df
    label_col = st.session_state.get("active_label_col")
    dataset_name = st.session_state.get("active_dataset_name", "dataset")
    dataset_id = st.session_state.get("active_dataset_id")

    st.caption(f"Active dataset: **{dataset_name}** — {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    with st.expander("🔬 Feature Extraction Module", expanded=False):
        if st.button("Run Feature Extraction"):
            with st.spinner("Ranking features by correlation and Random Forest importance..."):
                result, X, y, *_ = preprocessing.select_top_features(df, label_col)
            st.session_state.feature_extraction_result = result
        fx = st.session_state.get("feature_extraction_result")
        if fx:
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Numeric features:** {len(fx['numeric_features'])}")
                st.write(f"**Categorical features:** {len(fx['categorical_features'])}")
                st.write("**Top features (blended ranking):**")
                st.write(", ".join(fx["top_features"][:10]))
            with c2:
                if fx.get("rf_importance") is not None:
                    st.plotly_chart(
                        charts.feature_importance_bar(fx["rf_importance"]),
                        use_container_width=True,
                    )
                else:
                    st.caption("Random Forest importance requires a label column with 2+ classes.")

    st.markdown("<br/>", unsafe_allow_html=True)
    run_clicked = st.button("🚨 Run Deep Learning Detection", type="primary", use_container_width=False)

    if run_clicked:
        with st.spinner("Loading model and classifying traffic..."):
            model = _get_model()
            feature_df = df.drop(columns=[label_col]) if label_col and label_col in df.columns else df.copy()
            result = model.predict(feature_df)
        st.session_state.detection_result = result
        st.session_state.detection_feature_df = feature_df
        st.session_state.detection_model_kind = model.kind
        st.session_state.detection_dataset_id = dataset_id
        st.session_state.explanation_cache = {}
        database.log_action(user["id"], "detection_run", dataset_name)

    result = st.session_state.get("detection_result")
    if result is None:
        return

    labels, confidence = result["labels"], result["confidence"]
    batch = _build_batch_risk_table(labels, confidence)

    st.markdown(f"<span class='ss-badge low'>Model: {st.session_state.detection_model_kind}</span> "
                f"&nbsp; <span class='ss-badge medium'>Batch detection time: "
                f"{result['detection_time_ms']:.1f} ms for {len(labels)} rows</span>",
                unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    attack_count = int((batch["attack_type"] != "Benign").sum())
    critical_count = int((batch["risk_category"] == "Critical").sum())
    for col, label, value in zip(
        (k1, k2, k3, k4),
        ("Rows Analyzed", "Attacks Flagged", "Critical Risk Rows", "Avg Confidence"),
        (len(batch), attack_count, critical_count, f"{batch['confidence'].mean() * 100:.1f}%"),
    ):
        with col:
            st.markdown(
                f"<div class='ss-card ss-stat'><div class='value'>{value}</div>"
                f"<div class='label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.attack_distribution_pie(batch["attack_type"].value_counts().to_dict()),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            charts.severity_distribution_bar(batch["risk_category"].value_counts().to_dict()),
            use_container_width=True,
        )

    # ---------------- Drill into a single record ----------------
    st.markdown("##### 🔎 Inspect an Individual Record")
    default_idx = int(batch["risk_score"].idxmax())
    options = list(batch["row"])
    labels_map = {
        i: f"Row {i} — {batch.loc[i, 'attack_type']} ({batch.loc[i, 'confidence'] * 100:.1f}% confidence, "
           f"{batch.loc[i, 'risk_category']} risk)"
        for i in options
    }
    selected_idx = st.selectbox(
        "Select a record", options, index=options.index(default_idx),
        format_func=lambda i: labels_map[i],
    )

    row = batch.loc[selected_idx]
    attack_type, conf, severity = row["attack_type"], row["confidence"], row["severity"]
    risk_score, risk_category, risk_color = row["risk_score"], row["risk_category"], row["risk_color"]

    cache = st.session_state.setdefault("explanation_cache", {})
    if selected_idx not in cache:
        with st.spinner("Computing explanation..."):
            model = _get_model()
            predicted_class_idx = int(np.argmax(result["proba"][selected_idx]))
            cache[selected_idx] = explainability.explain_prediction(
                model, result["X_scaled"], result["X_scaled"], model.feature_list,
                predicted_class_idx, sample_idx=selected_idx,
            )
    explanation = cache[selected_idx]
    sentence = explainability.build_explanation_sentence(attack_type, explanation.human_explanation)
    recommendations = recommendation_engine.get_recommendations(attack_type)

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown(
            f"<div class='ss-card'>"
            f"<h4 style='margin-top:0;'>{attack_type} {_badge(risk_category, risk_category)}</h4>"
            f"<p><b>Confidence:</b> {conf * 100:.1f}%<br/>"
            f"<b>Severity:</b> {severity}<br/>"
            f"<b>Model:</b> {st.session_state.detection_model_kind}<br/>"
            f"<b>Explanation method:</b> {explanation.method}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(charts.risk_gauge(risk_score, risk_category, risk_color), use_container_width=True)

    with right:
        st.markdown("###### 🧩 Explainable AI — Why This Was Flagged")
        st.write(sentence)
        if explanation.bar_png:
            st.image(explanation.bar_png, use_container_width=True)
        st.markdown("###### 🛠️ Security Recommendations")
        for rec in recommendations:
            st.markdown(f"- {rec}")

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("📄 Generate Incident Report (PDF)", type="primary"):
        top_features_serializable = [(str(n), float(v)) for n, v in explanation.top_features]
        prediction_id = database.save_prediction(
            user_id=user["id"],
            dataset_id=dataset_id,
            attack_type=attack_type,
            confidence=float(conf),
            severity=severity,
            risk_score=float(risk_score),
            risk_category=risk_category,
            top_features=top_features_serializable,
            explanation_text=sentence,
            model_used=st.session_state.detection_model_kind,
        )
        output_path = report_generator.report_file_path(user["id"], prediction_id)
        report_generator.generate_incident_report_pdf(
            output_path=output_path,
            username=user["full_name"],
            dataset_name=dataset_name,
            attack_type=attack_type,
            confidence=float(conf),
            risk_score=float(risk_score),
            severity=severity,
            risk_category=risk_category,
            human_explanation=explanation.human_explanation,
            explanation_sentence=sentence,
            recommendations=recommendations,
            shap_chart_png=explanation.bar_png,
            detection_time_ms=float(result["detection_time_ms"]),
            report_id=prediction_id,
        )
        database.save_report(prediction_id, user["id"], output_path)
        database.log_action(user["id"], "report_generated", f"prediction_id={prediction_id}")
        st.success("Incident report generated.")
        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download Incident Report (PDF)", f.read(),
                file_name=f"shadowsec_incident_report_{prediction_id}.pdf",
                mime="application/pdf",
            )


# ---------------------------------------------------------------------
# Real-Time Monitor (simulated)
# ---------------------------------------------------------------------
def render_realtime_monitor(user):
    _section_title("Real-Time Attack Monitor", "📡")
    st.markdown(
        "<span class='ss-pulse'></span> *Simulated live feed — generates synthetic traffic events "
        "and classifies each one through the same Deep Learning IDS pipeline.*",
        unsafe_allow_html=True,
    )

    feed = st.session_state.setdefault("live_feed", [])

    c1, c2 = st.columns([1, 3])
    with c1:
        burst_size = st.slider("Events per simulation", 1, 10, 3)
        if st.button("▶️ Simulate Incoming Traffic", type="primary"):
            model = _get_model()
            seed = int(time.time() * 1000) % 1_000_000
            sample = data_utils.generate_synthetic_dataset(n_per_class=1, seed=seed)
            sample = sample.sample(n=min(burst_size, len(sample)), replace=True, random_state=seed).reset_index(drop=True)
            feature_df = sample.drop(columns=[data_utils.LABEL_COLUMN])
            result = model.predict(feature_df)
            now = datetime.utcnow()
            for i, (label, conf) in enumerate(zip(result["labels"], result["confidence"])):
                ev = risk_engine.evaluate(label, float(conf))
                feed.append({
                    "timestamp": now.strftime("%H:%M:%S"),
                    "attack_type": label,
                    "confidence": float(conf),
                    "severity": ev["severity"],
                    "risk_score": ev["risk_score"],
                    "risk_category": ev["risk_category"],
                })
            st.session_state.live_feed = feed[-40:]
            database.log_action(user["id"], "realtime_simulation", f"{burst_size} events")
            st.rerun()

        if st.button("🗑️ Clear Feed"):
            st.session_state.live_feed = []
            st.rerun()

    with c2:
        total = len(feed)
        critical = sum(1 for e in feed if e["risk_category"] == "Critical")
        high = sum(1 for e in feed if e["risk_category"] == "High")
        avg_risk = round(np.mean([e["risk_score"] for e in feed]), 1) if feed else 0.0
        m1, m2, m3, m4 = st.columns(4)
        for col, label, value in zip(
            (m1, m2, m3, m4),
            ("Total Events", "Critical Alerts", "High Alerts", "Avg Risk Score"),
            (total, critical, high, avg_risk),
        ):
            with col:
                st.markdown(
                    f"<div class='ss-card ss-stat'><div class='value'>{value}</div>"
                    f"<div class='label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )

    if feed:
        st.plotly_chart(
            charts.live_threat_timeline(
                [e["timestamp"] for e in feed], [e["severity"] for e in feed]
            ),
            use_container_width=True,
        )

        st.markdown("##### 🔔 Recent Alerts")
        for ev in reversed(feed[-8:]):
            css_class = ev["risk_category"].lower()
            st.markdown(
                f"<div class='ss-alert {css_class}'>"
                f"<b>{ev['timestamp']}</b> &nbsp; {ev['attack_type']} &nbsp; "
                f"{_badge(ev['risk_category'], ev['risk_category'])} &nbsp; "
                f"confidence {ev['confidence'] * 100:.1f}% &nbsp; risk score {ev['risk_score']:.0f}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Click **Simulate Incoming Traffic** to start the live feed.")


# ---------------------------------------------------------------------
# My Reports
# ---------------------------------------------------------------------
def render_my_reports(user):
    _section_title("My Reports & Detection History", "🗂️")

    reports = database.list_reports(user_id=user["id"])
    if not reports:
        st.info("No incident reports generated yet. Run a detection from **Threat Detection** first.")
        return

    for r in reports:
        css_class = r["risk_category"].lower()
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f"**{r['attack_type']}** {_badge(r['risk_category'], r['risk_category'])} "
                    f"&nbsp; Risk Score: {r['risk_score']:.0f}/1000 &nbsp; "
                    f"Generated: {r['created_at']}",
                    unsafe_allow_html=True,
                )
            with c2:
                try:
                    with open(r["pdf_path"], "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF", f.read(),
                            file_name=f"incident_report_{r['id']}.pdf",
                            mime="application/pdf",
                            key=f"dl_{r['id']}",
                        )
                except FileNotFoundError:
                    st.caption("Report file unavailable.")
