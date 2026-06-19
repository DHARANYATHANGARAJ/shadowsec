"""
admin_dashboard.py
---------------------
Administrator workspace. Render functions wired up by app.py:

    render_admin_overview(admin)      -> platform-wide KPI snapshot
    render_user_management(admin)     -> list/activate/deactivate/promote/delete users
    render_all_data(admin)            -> every dataset / prediction / report on the platform
    render_analytics(admin)           -> charts: attack/severity/risk distribution, monthly trend, top attacks
    render_model_performance(admin)   -> live-evaluated accuracy/precision/recall/F1/ROC-AUC,
                                          confusion matrix, ROC curve, training curve (if available)
    render_admin_settings(admin)      -> risk thresholds, attack-severity map, alert rules,
                                          password policy, OTP settings (all admin-editable,
                                          persisted via database.get_setting/set_setting)
"""

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

import charts
import config
import data_utils
import database
import model_utils


def _section_title(text, icon="🛡️"):
    st.markdown(
        f"<div class='ss-section-title'><span class='bar'></span>{icon} {text}</div>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _get_model():
    return model_utils.load_active_model()


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
def render_admin_overview(admin):
    _section_title("Platform Overview", "🛡️")
    stats = database.get_platform_stats()

    cols = st.columns(6)
    labels_values = [
        ("Total Users", stats["total_users"]),
        ("Datasets Uploaded", stats["total_datasets"]),
        ("Detections Run", stats["total_predictions"]),
        ("Attacks Flagged", stats["total_attacks"]),
        ("Critical Alerts", stats["critical_alerts"]),
        ("Avg Risk Score", stats["avg_risk_score"]),
    ]
    for col, (label, value) in zip(cols, labels_values):
        with col:
            st.markdown(
                f"<div class='ss-card ss-stat'><div class='value'>{value}</div>"
                f"<div class='label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("##### 🕘 Recent Activity")
    activity = database.recent_activity(limit=15)
    if activity:
        act_df = pd.DataFrame(activity)[["created_at", "username", "action", "details"]]
        act_df.columns = ["When", "User", "Action", "Details"]
        act_df["User"] = act_df["User"].fillna("—")
        st.dataframe(act_df, use_container_width=True, hide_index=True)
    else:
        st.info("No activity recorded yet.")


# ---------------------------------------------------------------------
# User Management
# ---------------------------------------------------------------------
def render_user_management(admin):
    _section_title("User Management", "👥")
    users = database.list_users()
    if not users:
        st.info("No users found.")
        return

    for u in users:
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([2.2, 1.4, 0.9, 1.1, 1.3])
            with c1:
                st.markdown(f"**{u['full_name']}**  \n{u['email']} · @{u['username']}")
            with c2:
                st.markdown(f"Role: **{u['role']}**  \nJoined: {u['created_at'][:10]}")
            with c3:
                status = "🟢 Active" if u["is_active"] else "🔴 Disabled"
                st.markdown(status)
            with c4:
                if u["id"] != admin["id"]:
                    new_role = "admin" if u["role"] == "user" else "user"
                    if st.button(f"Make {new_role}", key=f"role_{u['id']}"):
                        database.set_user_role(u["id"], new_role)
                        database.log_action(admin["id"], "role_changed", f"user={u['username']} -> {new_role}")
                        st.rerun()
                else:
                    st.caption("(you)")
            with c5:
                if u["id"] != admin["id"]:
                    toggle_label = "Disable" if u["is_active"] else "Enable"
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button(toggle_label, key=f"toggle_{u['id']}"):
                            database.set_user_active(u["id"], not u["is_active"])
                            database.log_action(admin["id"], "user_status_changed",
                                                 f"user={u['username']} -> {not u['is_active']}")
                            st.rerun()
                    with bc2:
                        if st.button("Delete", key=f"del_{u['id']}"):
                            database.delete_user(u["id"])
                            database.log_action(admin["id"], "user_deleted", f"user={u['username']}")
                            st.rerun()


# ---------------------------------------------------------------------
# All platform data
# ---------------------------------------------------------------------
def render_all_data(admin):
    _section_title("All Platform Data", "🗄️")
    tab_ds, tab_pred, tab_rep = st.tabs(["📂 Datasets", "🧠 Predictions", "📄 Reports"])

    with tab_ds:
        datasets = database.list_datasets(user_id=None)
        if datasets:
            df = pd.DataFrame(datasets)[
                ["id", "username", "filename", "rows", "cols", "missing_values", "duplicate_values", "uploaded_at"]
            ]
            df.columns = ["ID", "User", "Filename", "Rows", "Cols", "Missing", "Duplicates", "Uploaded At"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No datasets uploaded yet.")

    with tab_pred:
        predictions = database.list_predictions(user_id=None)
        if predictions:
            df = pd.DataFrame(predictions)[
                ["id", "username", "attack_type", "confidence", "severity",
                 "risk_score", "risk_category", "model_used", "created_at"]
            ]
            df.columns = ["ID", "User", "Attack Type", "Confidence", "Severity",
                          "Risk Score", "Risk Category", "Model", "When"]
            df["Confidence"] = (df["Confidence"] * 100).round(1).astype(str) + "%"
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = pd.DataFrame(predictions).to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export All Predictions (CSV)", csv_bytes,
                                file_name="shadowsec_predictions_export.csv", mime="text/csv")
        else:
            st.info("No predictions recorded yet.")

    with tab_rep:
        reports = database.list_reports(user_id=None)
        if reports:
            for r in reports:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(
                            f"**{r['attack_type']}** · {r['username']} · "
                            f"Risk {r['risk_score']:.0f}/1000 ({r['risk_category']}) · {r['created_at']}"
                        )
                    with c2:
                        try:
                            with open(r["pdf_path"], "rb") as f:
                                st.download_button(
                                    "⬇️ Download", f.read(),
                                    file_name=f"incident_report_{r['id']}.pdf",
                                    mime="application/pdf", key=f"admin_dl_{r['id']}",
                                )
                        except FileNotFoundError:
                            st.caption("File unavailable.")
        else:
            st.info("No incident reports generated yet.")


# ---------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------
def render_analytics(admin):
    _section_title("Threat Analytics Dashboard", "📊")
    predictions = database.list_predictions(user_id=None)
    if not predictions:
        st.info("No detection data yet — analytics will populate once users start running detections.")
        return

    df = pd.DataFrame(predictions)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["month"] = df["created_at"].dt.strftime("%Y-%m")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.attack_distribution_pie(df["attack_type"].value_counts().to_dict()),
                         use_container_width=True)
    with c2:
        st.plotly_chart(charts.severity_distribution_bar(df["risk_category"].value_counts().to_dict()),
                         use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.risk_distribution_histogram(df["risk_score"].tolist()),
                         use_container_width=True)
    with c4:
        st.plotly_chart(charts.top_attacks_bar(df["attack_type"].value_counts().to_dict()),
                         use_container_width=True)

    monthly = df.groupby("month").size().sort_index()
    st.plotly_chart(charts.monthly_threat_trend(monthly.index.tolist(), monthly.values.tolist()),
                     use_container_width=True)


# ---------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------
def _evaluate_model_performance(model, n_per_class=150, seed=123):
    """Honest, live evaluation of the *currently active* model on a freshly
    generated, held-out synthetic test split (the model never saw this seed
    during training/caching)."""
    test_df = data_utils.generate_synthetic_dataset(n_per_class=n_per_class, seed=seed)
    y_true = test_df[data_utils.LABEL_COLUMN].to_numpy()
    X = test_df.drop(columns=[data_utils.LABEL_COLUMN])

    result = model.predict(X)
    y_pred = result["labels"]
    classes = result["classes"]
    proba = result["proba"]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    roc_auc, fpr, tpr = None, None, None
    try:
        y_bin = label_binarize(y_true, classes=classes)
        roc_auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
    except Exception:
        pass

    return {"metrics": metrics, "cm": cm, "classes": classes, "roc_auc": roc_auc, "fpr": fpr, "tpr": tpr}


def render_model_performance(admin):
    _section_title("Model Performance", "📈")
    model = _get_model()
    st.caption(f"Active model backend: **{model.kind}**")

    if st.button("🔄 Run Live Evaluation", type="primary"):
        with st.spinner("Evaluating on a held-out synthetic test split..."):
            st.session_state.model_eval = _evaluate_model_performance(model)

    eval_result = st.session_state.get("model_eval")
    if not eval_result:
        st.info("Click **Run Live Evaluation** to score the active model on a fresh synthetic test split.")
        return

    st.caption(
        "Metrics below are computed live against a held-out synthetic test split generated with a "
        "different random seed than any training/caching run — they reflect the *currently active* "
        "model, not fixed placeholder numbers."
    )
    m = eval_result["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    labels_values = [
        ("Accuracy", m["accuracy"]), ("Precision", m["precision"]), ("Recall", m["recall"]),
        ("F1 Score", m["f1"]), ("ROC-AUC", eval_result["roc_auc"]),
    ]
    for col, (label, value) in zip((c1, c2, c3, c4, c5), labels_values):
        with col:
            text = f"{value * 100:.1f}%" if value is not None else "—"
            st.markdown(
                f"<div class='ss-card ss-stat'><div class='value'>{text}</div>"
                f"<div class='label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            charts.confusion_matrix_heatmap(eval_result["cm"], list(eval_result["classes"])),
            use_container_width=True,
        )
    with c2:
        if eval_result["fpr"] is not None:
            st.plotly_chart(
                charts.roc_curve_chart(eval_result["fpr"], eval_result["tpr"], eval_result["roc_auc"] or 0.0),
                use_container_width=True,
            )
        else:
            st.info("ROC curve unavailable for this model/data combination.")

    st.markdown("##### Training Curve")
    history = None
    try:
        import joblib
        history = joblib.load(config.MODELS_DIR + "/training_history.pkl")
    except Exception:
        history = None

    if history:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                charts.training_curve_chart(history, ("accuracy", "val_accuracy"), "Accuracy"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                charts.training_curve_chart(history, ("loss", "val_loss"), "Loss"),
                use_container_width=True,
            )
    else:
        st.info(
            "Epoch-level training curves are only available for the trained Keras LSTM "
            "(run `python train_model.py` with TensorFlow installed). The current active "
            "model is the scikit-learn fallback, which does not train in epochs."
        )


# ---------------------------------------------------------------------
# Admin Settings
# ---------------------------------------------------------------------
def render_admin_settings(admin):
    _section_title("Admin Settings", "⚙️")

    tab_risk, tab_alert, tab_pw, tab_otp = st.tabs(
        ["🎯 Risk & Severity", "🔔 Alert Rules", "🔐 Password Policy", "📧 OTP Settings"]
    )

    # ---- Risk thresholds & attack severity ----
    with tab_risk:
        st.markdown("###### Attack → Severity Mapping")
        severity_map = database.get_setting("attack_severity") or dict(config.DEFAULT_ATTACK_SEVERITY)
        new_map = {}
        cols = st.columns(2)
        for i, attack in enumerate(config.ATTACK_CATEGORIES):
            with cols[i % 2]:
                new_map[attack] = st.selectbox(
                    attack, ["Low", "Medium", "High", "Critical"],
                    index=["Low", "Medium", "High", "Critical"].index(severity_map.get(attack, "Medium")),
                    key=f"sev_{attack}",
                )
        if st.button("Save Severity Mapping"):
            database.set_setting("attack_severity", new_map)
            st.success("Attack severity mapping updated.")

        st.markdown("###### Risk Score Bands (0–1000)")
        bands = database.get_setting("risk_bands") or config.DEFAULT_RISK_BANDS
        b1, b2, b3 = st.columns(3)
        low_max = b1.number_input("Low max", 0, 999, int(bands[0]["max"]))
        med_max = b2.number_input("Medium max", low_max + 1, 999, int(bands[1]["max"]))
        high_max = b3.number_input("High max", med_max + 1, 999, int(bands[2]["max"]))
        if st.button("Save Risk Bands"):
            new_bands = [
                {"label": "Low", "min": 0, "max": low_max, "color": "#2ecc71"},
                {"label": "Medium", "min": low_max + 1, "max": med_max, "color": "#f1c40f"},
                {"label": "High", "min": med_max + 1, "max": high_max, "color": "#e67e22"},
                {"label": "Critical", "min": high_max + 1, "max": 1000, "color": "#e74c3c"},
            ]
            database.set_setting("risk_bands", new_bands)
            st.success("Risk bands updated.")

    # ---- Alert rules ----
    with tab_alert:
        alert_rules = database.get_setting("alert_rules") or {
            "min_risk_score_to_alert": 0, "notify_on_critical_only": False,
        }
        min_score = st.slider("Minimum risk score to raise an alert", 0, 1000,
                               int(alert_rules.get("min_risk_score_to_alert", 0)))
        critical_only = st.checkbox("Only notify on Critical-severity detections",
                                     value=alert_rules.get("notify_on_critical_only", False))
        if st.button("Save Alert Rules"):
            database.set_setting("alert_rules", {
                "min_risk_score_to_alert": min_score, "notify_on_critical_only": critical_only,
            })
            st.success("Alert rules updated.")

    # ---- Password policy ----
    with tab_pw:
        policy = database.get_setting("password_policy") or dict(config.DEFAULT_PASSWORD_POLICY)
        min_len = st.number_input("Minimum length", 6, 32, int(policy.get("min_length", 8)))
        require_upper = st.checkbox("Require uppercase letter", value=policy.get("require_upper", True))
        require_lower = st.checkbox("Require lowercase letter", value=policy.get("require_lower", True))
        require_number = st.checkbox("Require number", value=policy.get("require_number", True))
        require_special = st.checkbox("Require special character", value=policy.get("require_special", True))
        if st.button("Save Password Policy"):
            database.set_setting("password_policy", {
                "min_length": min_len, "require_upper": require_upper, "require_lower": require_lower,
                "require_number": require_number, "require_special": require_special,
            })
            st.success("Password policy updated.")

    # ---- OTP settings ----
    with tab_otp:
        otp_settings = database.get_setting("otp_settings") or dict(config.DEFAULT_OTP_SETTINGS)
        validity = st.number_input("OTP validity (minutes)", 1, 60, int(otp_settings.get("validity_minutes", 5)))
        max_attempts = st.number_input("Max verification attempts", 1, 10, int(otp_settings.get("max_attempts", 3)))
        otp_length = st.number_input("OTP code length", 4, 10, int(otp_settings.get("otp_length", 6)))
        if st.button("Save OTP Settings"):
            database.set_setting("otp_settings", {
                "validity_minutes": validity, "max_attempts": max_attempts, "otp_length": otp_length,
            })
            st.success("OTP settings updated.")

        st.markdown("---")
        st.caption(
            "SMTP delivery is configured via environment variables "
            "(SHADOWSEC_SMTP_HOST / PORT / USER / PASSWORD / FROM / USE_TLS). "
            "Without them, OTPs are shown directly in the UI (dev mode) so the demo always works."
        )
