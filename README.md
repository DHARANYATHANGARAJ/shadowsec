# 🛡️ ShadowSec — Explainable Intrusion Detection System

AI-powered SOC platform: Deep Learning IDS + Explainable AI (SHAP) + Risk
Scoring + Security Recommendations + downloadable PDF Incident Reports, all
behind email-OTP authentication with Admin/User RBAC.

---

## Quick Start

```bash
cd shadowsec
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first launch it creates the
SQLite database and seeds a demo admin account:

| Username | Password       |
|----------|----------------|
| `admin`  | `Admin@12345`  |

Register your own account from the landing page for the regular **User**
role, or log in as `admin` to see the **Admin** workspace.

---

## How login works (no mail server required)

Login is username + password, then a 6-digit email OTP. If you haven't
configured SMTP (see below), the app runs in **dev mode**: the OTP is shown
directly in a banner on screen instead of being emailed, so the whole flow
is testable end-to-end with zero external setup.

To send real emails, set these environment variables before launching:

```bash
export SHADOWSEC_SMTP_HOST=smtp.yourprovider.com
export SHADOWSEC_SMTP_PORT=587
export SHADOWSEC_SMTP_USER=you@yourdomain.com
export SHADOWSEC_SMTP_PASSWORD=your-app-password
export SHADOWSEC_SMTP_FROM=you@yourdomain.com
export SHADOWSEC_SMTP_USE_TLS=1
```

If SMTP is configured but a send fails for any reason, the app automatically
falls back to dev mode rather than locking you out.

---

## The model: it just works, with or without TensorFlow

`model_utils.py` resolves the active model in this priority order:

1. **Production LSTM** — `models/lstm_model.h5` + `scaler.pkl` +
   `label_encoder.pkl`, loaded with TensorFlow/Keras.
2. **Cached fallback** — a scikit-learn RandomForest trained once and
   cached to `models/fallback_rf_model.pkl`.
3. **Fresh fallback training** — if neither of the above exists yet (e.g.
   the very first run), a RandomForest is trained on the bundled synthetic
   CICIDS2017-style dataset (`data_utils.py`) and cached for next time.

Both paths expose the **same `.predict()` interface**, so every downstream
module (SHAP explainability, risk scoring, reporting) works identically
regardless of which backend is active. The active backend is always shown
in the UI (e.g. "Model: random_forest_fallback").

### Training the real LSTM (optional)

If you have TensorFlow installed:

```bash
python train_model.py                          # trains on the synthetic dataset
python train_model.py --epochs 30 --n-per-class 800
python train_model.py --csv my_cicids2017.csv --label-col Label   # real data
```

This writes `models/lstm_model.h5`, `scaler.pkl`, `label_encoder.pkl`,
`feature_list.pkl`, and `training_history.pkl` (epoch accuracy/loss, used by
the **Model Performance** page's training curve). Next time you launch
`app.py`, it automatically picks up the LSTM instead of the fallback.

### Explainability: SHAP, with a graceful fallback

`explainability.py` tries real SHAP (`TreeExplainer` for the RandomForest,
`KernelExplainer` for the LSTM) first. If the `shap` package isn't
installed, it falls back to the model's native feature importances and
labels itself `feature_importance_fallback` in the UI — so the page never
crashes, and you always know which explanation method produced the result.

### Model Performance numbers are computed live, not hardcoded

The **Model Performance** page evaluates whichever model is currently
active against a freshly generated, held-out synthetic test split (a
different random seed than any training run) and reports the real accuracy,
precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve for *your*
model — not fixed placeholder numbers.

---

## Bringing your own dataset

The **Upload Dataset** page accepts any CSV. It auto-detects a label/class
column (looks for `label`, `class`, `attack`, `attack_type`, `category`,
`target`) and works fine even if there isn't one — the IDS will still
classify every row. To use a real public dataset, download CICIDS2017 or
NSL-KDD as CSV and upload it directly; no preprocessing required.

---

## Project structure

```
shadowsec/
├── app.py                    # Main entry point — routing, landing page, sidebar
├── login.py                  # Register / Login / OTP / Forgot-password screens
├── otp_service.py            # OTP generation, SMTP delivery, dev-mode fallback
├── database.py                # SQLite persistence layer (users, otp, datasets,
│                               #   predictions, reports, settings, audit log)
├── security.py                # Password hashing (PBKDF2-HMAC) & policy checks
├── upload_module.py           # Dataset upload UI (CSV or synthetic sample)
├── preprocessing.py           # Feature-type detection, cleaning, correlation +
│                               #   Random Forest feature-importance ranking
├── data_utils.py              # Synthetic CICIDS2017-style dataset generator
├── model_utils.py             # Deep Learning IDS (LSTM + scikit-learn fallback)
├── train_model.py             # Optional script to train the real Keras LSTM
├── explainability.py          # SHAP (+ feature-importance fallback) explanations
├── risk_engine.py              # Risk Scoring Engine (confidence × severity → 0-1000)
├── recommendation_engine.py   # Mitigation playbook + alert-card builder
├── report_generator.py        # PDF incident report generation (reportlab)
├── user_dashboard.py           # Overview / Threat Detection / Real-Time Monitor /
│                               #   My Reports pages
├── admin_dashboard.py         # User management / platform data / analytics /
│                               #   model performance / settings pages
├── charts.py                   # Centralized Plotly chart factory
├── config.py                   # Shared constants: attack taxonomy, severity
│                               #   weights, risk bands, recommendations, etc.
├── assets/style.css            # Light glassmorphism SOC theme
├── .streamlit/config.toml      # Streamlit light-theme + upload-size config
├── models/                     # Saved model artifacts (created at runtime)
├── reports/                    # Generated PDF incident reports
├── uploads/                    # Saved copies of uploaded datasets
├── database/                   # SQLite database file
└── requirements.txt
```

---

## Roles

| Capability                                   | User | Admin |
|-----------------------------------------------|:----:|:-----:|
| Upload datasets, run detection                 | ✅   | ✅    |
| View own reports / detection history           | ✅   | ✅    |
| Real-time (simulated) monitor                   | ✅   | ✅    |
| View **all** users' datasets/predictions/reports | ❌  | ✅    |
| Manage users (promote/disable/delete)           | ❌   | ✅    |
| Platform-wide analytics dashboard               | ❌   | ✅    |
| Model performance evaluation                    | ❌   | ✅    |
| Edit risk thresholds / severity map / alert rules / password policy / OTP settings | ❌ | ✅ |

---

## Future scope (called out in code comments too)

- Streaming inference for live NetFlow/IPFIX records instead of batch CSV uploads.
- A real packet-capture connector (pcap → flow features) for the dataset pipeline.
- A transactional email API (SendGrid/SES/Postmark) in place of raw `smtplib`.
- OAuth/SSO identity providers alongside the username + password + OTP flow.
- Asset-criticality-aware recommendations (e.g. only suggest host isolation
  for non-production assets) once a CMDB integration exists.

---

## Notes

- Passwords are hashed with PBKDF2-HMAC-SHA256 (260,000 iterations, random
  salt) using only the Python standard library — no extra crypto dependency
  required.
- All admin-configurable values (risk bands, attack-severity mapping, alert
  rules, password policy, OTP settings) are stored in the `settings` table
  and take effect immediately, app-wide, without a restart.
