"""
model_utils.py
----------------
Deep Learning IDS Module.

Primary path (production): load a pre-trained LSTM classifier plus its
companion scaler / label-encoder from disk:

    models/lstm_model.h5
    models/scaler.pkl
    models/label_encoder.pkl

Run `python train_model.py` once (with TensorFlow installed) to produce
these three artifacts from the bundled synthetic dataset, or replace them
with artifacts trained on a real CICIDS2017 / NSL-KDD corpus.

Fallback path (so the app is always runnable): if TensorFlow or the model
files are not available -- e.g. evaluating this project on a machine
without a GPU/TF install -- a scikit-learn RandomForest is trained
on-the-fly from the synthetic dataset and cached to disk. The rest of the
pipeline (risk scoring, SHAP, recommendations, reporting) is identical
either way, because both paths expose the same `predict()` interface and
both support SHAP explainability.

   # FUTURE SCOPE: add a streaming inference path that scores live
   # NetFlow/IPFIX records as they arrive instead of batch CSV uploads.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
import data_utils

_TF_AVAILABLE = True
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model  # noqa: F401
except Exception:  # noqa: BLE001
    _TF_AVAILABLE = False


class LoadedModel:
    """Thin wrapper unifying the Keras-LSTM and scikit-learn fallback paths."""

    def __init__(self, kind, model, scaler, label_encoder, feature_list):
        self.kind = kind  # "lstm" or "random_forest_fallback"
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_list = feature_list

    # -- prediction -----------------------------------------------------
    def predict(self, df: pd.DataFrame):
        """
        Predict attack labels + confidence for every row of `df`.

        `df` must contain (a superset of) self.feature_list columns; missing
        columns are filled with 0, extra columns are ignored, ensuring the
        model never crashes on slightly different uploaded schemas.
        """
        X = self._align_features(df)
        X_scaled = self.scaler.transform(X)

        start = time.time()
        if self.kind == "lstm":
            X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            proba = self.model.predict(X_seq, verbose=0)
        else:
            proba = self.model.predict_proba(X_scaled)
        elapsed_ms = (time.time() - start) * 1000

        pred_idx = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)
        labels = self.label_encoder.inverse_transform(pred_idx)
        return {
            "labels": labels,
            "confidence": confidence,
            "proba": proba,
            "classes": list(self.label_encoder.classes_),
            "detection_time_ms": elapsed_ms,
            "X_scaled": X_scaled,
        }

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.feature_list:
            if col not in df.columns:
                df[col] = 0
        # Encode any leftover categorical/string columns numerically (best effort)
        for col in self.feature_list:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.factorize(df[col])[0]
        return df[self.feature_list]


def _train_fallback_model():
    """Train a RandomForest fallback model on the synthetic dataset and cache it."""
    df = data_utils.generate_synthetic_dataset(n_per_class=400)
    label_col = data_utils.LABEL_COLUMN
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].copy()
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_cols:
        X[c] = pd.factorize(X[c])[0]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y)

    joblib.dump(clf, config.FALLBACK_MODEL_PATH)
    joblib.dump(scaler, config.FALLBACK_MODEL_PATH + ".scaler.pkl")
    joblib.dump(label_encoder, config.FALLBACK_MODEL_PATH + ".label_encoder.pkl")
    joblib.dump(feature_cols, config.FEATURE_LIST_PATH)

    return LoadedModel("random_forest_fallback", clf, scaler, label_encoder, feature_cols)


def load_active_model(force_retrain_fallback=False) -> LoadedModel:
    """
    Resolve which model backend to use, in priority order:

      1. Pre-trained Keras LSTM + scaler.pkl + label_encoder.pkl (production).
      2. Cached scikit-learn fallback (models/fallback_rf_model.pkl).
      3. Train a brand-new fallback on synthetic data (first run).
    """
    if (
        _TF_AVAILABLE
        and os.path.exists(config.LSTM_MODEL_PATH)
        and os.path.exists(config.SCALER_PATH)
        and os.path.exists(config.LABEL_ENCODER_PATH)
        and os.path.exists(config.FEATURE_LIST_PATH)
    ):
        from tensorflow.keras.models import load_model

        model = load_model(config.LSTM_MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
        feature_list = joblib.load(config.FEATURE_LIST_PATH)
        return LoadedModel("lstm", model, scaler, label_encoder, feature_list)

    if not force_retrain_fallback and os.path.exists(config.FALLBACK_MODEL_PATH):
        try:
            clf = joblib.load(config.FALLBACK_MODEL_PATH)
            scaler = joblib.load(config.FALLBACK_MODEL_PATH + ".scaler.pkl")
            label_encoder = joblib.load(config.FALLBACK_MODEL_PATH + ".label_encoder.pkl")
            feature_list = joblib.load(config.FEATURE_LIST_PATH)
            return LoadedModel("random_forest_fallback", clf, scaler, label_encoder, feature_list)
        except Exception:
            pass  # cache corrupt/missing -> retrain below

    return _train_fallback_model()
