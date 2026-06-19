"""
preprocessing.py
------------------
Feature Extraction Module.

Given an arbitrary uploaded dataframe (CICIDS2017, NSL-KDD, or any CSV with
a label-like column), this module:

  1. Auto-detects numeric vs categorical columns.
  2. Encodes categorical columns (LabelEncoder) and scales numeric columns
     (StandardScaler).
  3. Ranks features via correlation-with-target AND Random Forest importance.
  4. Returns the Top-10 most important features for display + downstream
     modeling.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

CANDIDATE_LABEL_NAMES = {
    "label", "class", "attack", "attack_type", "category", "target", " label"
}


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristically find the label/target column, if any."""
    for col in df.columns:
        if col.strip().lower() in CANDIDATE_LABEL_NAMES:
            return col
    return None


def detect_feature_types(df: pd.DataFrame, label_col: Optional[str] = None):
    """Return (numeric_columns, categorical_columns)."""
    cols = [c for c in df.columns if c != label_col]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def basic_dataset_stats(df: pd.DataFrame, label_col: Optional[str] = None) -> dict:
    """Stats shown immediately after upload (preview, rows, cols, missing, dup, class dist)."""
    stats = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_values": int(df.duplicated().sum()),
    }
    if label_col and label_col in df.columns:
        stats["class_distribution"] = df[label_col].value_counts().to_dict()
    else:
        stats["class_distribution"] = {}
    return stats


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-empty columns, fill remaining NaNs, drop exact duplicate rows."""
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    non_numeric_cols = df.columns.difference(numeric_cols)
    for c in non_numeric_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "unknown")
    return df.reset_index(drop=True)


def encode_and_scale(df: pd.DataFrame, label_col: Optional[str] = None):
    """
    Encode categorical features and scale numeric ones.

    Returns:
        X_processed (pd.DataFrame), y (pd.Series or None),
        fitted encoders (dict), fitted scaler (StandardScaler)
    """
    df = df.copy()
    y = None
    label_encoder = None
    if label_col and label_col in df.columns:
        y_raw = df[label_col].astype(str)
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y_raw), name=label_col)
        df = df.drop(columns=[label_col])

    numeric_cols, categorical_cols = detect_feature_types(df)

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, y, encoders, scaler, label_encoder


def correlation_ranking(X: pd.DataFrame, y: pd.Series, top_n=10) -> pd.Series:
    """Absolute Pearson correlation of every feature with the (encoded) target."""
    combined = X.copy()
    combined["__target__"] = y.values
    corr = combined.corr(numeric_only=True)["__target__"].drop("__target__")
    return corr.abs().sort_values(ascending=False).head(top_n)


def random_forest_importance(X: pd.DataFrame, y: pd.Series, top_n=10, random_state=42):
    """Train a quick Random Forest purely to extract feature importances."""
    n_estimators = 100 if len(X) > 50 else 30
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=10, random_state=random_state, n_jobs=-1
    )
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(top_n), clf


def select_top_features(df: pd.DataFrame, label_col: Optional[str], top_n=10):
    """
    Full feature-extraction pipeline used by the 'Feature Extraction Module' page.

    Returns a dict with correlation ranking, RF importance ranking, and a
    blended top-N feature list (RF importance is the primary signal; ties
    broken by correlation).
    """
    X, y, encoders, scaler, label_encoder = encode_and_scale(df, label_col)

    result = {
        "numeric_features": detect_feature_types(df, label_col)[0],
        "categorical_features": detect_feature_types(df, label_col)[1],
        "correlation_ranking": None,
        "rf_importance": None,
        "top_features": list(X.columns)[:top_n],
    }

    if y is not None and y.nunique() > 1:
        try:
            result["correlation_ranking"] = correlation_ranking(X, y, top_n)
        except Exception:
            result["correlation_ranking"] = None
        try:
            rf_importance, _ = random_forest_importance(X, y, top_n)
            result["rf_importance"] = rf_importance
            result["top_features"] = list(rf_importance.index)
        except Exception:
            pass

    return result, X, y, encoders, scaler, label_encoder
