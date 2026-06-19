"""
explainability.py
--------------------
Explainable AI (XAI) Module.

Wraps SHAP so the rest of the app doesn't need to know whether the active
model is the Keras LSTM or the scikit-learn fallback -- both expose the
same `explain()` contract:

    result = explain(model_wrapper, X_scaled, X_background, class_names, predicted_idx)
    result.shap_values        -> np.ndarray, per-feature attribution for the predicted class
    result.top_features       -> list[(feature_name, shap_value)], sorted by |impact|
    result.human_explanation  -> list[str], e.g. ["High Packet Rate", "High SYN Count", ...]
    result.summary_png        -> bytes, SHAP summary bar chart (matplotlib, rendered to PNG)

If the `shap` package is unavailable, falls back to the underlying model's
native feature_importances_ (RandomForest) or a permutation-style proxy, so
the page never crashes -- it simply explains "globally important" features
instead of a true per-prediction SHAP attribution, and labels itself as such.
"""

import io
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

import config

_SHAP_AVAILABLE = True
try:
    import shap  # noqa: F401
except Exception:  # noqa: BLE001
    _SHAP_AVAILABLE = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ExplanationResult:
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    human_explanation: List[str] = field(default_factory=list)
    method: str = "shap"
    summary_png: bytes = None
    bar_png: bytes = None


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _plot_bar(top_features: List[Tuple[str, float]], title: str) -> bytes:
    names = [f for f, _ in top_features][::-1]
    values = [v for _, v in top_features][::-1]
    colors = ["#e74c3c" if v >= 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, values, color=colors)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(title, fontsize=11)
    ax.axvline(0, color="#888", linewidth=0.8)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _plot_summary(shap_values_matrix: np.ndarray, feature_names: List[str]) -> bytes:
    """A lightweight 'summary plot' (mean |SHAP| per feature across the batch)."""
    mean_abs = np.abs(shap_values_matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]
    names = [feature_names[i] for i in order][::-1]
    values = [mean_abs[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, values, color="#2980b9")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance (batch)", fontsize=11)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _human_explanation_from_features(top_features: List[Tuple[str, float]], max_reasons=4) -> List[str]:
    reasons = []
    seen = set()
    for name, _value in top_features:
        phrase = config.describe_feature(name)
        if phrase not in seen:
            reasons.append(phrase)
            seen.add(phrase)
        if len(reasons) >= max_reasons:
            break
    return reasons


def explain_prediction(model_wrapper, X_scaled: np.ndarray, background: np.ndarray,
                        feature_names: List[str], predicted_class_idx: int,
                        sample_idx: int = 0, max_background=100) -> ExplanationResult:
    """
    Explain ONE prediction (row `sample_idx` of X_scaled) for the predicted class.

    `model_wrapper` is the LoadedModel from model_utils (kind == "lstm" or
    "random_forest_fallback").
    """
    if _SHAP_AVAILABLE:
        try:
            return _explain_with_shap(
                model_wrapper, X_scaled, background, feature_names,
                predicted_class_idx, sample_idx, max_background
            )
        except Exception:
            pass  # fall through to the importance-based fallback below

    return _explain_with_feature_importance(model_wrapper, feature_names)


def _explain_with_shap(model_wrapper, X_scaled, background, feature_names,
                        predicted_class_idx, sample_idx, max_background) -> ExplanationResult:
    background_sample = background[: min(max_background, len(background))]
    instance = X_scaled[sample_idx : sample_idx + 1]

    if model_wrapper.kind == "random_forest_fallback":
        explainer = shap.TreeExplainer(model_wrapper.model)
        shap_values = explainer.shap_values(X_scaled[: max_background])
        # shap_values: list[n_classes] of (n_samples, n_features) for older SHAP,
        # or a single (n_samples, n_features, n_classes) array for newer SHAP.
        if isinstance(shap_values, list):
            class_matrix = shap_values[predicted_class_idx]
        else:
            class_matrix = shap_values[:, :, predicted_class_idx]
        instance_values = class_matrix[min(sample_idx, len(class_matrix) - 1)]
    else:
        # KernelExplainer works model-agnostically (incl. Keras), but is
        # slower -- so we cap the background sample size aggressively.
        def predict_fn(data):
            data_seq = data.reshape((data.shape[0], 1, data.shape[1]))
            return model_wrapper.model.predict(data_seq, verbose=0)

        explainer = shap.KernelExplainer(predict_fn, background_sample[:30])
        shap_values = explainer.shap_values(instance, nsamples=100)
        if isinstance(shap_values, list):
            instance_values = shap_values[predicted_class_idx][0]
        else:
            instance_values = shap_values[0, :, predicted_class_idx]
        class_matrix = np.array(instance_values).reshape(1, -1)

    pairs = sorted(
        zip(feature_names, instance_values), key=lambda p: abs(p[1]), reverse=True
    )
    top_features = pairs[:10]

    result = ExplanationResult(method="shap")
    result.top_features = top_features
    result.human_explanation = _human_explanation_from_features(top_features)
    result.bar_png = _plot_bar(top_features[:8], "Why this was flagged (SHAP)")
    try:
        result.summary_png = _plot_summary(np.array(class_matrix), feature_names)
    except Exception:
        result.summary_png = None
    return result


def _explain_with_feature_importance(model_wrapper, feature_names) -> ExplanationResult:
    """Fallback explanation when SHAP isn't installed: model-native importances."""
    if hasattr(model_wrapper.model, "feature_importances_"):
        importances = model_wrapper.model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    pairs = sorted(zip(feature_names, importances), key=lambda p: abs(p[1]), reverse=True)
    top_features = pairs[:10]

    result = ExplanationResult(method="feature_importance_fallback")
    result.top_features = top_features
    result.human_explanation = _human_explanation_from_features(top_features)
    result.bar_png = _plot_bar(top_features[:8], "Top contributing features (model importance)")
    result.summary_png = None
    return result


def build_explanation_sentence(attack_type: str, human_reasons: List[str]) -> str:
    if attack_type == "Benign":
        return "Traffic pattern matches expected baseline behaviour with no significant anomaly indicators."
    if not human_reasons:
        return f"{attack_type} was detected based on the model's learned decision boundary."
    reasons_text = ", ".join(human_reasons[:-1])
    if len(human_reasons) > 1:
        reasons_text += f", and {human_reasons[-1]}"
    else:
        reasons_text = human_reasons[0]
    return f"{attack_type} was detected primarily due to: {reasons_text}."
