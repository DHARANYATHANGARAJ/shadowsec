"""
charts.py
----------
Centralized Plotly chart builders. Keeping every chart factory here means
admin_dashboard.py / user_dashboard.py / app.py stay focused on layout and
just call `charts.xyz(...)`.

Color palette intentionally matches the light/cyan SOC theme (see assets/style.css).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PALETTE = {
    "primary": "#0EA5E9",      # soft cyan / sky blue
    "primary_dark": "#0369A1",
    "accent": "#22D3EE",
    "low": "#22C55E",
    "medium": "#FACC15",
    "high": "#FB923C",
    "critical": "#EF4444",
    "muted": "#64748B",
    "bg_card": "#FFFFFF",
}

ATTACK_COLOR_MAP = {
    "Benign": PALETTE["low"],
    "DDoS": PALETTE["critical"],
    "Botnet": "#B91C1C",
    "DoS": PALETTE["high"],
    "Brute Force": "#F97316",
    "Port Scan": "#FACC15",
    "Web Attack": "#A855F7",
}

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color="#1E293B"),
    margin=dict(l=20, r=20, t=50, b=20),
)


def risk_gauge(score: float, category: str, color: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 1000", "font": {"size": 30}},
            title={"text": f"Risk Score — {category}", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 1000], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, 200], "color": "#DCFCE7"},
                    {"range": [200, 500], "color": "#FEF9C3"},
                    {"range": [500, 800], "color": "#FFEDD5"},
                    {"range": [800, 1000], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": "#1E293B", "width": 3}, "value": score},
            },
        )
    )
    fig.update_layout(height=300, **_LAYOUT_DEFAULTS)
    return fig


def attack_distribution_pie(distribution: dict):
    labels = list(distribution.keys())
    values = list(distribution.values())
    colors = [ATTACK_COLOR_MAP.get(label, PALETTE["muted"]) for label in labels]
    fig = go.Figure(
        go.Pie(labels=labels, values=values, hole=0.55, marker=dict(colors=colors))
    )
    fig.update_layout(title="Attack Distribution", height=350, **_LAYOUT_DEFAULTS)
    return fig


def severity_distribution_bar(severity_counts: dict):
    order = ["Low", "Medium", "High", "Critical"]
    colors = {"Low": PALETTE["low"], "Medium": PALETTE["medium"],
              "High": PALETTE["high"], "Critical": PALETTE["critical"]}
    labels = [s for s in order if s in severity_counts]
    values = [severity_counts[s] for s in labels]
    fig = go.Figure(
        go.Bar(x=labels, y=values, marker_color=[colors[s] for s in labels])
    )
    fig.update_layout(title="Severity Distribution", height=350, **_LAYOUT_DEFAULTS)
    return fig


def risk_distribution_histogram(risk_scores: list):
    fig = go.Figure(go.Histogram(x=risk_scores, nbinsx=20, marker_color=PALETTE["primary"]))
    fig.update_layout(title="Risk Score Distribution", xaxis_title="Risk Score",
                       yaxis_title="Count", height=350, **_LAYOUT_DEFAULTS)
    return fig


def monthly_threat_trend(dates: list, counts: list):
    fig = go.Figure(
        go.Scatter(x=dates, y=counts, mode="lines+markers",
                   line=dict(color=PALETTE["primary"], width=3),
                   fill="tozeroy", fillcolor="rgba(14,165,233,0.15)")
    )
    fig.update_layout(title="Threat Trend Over Time", xaxis_title="Date",
                       yaxis_title="Detections", height=350, **_LAYOUT_DEFAULTS)
    return fig


def top_attacks_bar(attack_counts: dict, top_n=6):
    series = pd.Series(attack_counts).sort_values(ascending=False).head(top_n)
    colors = [ATTACK_COLOR_MAP.get(label, PALETTE["muted"]) for label in series.index]
    fig = go.Figure(go.Bar(x=series.index, y=series.values, marker_color=colors))
    fig.update_layout(title="Top Attack Types", height=350, **_LAYOUT_DEFAULTS)
    return fig


def confusion_matrix_heatmap(cm: np.ndarray, class_names: list):
    fig = go.Figure(
        go.Heatmap(
            z=cm, x=class_names, y=class_names, colorscale="Blues",
            texttemplate="%{z}", showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual",
        height=420, **_LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def roc_curve_chart(fpr: np.ndarray, tpr: np.ndarray, auc_value: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                              name=f"ROC (AUC = {auc_value:.3f})",
                              line=dict(color=PALETTE["primary"], width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                              line=dict(color="#CBD5E1", dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                       yaxis_title="True Positive Rate", height=400, **_LAYOUT_DEFAULTS)
    return fig


def training_curve_chart(history: dict, metric_keys=("accuracy", "val_accuracy"), title="Accuracy"):
    fig = go.Figure()
    colors = [PALETTE["primary"], PALETTE["critical"]]
    for key, color in zip(metric_keys, colors):
        if key in history:
            fig.add_trace(go.Scatter(
                y=history[key], mode="lines", name=key, line=dict(color=color, width=2.5)
            ))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=title, height=350, **_LAYOUT_DEFAULTS)
    return fig


def feature_importance_bar(importances: pd.Series, title="Top 10 Important Features"):
    series = importances.sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=series.values, y=series.index, orientation="h",
                            marker_color=PALETTE["primary"]))
    fig.update_layout(title=title, height=400, **_LAYOUT_DEFAULTS)
    return fig


def live_threat_timeline(timestamps: list, severities: list):
    color_map = {"Low": PALETTE["low"], "Medium": PALETTE["medium"],
                 "High": PALETTE["high"], "Critical": PALETTE["critical"]}
    colors = [color_map.get(s, PALETTE["muted"]) for s in severities]
    fig = go.Figure(go.Scatter(
        x=timestamps, y=[1] * len(timestamps), mode="markers",
        marker=dict(size=14, color=colors, line=dict(width=1, color="white")),
        text=severities, hoverinfo="text",
    ))
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Live Threat Timeline", height=180, showlegend=False, **_LAYOUT_DEFAULTS)
    return fig
