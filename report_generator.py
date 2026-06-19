"""
report_generator.py
----------------------
Incident Report Module.

Builds a polished PDF incident report (using reportlab's Platypus layer,
not raw canvas drawing) containing:

    Username, Dataset Name, Attack Type, Confidence, SHAP Explanation,
    Risk Score, Severity, Recommendations, Date and Time

The PNG produced by explainability.py (SHAP bar chart) is embedded
directly into the PDF as supporting evidence.
"""

import io
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    HRFlowable, PageBreak,
)

import config

SEVERITY_COLOR = {
    "Critical": colors.HexColor("#EF4444"),
    "High": colors.HexColor("#FB923C"),
    "Medium": colors.HexColor("#FACC15"),
    "Low": colors.HexColor("#22C55E"),
}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ShadowTitle", fontSize=22, leading=26, textColor=colors.HexColor("#0369A1"),
        fontName="Helvetica-Bold", spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ShadowSubtitle", fontSize=11, textColor=colors.HexColor("#64748B"),
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader", fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0F172A"), spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BodyText2", fontSize=10.5, leading=15, textColor=colors.HexColor("#1E293B"),
    ))
    return styles


def generate_incident_report_pdf(*, output_path: str, username: str, dataset_name: str,
                                  attack_type: str, confidence: float, risk_score: float,
                                  severity: str, risk_category: str,
                                  human_explanation: list, explanation_sentence: str,
                                  recommendations: list, shap_chart_png: bytes = None,
                                  detection_time_ms: float = None,
                                  report_id: int = None) -> str:
    """Render the incident report and write it to `output_path`. Returns the path."""
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=20 * mm, bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    story = []

    # --- Header -----------------------------------------------------
    story.append(Paragraph(f"{config.APP_NAME}", styles["ShadowTitle"]))
    story.append(Paragraph("Incident Detection Report — Explainable Intrusion Detection System",
                            styles["ShadowSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#0EA5E9")))
    story.append(Spacer(1, 10))

    meta_rows = [
        ["Report ID", str(report_id) if report_id else "—"],
        ["Generated For", username],
        ["Dataset", dataset_name],
        ["Date / Time (UTC)", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    meta_table = Table(meta_rows, colWidths=[140, 320])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748B")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    # --- Detection summary -------------------------------------------
    story.append(Paragraph("Detection Summary", styles["SectionHeader"]))
    sev_color = SEVERITY_COLOR.get(severity, colors.grey)
    summary_rows = [
        ["Attack Type", attack_type],
        ["Model Confidence", f"{confidence * 100:.1f}%"],
        ["Severity", severity],
        ["Risk Score", f"{risk_score:.0f} / 1000"],
        ["Risk Category", risk_category],
    ]
    if detection_time_ms is not None:
        summary_rows.append(["Detection Time", f"{detection_time_ms:.1f} ms"])

    summary_table = Table(summary_rows, colWidths=[160, 300])
    style_cmds = [
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ("TEXTCOLOR", (1, 2), (1, 2), sev_color),
        ("FONTNAME", (1, 2), (1, 2), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 4), (1, 4), sev_color),
        ("FONTNAME", (1, 4), (1, 4), "Helvetica-Bold"),
    ]
    summary_table.setStyle(TableStyle(style_cmds))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # --- Explainable AI -------------------------------------------------
    story.append(Paragraph("Explainable AI — Why This Was Flagged", styles["SectionHeader"]))
    story.append(Paragraph(explanation_sentence, styles["BodyText2"]))
    if human_explanation:
        bullet_items = "<br/>".join(f"&bull;&nbsp;&nbsp;{reason}" for reason in human_explanation)
        story.append(Spacer(1, 4))
        story.append(Paragraph(bullet_items, styles["BodyText2"]))

    if shap_chart_png:
        story.append(Spacer(1, 8))
        img_buf = io.BytesIO(shap_chart_png)
        story.append(Image(img_buf, width=420, height=260))

    # --- Recommendations --------------------------------------------
    story.append(Paragraph("Security Recommendations", styles["SectionHeader"]))
    rec_items = "<br/>".join(f"&bull;&nbsp;&nbsp;{rec}" for rec in recommendations)
    story.append(Paragraph(rec_items, styles["BodyText2"]))

    # --- Footer -------------------------------------------------------
    story.append(Spacer(1, 24))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated automatically by {config.APP_NAME} — AI Powered Intrusion Detection Platform.",
        ParagraphStyle(name="Footer", fontSize=8, textColor=colors.HexColor("#94A3B8")),
    ))

    doc.build(story)
    return output_path


def report_file_path(user_id: int, prediction_id: int) -> str:
    filename = f"incident_report_user{user_id}_pred{prediction_id}.pdf"
    return os.path.join(config.REPORTS_DIR, filename)
