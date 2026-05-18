"""
pdf_writer.py — Convert a structured markdown report string to a styled PDF bytes object.
Uses reportlab for reliable cross-platform PDF generation inside Docker.
"""
from __future__ import annotations
import io
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY      = HexColor("#0D1B2A")
TEAL      = HexColor("#00B4D8")
LIGHT_BG  = HexColor("#F0F7FF")
DARK_TEXT = HexColor("#1A1A2E")
GRAY      = HexColor("#6B7280")
GREEN     = HexColor("#057A55")
RED       = HexColor("#C81E1E")
AMBER     = HexColor("#92400E")
BORDER    = HexColor("#D1D5DB")


def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=white,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=11,
            textColor=HexColor("#CAE9FF"),
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=4,
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=3,
        ),
        "h3": ParagraphStyle(
            "h3",
            fontName="Helvetica-BoldOblique",
            fontSize=11,
            textColor=DARK_TEXT,
            spaceBefore=8,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK_TEXT,
            leading=15,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK_TEXT,
            leading=14,
            leftIndent=16,
            spaceAfter=3,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=GRAY,
            alignment=TA_CENTER,
            spaceBefore=6,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "meta",
            fontName="Helvetica",
            fontSize=9,
            textColor=GRAY,
            spaceAfter=2,
        ),
        "value": ParagraphStyle(
            "value",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=DARK_TEXT,
            spaceAfter=2,
        ),
        "label_green": ParagraphStyle(
            "label_green",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=GREEN,
        ),
        "label_red": ParagraphStyle(
            "label_red",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=RED,
        ),
    }
    return styles


def _header_flowable(title: str, subtitle: str):
    """Navy header banner as a mini-table."""
    data = [
        [Paragraph(title, _build_styles()["title"])],
        [Paragraph(subtitle, _build_styles()["subtitle"])],
    ]
    t = Table(data, colWidths=[17.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
    ]))
    return t


def _section_rule(styles):
    return HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceAfter=4)


def _kv_table(rows: list[tuple[str, str]], styles):
    """Compact two-column label→value table."""
    data = [[Paragraph(k, styles["meta"]), Paragraph(v, styles["value"])] for k, v in rows]
    t = Table(data, colWidths=[5 * cm, 12.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BG),
        ("GRID",       (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def markdown_to_pdf(
    report_md: str,
    predicted_dx: str = "",
    predicted_label: str = "",
    confidence: float = 0.0,
    malignant_prob: float = 0.0,
    patient_meta: dict | None = None,
) -> bytes:
    """
    Convert a markdown report string to a styled PDF.
    Returns raw PDF bytes suitable for st.download_button().
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=2.0 * cm,
        title="Clinical Decision Support Report",
        author="Agentic AI Medical Imaging System",
    )

    styles = _build_styles()
    story = []

    # ── Cover header ────────────────────────────────────────────────────────
    story.append(_header_flowable(
        "Agentic AI Medical Imaging System",
        "Clinical Decision Support Report  ·  Academic Prototype"
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── Meta info table ──────────────────────────────────────────────────────
    meta = patient_meta or {}
    label_color = "label_red" if predicted_label.lower() == "malignant" else "label_green"
    kv_rows = [
        ("Date",              datetime.now().strftime("%Y-%m-%d  %H:%M")),
        ("Binary Result",     predicted_label.upper()),
        ("Primary Diagnosis", predicted_dx.upper()),
        ("Confidence",        f"{confidence:.1%}"),
        ("Malignant Prob.",   f"{malignant_prob:.1%}"),
        ("Patient Age",       str(meta.get("age", "—"))),
        ("Sex",               str(meta.get("sex", "—")).capitalize()),
        ("Localization",      str(meta.get("localization", "—")).capitalize()),
    ]
    story.append(_kv_table(kv_rows, styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_section_rule(styles))

    # ── Parse markdown ──────────────────────────────────────────────────────
    lines = report_md.split("\n")
    for line in lines:
        line = line.rstrip()

        if not line:
            story.append(Spacer(1, 0.2 * cm))
            continue

        # Headings
        if line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), styles["h3"]))
        elif line.startswith("## "):
            story.append(_section_rule(styles))
            story.append(Paragraph(line[3:].strip(), styles["h1"]))
        elif line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), styles["h1"]))

        # Horizontal rule
        elif line.startswith("---"):
            story.append(_section_rule(styles))

        # Bullets
        elif line.startswith("- ") or line.startswith("* "):
            text = line[2:].strip()
            # Bold inline: **text**
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
            story.append(Paragraph(f"• {text}", styles["bullet"]))

        # Blockquote
        elif line.startswith("> "):
            text = line[2:].strip()
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            q_style = ParagraphStyle(
                "quote", parent=styles["body"],
                leftIndent=20, fontName="Helvetica-Oblique",
                textColor=HexColor("#374151"),
                borderPadding=(4, 4, 4, 8),
            )
            story.append(Paragraph(f"❝ {text}", q_style))

        # Normal paragraph
        else:
            text = line.strip()
            if not text:
                continue
            # Bold, italic, inline code
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
            text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
            text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
            story.append(Paragraph(text, styles["body"]))

    # ── Disclaimer footer ────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(_section_rule(styles))
    story.append(Paragraph(
        "⚠  ACADEMIC PROTOTYPE — FOR RESEARCH AND EDUCATIONAL USE ONLY. "
        "This system does not constitute medical advice and must not be used "
        "for clinical decision-making without qualified medical supervision.",
        styles["disclaimer"]
    ))
    story.append(Paragraph(
        f"Generated by Agentic AI Medical Imaging System  ·  {datetime.now().strftime('%Y-%m-%d')}",
        styles["disclaimer"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()
