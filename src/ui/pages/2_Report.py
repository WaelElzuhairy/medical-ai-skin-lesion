"""
Phase 2 Streamlit page — Full Agentic Report Pipeline.

Upload an image + enter patient metadata → runs the full orchestration:

  HIGH   → Diagnosis Agent + Evidence Agent + Guard + Report Agent → report
  MEDIUM → Uncertainty Agent → escalation banner
  LOW    → rejection message

VGG16 always runs automatically alongside the primary model.
If they disagree on binary label → escalation is forced regardless of tier.
No free LLM narrative — the Report Agent fills template slots only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import config
from src.input.image_loader import load_from_bytes

st.set_page_config(page_title="Agentic Report", layout="wide")
st.title("Phase 2 — Agentic Report")
st.caption("Full pipeline: model → router → agents → guard → report")

# ---------------------------------------------------------------------------
# Sidebar: primary model (ViT or EfficientNet)
# VGG16 always runs automatically for disagreement detection
# ---------------------------------------------------------------------------
vit_ckpts = sorted(config.CHECKPOINTS_DIR.glob("vit_*.pt"))             if config.CHECKPOINTS_DIR.exists() else []
eff_ckpts = sorted(config.CHECKPOINTS_DIR.glob("efficientnet_b4_*.pt")) if config.CHECKPOINTS_DIR.exists() else []
primary_ckpts = vit_ckpts + eff_ckpts

if not primary_ckpts:
    st.error("No trained model found. Train a model first via `scripts/train_vit.py`.")
    st.stop()

ckpt_names  = [p.name for p in primary_ckpts]
default_idx = max((i for i, n in enumerate(ckpt_names) if n.startswith("vit_")), default=len(ckpt_names)-1)
selected    = st.sidebar.selectbox("Primary model", ckpt_names, index=default_idx)
ckpt_path   = config.CHECKPOINTS_DIR / selected
is_vit      = selected.startswith("vit_")
model_label = "ViT-Base-16" if is_vit else "EfficientNet-B4"
version_tag = ckpt_path.stem.replace("vit_", "", 1) if is_vit else ckpt_path.stem.replace("efficientnet_b4_", "", 1)
arch_tag    = f"vit_{version_tag}" if is_vit else version_tag

st.sidebar.markdown(f"**Architecture:** {model_label}")
st.sidebar.markdown(f"**LLM:** `{config.LLM_PROVIDER}` / `{config.GROQ_MODEL if config.LLM_PROVIDER == 'groq' else config.ANTHROPIC_MODEL}`")

vgg_ckpt   = config.CHECKPOINTS_DIR / "vgg16.pt"
vgg_exists = vgg_ckpt.exists()
if vgg_exists:
    st.sidebar.success("VGG16 active — disagreement detection on")
else:
    st.sidebar.caption("VGG16 not found — single model mode")

# ---------------------------------------------------------------------------
# Patient metadata
# ---------------------------------------------------------------------------
st.subheader("Patient Information")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
with col2:
    sex = st.selectbox("Sex", ["male", "female", "unknown"])
with col3:
    localization = st.selectbox("Lesion localization", [
        "back", "lower extremity", "trunk", "upper extremity",
        "abdomen", "face", "chest", "foot", "hand",
        "neck", "scalp", "ear", "genital", "acral", "unknown",
    ])

metadata = {"age": int(age), "sex": sex, "localization": localization}

# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------
st.subheader("Upload Image")
uploaded = st.file_uploader(
    "Dermatoscopic image (PNG/JPEG) or DICOM",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
)

if uploaded is None:
    st.info("Upload an image and fill in patient details to generate a report.")
    st.stop()

loaded = load_from_bytes(uploaded.getvalue(), uploaded.name)

col_img, col_meta = st.columns([1, 1])
with col_img:
    st.image(loaded.image, caption="Input image", use_column_width=True)
with col_meta:
    st.markdown("**Patient summary**")
    st.markdown(f"- Age: **{age}**")
    st.markdown(f"- Sex: **{sex}**")
    st.markdown(f"- Localization: **{localization}**")
    st.markdown(f"- Primary model: **{model_label}**")
    if vgg_exists:
        st.markdown("- Secondary model: **VGG16** (disagreement check)")

st.divider()

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
if st.button("Run Agentic Pipeline", type="primary"):

    # ── Step 1: Primary model inference ─────────────────────────────────────
    with st.spinner(f"Running {model_label}…"):
        if is_vit:
            from src.deep_learning.vit_infer import vit_infer
            result = vit_infer(loaded.image, ckpt_path, version_tag=arch_tag)
        else:
            from src.deep_learning.infer import infer
            result = infer(loaded.image, ckpt_path, version_tag=arch_tag)

    # ── Step 2: VGG16 comparison ─────────────────────────────────────────────
    vgg_result = None
    models_disagree = False
    if vgg_exists:
        with st.spinner("Running VGG16 for disagreement check…"):
            from src.deep_learning.vgg_infer import vgg_infer
            vgg_result = vgg_infer(loaded.image, vgg_ckpt)
        models_disagree = (result.predicted_label != vgg_result.predicted_label)

    # ── Quick inference summary ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Binary",            result.predicted_label.upper())
    m2.metric("Most likely dx",    result.predicted_dx.upper())
    m3.metric("Router confidence", f"{result.confidence:.2%}")
    m4.metric("Malignant prob",    f"{result.malignant_prob:.2%}")

    if vgg_result is not None:
        if models_disagree:
            st.error(
                f"⚠️ **Model Disagreement Detected** — "
                f"{model_label}: **{result.predicted_label.upper()}** ({result.confidence:.1%})  ·  "
                f"VGG16: **{vgg_result.predicted_label.upper()}** ({vgg_result.confidence:.1%})  —  "
                f"**Pipeline will escalate for clinician review regardless of confidence tier.**"
            )
        else:
            st.success(
                f"✅ Models agree: **{result.predicted_label.upper()}**  —  "
                f"{model_label}: {result.confidence:.1%}  ·  VGG16: {vgg_result.confidence:.1%}"
            )

    # ── Step 3: Orchestrator ─────────────────────────────────────────────────
    with st.spinner("Running agentic pipeline (Diagnosis → Evidence → Guard → Report)…"):
        from src.agentic.orchestrator import run as orchestrate
        orch = orchestrate(result, metadata)

    st.divider()

    # ── DISAGREEMENT OVERRIDE: force escalation regardless of tier ───────────
    if models_disagree:
        st.warning("### ⚠️ Clinician Review Required — Models Disagree")
        st.markdown(
            f"The two models reached **opposite binary conclusions** on this image:\n\n"
            f"| Model | Prediction | Confidence | Trained on |\n"
            f"|-------|-----------|------------|------------|\n"
            f"| {model_label} | **{result.predicted_label.upper()}** | {result.confidence:.1%} | ISIC 2019 |\n"
            f"| VGG16 | **{vgg_result.predicted_label.upper()}** | {vgg_result.confidence:.1%} | HAM10000 |\n\n"
            f"This case **cannot be auto-reported**. A dermatologist must review before any clinical action."
        )

        with st.expander(f"{model_label} 7-class breakdown"):
            for label, prob in zip(result.dx_labels, result.dx_probs):
                badge = "🔴" if label in config.HAM10000_MALIGNANT_DX else "🟢"
                tag   = " ⬅" if label == result.predicted_dx else ""
                st.write(f"{badge} **{label.upper()}**{tag}")
                st.progress(float(prob), text=f"{prob:.2%}")

        with st.expander("VGG16 7-class breakdown"):
            for label, prob in zip(vgg_result.dx_labels, vgg_result.dx_probs):
                badge = "🔴" if label in config.HAM10000_MALIGNANT_DX else "🟢"
                tag   = " ⬅" if label == vgg_result.predicted_dx else ""
                st.write(f"{badge} **{label.upper()}**{tag}")
                st.progress(float(prob), text=f"{prob:.2%}")

        st.caption(config.CLINICAL_DISCLAIMER)

    # ── LOW tier ─────────────────────────────────────────────────────────────
    elif orch.tier.value == "LOW":
        st.error("Case Rejected — LOW Confidence")
        st.markdown(f"""
**The model is genuinely uncertain about this image.**

- Router confidence: `{result.confidence:.2%}`
- Predicted: `{result.predicted_label.upper()}` / `{result.predicted_dx.upper()}`
- Malignant probability: `{result.malignant_prob:.2%}`

This case cannot be auto-reported. Please submit for manual review.
        """)
        st.caption(config.CLINICAL_DISCLAIMER)

    # ── MEDIUM tier ───────────────────────────────────────────────────────────
    elif orch.tier.value == "MEDIUM":
        st.warning("MEDIUM Confidence — Escalated to Clinician")

        if orch.error:
            st.error(f"Pipeline error: {orch.error}")
        elif orch.escalation:
            esc = orch.escalation
            st.markdown(f"**Escalation reason:** {esc.get('escalation_reason', '')}")
            st.markdown(f"**Recommended action:** {esc.get('recommended_action', '')}")

            with st.expander("Ambiguities detected"):
                for amb in esc.get("ambiguities", []):
                    st.markdown(f"- {amb}")

            with st.expander("Model output details"):
                st.json({
                    "predicted_dx":          esc.get("predicted_dx"),
                    "predicted_binary":      esc.get("predicted_binary"),
                    "confidence":            f"{esc.get('confidence_value', 0):.2%}",
                    "malignant_probability": f"{esc.get('malignant_probability', 0):.2%}",
                })

        st.caption(config.CLINICAL_DISCLAIMER)

    # ── HIGH tier ─────────────────────────────────────────────────────────────
    elif orch.tier.value == "HIGH":
        if orch.error:
            st.error(f"Pipeline error: {orch.error}")

        elif not orch.guard_passed:
            st.error("Report Blocked by Guard Agent")
            for reason in orch.guard_blocks:
                st.markdown(f"- ❌ {reason}")
            with st.expander("Checks passed"):
                for reason in orch.guard_reasons:
                    st.markdown(f"- ✅ {reason}")

        else:
            st.success("HIGH Confidence — Report Generated")
            st.divider()

            st.subheader("Generated Report")
            st.markdown(orch.report)

            st.download_button(
                label="Download Report (.md)",
                data=orch.report,
                file_name=f"report_{result.predicted_dx}_{result.predicted_label}.md",
                mime="text/markdown",
            )

            st.divider()
            st.markdown("##### Agent Details")
            col_diag, col_ev = st.columns(2)

            with col_diag:
                with st.expander("Diagnosis Agent", expanded=True):
                    if orch.diagnosis:
                        diag = orch.diagnosis
                        icon = "✅" if diag.get("agrees") else "⚠️"
                        st.markdown(f"{icon} **CNN agreement:** {diag.get('agrees')}")
                        st.markdown(f"**Rationale:** {diag.get('rationale', '')}")
                        if diag.get("contradictions"):
                            st.markdown("**Contradictions noted:**")
                            for c in diag["contradictions"]:
                                st.markdown(f"- {c}")
                        if diag.get("supporting_factors"):
                            st.markdown("**Supporting factors:**")
                            for s in diag["supporting_factors"]:
                                st.markdown(f"- {s}")

            with col_ev:
                with st.expander("Evidence Agent", expanded=True):
                    if orch.evidence:
                        ev = orch.evidence
                        if ev.get("status") == "insufficient_evidence":
                            st.warning("No literature found above cosine 0.6 threshold.")
                        else:
                            quotes = ev.get("quotes", [])
                            st.markdown(f"**{len(quotes)} quote(s) from corpus:**")
                            for q in quotes:
                                st.markdown(f"> {q.get('text', '')}")
                                meta_str = []
                                if q.get("doi"):   meta_str.append(f"DOI: `{q['doi']}`")
                                if q.get("pub_date"): meta_str.append(q["pub_date"])
                                if meta_str: st.caption(" | ".join(meta_str))

            with st.expander("Guard Agent checks", expanded=False):
                for r in orch.guard_reasons:
                    st.markdown(f"- ✅ {r}")

        st.caption(config.CLINICAL_DISCLAIMER)
