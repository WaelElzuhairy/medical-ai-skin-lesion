"""
Phase 1 Streamlit page: upload an image, run the CNN, display:
  - Original image + Grad-CAM / attention heatmap
  - Binary prediction with calibrated confidence
  - Full 7-class dx breakdown
  - Confidence tier preview (LOW / MEDIUM / HIGH)
  - If vgg16.pt exists: auto-runs VGG16 in parallel, shows disagreement banner
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import config
from src.input.image_loader import load_from_bytes

st.set_page_config(page_title="Analyze", layout="wide")
st.title("Phase 1 — Analyze")

# ---------------------------------------------------------------------------
# Sidebar: primary model selection (ViT / EfficientNet only)
# VGG16 always runs automatically in the background for comparison
# ---------------------------------------------------------------------------
eff_ckpts = sorted(config.CHECKPOINTS_DIR.glob("efficientnet_b4_*.pt")) if config.CHECKPOINTS_DIR.exists() else []
vit_ckpts = sorted(config.CHECKPOINTS_DIR.glob("vit_*.pt"))             if config.CHECKPOINTS_DIR.exists() else []
primary_ckpts = eff_ckpts + vit_ckpts

if not primary_ckpts:
    st.warning("No trained model found in `models/checkpoints/`. Train a ViT first.")
    st.stop()

ckpt_names = [p.name for p in primary_ckpts]

# Default to the latest ViT
default_idx = len(ckpt_names) - 1
for i, n in enumerate(ckpt_names):
    if n.startswith("vit_"):
        default_idx = i

selected    = st.sidebar.selectbox("Primary model", ckpt_names, index=default_idx)
ckpt_path   = config.CHECKPOINTS_DIR / selected
is_vit      = selected.startswith("vit_")
model_label = "ViT-Base-16" if is_vit else "EfficientNet-B4"
version_tag = ckpt_path.stem.replace("vit_", "", 1) if is_vit else ckpt_path.stem.replace("efficientnet_b4_", "", 1)

st.sidebar.markdown(f"**Architecture:** {model_label}")
st.sidebar.markdown(f"**Version tag:** `{version_tag}`")

vgg_ckpt   = config.CHECKPOINTS_DIR / "vgg16.pt"
vgg_exists = vgg_ckpt.exists()
if vgg_exists:
    st.sidebar.success("VGG16 loaded — disagreement detection active")
else:
    st.sidebar.caption("VGG16 not found — single model mode")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ViT checkpoint…")
def _load_vit(checkpoint: str):
    from transformers import AutoImageProcessor
    from src.deep_learning.vit_train import load_vit_checkpoint
    model, meta = load_vit_checkpoint(Path(checkpoint))
    processor   = AutoImageProcessor.from_pretrained(
        meta.get("model_name", config.VIT_BASE_MODEL)
    )
    return model, meta, processor

@st.cache_resource(show_spinner="Loading EfficientNet checkpoint…")
def _load_efficientnet(checkpoint: str):
    from src.deep_learning.model import load_checkpoint
    return load_checkpoint(Path(checkpoint))


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a dermatoscopic image (PNG/JPEG) or DICOM file",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
)

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

loaded = load_from_bytes(uploaded.getvalue(), uploaded.name)

col_in, col_out = st.columns(2)
with col_in:
    st.subheader("Input")
    st.image(
        loaded.image,
        caption=f"{loaded.modality} — {loaded.metadata.get('filename')}",
        use_column_width=True,
    )
    if loaded.modality == "DICOM":
        st.json({k: v for k, v in loaded.metadata.items() if v is not None})

# ---------------------------------------------------------------------------
# Run primary model
# ---------------------------------------------------------------------------
with st.spinner(f"Running {model_label}…"):
    if is_vit:
        from src.deep_learning.vit_infer import vit_infer
        from src.deep_learning.gradcam import generate_vit_heatmap
        result  = vit_infer(loaded.image, ckpt_path, version_tag=version_tag)
        vit_model, vit_meta, vit_processor = _load_vit(str(ckpt_path))
        overlay = generate_vit_heatmap(vit_model, loaded.image, result.predicted_dx_idx, vit_processor)
        cam_label = f"Attention Map — {model_label}"
    else:
        from src.deep_learning.infer import infer
        from src.deep_learning.gradcam import generate_heatmap
        result    = infer(loaded.image, ckpt_path, version_tag=version_tag)
        eff_model, _ = _load_efficientnet(str(ckpt_path))
        overlay   = generate_heatmap(eff_model, loaded.image, result.predicted_dx_idx)
        cam_label = f"Grad-CAM — {model_label}"

with col_out:
    st.subheader(cam_label)
    st.image(overlay, caption=f"Predicted: {result.predicted_dx.upper()}", use_column_width=True)

# ---------------------------------------------------------------------------
# Run VGG16 automatically if available
# ---------------------------------------------------------------------------
vgg_result = None
if vgg_exists:
    with st.spinner("Running VGG16 for comparison…"):
        from src.deep_learning.vgg_infer import vgg_infer
        vgg_result = vgg_infer(loaded.image, vgg_ckpt)

st.divider()

# ---------------------------------------------------------------------------
# Disagreement banner (most prominent element)
# ---------------------------------------------------------------------------
if vgg_result is not None:
    if result.predicted_label != vgg_result.predicted_label:
        st.error(
            f"### ⚠️ Models Disagree — Clinician Review Required\n\n"
            f"**{model_label}** predicts **{result.predicted_label.upper()}** "
            f"({result.confidence:.1%} confidence)  \n"
            f"**VGG16** predicts **{vgg_result.predicted_label.upper()}** "
            f"({vgg_result.confidence:.1%} confidence)  \n\n"
            f"When two independently trained models disagree on the binary classification, "
            f"this case must be reviewed by a clinician before any action is taken."
        )
    else:
        st.success(
            f"✅ **Both models agree: {result.predicted_label.upper()}**  —  "
            f"{model_label}: {result.confidence:.1%} confidence  ·  "
            f"VGG16: {vgg_result.confidence:.1%} confidence"
        )

# ---------------------------------------------------------------------------
# Primary model metrics
# ---------------------------------------------------------------------------
st.subheader(f"{model_label} Results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Binary prediction",     result.predicted_label.upper())
m2.metric("Most likely dx",        result.predicted_dx.upper())
m3.metric("Router confidence",     f"{result.confidence:.2%}")
m4.metric("Malignant probability", f"{result.malignant_prob:.2%}")

conf = result.confidence
if conf < config.ROUTER_LOW_MAX:
    tier_label, tier_color = "LOW — genuinely uncertain, would reject", "red"
elif conf <= config.ROUTER_HIGH_MIN:
    tier_label, tier_color = "MEDIUM — moderate confidence, would escalate to clinician", "orange"
else:
    tier_label, tier_color = "HIGH — full report path", "green"
st.markdown(f"**Confidence tier:** :{tier_color}[{tier_label}]  `{conf:.2%}`")

# ---------------------------------------------------------------------------
# VGG16 metrics (secondary, always shown if available)
# ---------------------------------------------------------------------------
if vgg_result is not None:
    st.divider()
    st.subheader("VGG16 Results (HAM10000)")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Binary prediction",     vgg_result.predicted_label.upper())
    v2.metric("Most likely dx",        vgg_result.predicted_dx.upper())
    v3.metric("Confidence",            f"{vgg_result.confidence:.2%}")
    v4.metric("Malignant probability", f"{vgg_result.malignant_prob:.2%}")
    st.caption("VGG16 trained on HAM10000 · uncalibrated confidence (T=1.0)")

st.divider()

# ---------------------------------------------------------------------------
# 7-class breakdown — side by side if VGG16 ran, single column otherwise
# ---------------------------------------------------------------------------
DX_FULLNAMES = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus (mole)",
    "vasc":  "Vascular Lesion",
}
MALIGNANT_DX = config.HAM10000_MALIGNANT_DX

if vgg_result is not None:
    col_primary, col_vgg = st.columns(2)
    with col_primary:
        st.subheader(f"{model_label}")
        for label, prob in zip(result.dx_labels, result.dx_probs):
            is_top = (label == result.predicted_dx)
            badge  = " 🔴" if label in MALIGNANT_DX else " 🟢"
            tag    = " ⬅" if is_top else ""
            st.write(f"**{label.upper()}**{badge} — {DX_FULLNAMES.get(label, label)}{tag}")
            st.progress(float(prob), text=f"{prob:.2%}")
    with col_vgg:
        st.subheader("VGG16")
        for label, prob in zip(vgg_result.dx_labels, vgg_result.dx_probs):
            is_top = (label == vgg_result.predicted_dx)
            badge  = " 🔴" if label in MALIGNANT_DX else " 🟢"
            tag    = " ⬅" if is_top else ""
            st.write(f"**{label.upper()}**{badge} — {DX_FULLNAMES.get(label, label)}{tag}")
            st.progress(float(prob), text=f"{prob:.2%}")
else:
    st.subheader("7-Class Dx Breakdown")
    col_a, col_b = st.columns(2)
    for i, (label, prob) in enumerate(zip(result.dx_labels, result.dx_probs)):
        col = col_a if i % 2 == 0 else col_b
        badge = " 🔴" if label in MALIGNANT_DX else " 🟢"
        tag   = " ⬅ predicted" if label == result.predicted_dx else ""
        col.write(f"**{label.upper()}**{badge} — {DX_FULLNAMES.get(label, label)}{tag}")
        col.progress(float(prob), text=f"{prob:.2%}")

st.divider()

with st.expander("Binary collapse detail"):
    for label, p in zip(config.BINARY_LABELS, result.binary_probs):
        st.write(f"- **{label}**: {p:.4f}")
        st.progress(float(p))

st.divider()
st.info("Want the full agentic report? → Go to **Report** in the sidebar.")
st.caption(config.CLINICAL_DISCLAIMER)
