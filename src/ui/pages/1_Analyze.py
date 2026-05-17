"""
Phase 1 Streamlit page: upload an image, run the CNN, display:
  - Original image
  - Grad-CAM / attention heatmap
  - Binary prediction (benign / malignant) with calibrated confidence
  - Full 7-class dx breakdown with probability bars
  - Confidence tier preview (LOW / MEDIUM / HIGH)

Supports both EfficientNet-B4 (efficientnet_b4_*.pt) and the locally
trained ViT-Base-16 (vit_*.pt) checkpoints.  The correct inference
function and Grad-CAM variant are selected automatically from the filename.

No agentic layer is invoked here — this page validates the ML pipeline.
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
# Sidebar: checkpoint selection
# ---------------------------------------------------------------------------
eff_ckpts = sorted(config.CHECKPOINTS_DIR.glob("efficientnet_b4_*.pt")) if config.CHECKPOINTS_DIR.exists() else []
vit_ckpts = sorted(config.CHECKPOINTS_DIR.glob("vit_*.pt"))             if config.CHECKPOINTS_DIR.exists() else []
all_ckpts = eff_ckpts + vit_ckpts

if not all_ckpts:
    st.warning(
        "No trained model found in `models/checkpoints/`.\n\n"
        "Train EfficientNet:\n"
        "```\n"
        "python scripts/prepare_ham10000.py\n"
        "python scripts/train_cnn.py v2\n"
        "python scripts/calibrate_model.py v2\n"
        "```\n\n"
        "Or train ViT:\n"
        "```\n"
        "python scripts/train_vit.py vit_v1\n"
        "python scripts/calibrate_model.py vit_v1\n"
        "```"
    )
    st.stop()

ckpt_names = [p.name for p in all_ckpts]

# Default to the latest ViT checkpoint if available, otherwise last entry
default_idx = len(ckpt_names) - 1
for i, n in enumerate(ckpt_names):
    if n.startswith("vit_"):
        default_idx = i   # keep updating -> ends on the last ViT

selected    = st.sidebar.selectbox("Model checkpoint", ckpt_names, index=default_idx)
ckpt_path   = config.CHECKPOINTS_DIR / selected
is_vit      = selected.startswith("vit_")
model_label = "ViT-Base-16" if is_vit else "EfficientNet-B4"

# Derive version tag from filename
if is_vit:
    version_tag = ckpt_path.stem.replace("vit_", "", 1)
else:
    version_tag = ckpt_path.stem.replace("efficientnet_b4_", "", 1)

st.sidebar.markdown(f"**Architecture:** {model_label}")
st.sidebar.markdown(f"**Version tag:** `{version_tag}`")


# ---------------------------------------------------------------------------
# Cached model / processor load
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading EfficientNet checkpoint…")
def _load_efficientnet(checkpoint: str):
    from src.deep_learning.model import load_checkpoint
    return load_checkpoint(Path(checkpoint))


@st.cache_resource(show_spinner="Loading ViT checkpoint…")
def _load_vit(checkpoint: str):
    from transformers import AutoImageProcessor
    from src.deep_learning.vit_train import load_vit_checkpoint
    model, meta = load_vit_checkpoint(Path(checkpoint))
    processor   = AutoImageProcessor.from_pretrained(
        meta.get("model_name", config.VIT_BASE_MODEL)
    )
    return model, meta, processor


# ---------------------------------------------------------------------------
# Upload + analyze
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

# --- Run inference ---
with st.spinner("Running inference…"):
    if is_vit:
        from src.deep_learning.vit_infer import vit_infer
        from src.deep_learning.gradcam import generate_vit_heatmap
        result = vit_infer(loaded.image, ckpt_path, version_tag=version_tag)
        vit_model, vit_meta, vit_processor = _load_vit(str(ckpt_path))
        overlay = generate_vit_heatmap(
            vit_model, loaded.image, result.predicted_dx_idx, vit_processor
        )
        cam_caption = f"Attention map ({model_label}) — predicted dx: {result.predicted_dx.upper()}"
    else:
        from src.deep_learning.infer import infer
        from src.deep_learning.gradcam import generate_heatmap
        result = infer(loaded.image, ckpt_path, version_tag=version_tag)
        eff_model, _ = _load_efficientnet(str(ckpt_path))
        overlay = generate_heatmap(eff_model, loaded.image, result.predicted_dx_idx)
        cam_caption = f"Grad-CAM ({model_label}) — predicted dx: {result.predicted_dx.upper()}"

# --- Heatmap ---
with col_out:
    st.subheader("Grad-CAM" if not is_vit else "Attention Map")
    st.image(overlay, caption=cam_caption, use_column_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Binary prediction",     result.predicted_label.upper())
m2.metric("Most likely dx",        result.predicted_dx.upper())
m3.metric("Router confidence",     f"{result.confidence:.2%}")
m4.metric("Malignant probability", f"{result.malignant_prob:.2%}")

# Confidence tier preview (uses max(P(benign), P(malignant)))
conf = result.confidence
if conf < config.ROUTER_LOW_MAX:
    tier_label, tier_color = "LOW — genuinely uncertain, would reject", "red"
elif conf <= config.ROUTER_HIGH_MIN:
    tier_label, tier_color = "MEDIUM — moderate confidence, would escalate to clinician", "orange"
else:
    tier_label, tier_color = "HIGH — full report path", "green"
st.markdown(f"**Confidence tier:** :{tier_color}[{tier_label}]  `{conf:.2%}`")

st.divider()

# ---------------------------------------------------------------------------
# 7-class breakdown
# ---------------------------------------------------------------------------
st.subheader("7-Class Dx Breakdown")

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

col_a, col_b = st.columns(2)
for i, (label, prob) in enumerate(zip(result.dx_labels, result.dx_probs)):
    col    = col_a if i % 2 == 0 else col_b
    is_top = (label == result.predicted_dx)
    is_mal = (label in MALIGNANT_DX)
    tag    = " ⬅ predicted" if is_top else ""
    badge  = " \U0001f534" if is_mal else " \U0001f7e2"
    full   = DX_FULLNAMES.get(label, label)
    col.write(f"**{label.upper()}**{badge} — {full}{tag}")
    col.progress(float(prob), text=f"{prob:.2%}")

st.divider()

# ---------------------------------------------------------------------------
# Binary collapse detail
# ---------------------------------------------------------------------------
with st.expander("Binary collapse detail"):
    for label, p in zip(config.BINARY_LABELS, result.binary_probs):
        st.write(f"- **{label}**: {p:.4f}")
        st.progress(float(p))

st.divider()
st.info("Want the full agentic report with diagnosis cross-reference and literature evidence? → Go to **Report** in the sidebar.")
st.caption(config.CLINICAL_DISCLAIMER)
