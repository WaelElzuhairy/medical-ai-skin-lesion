"""
Streamlit entry point.

Streamlit auto-discovers the multipage layout under `pages/`. This file is
the landing page. Phase 1 ships only the Analyze page; Phase 2 adds Report,
Phase 3 adds Clinician Review.

Run from the project root:
    streamlit run src/ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402, F401  (imported so .env loads early)

# Download model checkpoints from HF Hub if running in a cloud environment
try:
    import startup
    startup.download_models()
except Exception:
    pass  # local dev: models already present


st.set_page_config(page_title="Agentic Medical Imaging", layout="wide")

st.title("Agentic AI Medical Imaging System")
st.caption("Academic prototype — clinical decision support, not a diagnostic tool.")

st.markdown(
    """
    ### Phases
    - **Analyze** — upload an image, see classification + Grad-CAM (Phase 1 ✅).
    - **Report** — full agentic pipeline: CNN → router → agents → guard → report (Phase 2 ✅).
    - **Clinician Review** — *(Phase 3, not yet enabled)* correction interface.

    Use the sidebar to navigate.
    """
)

st.info(
    "Disclaimer: " + config.CLINICAL_DISCLAIMER,
    icon=":material/warning:",
)
