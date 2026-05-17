# Agentic AI Medical Imaging System — Skin Lesion Classification

An end-to-end clinical decision-support prototype that classifies dermatoscopic skin-lesion images across 7 classes (benign vs malignant), explains decisions via Grad-CAM, and routes cases through a fully agentic pipeline backed by a PubMed RAG corpus and a deterministic safety layer.

**Academic project — not a diagnostic tool.** Every output carries a mandatory clinical disclaimer.

---

## Key Results

| Model | Dataset | 7-Class Accuracy | 7-Class Macro-F1 | Binary Accuracy | ECE (calibrated) |
|-------|---------|-----------------|-----------------|----------------|-----------------|
| EfficientNet-B4 (v2) | HAM10000 | — | — | — | — |
| ViT-Base-16 (v1) | HAM10000 | 53.3% | 0.508 | 86.0% | 0.046 ✅ |
| ViT-Base-16 (v3) | HAM10000 | 53.7% | 0.523 | 87.4% | 0.026 ✅ |
| **ViT-Base-16 (v4)** | **ISIC 2019** | **73.9%** | **0.631** | **89.6%** | **0.030 ✅** |
| VGG16 (ensemble) | HAM10000 | — | — | — | — |

Binary sensitivity: **80.4%** (malignant recall) · Specificity: **95.0%** (benign recall)

---

## System Architecture

```
Image + Patient Metadata
        │
        ▼
   Preprocessing ──► ViT-Base-16 (vit_v4) ──► Grad-CAM heatmap
                            │
                            ▼
               Temperature-scaled Softmax (T=0.58)
                            │
                            ▼
               Confidence Router (rule-based, no LLM)
              ┌─────────────┼──────────────┐
            <50%         50–85%          >85%
           REJECT       ESCALATE      FULL PIPELINE
                            │               │
                   Uncertainty Agent   Diagnosis Agent
                   (escalation report) (CNN × metadata)
                                           │
                                   Evidence Agent
                                   (RAG: PubMed/WHO)
                                           │
                               Guard Agent (hard rules)
                                           │
                                    Report Agent
                                  (slot-fill template)
```

---

## Project Structure

```
medical-ai/
├── config.py                     # ALL thresholds, paths, model names
├── requirements.txt
├── .env.example
│
├── scripts/
│   ├── prepare_ham10000.py       # Build lesion-id-grouped train/val/test
│   ├── prepare_isic2019.py       # Build ISIC 2019 splits (current)
│   ├── train_vit.py              # ViT-Base-16 fine-tuning
│   ├── train_cnn.py              # EfficientNet-B4 fine-tuning
│   ├── calibrate_model.py        # Temperature scaling
│   ├── collect_test_logits.py    # Logit collection (subprocess, avoids segfault)
│   ├── generate_reports.py       # Confusion matrices, reliability diagrams
│   └── build_corpus.py           # PubMed RAG corpus ingestion
│
├── src/
│   ├── deep_learning/            # Models, training, inference, calibration, GradCAM
│   ├── agentic/                  # 8-agent orchestration pipeline
│   ├── rag/                      # ChromaDB + S-PubMedBERT retriever
│   ├── preprocessing/            # Transforms + augmentation
│   ├── input/                    # Image loader (JPEG + DICOM)
│   └── ui/                       # Streamlit app (Analyze + Report pages)
│
├── collaborators/
│   └── vgg16/                    # VGG16 fine-tune by team member
│
├── models/
│   ├── calibration/              # Temperature scalars (JSON)
│   └── reports/                  # Confusion matrices, reliability diagrams, curves
│
└── tests/
    ├── test_router.py            # Confidence tier boundary tests
    ├── test_guard_agent.py       # Deterministic rule tests
    └── test_evidence_threshold.py
```

---

## Methodology

1. **Dataset**: ISIC 2019 (25,331 dermoscopic images, 7 classes + SCC merged into akiec)
2. **Model**: ViT-Base-16 fine-tuned with 2-stage training (frozen head → full backbone unfreeze)
3. **Class imbalance**: WeightedRandomSampler capped at 10× majority class + class-weighted CrossEntropyLoss
4. **Calibration**: Temperature scaling fitted on val set (ECE < 0.05 target)
5. **Agentic layer**: Rule-based confidence router → 3-tier pipeline (High/Medium/Low confidence)
6. **RAG**: 1,759 PubMed chunks embedded with S-PubMedBERT-MS-MARCO, cosine ≥ 0.6 threshold
7. **Safety**: Deterministic Guard Agent — DOI resolution, disclaimer check, contradiction detection

---

## Non-negotiable Design Rules

- Confidence router is **rule-based, never LLM** (`src/agentic/confidence_router.py`)
- Guard Agent block/pass uses **deterministic hard rules** — no LLM in the decision path
- Report Agent uses **template slot-filling only** — no free LLM narrative
- Evidence Agent returns `insufficient_evidence` when no chunk clears the cosine threshold
- Temperature-scaled softmax feeds the router — raw softmax is forbidden
- Retraining is **always human-gated**

---

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env   # fill in ANTHROPIC_API_KEY and GROQ_API_KEY
```

### Download ISIC 2019 dataset
```bash
kaggle datasets download -d nischaydnk/isic-2019-jpg-224x224-resized -p data/raw/
# Extract to data/raw/ISIC2019/images/
```

### Train & evaluate
```bash
python scripts/prepare_isic2019.py
python scripts/train_vit.py vit_v4
python scripts/calibrate_model.py vit_v4
python scripts/generate_reports.py vit_v4
```

### Run the app
```bash
streamlit run src/ui/streamlit_app.py
```

---

## Tech Stack

- PyTorch · HuggingFace Transformers · timm
- ChromaDB · sentence-transformers (S-PubMedBERT-MS-MARCO)
- Anthropic Claude (Haiku) · Groq (LLaMA-3.3-70B)
- Streamlit · pytorch-grad-cam · albumentations
- scikit-learn · pandas · numpy
