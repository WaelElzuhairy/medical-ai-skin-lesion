"""
Single source of truth for every tunable value in the system.

Logic files MUST import constants from here — never hardcode thresholds,
model names, paths, or the disclaimer string at the call site. The router,
Guard Agent, Evidence Agent, and calibration code all read from this module
so a single edit here updates the whole pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
HAM10000_DIR = RAW_DIR / "HAM10000"
PROCESSED_DIR = DATA_DIR / "processed"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = CORPUS_DIR / "chroma"

MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
CALIBRATION_DIR = MODELS_DIR / "calibration"
VERSIONS_FILE = MODELS_DIR / "versions.json"

# ---------------------------------------------------------------------------
# CNN / training
# ---------------------------------------------------------------------------
CNN_BACKBONE = "tf_efficientnet_b4"
CNN_INPUT_SIZE = 380  # native resolution for EfficientNet-B4
NUM_BINARY_CLASSES = 2
NUM_FINEGRAINED_CLASSES = 7  # HAM10000 dx codes

# HAM10000 7-class -> binary mapping. Pre-malignant akiec is grouped with malignant.
HAM10000_MALIGNANT_DX = {"mel", "bcc", "akiec"}
HAM10000_BENIGN_DX    = {"nv", "bkl", "df", "vasc"}

# Canonical label order (alphabetical by dx code — must match prepare_ham10000.py)
HAM10000_DX_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Indices into HAM10000_DX_LABELS that map to malignant
MALIGNANT_CLASS_INDICES = [
    HAM10000_DX_LABELS.index(dx) for dx in sorted(HAM10000_MALIGNANT_DX)
]  # [0, 1, 4]  (akiec, bcc, mel)

BINARY_LABELS = ["benign", "malignant"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE  = 32
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-4
NUM_EPOCHS       = 30
EARLY_STOP_PATIENCE = 7
NUM_WORKERS = 0  # 0 = main process only (required on Windows to avoid DataLoader spawn deadlock)

# Fine-tuning strategy — v2
LABEL_SMOOTHING       = 0.1   # reduces overconfidence on 7-class task
UNFREEZE_EPOCH        = 4     # freeze backbone for first N epochs, then unfreeze
BACKBONE_LR_MULT      = 0.1   # backbone gets lr * this; head gets full lr

# Fine-tuning strategy — v3 (focal loss warm-start from v2)
V3_LEARNING_RATE      = 1e-4  # lower LR: fine-tuning an already-trained model
V3_BACKBONE_LR_MULT   = 0.1   # backbone at 1e-5, head at 1e-4
V3_FOCAL_GAMMA        = 2.0   # focal loss focusing parameter (2 = standard)
V3_UNFREEZE_EPOCH     = 0     # unfreeze from epoch 1 (already trained backbone)
V3_EARLY_STOP_PATIENCE = 8    # more patience for fine-tuning
V3_NUM_EPOCHS         = 20    # max additional epochs

# ---------------------------------------------------------------------------
# ViT (Vision Transformer) training — vit_v1
# Base: google/vit-base-patch16-224 (86 M params, ImageNet-21k pretrained)
# ---------------------------------------------------------------------------
VIT_BASE_MODEL         = "google/vit-base-patch16-224"
VIT_INPUT_SIZE         = 224            # ViT-Base-16 native resolution
VIT_TRAIN_BATCH_SIZE   = 16             # same as EfficientNet (fits 6 GB VRAM)
VIT_EVAL_BATCH_SIZE    = 32
VIT_LEARNING_RATE      = 2e-4           # slightly higher than EffNet: ViT head needs more signal
VIT_BACKBONE_LR_MULT   = 0.1           # transformer blocks at lr * 0.1 after unfreezing
VIT_UNFREEZE_EPOCH     = 3             # freeze transformer for 3 epochs, then unfreeze
VIT_LABEL_SMOOTHING    = 0.1
VIT_WEIGHT_DECAY       = 1e-4
VIT_NUM_EPOCHS         = 25
VIT_EARLY_STOP_PATIENCE = 7

# Stratified split fractions, applied AFTER lesion_id grouping.
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Confidence routing — RULE-BASED, never LLM
# ---------------------------------------------------------------------------
# Boundaries are inclusive on the lower side: [0, LOW_MAX) -> LOW,
# [LOW_MAX, HIGH_MIN] -> MEDIUM, (HIGH_MIN, 1] -> HIGH.
ROUTER_LOW_MAX = 0.50
ROUTER_HIGH_MIN = 0.85

# Hard confidence floor enforced by Guard Agent before any report is issued.
GUARD_MIN_CONFIDENCE = ROUTER_HIGH_MIN  # high tier only path produces reports

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
ECE_TARGET = 0.05  # report Phase 1 acceptance target
ECE_NUM_BINS = 15

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
RETRIEVAL_TOP_K = 5
RETRIEVAL_MIN_COSINE = 0.6  # below this -> "insufficient_evidence", no LLM call
CHROMA_COLLECTION = "skin_lesion_literature"

# Scope of PubMed ingestion — aligned to HAM10000.
PUBMED_QUERY_TERMS = [
    "dermoscopy",
    "melanoma",
    "basal cell carcinoma",
    "actinic keratosis",
    "benign nevus",
    "skin cancer screening",
]
PUBMED_MAX_RESULTS_PER_TERM = 350  # ~2k abstracts total
NCBI_API_KEY = os.getenv("NCBI_API_KEY") or None

# DOI resolution for Guard Agent citation check.
DOI_RESOLVER_URL = "https://doi.org/api/handles/{doi}"
DOI_RESOLVE_TIMEOUT = 5

# ---------------------------------------------------------------------------
# LLM — provider-agnostic (Groq by default, Anthropic as fallback)
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "groq" or "anthropic"

# Groq (free tier)
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"

# Anthropic (paid fallback)
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL    = os.getenv("ANTHROPIC_MODEL") or "claude-haiku-4-5"

LLM_MAX_TOKENS = 1024
LLM_TIMEOUT    = 30

# ---------------------------------------------------------------------------
# Reports / safety
# ---------------------------------------------------------------------------
# Mandatory disclaimer — Guard Agent blocks any report missing this exact string.
CLINICAL_DISCLAIMER = (
    "This report is generated by an AI clinical decision support tool. "
    "It is not a diagnosis. All findings must be reviewed and confirmed "
    "by a qualified clinician before any clinical action is taken."
)
