import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Auto-create writable directories so downstream code never has to mkdir manually
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE              = 32
EPOCHS_PHASE1           = 10      # feature-extraction phase
EPOCHS_PHASE2           = 20      # fine-tuning phase (subject to early stopping)
LR_HEAD                 = 1e-3    # new classifier head, phase 1
LR_FINETUNE             = 1e-5    # backbone layers, phase 2
LR_CLASSIFIER_FINETUNE  = 1e-4    # classifier layers, phase 2

EARLY_STOPPING_PATIENCE = 5
SEED                    = 42

# ── Model / data ──────────────────────────────────────────────────────────────
NUM_CLASSES  = 7
CLASS_NAMES  = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
LABEL_MAP    = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Imbalance strategy ────────────────────────────────────────────────────────
# Options: "weighted_loss" | "oversampling" | "augment_minority" | "combined"
IMBALANCE_STRATEGY = "weighted_loss"
