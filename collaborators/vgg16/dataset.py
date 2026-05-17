"""
dataset.py — HAM10000 PyTorch Dataset and stratified data-loading utilities.

Expected layout inside DATA_DIR:
    data/
    ├── HAM10000_metadata.csv   (columns: image_id, dx, ...)
    └── HAM10000_images/        (all .jpg images)

Class labels (7):  mel, nv, bcc, akiec, bkl, df, vasc
Split ratio:        70 % train / 15 % val / 15 % test  (stratified by label)
"""

import os
from typing import Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
import torch

from config import DATA_DIR, IMBALANCE_STRATEGY, LABEL_MAP, CLASS_NAMES, SEED
from imbalance import (
    get_class_weights,
    get_minority_transform,
    get_sampler,
    get_val_transform,
    print_imbalance_summary,
)


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 dermoscopy images.

    When strategy is 'augment_minority' or 'combined', each sample's transform
    is chosen individually based on whether that sample belongs to a minority
    class (anything that is not 'nv').  Otherwise a single shared transform
    (passed via the `transform` argument) is applied to every sample.
    """

    def __init__(self, df: pd.DataFrame, img_dir: str,
                 transform=None, strategy: Optional[str] = None):
        self.df      = df.reset_index(drop=True)
        self.img_dir = str(img_dir)
        self.transform = transform

        # Per-sample transform path is active for augment_minority and combined
        self._per_sample = strategy in ("augment_minority", "combined")
        if self._per_sample:
            # Precompute minority flag for every row to avoid per-call overhead
            self._minority = [row["dx"] != "nv" for _, row in self.df.iterrows()]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.img_dir, row["image_id"] + ".jpg")
        image = Image.open(path).convert("RGB")
        label = int(row["label"])

        if self._per_sample:
            tfm = get_minority_transform(self._minority[idx])
        else:
            tfm = self.transform

        if tfm is not None:
            image = tfm(image)

        return image, label


# ── Data-loading entry point ──────────────────────────────────────────────────

def load_and_split(
    data_dir=None,
    strategy: Optional[str] = None,
) -> Tuple[HAM10000Dataset, HAM10000Dataset, HAM10000Dataset,
           Optional[WeightedRandomSampler], torch.Tensor]:
    """
    Load HAM10000 metadata, perform stratified 70/15/15 split, and return:
        train_dataset, val_dataset, test_dataset, sampler, class_weights

    `sampler` is None when the active strategy does not use oversampling.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if strategy is None:
        strategy = IMBALANCE_STRATEGY

    csv_path = data_dir / "HAM10000_metadata.csv"
    img_dir  = data_dir / "HAM10000_images"

    # ── Load metadata and encode labels ──────────────────────────────────────
    df = pd.read_csv(csv_path)
    df["label"] = df["dx"].map(LABEL_MAP)

    # Drop any rows whose dx value isn't in our label map
    unknown = df["label"].isna()
    if unknown.any():
        print(f"[dataset] Dropping {unknown.sum()} rows with unrecognised 'dx' values.")
        df = df[~unknown].copy()
    df["label"] = df["label"].astype(int)

    # ── Stratified 70 / 15 / 15 split ────────────────────────────────────────
    # Step 1: carve out 15 % test
    train_val_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=SEED
    )
    # Step 2: from the remaining 85 %, take 70/85 ≈ 82.35 % as train → 70 % overall
    train_size_frac = 0.70 / 0.85
    train_df, val_df = train_test_split(
        train_val_df,
        train_size=train_size_frac,
        stratify=train_val_df["label"],
        random_state=SEED,
    )

    # ── Class statistics (computed on train split only) ───────────────────────
    class_counts = {cls: 0 for cls in CLASS_NAMES}
    for dx_val in train_df["dx"]:
        if dx_val in class_counts:
            class_counts[dx_val] += 1

    print_imbalance_summary(class_counts, strategy)

    # ── Imbalance artefacts ───────────────────────────────────────────────────
    class_weights  = get_class_weights(class_counts)
    train_labels   = train_df["label"].tolist()
    sampler        = get_sampler(train_labels)
    val_transform  = get_val_transform()

    # Training transform: per-sample when strategy needs it, shared otherwise
    if strategy in ("augment_minority", "combined"):
        train_tfm = None          # HAM10000Dataset will dispatch per sample
    else:
        train_tfm = get_minority_transform(is_minority=False)  # standard train tfm

    train_dataset = HAM10000Dataset(train_df, img_dir, transform=train_tfm, strategy=strategy)
    val_dataset   = HAM10000Dataset(val_df,   img_dir, transform=val_transform)
    test_dataset  = HAM10000Dataset(test_df,  img_dir, transform=val_transform)

    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    print(f"[dataset] Split — train: {n_train}  val: {n_val}  test: {n_test}  "
          f"(total: {n_train + n_val + n_test})")

    return train_dataset, val_dataset, test_dataset, sampler, class_weights
