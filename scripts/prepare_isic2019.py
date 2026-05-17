"""
Prepare ISIC 2019 train/val/test splits for the medical-AI pipeline.

Replaces HAM10000 as the training dataset. Uses the same 7-class label
scheme as HAM10000 (SCC merged into akiec; UNK dropped).

Label mapping (matches config.HAM10000_DX_LABELS alphabetical order):
  AK  -> akiec (0)  malignant
  BCC -> bcc   (1)  malignant
  BKL -> bkl   (2)  benign
  DF  -> df    (3)  benign
  MEL -> mel   (4)  malignant
  NV  -> nv    (5)  benign
  VASC-> vasc  (6)  benign
  SCC -> akiec (0)  malignant (both are keratinocyte carcinomas)
  UNK -> dropped

Split: 70 / 15 / 15 stratified by dx (no patient grouping — all patient_ids are dummy).

Usage:
    python scripts/prepare_isic2019.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ---------------------------------------------------------------------------
GROUND_TRUTH_CSV = config.DATA_DIR / "raw" / "ISIC_2019_Training_GroundTruth.csv"
IMAGE_DIR        = config.DATA_DIR / "raw" / "ISIC2019" / "images"
OUT_DIR          = config.PROCESSED_DIR

ISIC_TO_DX = {
    "MEL":  "mel",
    "NV":   "nv",
    "BCC":  "bcc",
    "AK":   "akiec",
    "BKL":  "bkl",
    "DF":   "df",
    "VASC": "vasc",
    "SCC":  "akiec",   # merge SCC -> akiec
}
BINARY_MAL = {"mel", "bcc", "akiec"}   # malignant classes


def build_manifest() -> pd.DataFrame:
    gt = pd.read_csv(GROUND_TRUTH_CSV)

    # one-hot -> dx label
    class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    rows = []
    for _, row in gt.iterrows():
        isic_id = row["image"]
        dx_col  = None
        for col in class_cols:
            if row.get(col, 0) == 1.0:
                dx_col = col
                break
        if dx_col is None or dx_col == "UNK":
            continue
        dx = ISIC_TO_DX[dx_col]
        img_path = IMAGE_DIR / f"{isic_id}.jpg"
        if not img_path.is_file():
            continue
        rows.append({
            "image_id":   isic_id,
            "image_path": str(img_path),
            "lesion_id":  isic_id,
            "dx":         dx,
            "dx_label":   config.HAM10000_DX_LABELS.index(dx),
            "binary":     1 if dx in BINARY_MAL else 0,
        })

    df = pd.DataFrame(rows)
    print(f"Total usable images: {len(df)}")
    print("Class distribution:")
    for dx, grp in df.groupby("dx"):
        print(f"  {dx:6s}: {len(grp):6d}  ({len(grp)/len(df)*100:.1f}%)")
    return df


def split_and_save(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 70 / 30 first, then 30 -> 15 val / 15 test
    train_df, tmp_df = train_test_split(
        df, test_size=0.30, stratify=df["dx_label"], random_state=42
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, stratify=tmp_df["dx_label"], random_state=42
    )

    for split, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = OUT_DIR / f"{split}.parquet"
        sdf.reset_index(drop=True).to_parquet(out, index=False)
        print(f"{split:5s}: {len(sdf):6d} rows -> {out}")

    # Summary
    print("\nSplit class distribution:")
    labels = config.HAM10000_DX_LABELS
    for split, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = sdf["dx_label"].value_counts().sort_index()
        row = "  " + split + ": " + "  ".join(f"{labels[i]}={counts.get(i,0)}" for i in range(len(labels)))
        print(row)


def main() -> None:
    print("Building ISIC 2019 manifest...")
    df = build_manifest()
    print("\nSplitting 70/15/15 ...")
    split_and_save(df)
    print("\nDone. Next steps:")
    print("  python scripts/train_vit.py vit_v4")
    print("  python scripts/calibrate_model.py vit_v4")
    print("  python scripts/collect_test_logits.py vit_v4")
    print("  python scripts/generate_reports.py vit_v4")


if __name__ == "__main__":
    main()
