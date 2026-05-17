"""
Build train/val/test manifests for HAM10000.

Critical rule: split by `lesion_id`, never by `image_id`. The dataset contains
multiple images of the same lesion; splitting at the image level leaks the same
lesion across train and test, inflating accuracy by ~5-10pp. Splits are
stratified at the lesion level on the binary benign/malignant label.

Output: parquet manifests at data/processed/{train,val,test}.parquet with
columns:
  image_id, image_path, lesion_id, dx, dx_label, binary_label,
  age, sex, localization

  dx_label   — integer 0-6 (index into config.HAM10000_DX_LABELS, alphabetical)
  binary_label — 0=benign, 1=malignant (collapsed from dx_label)

Run:
    python scripts/prepare_ham10000.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Make project root importable when running via `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


METADATA_CSV = config.HAM10000_DIR / "HAM10000_metadata.csv"
IMAGE_SUBDIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]


_DX_TO_IDX: dict[str, int] = {dx: i for i, dx in enumerate(config.HAM10000_DX_LABELS)}


def _dx_label(dx: str) -> int:
    """Return 0-6 integer index (alphabetical HAM10000_DX_LABELS order)."""
    if dx not in _DX_TO_IDX:
        raise ValueError(f"Unknown dx code: {dx!r}")
    return _DX_TO_IDX[dx]


def _binary_label(dx: str) -> int:
    """Return 1 for malignant (incl. pre-malignant akiec), 0 for benign."""
    if dx in config.HAM10000_MALIGNANT_DX:
        return 1
    if dx in config.HAM10000_BENIGN_DX:
        return 0
    raise ValueError(f"Unknown dx code: {dx!r}")


def _index_images() -> dict[str, Path]:
    """Map image_id -> absolute path by walking both HAM10000 image folders."""
    index: dict[str, Path] = {}
    for subdir in IMAGE_SUBDIRS:
        folder = config.HAM10000_DIR / subdir
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Expected HAM10000 folder not found: {folder}. "
                "Download the dataset from Kaggle and unzip into data/raw/HAM10000/."
            )
        for path in folder.iterdir():
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[path.stem] = path.resolve()
    return index


def _split_by_lesion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split at the lesion_id level."""
    lesions = (
        df.groupby("lesion_id", as_index=False)
        .agg(binary_label=("binary_label", "first"))
    )

    val_test_frac = config.SPLIT_VAL + config.SPLIT_TEST
    train_lesions, holdout_lesions = train_test_split(
        lesions,
        test_size=val_test_frac,
        stratify=lesions["binary_label"],
        random_state=config.SPLIT_SEED,
    )

    val_frac_within_holdout = config.SPLIT_VAL / val_test_frac
    val_lesions, test_lesions = train_test_split(
        holdout_lesions,
        test_size=1.0 - val_frac_within_holdout,
        stratify=holdout_lesions["binary_label"],
        random_state=config.SPLIT_SEED,
    )

    train = df[df["lesion_id"].isin(train_lesions["lesion_id"])].reset_index(drop=True)
    val = df[df["lesion_id"].isin(val_lesions["lesion_id"])].reset_index(drop=True)
    test = df[df["lesion_id"].isin(test_lesions["lesion_id"])].reset_index(drop=True)
    return train, val, test


def main() -> None:
    if not METADATA_CSV.is_file():
        raise FileNotFoundError(
            f"Missing {METADATA_CSV}. Download HAM10000 from Kaggle and unzip "
            "into data/raw/HAM10000/."
        )

    df = pd.read_csv(METADATA_CSV)
    df["dx_label"]    = df["dx"].map(_dx_label)
    df["binary_label"] = df["dx"].map(_binary_label)

    image_index = _index_images()
    df["image_path"] = df["image_id"].map(lambda x: str(image_index[x]) if x in image_index else None)
    missing = df["image_path"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} metadata rows have no matching image file on disk.")

    keep_cols = ["image_id", "image_path", "lesion_id", "dx", "dx_label",
                 "binary_label", "age", "sex", "localization"]
    df = df[keep_cols]

    train, val, test = _split_by_lesion(df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(config.PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(config.PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(config.PROCESSED_DIR / "test.parquet", index=False)

    print(f"train: {len(train):>5}  malignant={train['binary_label'].mean():.3f}")
    print(f"val:   {len(val):>5}  malignant={val['binary_label'].mean():.3f}")
    print(f"test:  {len(test):>5}  malignant={test['binary_label'].mean():.3f}")
    print("\nClass distribution (train):")
    for dx, idx in _DX_TO_IDX.items():
        n = (train["dx_label"] == idx).sum()
        print(f"  {idx} {dx:<6}  {n:>4}  ({n/len(train)*100:.1f}%)")
    print(f"\nManifests written to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
