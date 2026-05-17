"""
Fetch extra images for under-represented classes from the ISIC Archive API v2.

Target classes (both need supplementing):
  df    — 65 train samples, F1=0.086  -> fetch 300 more
  akiec — 232 train samples, F1=0.451 -> fetch 200 more

Downloads only images NOT already in HAM10000, appends to train.parquet only.

Usage:
    python scripts/fetch_isic_extra.py [--dry-run]
    python scripts/fetch_isic_extra.py --dry-run
    python scripts/fetch_isic_extra.py --classes df akiec --max-df 300 --max-akiec 200

ISIC API: https://api.isic-archive.com/api/v2/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# ---------------------------------------------------------------------------
# ISIC diagnosis strings that map to our HAM10000 labels
# ---------------------------------------------------------------------------
ISIC_DIAGNOSIS_MAP = {
    "df":    "dermatofibroma",
    "akiec": "actinic keratosis",
}

DEFAULT_MAX = {
    "df":    300,
    "akiec": 200,
}

ISIC_API = "https://api.isic-archive.com/api/v2"
SESSION  = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_metadata(diagnosis: str, max_images: int) -> list[dict]:
    """Query ISIC API for images with the given diagnosis string."""
    records = []
    params  = {"diagnosis": diagnosis, "limit": 100, "offset": 0}

    while len(records) < max_images:
        resp = SESSION.get(f"{ISIC_API}/images/", params=params, timeout=30)
        resp.raise_for_status()
        data  = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        records.extend(batch)
        print(f"  [{diagnosis}] fetched {len(records)} ...", flush=True)
        if not data.get("next"):
            break
        params["offset"] += params["limit"]
        time.sleep(0.2)

    return records[:max_images]


def load_existing_ids() -> set[str]:
    """All image_ids already in our train/val/test splits."""
    ids = set()
    for split in ("train", "val", "test"):
        p = config.PROCESSED_DIR / f"{split}.parquet"
        if p.is_file():
            ids.update(pd.read_parquet(p, columns=["image_id"])["image_id"].tolist())
    return ids


def download_image(isic_id: str, files: dict, out_dir: Path) -> Path | None:
    out_path = out_dir / f"{isic_id}.jpg"
    if out_path.is_file():
        return out_path

    img_url = (
        files.get("full", {}).get("url")
        or files.get("thumbnail_256", {}).get("url")
        or f"{ISIC_API}/images/{isic_id}/download/"
    )
    try:
        r = SESSION.get(img_url, timeout=30, stream=True)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return out_path
    except Exception as e:
        print(f"  [warn] {isic_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-class fetch + download
# ---------------------------------------------------------------------------

def process_class(
    dx: str,
    max_images: int,
    existing_ids: set[str],
    dry_run: bool,
) -> list[dict]:
    """Fetch, filter, optionally download. Returns enriched records."""

    isic_diagnosis = ISIC_DIAGNOSIS_MAP[dx]
    dx_label       = config.HAM10000_DX_LABELS.index(dx)
    out_dir        = config.DATA_DIR / "raw" / "HAM10000" / f"isic_{dx}_extra"

    print(f"\n[{dx.upper()}] Querying ISIC for '{isic_diagnosis}' (max {max_images}) ...")
    records = fetch_metadata(isic_diagnosis, max_images)

    new = [r for r in records
           if (r.get("isic_id") or r.get("id", "")) not in existing_ids]
    print(f"[{dx.upper()}] {len(records)} found -> {len(new)} new (not in HAM10000)")

    if dry_run:
        print(f"  [dry-run] would download {len(new)} images to {out_dir}")
        for r in new[:5]:
            iid = r.get("isic_id") or r.get("id")
            age = r.get("metadata", {}).get("clinical", {}).get("age_approx", "?")
            sex = r.get("metadata", {}).get("clinical", {}).get("sex", "?")
            print(f"    {iid}  age={age}  sex={sex}")
        if len(new) > 5:
            print(f"    ... and {len(new)-5} more")
        return []

    # Download
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched = []
    for i, rec in enumerate(new):
        isic_id = rec.get("isic_id") or rec.get("id", "")
        path = download_image(isic_id, rec.get("files", {}), out_dir)
        if path:
            enriched.append({
                "image_id":   isic_id,
                "image_path": str(path),
                "lesion_id":  isic_id,
                "dx":         dx,
                "dx_label":   dx_label,
            })
        if (i + 1) % 50 == 0:
            print(f"  [{dx.upper()}] downloaded {i+1}/{len(new)}", flush=True)
        time.sleep(0.05)

    print(f"[{dx.upper()}] {len(enriched)}/{len(new)} images saved to {out_dir}")
    return enriched


# ---------------------------------------------------------------------------
# Append to train.parquet
# ---------------------------------------------------------------------------

def append_to_train(all_new_rows: list[dict]) -> None:
    train_df = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
    before   = len(train_df)

    new_df   = pd.DataFrame(all_new_rows)
    combined = pd.concat([train_df, new_df], ignore_index=True)
    combined.to_parquet(config.PROCESSED_DIR / "train.parquet", index=False)

    print(f"\n[manifest] train.parquet: {before} -> {len(combined)} rows")

    # Summary by class
    labels = config.HAM10000_DX_LABELS
    counts = combined["dx_label"].value_counts().sort_index()
    print("Updated class distribution in train:")
    for i, lbl in enumerate(labels):
        n = counts.get(i, 0)
        print(f"  {lbl:6s}: {n:5d} ({n/len(combined)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch extra ISIC images for df and akiec classes"
    )
    parser.add_argument("--classes",    nargs="+", default=["df", "akiec"],
                        choices=list(ISIC_DIAGNOSIS_MAP.keys()),
                        help="Which classes to supplement")
    parser.add_argument("--max-df",    type=int, default=300)
    parser.add_argument("--max-akiec", type=int, default=200)
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    max_per_class = {"df": args.max_df, "akiec": args.max_akiec}

    existing_ids = load_existing_ids()
    print(f"[HAM10000] {len(existing_ids)} existing image IDs across all splits")

    all_new_rows = []
    for dx in args.classes:
        rows = process_class(
            dx=dx,
            max_images=max_per_class.get(dx, DEFAULT_MAX[dx]),
            existing_ids=existing_ids,
            dry_run=args.dry_run,
        )
        all_new_rows.extend(rows)

    if args.dry_run:
        print("\n[dry-run] No files written. Remove --dry-run to download.")
        return

    if not all_new_rows:
        print("Nothing new downloaded.")
        return

    append_to_train(all_new_rows)

    print("\nNext steps:")
    print("  python scripts/train_vit.py vit_v2")
    print("  python scripts/calibrate_model.py vit_v2")
    print("  python scripts/collect_test_logits.py vit_v2")
    print("  python scripts/generate_reports.py vit_v2")


if __name__ == "__main__":
    main()
