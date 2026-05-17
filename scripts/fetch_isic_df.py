"""
Fetch additional Dermatofibroma (DF) images from the ISIC Archive API v2.

Downloads only DF-labelled dermoscopic images not already in HAM10000,
then appends them to the training manifest (train.parquet only).

Usage:
    python scripts/fetch_isic_df.py [--max N] [--dry-run]

    --max N     max images to download (default 300)
    --dry-run   show what would be downloaded, don't write anything

ISIC API docs: https://api.isic-archive.com/api/v2/
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ISIC_API      = "https://api.isic-archive.com/api/v2"
DF_OUT_DIR    = config.DATA_DIR / "raw" / "HAM10000" / "isic_df_extra"
TRAIN_PARQUET = config.PROCESSED_DIR / "train.parquet"
DX_LABEL_DF   = config.HAM10000_DX_LABELS.index("df")   # integer label for df
SESSION       = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ---------------------------------------------------------------------------
# Step 1: query ISIC API for DF images
# ---------------------------------------------------------------------------

def fetch_isic_df_metadata(max_images: int = 300) -> list[dict]:
    """Return ISIC image records diagnosed as dermatofibroma."""
    records  = []
    url      = f"{ISIC_API}/images/"
    params   = {
        "diagnosis":         "dermatofibroma",
        "limit":             100,
        "offset":            0,
    }

    print(f"[ISIC] Querying for dermatofibroma images (max {max_images}) ...")
    while len(records) < max_images:
        resp = SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data  = resp.json()
        batch = data.get("results", [])
        if not batch:
            break
        records.extend(batch)
        print(f"  fetched {len(records)} so far ...", flush=True)
        if not data.get("next"):
            break
        params["offset"] += params["limit"]
        time.sleep(0.2)

    return records[:max_images]


# ---------------------------------------------------------------------------
# Step 2: filter out HAM10000 duplicates
# ---------------------------------------------------------------------------

def load_ham10000_ids() -> set[str]:
    """Return all image_ids already in our processed splits."""
    ids = set()
    for split in ("train", "val", "test"):
        p = config.PROCESSED_DIR / f"{split}.parquet"
        if p.is_file():
            df = pd.read_parquet(p, columns=["image_id"])
            ids.update(df["image_id"].tolist())
    return ids


def filter_new(records: list[dict], existing_ids: set[str]) -> list[dict]:
    new = [r for r in records if r.get("isic_id", r.get("id", "")) not in existing_ids]
    print(f"[filter] {len(records)} ISIC records -> {len(new)} new (not in HAM10000)")
    return new


# ---------------------------------------------------------------------------
# Step 3: download images
# ---------------------------------------------------------------------------

def download_image(record: dict, out_dir: Path) -> Path | None:
    """Download one ISIC image. Returns local path or None on failure."""
    isic_id = record.get("isic_id") or record.get("id", "")
    if not isic_id:
        return None

    out_path = out_dir / f"{isic_id}.jpg"
    if out_path.is_file():
        return out_path   # already downloaded

    # ISIC v2 download URL
    files = record.get("files", {})
    img_url = (
        files.get("full", {}).get("url")
        or files.get("thumbnail_256", {}).get("url")
    )

    if not img_url:
        # Fall back to the dedicated download endpoint
        img_url = f"{ISIC_API}/images/{isic_id}/download/"

    try:
        r = SESSION.get(img_url, timeout=30, stream=True)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return out_path
    except Exception as e:
        print(f"  [warn] failed to download {isic_id}: {e}")
        return None


def download_all(records: list[dict], out_dir: Path) -> list[dict]:
    """Download all records and return enriched list with local paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched = []
    for i, rec in enumerate(records):
        path = download_image(rec, out_dir)
        if path:
            enriched.append({**rec, "_local_path": str(path)})
        if (i + 1) % 25 == 0:
            print(f"  downloaded {i+1}/{len(records)}", flush=True)
        time.sleep(0.05)   # be polite to the API
    return enriched


# ---------------------------------------------------------------------------
# Step 4: append to train.parquet
# ---------------------------------------------------------------------------

def append_to_train(enriched: list[dict]) -> None:
    """Add new DF rows to train.parquet."""
    if not TRAIN_PARQUET.is_file():
        raise FileNotFoundError(f"train.parquet not found at {TRAIN_PARQUET}")

    train_df = pd.read_parquet(TRAIN_PARQUET)

    new_rows = []
    for rec in enriched:
        isic_id = rec.get("isic_id") or rec.get("id", "")
        new_rows.append({
            "image_id":   isic_id,
            "image_path": rec["_local_path"],
            "lesion_id":  isic_id,           # no lesion grouping for ISIC extras
            "dx":         "df",
            "dx_label":   DX_LABEL_DF,
        })

    new_df   = pd.DataFrame(new_rows)
    combined = pd.concat([train_df, new_df], ignore_index=True)
    combined.to_parquet(TRAIN_PARQUET, index=False)

    print(f"\n[manifest] train.parquet: {len(train_df)} → {len(combined)} rows")
    print(f"  Added {len(new_rows)} new DF images from ISIC")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch extra DF images from ISIC API")
    parser.add_argument("--max",     type=int, default=300, help="Max images to fetch")
    parser.add_argument("--dry-run", action="store_true",   help="Query only, don't download")
    args = parser.parse_args()

    # 1. Query API
    records = fetch_isic_df_metadata(max_images=args.max)
    print(f"\n[ISIC] Found {len(records)} DF records in ISIC archive")

    # 2. Filter duplicates
    existing_ids = load_ham10000_ids()
    print(f"[HAM10000] {len(existing_ids)} existing image IDs in our splits")
    new_records  = filter_new(records, existing_ids)

    if not new_records:
        print("Nothing new to add — all ISIC DF images already in dataset.")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would download {len(new_records)} images:")
        for r in new_records[:10]:
            print(f"  {r.get('isic_id') or r.get('id')}  "
                  f"age={r.get('metadata', {}).get('clinical', {}).get('age_approx', '?')}  "
                  f"sex={r.get('metadata', {}).get('clinical', {}).get('sex', '?')}")
        if len(new_records) > 10:
            print(f"  ... and {len(new_records) - 10} more")
        return

    # 3. Download
    print(f"\n[download] Downloading {len(new_records)} images to {DF_OUT_DIR} ...")
    enriched = download_all(new_records, DF_OUT_DIR)
    print(f"[download] {len(enriched)}/{len(new_records)} succeeded")

    if not enriched:
        print("[error] No images downloaded — check network / API availability")
        sys.exit(1)

    # 4. Append to train manifest
    append_to_train(enriched)

    print("\nDone. Next steps:")
    print("  python scripts/train_vit.py vit_v2")
    print("  python scripts/calibrate_model.py vit_v2")
    print("  python scripts/collect_test_logits.py vit_v2")
    print("  python scripts/generate_reports.py vit_v2")


if __name__ == "__main__":
    main()
