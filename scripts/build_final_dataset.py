"""
Final Unified ContextShield Dataset Builder.

Combines the three verified, processed multimodal datasets:
  1. MultiOFF   (dataset/processed/multioff/) -> SAFE, OFFENSIVE
  2. Multi3Hate (dataset/processed/multi3hate/) -> HATE
  3. HarMeme    (dataset/processed/harmeme/)    -> HARASSMENT

Rules:
  - Preserves source train/validation/test splits for MultiOFF and HarMeme.
  - Multi3Hate (no official 3-way split) is deterministically partitioned 70/15/15
    stratified by language with fixed seed=42.
  - severity is kept blank/NULL ("") because sources do not provide compatible 0-3 severity.
  - Copies referenced images into dataset/final/images/ to produce a fully self-contained dataset.
  - Verifies: image presence, caption completeness, zero duplicate IDs, class distributions.

Output:
  dataset/final/
    train.csv
    validation.csv
    test.csv
    README.md
    images/
"""

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CSV_FIELDS = ["id", "image", "caption", "label", "severity", "language"]
VALID_LABELS = {"SAFE", "OFFENSIVE", "HATE", "HARASSMENT"}


def load_csv_split(csv_path: Path) -> List[Dict]:
    """Loads records from a processed CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)
    # Replace NaN with empty string
    df = df.fillna("")
    return df.to_dict(orient="records")


def split_multi3hate(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Deterministically partitions Multi3Hate into train/val/test splits,
    stratified by language so each language is proportionally represented.
    """
    rng = random.Random(seed)

    # Group by language
    by_lang: Dict[str, List[Dict]] = {}
    for r in records:
        lang = r.get("language", "OTHER")
        by_lang.setdefault(lang, []).append(r)

    train_all, val_all, test_all = [], [], []

    for lang, lang_records in sorted(by_lang.items()):
        shuffled = list(lang_records)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_part = shuffled[:n_train]
        val_part = shuffled[n_train : n_train + n_val]
        test_part = shuffled[n_train + n_val :]

        train_all.extend(train_part)
        val_all.extend(val_part)
        test_all.extend(test_part)

    # Shuffle combined partitions with seed for reproducibility
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    return train_all, val_all, test_all


def write_csv(filepath: Path, rows: List[Dict]):
    """Writes records to a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_final_dataset(
    project_root: Path,
    multi3hate_subset: str = "all",
    copy_images: bool = True,
):
    print("=" * 75)
    print(" Building Final Unified ContextShield Dataset")
    print("=" * 75)

    processed_dir = project_root / "dataset" / "processed"
    external_dir = project_root / "dataset" / "external"
    final_dir = project_root / "dataset" / "final"
    final_images_dir = final_dir / "images"

    multioff_dir = processed_dir / "multioff"
    multi3hate_dir = processed_dir / "multi3hate"
    harmeme_dir = processed_dir / "harmeme"

    # 1. Load MultiOFF splits
    print("\n[1] Loading MultiOFF splits...")
    mo_train = load_csv_split(multioff_dir / "train.csv")
    mo_val = load_csv_split(multioff_dir / "validation.csv")
    mo_test = load_csv_split(multioff_dir / "test.csv")
    mo_total = len(mo_train) + len(mo_val) + len(mo_test)
    print(f"    MultiOFF Loaded: {mo_total} samples (train={len(mo_train)}, val={len(mo_val)}, test={len(mo_test)})")

    # 2. Load Multi3Hate
    print(f"\n[2] Loading Multi3Hate ({multi3hate_subset})...")
    m3_file = multi3hate_dir / ("all.csv" if multi3hate_subset == "all" else "hindi_english.csv")
    m3_all = load_csv_split(m3_file)
    m3_train, m3_val, m3_test = split_multi3hate(m3_all, train_ratio=0.70, val_ratio=0.15, seed=42)
    print(f"    Multi3Hate Loaded: {len(m3_all)} samples")
    print(f"    Multi3Hate Stratified Split: train={len(m3_train)}, val={len(m3_val)}, test={len(m3_test)}")

    # 3. Load HarMeme splits
    print("\n[3] Loading HarMeme splits...")
    hm_train = load_csv_split(harmeme_dir / "train.csv")
    hm_val = load_csv_split(harmeme_dir / "validation.csv")
    hm_test = load_csv_split(harmeme_dir / "test.csv")
    hm_total = len(hm_train) + len(hm_val) + len(hm_test)
    print(f"    HarMeme Loaded: {hm_total} samples (train={len(hm_train)}, val={len(hm_val)}, test={len(hm_test)})")

    # 4. Source Image Locations map
    image_sources = {
        "multioff": external_dir / "multioff" / "Labelled Images",
        "multi3hate": external_dir / "multi3hate" / "images",
        "harmeme": external_dir / "harmeme" / "images",
    }

    # Tag records with source for image location tracking
    for r in mo_train + mo_val + mo_test:
        r["_src"] = "multioff"
    for r in m3_train + m3_val + m3_test:
        r["_src"] = "multi3hate"
    for r in hm_train + hm_val + hm_test:
        r["_src"] = "harmeme"

    # Combine splits
    final_train = mo_train + m3_train + hm_train
    final_val = mo_val + m3_val + hm_val
    final_test = mo_test + m3_test + hm_test
    final_all = final_train + final_val + final_test

    # Normalize severity to blank ("")
    for r in final_all:
        r["severity"] = ""

    # 5. Validation Checks
    print("\n" + "-" * 75)
    print(" Running Pre-Write Verification Checks")
    print("-" * 75)

    all_ids: Set[str] = set()
    duplicate_ids: List[str] = []
    missing_captions: List[str] = []
    invalid_labels: List[str] = []

    for r in final_all:
        sid = r["id"]
        if sid in all_ids:
            duplicate_ids.append(sid)
        all_ids.add(sid)

        if not r.get("caption", "").strip():
            missing_captions.append(sid)

        if r.get("label") not in VALID_LABELS:
            invalid_labels.append(f"{sid}: {r.get('label')}")

    print(f"  Total Unified Samples:       {len(final_all)}")
    print(f"  Unique Sample IDs:           {len(all_ids)} (duplicates: {len(duplicate_ids)})")
    print(f"  Missing / Empty Captions:    {len(missing_captions)}")
    print(f"  Invalid Labels:              {len(invalid_labels)}")

    if duplicate_ids:
        raise ValueError(f"Aborting: Found {len(duplicate_ids)} duplicate IDs!")
    if missing_captions:
        raise ValueError(f"Aborting: Found {len(missing_captions)} missing captions!")
    if invalid_labels:
        raise ValueError(f"Aborting: Found {len(invalid_labels)} invalid labels: {invalid_labels[:5]}")

    # 6. Copy and Verify Images
    print("\n[6] Verifying & Copying Images into dataset/final/images/...")
    if copy_images:
        final_images_dir.mkdir(parents=True, exist_ok=True)

    missing_images: List[str] = []
    copied_images = 0

    for r in final_all:
        src_dir = image_sources[r["_src"]]
        img_filename = r["image"]
        src_path = src_dir / img_filename

        if not src_path.exists() or src_path.stat().st_size == 0:
            missing_images.append(f"{r['id']}: {src_path}")
            continue

        if copy_images:
            dest_path = final_images_dir / img_filename
            if not dest_path.exists() or dest_path.stat().st_size == 0:
                shutil.copy2(str(src_path), str(dest_path))
                copied_images += 1

    print(f"  Missing Source Images:       {len(missing_images)}")
    if copy_images:
        total_on_disk = len(list(final_images_dir.glob("*.*")))
        print(f"  Total Images in Final Dir:   {total_on_disk}")

    if missing_images:
        raise FileNotFoundError(f"Aborting: {len(missing_images)} source images are missing!")

    # 7. Write Final CSVs (clean internal helper fields)
    print("\n[7] Writing Final CSV Splits...")
    for split_rows, split_name in [
        (final_train, "train.csv"),
        (final_val, "validation.csv"),
        (final_test, "test.csv"),
    ]:
        cleaned_rows = []
        for r in split_rows:
            cleaned = {k: r[k] for k in CSV_FIELDS}
            cleaned_rows.append(cleaned)
        out_file = final_dir / split_name
        write_csv(out_file, cleaned_rows)
        print(f"    Wrote {out_file.name}: {len(cleaned_rows)} rows")

    # 8. Print Comprehensive Summary Reports
    print("\n" + "=" * 75)
    print(" Final Unified ContextShield Dataset Report")
    print("=" * 75)

    # Sample counts from each source
    print("\nSamples by Source Dataset:")
    print(f"  - MultiOFF   (SAFE, OFFENSIVE) : {mo_total:>5} samples")
    print(f"  - Multi3Hate (HATE)            : {len(m3_all):>5} samples")
    print(f"  - HarMeme    (HARASSMENT)      : {hm_total:>5} samples")
    print(f"  - TOTAL                        : {len(final_all):>5} samples")

    # Overall label counts
    df_all = pd.DataFrame(final_all)
    print("\nOverall Label Distribution (4-Class):")
    label_order = ["SAFE", "OFFENSIVE", "HATE", "HARASSMENT"]
    all_counts = df_all["label"].value_counts().to_dict()
    for lbl in label_order:
        cnt = all_counts.get(lbl, 0)
        pct = cnt / len(final_all) * 100
        print(f"  - {lbl:<14}: {cnt:>5} samples ({pct:>5.1f}%)")

    # Language breakdown
    print("\nLanguage Distribution:")
    lang_counts = df_all["language"].value_counts().to_dict()
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(final_all) * 100
        print(f"  - {lang:<14}: {cnt:>5} samples ({pct:>5.1f}%)")

    # Split breakdown
    print("\nPartition Breakdown:")
    df_train = pd.DataFrame(final_train)
    df_val = pd.DataFrame(final_val)
    df_test = pd.DataFrame(final_test)

    print(f"{'Split':<14} {'SAFE':<8} {'OFFENSIVE':<11} {'HATE':<8} {'HARASSMENT':<12} {'Total':<8} {'Pct':<6}")
    print("-" * 75)
    for sp_name, df_sp in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        c = df_sp["label"].value_counts().to_dict()
        n = len(df_sp)
        pct = n / len(final_all) * 100
        print(
            f"{sp_name:<14} {c.get('SAFE', 0):<8} {c.get('OFFENSIVE', 0):<11} "
            f"{c.get('HATE', 0):<8} {c.get('HARASSMENT', 0):<12} {n:<8} {pct:>5.1f}%"
        )
    print("-" * 75)
    print(
        f"{'TOTAL':<14} {all_counts.get('SAFE', 0):<8} {all_counts.get('OFFENSIVE', 0):<11} "
        f"{all_counts.get('HATE', 0):<8} {all_counts.get('HARASSMENT', 0):<12} {len(final_all):<8} 100.0%"
    )

    print("\nIntegrity Verification:")
    print("  - Missing Images:        0 (ALL verified on disk)")
    print("  - Missing Captions:      0")
    print("  - Duplicate IDs:         0")
    print("  - Severity Populated:    Preserved blank/NULL (0 fabricated)")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build final unified ContextShield dataset")
    parser.add_argument(
        "--multi3hate-subset",
        type=str,
        choices=["all", "hindi_english"],
        default="all",
        help="Multi3Hate subset to include: 'all' (870) or 'hindi_english' (334). Default: 'all'",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Skip copying images to dataset/final/images/",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    build_final_dataset(
        project_root,
        multi3hate_subset=args.multi3hate_subset,
        copy_images=not args.no_copy_images,
    )
