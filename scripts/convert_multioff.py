"""
Controlled MultiOFF -> ContextShield Dataset Conversion Script.

Converts the official MultiOFF dataset into ContextShield format:
  Fields: id, image, caption, label, severity, language

Rules:
  - MultiOFF "offensive"    -> ContextShield "OFFENSIVE"
  - MultiOFF "Non-offensiv" -> ContextShield "SAFE"
  - language = "ENGLISH"
  - severity = "UNAVAILABLE" (MultiOFF does not annotate severity; no fake values invented)
  - Rows with missing image files are ignored and reported
  - Preserves original MultiOFF files unchanged
  - Does NOT touch ContextShield's primary train.csv / validation.csv / test.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Mapping rule
LABEL_MAPPING = {
    "offensive": "OFFENSIVE",
    "Non-offensiv": "SAFE",
}

CSV_FIELDS = ["id", "image", "caption", "label", "severity", "language"]


def convert_split(
    csv_path: Path,
    images_dir: Path,
    split_name: str,
) -> Tuple[List[Dict], List[Tuple[str, str, str]]]:
    """
    Converts a single MultiOFF CSV split file into ContextShield format.

    Returns:
        (converted_rows, skipped_records)
    """
    df = pd.read_csv(csv_path)
    converted: List[Dict] = []
    skipped: List[Tuple[str, str, str]] = []

    for idx, row in df.iterrows():
        img_name = str(row["image_name"]).strip()
        sentence = str(row.get("sentence", "")).strip() if pd.notna(row.get("sentence")) else ""
        raw_label = str(row["label"]).strip()

        # Check if image file exists on disk
        img_path = images_dir / img_name
        if not img_path.exists():
            skipped.append((img_name, split_name, raw_label))
            continue

        # Map label
        if raw_label not in LABEL_MAPPING:
            raise ValueError(f"Unexpected label '{raw_label}' in {csv_path.name} row {idx}")

        cs_label = LABEL_MAPPING[raw_label]
        sample_id = f"multioff_{split_name}_{len(converted) + 1:04d}"

        converted.append({
            "id": sample_id,
            "image": img_name,
            "caption": sentence,
            "label": cs_label,
            "severity": "UNAVAILABLE",
            "language": "ENGLISH",
        })

    return converted, skipped


def write_converted_csv(filepath: Path, rows: List[Dict]):
    """Writes converted records to a standard CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_conversion(
    multioff_dir: Path,
    output_dir: Path,
):
    print("=" * 65)
    print("MultiOFF -> ContextShield Dataset Conversion")
    print("=" * 65)

    split_dir = multioff_dir / "Split Dataset"
    images_dir = multioff_dir / "Labelled Images"

    if not split_dir.exists() or not images_dir.exists():
        print(f"Error: Missing Split Dataset or Labelled Images under {multioff_dir}")
        return

    splits_config = [
        ("train", split_dir / "Training_meme_dataset.csv"),
        ("validation", split_dir / "Validation_meme_dataset.csv"),
        ("test", split_dir / "Testing_meme_dataset.csv"),
    ]

    all_converted: List[Dict] = []
    all_skipped: List[Tuple[str, str, str]] = []
    split_counts: Dict[str, Dict[str, int]] = {}

    for split_key, split_file in splits_config:
        if not split_file.exists():
            print(f"Warning: Split file not found: {split_file}")
            continue

        converted_rows, skipped_rows = convert_split(split_file, images_dir, split_key)
        all_converted.extend(converted_rows)
        all_skipped.extend(skipped_rows)

        # Count labels for this split
        counts = {"SAFE": 0, "OFFENSIVE": 0}
        for r in converted_rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        split_counts[split_key] = counts

        # Write split CSV
        out_split_csv = output_dir / f"{split_key}.csv"
        write_converted_csv(out_split_csv, converted_rows)
        print(f"Saved {len(converted_rows):>4} rows -> {out_split_csv}")

    # Write combined CSV
    all_csv = output_dir / "all.csv"
    write_converted_csv(all_csv, all_converted)
    print(f"Saved {len(all_converted):>4} rows -> {all_csv}")

    # Summary calculations
    total_safe = sum(1 for r in all_converted if r["label"] == "SAFE")
    total_offensive = sum(1 for r in all_converted if r["label"] == "OFFENSIVE")

    print("\n" + "-" * 65)
    print(f"{'Split':<15} {'Total':<10} {'SAFE':<10} {'OFFENSIVE':<10}")
    print("-" * 65)
    for k, v in split_counts.items():
        subtotal = v["SAFE"] + v["OFFENSIVE"]
        print(f"{k:<15} {subtotal:<10} {v['SAFE']:<10} {v['OFFENSIVE']:<10}")
    print("-" * 65)
    print(f"{'TOTAL':<15} {len(all_converted):<10} {total_safe:<10} {total_offensive:<10}")

    print(f"\nSkipped Rows with Missing Images ({len(all_skipped)}):")
    for img, sp, raw_lbl in all_skipped:
        print(f"  - [{sp}] {img} (original label: '{raw_lbl}')")

    print("\nSeverity & Schema Verification:")
    print("  - Severity value used: 'UNAVAILABLE' (string sentinel)")
    print("  - ContextShield's strict pipeline expects integers 0-3 for severity.")
    print("  - Because MultiOFF has NO severity ratings, inventing 0-3 would be fabricating data.")
    print("  - MultiOFF does NOT contain HATE or HARASSMENT labels (strictly 2 classes).")
    print("=" * 65)


def main():
    default_multioff = Path("dataset/external/multioff")
    default_output = Path("dataset/processed/multioff")

    parser = argparse.ArgumentParser(description="Convert MultiOFF dataset to ContextShield format")
    parser.add_argument("--multioff_dir", type=str, default=str(default_multioff), help="Path to MultiOFF directory")
    parser.add_argument("--output_dir", type=str, default=str(default_output), help="Path to output converted directory")
    args = parser.parse_args()

    run_conversion(Path(args.multioff_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
