"""
Inspection script for the official MultiOFF dataset in ContextShield.

Inspects:
  - Total samples across splits
  - Total image files on disk
  - CSV/annotation files and their row counts
  - CSV column names
  - Original label distribution (per split and overall)
  - Missing images referenced by the CSVs
  - Missing or empty captions/text
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd


def inspect_multioff(dataset_dir: Path):
    print("=" * 65)
    print("MultiOFF Dataset Inspection Report")
    print("=" * 65)

    images_dir = dataset_dir / "Labelled Images"
    split_dir = dataset_dir / "Split Dataset"

    # 1. Inspect image files on disk
    if not images_dir.exists():
        print(f"Error: Images directory not found at '{images_dir}'")
        return

    image_files = [f for f in os.listdir(images_dir) if (images_dir / f).is_file()]
    disk_image_set: Set[str] = set(image_files)
    print(f"Image Directory:            {images_dir}")
    print(f"Total Image Files on Disk:   {len(image_files)}")

    # 2. Inspect CSV files
    if not split_dir.exists():
        print(f"Error: Split dataset directory not found at '{split_dir}'")
        return

    csv_files = sorted(list(split_dir.glob("*.csv")))
    print(f"\nCSV Files Found ({len(csv_files)}):")
    for csv_path in csv_files:
        print(f"  - {csv_path.name}")

    total_samples = 0
    all_referenced_images: Set[str] = set()
    overall_label_counts: Dict[str, int] = {}
    total_missing_captions = 0

    print("\n" + "-" * 65)
    print(f"{'Split / File':<30} {'Rows':<8} {'Columns':<25}")
    print("-" * 65)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        num_rows = len(df)
        total_samples += num_rows
        cols_str = ", ".join(df.columns.tolist())
        print(f"{csv_path.name:<30} {num_rows:<8} {cols_str:<25}")

        # Track referenced images
        ref_imgs = df["image_name"].astype(str).tolist()
        all_referenced_images.update(ref_imgs)

        # Check missing captions
        missing_cap = df["sentence"].isna().sum() + (df["sentence"].astype(str).str.strip() == "").sum()
        total_missing_captions += missing_cap

        # Label counts per split
        label_dist = df["label"].value_counts().to_dict()
        for lbl, count in label_dist.items():
            overall_label_counts[lbl] = overall_label_counts.get(lbl, 0) + count

    print("-" * 65)
    print(f"Total Combined Samples:      {total_samples}")
    print(f"Unique Images in CSVs:       {len(all_referenced_images)}")

    # 3. Label distribution
    print("\nOriginal Label Distribution:")
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        dist = df["label"].value_counts().to_dict()
        dist_str = ", ".join(f"'{k}': {v}" for k, v in sorted(dist.items()))
        print(f"  - {csv_path.stem:<26}: {dist_str}")

    overall_dist_str = ", ".join(f"'{k}': {v}" for k, v in sorted(overall_label_counts.items()))
    print(f"  - {'Total (All Splits)':<26}: {overall_dist_str}")

    # 4. Check for missing images referenced by CSVs
    missing_images = sorted(list(all_referenced_images - disk_image_set))
    unreferenced_images = sorted(list(disk_image_set - all_referenced_images))

    print("\nImage Integrity Check:")
    if missing_images:
        print(f"  Missing Images Referenced in CSVs: {len(missing_images)}")
        for img in missing_images:
            # Locate which split references this missing image
            split_names = []
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                if img in df["image_name"].values:
                    split_names.append(csv_path.name)
            print(f"    - {img} (referenced in: {', '.join(split_names)})")
    else:
        print("  Missing Images Referenced in CSVs: 0 (All referenced images found)")

    print(f"  Extra/Unreferenced Images on Disk: {len(unreferenced_images)}")

    # 5. Check for missing captions
    print("\nCaption Integrity Check:")
    print(f"  Missing or Blank Captions:         {total_missing_captions}")

    print("=" * 65)


def main():
    default_dir = Path("dataset/external/multioff")
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    inspect_multioff(target_dir)


if __name__ == "__main__":
    main()
