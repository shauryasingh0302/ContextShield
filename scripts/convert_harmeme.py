"""
Controlled HarMeme -> ContextShield Dataset Conversion Script.

Converts the official HarMeme dataset into ContextShield format:
  Fields: id, image, caption, label, severity, language

Conversion Rules:
  1. Target Filter:
     - Target == "individual" AND harmfulness in ("somewhat harmful", "very harmful")
       -> ContextShield label = "HARASSMENT"
     - All other samples (non-harmful, or harmful targeting organization/community/society)
       are ignored.

  2. Language:
     - language = "ENGLISH"

  3. Severity:
     - severity = "" (blank/NULL, because HarMeme's harmfulness categories
       are not directly compatible with ContextShield's 0-3 severity scale)

  4. Splits:
     - Preserves the original HarMeme train / val / test split structure:
       * train.csv
       * validation.csv
       * test.csv
       * all.csv (combined converted dataset)

  5. Output:
     - dataset/processed/harmeme/
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CSV_FIELDS = ["id", "image", "caption", "label", "severity", "language"]


def write_csv(filepath: Path, rows: List[Dict]):
    """Writes converted records to a standard CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def convert_split(
    jsonl_path: Path,
    images_dir: Path,
    split_name: str,
    seen_ids: Set[str],
) -> Tuple[List[Dict], int, List[str], List[str]]:
    """
    Converts a single HarMeme split into ContextShield format.

    Returns:
        (converted_rows, ignored_count, missing_images, missing_captions)
    """
    converted: List[Dict] = []
    ignored_count = 0
    missing_images: List[str] = []
    missing_captions: List[str] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for rec in records:
        labels = rec.get("labels", [])
        is_individual = "individual" in labels
        is_harmful = "somewhat harmful" in labels or "very harmful" in labels

        # Rule 1: Filter only individual-targeted harmful memes
        if not (is_individual and is_harmful):
            ignored_count += 1
            continue

        raw_id = rec.get("id", "")
        img_name = rec.get("image", "")
        raw_text = rec.get("text", "")

        # Verify image on disk
        img_path = images_dir / img_name
        if not img_path.exists() or img_path.stat().st_size == 0:
            missing_images.append(img_name)
            continue

        # Verify caption
        if raw_text is None or not str(raw_text).strip():
            missing_captions.append(raw_id)
            continue

        # Clean whitespace in caption (replace multi-newlines with clean single line)
        cleaned_caption = " ".join(str(raw_text).split())

        # ID generation and uniqueness check
        sample_id = f"harmeme_{raw_id}"
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID detected: {sample_id}")
        seen_ids.add(sample_id)

        converted.append({
            "id": sample_id,
            "image": img_name,
            "caption": cleaned_caption,
            "label": "HARASSMENT",
            "severity": "",  # Blank / NULL as specified
            "language": "ENGLISH",
        })

    return converted, ignored_count, missing_images, missing_captions


def run_conversion(
    harmeme_dir: Path,
    output_dir: Path,
) -> Dict:
    print("=" * 70)
    print(" HarMeme -> ContextShield Controlled Conversion (HARASSMENT)")
    print("=" * 70)

    annotations_dir = harmeme_dir / "annotations"
    images_dir = harmeme_dir / "images"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations not found at: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found at: {images_dir}")

    splits_map = {
        "train": annotations_dir / "train.jsonl",
        "validation": annotations_dir / "val.jsonl",
        "test": annotations_dir / "test.jsonl",
    }

    seen_ids: Set[str] = set()
    split_converted: Dict[str, List[Dict]] = {}
    all_converted: List[Dict] = []
    total_ignored = 0
    all_missing_images: List[str] = []
    all_missing_captions: List[str] = []

    for split_key, split_file in splits_map.items():
        if not split_file.exists():
            raise FileNotFoundError(f"Split file missing: {split_file}")

        rows, ignored, miss_img, miss_cap = convert_split(
            split_file, images_dir, split_key, seen_ids
        )
        split_converted[split_key] = rows
        all_converted.extend(rows)
        total_ignored += ignored
        all_missing_images.extend(miss_img)
        all_missing_captions.extend(miss_cap)

        out_csv = output_dir / f"{split_key}.csv"
        write_csv(out_csv, rows)
        print(f"  {split_key:<12}: {len(rows):>4} converted rows (ignored {ignored}) -> {out_csv.name}")

    # Write combined all.csv
    all_csv = output_dir / "all.csv"
    write_csv(all_csv, all_converted)
    print(f"  {'all':<12}: {len(all_converted):>4} converted rows -> {all_csv.name}")

    # Verification checks
    print("\n" + "-" * 70)
    print(" Verification Summary")
    print("-" * 70)
    print(f"Total HarMeme Samples Processed: {len(all_converted) + total_ignored}")
    print(f"Total Converted HARASSMENT:      {len(all_converted)}")
    print(f"Total Ignored Non-Targeted:      {total_ignored}")
    print(f"Missing Images:                  {len(all_missing_images)}")
    print(f"Missing/Empty Captions:          {len(all_missing_captions)}")
    print(f"Unique Converted IDs:            {len(seen_ids)} (duplicates: {len(all_converted) - len(seen_ids)})")

    # Label distribution
    label_dist: Dict[str, int] = {}
    for r in all_converted:
        label_dist[r["label"]] = label_dist.get(r["label"], 0) + 1
    print(f"Label Distribution:              {label_dist}")

    # Split breakdown
    print("\nSplit Counts Breakdown:")
    for sp_name, rows in split_converted.items():
        pct = len(rows) / len(all_converted) * 100
        print(f"  - {sp_name:<12}: {len(rows):>4} samples ({pct:.1f}%)")

    print("=" * 70)
    print("Conversion Complete: PASS")
    print("=" * 70)

    return {
        "total_converted": len(all_converted),
        "split_counts": {k: len(v) for k, v in split_converted.items()},
        "missing_images": len(all_missing_images),
        "missing_captions": len(all_missing_captions),
        "duplicate_ids": len(all_converted) - len(seen_ids),
        "label_distribution": label_dist,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    harmeme_path = base_dir / "dataset" / "external" / "harmeme"
    output_path = base_dir / "dataset" / "processed" / "harmeme"

    run_conversion(harmeme_path, output_path)
