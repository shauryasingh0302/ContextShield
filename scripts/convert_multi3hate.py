"""
Controlled Multi3Hate -> ContextShield Dataset Conversion Script.

Converts the official Multi3Hate dataset into ContextShield format:
  Fields: id, image, caption, label, severity, language

Conversion Rules:
  1. Cultural Consensus Fields:
     - Hindi ('hi')   -> 'IN' (India native annotators)
     - English ('en') -> 'US' (USA native annotators)
     - German ('de')  -> 'DE' (Germany native annotators)
     - Spanish ('es') -> 'MX' (Mexico native annotators)
     - Chinese ('zh') -> 'CN' (China native annotators)

  2. Label Mapping:
     - consensus == 1.0 -> 'HATE'
     - consensus == 0.0 -> SKIPPED (do NOT assume non-hate equals SAFE)
     - No HARASSMENT label is assigned (Multi3Hate does not annotate harassment)

  3. Language Mapping:
     - Hindi ('hi')   -> 'HINDI'
     - English ('en') -> 'ENGLISH'
     - German/Spanish/Chinese ('de', 'es', 'zh') -> 'OTHER'

  4. Severity:
     - 'UNAVAILABLE' (Multi3Hate provides binary 0.0/1.0; no fake 0-3 severity invented)

  5. Image & Caption Verification:
     - Checks image existence in dataset/external/multi3hate/images/
     - Checks caption presence and non-emptiness

Outputs:
  - dataset/processed/multi3hate/all.csv (all converted HATE samples across all languages)
  - dataset/processed/multi3hate/hindi_english.csv (Hindi and English focus subset)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CSV_FIELDS = ["id", "image", "caption", "label", "severity", "language"]

CULTURAL_CONSENSUS_COLUMNS = {
    "hi": "IN",
    "en": "US",
    "de": "DE",
    "es": "MX",
    "zh": "CN",
}

LANGUAGE_MAPPING = {
    "hi": "HINDI",
    "en": "ENGLISH",
    "de": "OTHER",
    "es": "OTHER",
    "zh": "OTHER",
}


def write_csv(filepath: Path, rows: List[Dict]):
    """Writes converted records to a standard CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def convert_multi3hate(
    metadata_csv: Path,
    images_dir: Path,
    output_dir: Path,
) -> Dict:
    print("=" * 70)
    print(" Multi3Hate -> ContextShield Controlled Conversion")
    print("=" * 70)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found at '{metadata_csv}'")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at '{images_dir}'")

    df = pd.read_csv(metadata_csv)
    total_raw_samples = len(df)
    print(f"Loaded raw metadata: {total_raw_samples} samples from {metadata_csv.name}")

    all_converted: List[Dict] = []
    hi_en_converted: List[Dict] = []

    skipped_non_hate = 0
    skipped_missing_image = 0
    skipped_missing_caption = 0
    skipped_unknown_lang = 0

    lang_hate_counts: Dict[str, int] = {}
    lang_total_counts: Dict[str, int] = {}
    missing_image_files: List[str] = []
    missing_caption_ids: List[str] = []

    for idx, row in df.iterrows():
        raw_lang = str(row["Language"]).strip().lower()
        meme_id = int(row["Meme ID"])
        caption = str(row["Caption"]).strip() if pd.notna(row["Caption"]) else ""
        img_name = str(row.get("image_filename", f"{meme_id}_{raw_lang}.jpg")).strip()

        lang_total_counts[raw_lang] = lang_total_counts.get(raw_lang, 0) + 1

        # 1. Validate Language & Consensus column
        if raw_lang not in CULTURAL_CONSENSUS_COLUMNS:
            skipped_unknown_lang += 1
            continue

        consensus_col = CULTURAL_CONSENSUS_COLUMNS[raw_lang]
        consensus_val = row.get(consensus_col)

        # 2. Check Hate Annotation: Only consensus == 1.0 is converted
        # If 0.0 or NaN, we do NOT call it SAFE; we skip it
        if pd.isna(consensus_val) or float(consensus_val) != 1.0:
            skipped_non_hate += 1
            continue

        # 3. Check Image on Disk
        img_path = images_dir / img_name
        if not img_path.exists() or img_path.stat().st_size == 0:
            skipped_missing_image += 1
            missing_image_files.append(img_name)
            continue

        # 4. Check Caption
        if not caption:
            skipped_missing_caption += 1
            missing_caption_ids.append(f"{meme_id}_{raw_lang}")
            continue

        # 5. Build Record
        cs_language = LANGUAGE_MAPPING[raw_lang]
        sample_id = f"multi3hate_{raw_lang}_{meme_id:03d}"

        record = {
            "id": sample_id,
            "image": img_name,
            "caption": caption,
            "label": "HATE",
            "severity": "UNAVAILABLE",
            "language": cs_language,
        }

        all_converted.append(record)
        lang_hate_counts[cs_language] = lang_hate_counts.get(cs_language, 0) + 1

        if raw_lang in ("hi", "en"):
            hi_en_converted.append(record)

    # Output paths
    all_csv_path = output_dir / "all.csv"
    hi_en_csv_path = output_dir / "hindi_english.csv"

    write_csv(all_csv_path, all_converted)
    write_csv(hi_en_csv_path, hi_en_converted)

    total_skipped = (
        skipped_non_hate
        + skipped_missing_image
        + skipped_missing_caption
        + skipped_unknown_lang
    )

    # Print Summary Report
    print("\n" + "-" * 70)
    print(" Conversion Results")
    print("-" * 70)
    print(f"Total Raw Multi3Hate Samples:   {total_raw_samples}")
    print(f"Total Converted (All Languages): {len(all_converted)}")
    print(f"  - Total HATE Count:            {len(all_converted)}")
    print(f"Total Skipped:                   {total_skipped}")
    print(f"  - Non-Hate (consensus != 1.0): {skipped_non_hate} (not assumed SAFE)")
    print(f"  - Missing Images:              {skipped_missing_image}")
    print(f"  - Missing Captions:            {skipped_missing_caption}")
    print(f"  - Unknown Language:            {skipped_unknown_lang}")

    print("\nHATE Count by ContextShield Language:")
    for lang, count in sorted(lang_hate_counts.items()):
        print(f"  - {lang:<12}: {count} samples")

    print("\nSubsets Written:")
    print(f"  - All Languages:               {all_csv_path} ({len(all_converted)} rows)")
    print(f"  - Hindi & English Subset:      {hi_en_csv_path} ({len(hi_en_converted)} rows)")
    print(f"    * HINDI:                     {sum(1 for r in hi_en_converted if r['language'] == 'HINDI')}")
    print(f"    * ENGLISH:                   {sum(1 for r in hi_en_converted if r['language'] == 'ENGLISH')}")

    print("\nIntegrity Verification:")
    print(f"  - Missing Images:              {len(missing_image_files)}")
    print(f"  - Missing/Empty Captions:      {len(missing_caption_ids)}")
    print("=" * 70)

    return {
        "total_raw": total_raw_samples,
        "total_converted": len(all_converted),
        "total_hate": len(all_converted),
        "total_skipped": total_skipped,
        "skipped_non_hate": skipped_non_hate,
        "missing_images": len(missing_image_files),
        "missing_captions": len(missing_caption_ids),
        "lang_hate_counts": lang_hate_counts,
        "hi_en_converted_count": len(hi_en_converted),
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    metadata_path = base_dir / "dataset" / "external" / "multi3hate" / "metadata.csv"
    images_dir = base_dir / "dataset" / "external" / "multi3hate" / "images"
    output_dir = base_dir / "dataset" / "processed" / "multi3hate"

    convert_multi3hate(metadata_path, images_dir, output_dir)
