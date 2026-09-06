"""
Dataset Preparation and Splitting Script for ContextShield.

Reads raw annotations from dataset/raw/annotations.jsonl, validates every record,
and performs a reproducible 70% train / 15% validation / 15% test split.
Saves the results into dataset/train.csv, dataset/validation.csv, and dataset/test.csv.
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Field definitions
VALID_LABELS = {"SAFE", "OFFENSIVE", "HATE", "HARASSMENT"}
VALID_SEVERITIES = {0, 1, 2, 3}
VALID_LANGUAGES = {"ENGLISH", "HINDI", "HINGLISH", "OTHER"}
CSV_FIELDS = ["id", "image", "caption", "label", "severity", "language"]


def validate_sample(sample: Dict, images_dir: Path) -> Tuple[bool, str]:
    """
    Validates a single annotated sample against required schema and constraints.

    Returns:
        (is_valid, error_message)
    """
    # 1. Check all required fields are present
    for field in CSV_FIELDS:
        if field not in sample or sample[field] is None:
            return False, f"Missing required field: '{field}'"

    sample_id = str(sample["id"]).strip()
    if not sample_id:
        return False, "Field 'id' cannot be empty."

    # 2. Validate image existence
    img_name = str(sample["image"]).strip()
    if not img_name:
        return False, "Field 'image' cannot be empty."

    img_path = images_dir / img_name
    if not img_path.exists():
        # Check if absolute path was saved
        if not Path(img_name).exists():
            return False, f"Image file not found: '{img_name}' in {images_dir}"

    # 3. Validate label
    label = str(sample["label"]).strip().upper()
    if label not in VALID_LABELS:
        return False, f"Invalid label: '{label}'. Allowed: {sorted(list(VALID_LABELS))}"

    # 4. Validate severity
    try:
        severity = int(sample["severity"])
        if severity not in VALID_SEVERITIES:
            return False, f"Invalid severity: {severity}. Allowed: 0, 1, 2, 3"
    except (ValueError, TypeError):
        return False, f"Severity must be an integer (0-3), got: '{sample['severity']}'"

    # 5. Validate language
    lang = str(sample["language"]).strip().upper()
    if lang not in VALID_LANGUAGES:
        return False, f"Invalid language: '{lang}'. Allowed: {sorted(list(VALID_LANGUAGES))}"

    return True, ""


def load_raw_annotations(raw_path: Path) -> List[Dict]:
    """
    Loads raw annotations from a JSONL or JSON file.
    """
    if not raw_path.exists():
        return []

    samples = []
    with open(raw_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []

        if content.startswith("["):
            samples = json.loads(content)
        else:
            for line_no, line in enumerate(content.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON on line {line_no}: {e}")

    return samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Randomly splits samples into train, validation, and test sets.
    Handles small sample sizes gracefully.
    """
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        return [], [], []

    if n == 1:
        return shuffled, [], []

    if n == 2:
        return [shuffled[0]], [shuffled[1]], []

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Ensure train and val have at least 1 sample if N >= 3
    n_train = max(1, min(n - 2, n_train))
    n_val = max(1, min(n - n_train - 1, n_val))
    n_test = n - n_train - n_val

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val :]

    return train_set, val_set, test_set


def write_csv(filepath: Path, samples: List[Dict]):
    """
    Writes a list of sample dictionaries to a CSV file with standard fields.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for s in samples:
            clean_row = {
                "id": str(s["id"]).strip(),
                "image": Path(str(s["image"])).name,
                "caption": str(s.get("caption", "")).strip(),
                "label": str(s["label"]).strip().upper(),
                "severity": int(s["severity"]),
                "language": str(s["language"]).strip().upper(),
            }
            writer.writerow(clean_row)


def prepare_dataset(
    raw_path: Path,
    images_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    print("=" * 60)
    print("ContextShield Dataset Preparation & Splitting")
    print("=" * 60)

    raw_samples = load_raw_annotations(raw_path)
    print(f"Loaded {len(raw_samples)} raw annotations from: {raw_path}")

    if not raw_samples:
        print("\nNote: No annotations found to process.")
        print("Use the web annotation tool at /annotate to add samples.")
        print("=" * 60)
        return

    # Validate each sample
    valid_samples: List[Dict] = []
    invalid_samples: List[Tuple[Dict, str]] = []

    for s in raw_samples:
        is_valid, err = validate_sample(s, images_dir)
        if is_valid:
            valid_samples.append(s)
        else:
            invalid_samples.append((s, err))

    print(f"\nValidation Summary:")
    print(f"  Valid samples:   {len(valid_samples)}")
    print(f"  Invalid samples: {len(invalid_samples)}")

    if invalid_samples:
        print("\nInvalid Samples Details:")
        for idx, (inv, reason) in enumerate(invalid_samples, start=1):
            sample_id = inv.get("id", f"sample_{idx}")
            print(f"  - [{sample_id}] {reason}")

    if not valid_samples:
        print("\nError: 0 valid samples available for splitting.")
        return

    # Perform train/val/test split
    train_set, val_set, test_set = split_dataset(
        valid_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Save to CSV files
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "validation.csv"
    test_csv = output_dir / "test.csv"

    write_csv(train_csv, train_set)
    write_csv(val_csv, val_set)
    write_csv(test_csv, test_set)

    print("\nDataset Splits Generated:")
    print(f"  Train:      {len(train_set):>4} samples -> {train_csv}")
    print(f"  Validation: {len(val_set):>4} samples -> {val_csv}")
    print(f"  Test:       {len(test_set):>4} samples -> {test_csv}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare and split ContextShield dataset")
    parser.add_argument("--raw_path", type=str, default="dataset/raw/annotations.jsonl", help="Path to raw annotations JSONL")
    parser.add_argument("--images_dir", type=str, default="dataset/images", help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Directory to save train/validation/test CSVs")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")
    args = parser.parse_args()

    prepare_dataset(
        raw_path=Path(args.raw_path),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
