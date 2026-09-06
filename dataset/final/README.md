# ContextShield Final Unified Multimodal Dataset

## 1. Overview

This directory contains the **final unified multimodal benchmark dataset** for ContextShield risk classification. It combines three real, verified public multimodal datasets to establish complete coverage across ContextShield's four risk categories:

| Target Class | Primary Source Dataset | Selection Criterion |
| :--- | :--- | :--- |
| **`SAFE`** | **MultiOFF** | Label = `Non-offensiv` |
| **`OFFENSIVE`** | **MultiOFF** | Label = `offensive` |
| **`HATE`** | **Multi3Hate** | Native regional consensus = `1.0` (Hateful) |
| **`HARASSMENT`** | **HarMeme** | Target = `individual` AND Harmfulness $\in$ {`somewhat harmful`, `very harmful`} |

---

## 2. Directory Layout

```
dataset/final/
├── README.md          # This documentation
├── train.csv          # Training split (1,465 samples)
├── validation.csv     # Validation split (329 samples)
├── test.csv           # Test evaluation split (398 samples)
└── images/            # Self-contained image directory (2,192 verified images)
```

---

## 3. Dataset Schema & Field Definitions

Every record adheres to ContextShield's standard 6-column specification:

```csv
id,image,caption,label,severity,language
```

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Globally unique, traceable identifier (`multioff_*`, `multi3hate_*`, `harmeme_*`). |
| `image` | `string` | Filename of the meme image inside `images/`. |
| `caption` | `string` | Text string extracted from or paired with the meme. |
| `label` | `string` | One of `SAFE`, `OFFENSIVE`, `HATE`, `HARASSMENT`. |
| `severity` | `string` | **Blank / NULL (`""`)**. Left empty because source datasets do not annotate on ContextShield's 0–3 scale. No synthetic values were fabricated. |
| `language` | `string` | `ENGLISH`, `HINDI`, or `OTHER`. |

---

## 4. Partition Methodology

Rather than performing a global random reshuffle (which would contaminate official evaluation benchmarks), the final dataset **preserves original benchmark splits**:

1. **MultiOFF**: Uses original `train.csv` (444), `validation.csv` (148), and `test.csv` (148).
2. **HarMeme**: Uses original `train.jsonl` (413), `val.jsonl` (50), and `test.jsonl` (119).
3. **Multi3Hate**: Multi3Hate had no official 3-way split. It was partitioned deterministically using a stratified 70% train (608), 15% validation (131), and 15% test (131) split with fixed `seed=42`, ensuring equal proportional representation of Hindi, English, and other languages across splits.

---

## 5. Dataset Statistics & Class Distributions

### Summary by Source
- **MultiOFF**: 740 samples (440 `SAFE`, 300 `OFFENSIVE`)
- **Multi3Hate**: 870 samples (870 `HATE`)
- **HarMeme**: 582 samples (582 `HARASSMENT`)
- **Total Unified Dataset**: **2,192 samples**

### Partition Breakdown (4-Class Matrix)

| Partition | `SAFE` | `OFFENSIVE` | `HATE` | `HARASSMENT` | Total Samples | Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 258 | 186 | 608 | 413 | **1,465** | 66.8% |
| **Validation** | 91 | 57 | 131 | 50 | **329** | 15.0% |
| **Test** | 91 | 57 | 131 | 119 | **398** | 18.2% |
| **Total** | **440** | **300** | **870** | **582** | **2,192** | **100.0%** |

### Language Representation
- **`ENGLISH`**: 1,476 samples (67.3%)
- **`OTHER`** (German, Spanish, Mandarin): 536 samples (24.5%)
- **`HINDI`**: 180 samples (8.2%)

---

## 6. Verification & Data Integrity Guarantees

- **Missing Images**: **0** (All 2,192 image files verified and present in `dataset/final/images/`).
- **Missing / Empty Captions**: **0** (100% paired with text).
- **Duplicate Sample IDs**: **0** (All 2,192 sample IDs are strictly unique).
- **Class Balance**: All 4 target classes are well-represented with hundreds of genuine samples each.
- **Severity Integrity**: Preserved as blank/NULL; no artificial ratings were fabricated.
- **Self-Contained**: The dataset can be loaded by `backend/dataset.py` directly using `dataset_path="dataset/final/train.csv"`.
