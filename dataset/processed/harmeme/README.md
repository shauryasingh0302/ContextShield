# Processed HarMeme Dataset (ContextShield Format)

## 1. Overview

This directory contains the **controlled conversion** of the official **HarMeme** dataset into ContextShield's unified 6-column dataset format:

```csv
id,image,caption,label,severity,language
```

This dataset supplies ground-truth samples for ContextShield's **`HARASSMENT`** risk category (interpersonal bullying, targeted attacks, and harassment directed at specific individuals).

---

## 2. Directory Contents

```
dataset/processed/harmeme/
├── README.md          # This documentation file
├── all.csv            # Combined converted HARASSMENT dataset (582 rows)
├── train.csv          # Converted training split (413 rows)
├── validation.csv     # Converted validation split (50 rows)
└── test.csv           # Converted test split (119 rows)
```

The underlying image files remain stored in the external directory:
```
dataset/external/harmeme/images/
```

---

## 3. Conversion Rules & Methodology

### A. Label Mapping
- **Condition**: A meme is included **only if**:
  1. `target == "individual"` (the meme targets a specific, identifiable individual person)
  2. `harmfulness in ("somewhat harmful", "very harmful")`
- **Assigned Label**: **`HARASSMENT`**
- **All other samples are ignored**: Memes labeled `not harmful` or memes targeting `organization`, `community`, or `society` are ignored for this conversion to prevent mislabeling and label leakage.

### B. Severity Handling
- **`severity = ""` (blank / NULL)**:
  HarMeme annotates harm using a two-tier scale (`somewhat harmful`, `very harmful`). These do not map cleanly to ContextShield's 4-point severity rating (`0=None`, `1=Low`, `2=Moderate`, `3=High`). No synthetic numerical severities were fabricated.

### C. Language Mapping
- **`language = ENGLISH`**: HarMeme consists of English-language social media memes collected from Twitter and Instagram.

### D. Original Split Preservation
The conversion strictly preserves the official partition defined by the authors in `train.jsonl`, `val.jsonl`, and `test.jsonl`:
- **Train split**: 413 samples (71.0%)
- **Validation split**: 50 samples (8.6%)
- **Test split**: 119 samples (20.4%)
- **Combined total**: 582 samples (100%)

### E. Traceable Sample IDs
IDs are formatted as `harmeme_{original_id}` (e.g., `harmeme_covid_memes_5631`), ensuring direct provenance to the original dataset.

---

## 4. Verification & Integrity Results

| Metric | Verified Value | Status |
| :--- | :--- | :--- |
| **Total Processed** | 3,544 | Verified |
| **Converted HARASSMENT** | **582** | Verified |
| **Ignored (Non-Targeted / Benign)** | 2,962 | Verified |
| **Missing Images on Disk** | **0** | Verified |
| **Missing / Empty Captions** | **0** | Verified |
| **Duplicate IDs** | **0** | Verified (582 unique IDs) |
| **Label Distribution** | 100% `HARASSMENT` | Verified |

---

## 5. Limitations

1. **Context Specificity**: Memes are centered on the COVID-19 pandemic and political discourse (targeting politicians, medical officials, celebrities, and public social media figures).
2. **Monolingual (English)**: Does not contain Hindi or Hinglish examples.
3. **Absence of Numerical Severity**: Severity is left blank/NULL to preserve data honesty.
