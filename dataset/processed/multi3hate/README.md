# Processed Multi3Hate Dataset (ContextShield Format)

## 1. Overview

This directory contains the **controlled conversion** of the official **Multi3Hate** dataset into ContextShield's unified 6-column dataset format:

```csv
id,image,caption,label,severity,language
```

The conversion strictly adheres to ContextShield's data integrity rules:
- **No fabricated labels**: Non-hate samples (`0.0`) are **NOT** automatically classified as `SAFE`.
- **No fabricated severities**: Multi3Hate annotators did not rate severity on a 0–3 scale, so `severity = UNAVAILABLE`.
- **No harassment extrapolation**: No sample is assigned `HARASSMENT`.
- **Culturally grounded annotations**: Annotations match the regional consensus of native speakers corresponding to each language.

---

## 2. Directory Contents

```
dataset/processed/multi3hate/
├── README.md              # This documentation file
├── all.csv                # All converted HATE samples across all 5 languages (870 rows)
└── hindi_english.csv      # Converted HATE samples for Hindi and English only (334 rows)
```

The underlying image files remain stored in the external directory:
```
dataset/external/multi3hate/images/
```

---

## 3. Conversion Rules & Methodology

### A. Cultural Consensus Mapping
Each meme language is paired with the majority vote annotation from annotators from that language's primary geographic context:

| Meme Language | Code | Annotator Country | Column Checked | ContextShield Language |
| :--- | :--- | :--- | :--- | :--- |
| **Hindi** | `hi` | India | `IN` | `HINDI` |
| **English** | `en` | United States | `US` | `ENGLISH` |
| **German** | `de` | Germany | `DE` | `OTHER` |
| **Spanish** | `es` | Mexico | `MX` | `OTHER` |
| **Mandarin** | `zh` | China | `CN` | `OTHER` |

### B. Label Mapping
- **`consensus == 1.0` $\rightarrow$ `HATE`**: Converted into the ContextShield `HATE` class.
- **`consensus == 0.0` $\rightarrow$ SKIPPED**:
  - *Rationale*: In Multi3Hate, `0.0` indicates that native annotators did not reach consensus that the meme constituted hate speech. However, a non-hateful meme may still be offensive, harassing, or benign. Silently labeling `0.0` as `SAFE` would introduce unverified label noise. Only samples with explicit positive hate evidence (`1.0`) are included.
- **`HARASSMENT`**: Not assigned to any sample, as Multi3Hate does not measure targeted individual harassment.

### C. Severity Handling
- **`severity = UNAVAILABLE`**:
  Multi3Hate does not provide severity annotations on ContextShield's 4-tier scale (`0=None`, `1=Low`, `2=Moderate`, `3=High`). Fabricating numerical severities would compromise dataset credibility.

### D. ID Scheme
Each converted record is assigned a traceable ID:
`multi3hate_{language}_{meme_id:03d}` (e.g., `multi3hate_hi_001`, `multi3hate_en_100`).

---

## 4. Conversion Statistics

### Overall Conversion (`all.csv`)
- **Total Raw Multi3Hate Memes**: 1,500
- **Total Converted Samples**: **870**
- **Total HATE Labels**: **870** (100% of converted)
- **Total Skipped Samples**: **630** (all `0.0` non-hate consensus samples)
- **Missing Images / Captions**: **0**

### Breakdown by ContextShield Language
| ContextShield Language | Original Languages | HATE Count | Notes |
| :--- | :--- | :--- | :--- |
| **HINDI** | Hindi (`hi`) evaluated by `IN` | **180** | Native Devanagari script memes |
| **ENGLISH** | English (`en`) evaluated by `US` | **154** | English cultural memes |
| **OTHER** | German (`de`), Spanish (`es`), Chinese (`zh`) | **536** | German (179), Spanish (167), Chinese (190) |
| **Total** | | **870** | |

### Hindi & English Focus Subset (`hindi_english.csv`)
- **Total Rows**: **334**
  - `HINDI`: 180 (53.9%)
  - `ENGLISH`: 154 (46.1%)

---

## 5. Limitations

1. **Class Asymmetry**: Multi3Hate conversion only outputs `HATE`. It does not provide `OFFENSIVE`, `HARASSMENT`, or verified `SAFE` samples.
2. **Binary Granularity**: Lacks nuanced severity ratings (marked `UNAVAILABLE`).
3. **Meme Specificity**: Content consists of satirical meme templates that rely heavily on cultural references.
4. **License (CC BY-NC-ND 4.0)**: Use is limited to non-commercial academic research; derivatives cannot be publicly distributed.
