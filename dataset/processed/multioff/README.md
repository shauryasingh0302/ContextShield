# Converted MultiOFF Dataset (ContextShield Format) 🛡️

This directory contains the controlled conversion of the **MultiOFF (Multimodal Offensive Meme Dataset)** into ContextShield's standard 6-field schema:
`id, image, caption, label, severity, language`

---

## 1. Conversion Rules

| MultiOFF Original Field | Value | ContextShield Converted Field | Converted Value | Rationale |
|---|---|---|---|---|
| `image_name` | String (e.g. `qDnIIHA.png`) | `image` | String (e.g. `qDnIIHA.png`) | Direct filename reference to image in `dataset/external/multioff/Labelled Images/` |
| `sentence` | String (meme text) | `caption` | String | Preserved original cleaned meme text |
| `label` | `"Non-offensiv"` | `label` | `"SAFE"` | Mapped non-offensive political discourse to SAFE |
| `label` | `"offensive"` | `label` | `"OFFENSIVE"` | Mapped offensive insults/vulgarity to OFFENSIVE |
| *(None)* | N/A | `language` | `"ENGLISH"` | All 2016 US presidential election memes are in English |
| *(None)* | N/A | `severity` | `"UNAVAILABLE"` | **No severity data exists in MultiOFF.** We do NOT invent synthetic 0–3 numbers |
| *(None)* | N/A | `id` | `multioff_<split>_<idx>` | Unique identifier generated per sample |

---

## 2. Dataset Files Generated

```text
dataset/processed/multioff/
├── train.csv         # 444 converted training samples
├── validation.csv    # 148 converted validation samples
├── test.csv          # 148 converted test samples
├── all.csv           # 740 total converted samples
└── README.md         # This documentation file
```

---

## 3. Label Breakdown by Split

| Split | SAFE (`Non-offensiv`) | OFFENSIVE (`offensive`) | Total Converted |
|---|---|---|---|
| **Train** | 258 (58.1%) | 186 (41.9%) | **444** |
| **Validation** | 91 (61.5%) | 57 (38.5%) | **148** |
| **Test** | 91 (61.5%) | 57 (38.5%) | **148** |
| **Total** | **440 (59.5%)** | **300 (40.5%)** | **740** |

---

## 4. Skipped Records (Missing Images)

MultiOFF's original CSV split files reference 3 images that are absent from the official image bundle (`dataset/external/multioff/Labelled Images/`). These 3 rows were cleanly skipped during conversion:

1. **`80NRcEf.png`** (Training split, original label: `'offensive'`)
2. **`u4QjzUI.png`** (Validation split, original label: `'offensive'`)
3. **`XtxfPFR.png`** (Testing split, original label: `'offensive'`)

All 740 converted samples have their image files verified present on disk.

---

## 5. Limitations & Design Considerations

### A. Why MultiOFF Does NOT Provide `HATE` or `HARASSMENT`
- MultiOFF was annotated strictly as a **binary offensive detection task** (Offensive vs. Non-offensive).
- The annotators did **not** distinguish between:
  - Identity-based hate speech targeting protected groups (ContextShield `HATE`)
  - Targeted bullying, persistent abuse, or stalking of individuals (ContextShield `HARASSMENT`)
  - General crude vulgarity, sarcasm, or profanity (ContextShield `OFFENSIVE`)
- Assigning `HATE` or `HARASSMENT` to any of these samples would be **fabricating labels** without ground truth annotation. Therefore, only `SAFE` and `OFFENSIVE` are populated.

### B. Why `severity` is `UNAVAILABLE` & Schema Impact
- MultiOFF contains **no severity ratings** whatsoever.
- ContextShield's schema includes `severity` (`0` None, `1` Low, `2` Moderate, `3` High).
- Rather than inventing arbitrary severity scores (e.g. guessing that every offensive meme is level 2), we explicitly set `severity = UNAVAILABLE`.
- **Important Note**: If these files are later merged with ContextShield's primary training data, the pipeline must either accept `"UNAVAILABLE"` as a valid sentinel or provide a dedicated severity imputation step.

### C. Isolation from Primary Training Sets
- These files are stored in `dataset/processed/multioff/`.
- ContextShield's primary `dataset/train.csv`, `dataset/validation.csv`, and `dataset/test.csv` remain **untouched**.
