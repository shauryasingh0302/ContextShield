# MultiOFF Dataset 🛡️

This directory stores the official **MultiOFF (Multimodal Offensive Meme Dataset)** imported as an external reference dataset for ContextShield.

---

## 1. Dataset Source & Citation

- **Authors**: Shardul Suryawanshi, Bharathi Raja Chakravarthi, Mihael Arcan, Paul Buitelaar
- **Paper**: *"Multimodal Meme Dataset (MultiOFF) for Identifying Offensive Content in Image and Text"*
- **Conference**: Proceedings of the Second Workshop on Trolling, Aggression and Cyberbullying (TRAC-2020), ACL Anthology ID: `2020.trac-1.6`
- **Official Zenodo Record**: [https://zenodo.org/records/3899868](https://zenodo.org/records/3899868) (DOI: `10.5281/zenodo.3899868`)
- **Official Code Repository**: [https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text](https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text)

---

## 2. Dataset Size & Structure

The dataset contains a total of **743 annotated meme samples** across three official splits:

```text
dataset/external/multioff/
├── Labelled Images/               # 746 image files (.png / .jpg)
├── Split Dataset/
│   ├── Training_meme_dataset.csv  # 445 samples
│   ├── Validation_meme_dataset.csv# 149 samples
│   └── Testing_meme_dataset.csv   # 149 samples
├── shardul_MultiOFF.pdf           # Original paper PDF from Zenodo
└── README.md                      # This documentation file
```

---

## 3. Original Fields

Each CSV split file contains exactly three columns:

| Column Name | Data Type | Description |
|---|---|---|
| `image_name` | String | Filename of the meme image inside `Labelled Images/` |
| `sentence` | String | Cleaned textual caption / transcription of the meme text |
| `label` | String | Original annotation class (`Non-offensiv` or `offensive`) |

---

## 4. Original Label Distribution

The dataset uses binary offensive classification:

| Split | Non-offensiv | offensive | Total |
|---|---|---|---|
| **Training** | 258 | 187 | 445 |
| **Validation** | 91 | 58 | 149 |
| **Testing** | 91 | 58 | 149 |
| **Total** | **440 (59.2%)** | **303 (40.8%)** | **743** |

*Note on original label spelling*: The original CSV files use `'Non-offensiv'` (without trailing 'e') and lowercase `'offensive'`.

---

## 5. License Information

- **Paper & Dataset Publication**: Published via ACL Anthology and Zenodo under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- Free to share, adapt, and use for academic and research purposes with appropriate citation.

---

## 6. Known Image Anomalies & Integrity Notes

- **Referenced Images in CSVs**: 743 unique filenames.
- **Images Present in Folder**: 746 image files.
- **Missing Images**: Exactly 3 image files referenced in the CSVs are missing from the official image bundle:
  - `80NRcEf.png` (referenced in `Training_meme_dataset.csv`)
  - `u4QjzUI.png` (referenced in `Validation_meme_dataset.csv`)
  - `XtxfPFR.png` (referenced in `Testing_meme_dataset.csv`)
- **Extra Unreferenced Images**: 6 image files present on disk are not referenced in the three CSV splits.
- **Captions**: 0 missing or empty captions (all 743 samples have non-empty text).

---

## 7. Limitations for ContextShield

1. **Binary vs. 4-Class Architecture**:
   - MultiOFF has only two categories: `Non-offensiv` and `offensive`.
   - ContextShield requires four distinct risk classes: `SAFE`, `OFFENSIVE`, `HATE`, and `HARASSMENT`.
   - MultiOFF does **not** distinguish identity-based hate speech or targeted harassment from general offensive vulgarity.
2. **No Severity Ratings**:
   - MultiOFF contains no severity information (`0` None, `1` Low, `2` Moderate, `3` High).
3. **English Only**:
   - All memes are English-language political memes from the 2016 US presidential election; no multilingual/Hinglish content is present.
4. **Current Status in Project**:
   - MultiOFF is stored in `dataset/external/multioff/` strictly as an external reference dataset.
   - It is **not** copied into `dataset/train.csv` or used to train models without an explicit mapping strategy.
