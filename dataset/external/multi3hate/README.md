# Multi3Hate External Dataset

## 1. Dataset Overview

**Multi<sup>3</sup>Hate** is a multimodal, multilingual, and multicultural benchmark dataset designed to evaluate hate speech detection across visual and textual modalities under culturally grounded human perceptions.

- **Official Repository**: [GitHub - MinhDucBui/Multi3Hate](https://github.com/MinhDucBui/Multi3Hate)
- **Hugging Face**: [MinhDucBui/Multi3Hate](https://huggingface.co/datasets/MinhDucBui/Multi3Hate)
- **Paper**: *Multi<sup>3</sup>Hate: Advancing Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision–Language Models* (Bui, von der Wense, & Lauscher; arXiv:2411.03888, 2024)
- **License**: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (**CC BY-NC-ND 4.0**)

---

## 2. Dataset Structure & Formats

The official dataset distribution is packaged as an Apache Parquet table (`train-00000-of-00001.parquet`), where each row embeds both the binary image bytes and tabular metadata. ContextShield extracts these images into `images/` and stores an unbundled reference table in `metadata.csv`.

### Directory Layout

```
dataset/external/multi3hate/
├── train-00000-of-00001.parquet   # Official downloaded Hugging Face parquet archive (62.8 MB)
├── HF_README.md                   # Original repository documentation from Hugging Face
├── README.md                      # This ContextShield documentation
├── metadata.csv                   # Extracted metadata table without raw image bytes
└── images/                        # Extracted image files ({Meme ID}_{Language}.jpg, 1,500 files)
```

---

## 3. Dataset Fields & Schema

The dataset contains **1,500 total rows** across 12 primary features:

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `image` | `dict` / `image` | Contains raw binary bytes (`bytes`) and internal relative path (`path`). Extracted to `images/{Meme ID}_{Language}.jpg`. |
| `Meme ID` | `int64` | Template ID (values `0` to `299`). Each template appears across all 5 languages. |
| `Language` | `string` | ISO language code: `en` (English), `hi` (Hindi), `de` (German), `es` (Spanish), `zh` (Mandarin). |
| `Caption` | `string` | The text overlaid or paired with the meme in the specified language. |
| `US` | `float64` | Binary consensus annotation by native annotators from the United States (`1.0` = Hateful, `0.0` = Non-Hateful). |
| `DE` | `float64` | Binary consensus annotation by native annotators from Germany (`1.0` = Hateful, `0.0` = Non-Hateful). |
| `MX` | `float64` | Binary consensus annotation by native annotators from Mexico (`1.0` = Hateful, `0.0` = Non-Hateful). |
| `CN` | `float64` | Binary consensus annotation by native annotators from China (`1.0` = Hateful, `0.0` = Non-Hateful). |
| `IN` | `float64` | Binary consensus annotation by native annotators from India (`1.0` = Hateful, `0.0` = Non-Hateful). |
| `Template Name` | `string` | The common meme template descriptor (e.g., *Disaster Girl*, *Distracted Boyfriend*, *Drake Hotline Bling*). |
| `Category` | `string` | High-level sociopolitical target category. |
| `Subcategory` | `string` | Specific target identity or topic. |

---

## 4. Languages and Sample Distribution

The dataset comprises **300 unique base meme templates** rendered into 5 languages, giving **1,500 parallel samples**:

- **English (`en`)**: 300 samples
- **Hindi (`hi`)**: 300 samples
- **German (`de`)**: 300 samples
- **Spanish (`es`)**: 300 samples
- **Mandarin (`zh`)**: 300 samples

---

## 5. Annotations and Cultural Consensus

Each sample is annotated by at least **5 native speakers** from each of the 5 corresponding countries:
- `US` = United States
- `DE` = Germany
- `MX` = Mexico
- `CN` = China
- `IN` = India

The score in each country column is the majority consensus label:
- `1.0`: **Hateful**
- `0.0`: **Non-Hateful**

### Cultural Variance Highlight
The authors observed significant cultural divergence in hate perception. For instance, the pairwise label agreement between US and Indian annotators was only ~67%, demonstrating that what constitutes hate or acceptable humor depends heavily on regional sociopolitical norms.

---

## 6. Relevance to ContextShield

ContextShield requires 4 target classes: `SAFE`, `OFFENSIVE`, `HATE`, `HARASSMENT`, along with Hindi / Hinglish multilingual coverage:

1. **High-Quality Ground Truth for `HATE`**:
   Multi3Hate was specifically created and annotated for hate speech (attacks based on protected identities, race, religion, nationality, gender). Unlike generic toxicity datasets, its `1.0` labels map directly to `HATE`.

2. **Native Hindi Coverage**:
   Contains **300 real Hindi memes** written in Devanagari script. Evaluated by native Indian annotators (`IN`), this yields:
   - **180 Hateful samples (`1.0`)**
   - **120 Non-Hateful samples (`0.0`)**

3. **Multimodal Native Memes**:
   Every sample has an actual visual meme image and corresponding text/caption, fully compatible with ContextShield's XLM-RoBERTa + CLIP fusion architecture.

---

## 7. Limitations & Usage Guidelines for ContextShield

1. **License Restrictions (CC BY-NC-ND 4.0)**:
   - Permitted: Non-commercial academic research, student evaluation, internal model training.
   - Prohibited: Commercial exploitation, public re-distribution of modified derivative datasets.
2. **Missing `HARASSMENT` Category**:
   Multi3Hate focuses strictly on hate speech vs. non-hate speech. It does not annotate interpersonal targeted harassment.
3. **Missing Fine-Grained Severity**:
   Labels are binary (`0.0` or `1.0`). Numerical severity (0–3) is not provided by the authors.
4. **Cultural Perspective Alignment**:
   When using Hindi samples (`hi`), use the India consensus column (`IN`). When using English samples (`en`), use the US consensus column (`US`).

---

## 8. Citation

```bibtex
@misc{bui2024multi3hatemultimodalmultilingualmulticultural,
      title={Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection with Vision-Language Models}, 
      author={MinhDuc Bui and Katharina von der Wense and Anne Lauscher},
      year={2024},
      eprint={2411.03888},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.03888}, 
}
```
