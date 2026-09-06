# HarMeme External Dataset

## 1. Dataset Overview

**HarMeme** is a benchmark multimodal dataset specifically formulated for two tasks:
1. **Detecting Harmful Memes** (distinguishing benign humor from harmful attacks)
2. **Identifying Target Social Entities** (detecting whether harm is directed at an **individual**, an **organization**, a **community**, or **society at large**)

- **Official Repository**: [GitHub - di-dimitrov/harmeme](https://github.com/di-dimitrov/harmeme)
- **Data Source Fork**: [GitHub - di-dimitrov/mmf](https://github.com/di-dimitrov/mmf/tree/master/data/datasets/memes)
- **Foundational Papers**:
  - *Detecting Harmful Memes and Their Targets* (Dimitar Dimitrov, Bishr Bin Ali, Shraman Pramanick, Fabrizio Silvestri, Preslav Nakov, Giovanni Da San Martino; Findings of ACL-IJCNLP 2021, pp. 2783–2796)
  - *MOMENTA: A Multimodal Framework for Detecting Harmful Memes and Their Targets* (Shraman Pramanick, Dimitar Dimitrov, Rituparna Mukherjee, Shivam Sharma, Md. Shad Akhtar, Preslav Nakov, Tanmoy Chakraborty; Findings of EMNLP 2021)
- **License**: Open Research License (MIT / Facebook MMF Research License)

---

## 2. Directory Structure

```
dataset/external/harmeme/
├── README.md                      # This documentation file
├── annotations/
│   ├── train.jsonl                # Task 1 training split (3,013 samples)
│   ├── val.jsonl                  # Task 1 validation split (177 samples)
│   ├── test.jsonl                 # Task 1 test split (354 samples)
│   ├── target_train.jsonl         # Task 2 target training split (1,063 harmful samples)
│   ├── target_val.jsonl           # Task 2 target validation split (62 harmful samples)
│   └── target_test.jsonl          # Task 2 target test split (124 harmful samples)
└── images/                        # All 3,544 image files (covid_memes_*.png)
```

---

## 3. Dataset Annotations & Schema

Annotations are structured in standard JSON Lines (`.jsonl`) format:

```json
{
  "id": "covid_memes_5631",
  "image": "covid_memes_5631.png",
  "text": "LET ME GET THIS STRAIGHT, YOU THINK THE PRESIDENT GETTING COVID IS FUNNY? I DO. AND I'M TIRED OF PRETENDING IT'S NOT.",
  "labels": ["somewhat harmful", "individual"]
}
```

### Fields
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique sample identifier (e.g., `covid_memes_146`). |
| `image` | `string` | Relative filename of the corresponding PNG image in `images/`. |
| `text` | `string` | Extracted text / caption overlaid on or paired with the meme. |
| `labels` | `list[str]` | List of assigned category labels (harmfulness level and/or target entity). |

---

## 4. Original Labels & Distributions

The dataset encompasses **3,544 multimodal memes** with two levels of annotations:

### Level 1: Harmfulness Level (Full Dataset: 3,544 samples)
- **`not harmful`**: 2,295 memes (64.8%)
- **`somewhat harmful`**: 1,036 memes (29.2%)
- **`very harmful`**: 213 memes (6.0%)

Total harmful memes (`somewhat harmful` + `very harmful`) = **1,249 memes**.

### Level 2: Target Social Entity (Harmful Memes Subset: 1,249 samples)
- **`individual`**: **582 samples** (46.6% of harmful memes) $\rightarrow$ *Targeted personal attack / harassment*
- **`community`**: 327 samples (26.2%) $\rightarrow$ *Attacks against social/demographic groups (hate speech)*
- **`society`**: 265 samples (21.2%) $\rightarrow$ *Attacks on public welfare, science, conspiracy theories*
- **`organization`**: 75 samples (6.0%) $\rightarrow$ *Attacks on specific institutions, agencies, or companies*

---

## 5. Relevance to ContextShield's HARASSMENT Category

ContextShield requires distinguishing **HARASSMENT** (targeted bullying against a specific person) from **HATE** (attacks against protected identity groups):

1. **Explicit `individual` Target Annotation**:
   HarMeme explicitly annotates whether a meme targets an **individual** person (e.g., public officials, politicians, doctors, celebrities, private individuals) rather than a demographic community.
2. **582 Verified Multimodal Harassment Samples**:
   Every one of the 582 individual-targeted memes has both an actual image file on disk (`images/covid_memes_*.png`) and an extracted text caption.
3. **Harm Severity Gradients**:
   Within the 582 individual-targeted memes:
   - `somewhat harmful`: 490 samples
   - `very harmful`: 92 samples

---

## 6. Limitations & ContextShield Usage Guidelines

1. **Western / English Focus**:
   Memes are primarily in English and centered on COVID-19 and political figures. They do not provide Hindi/Hinglish Indian context.
2. **Public Figures vs. Private Individuals**:
   Because COVID-19 memes were sourced from public social media (Twitter/Instagram), many individual targets are prominent public figures (e.g., Anthony Fauci, Donald Trump, Boris Johnson) alongside private social media users.
3. **No Direct Mapping for `not harmful`**:
   Like Multi3Hate, a meme that is `not harmful` in a COVID-19 context cannot automatically be assumed to be universally `SAFE` across all dimensions without verification.
4. **Controlled Conversion Required**:
   ContextShield will define an explicit, controlled mapping policy for HarMeme rather than automatically importing unverified samples.

---

## 7. Citations

```bibtex
@inproceedings{dimitrov-etal-2021-detecting,
    title = "Detecting Harmful Memes and Their Targets",
    author = "Dimitrov, Dimitar  and
      Bin Ali, Bishr  and
      Pramanick, Shraman  and
      Silvestri, Fabrizio  and
      Nakov, Preslav  and
      Da San Martino, Giovanni",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    year = "2021",
    pages = "2783--2796",
    doi = "10.18653/v1/2021.findings-acl.246"
}

@inproceedings{pramanick-etal-2021-momenta,
    title = "{MOMENTA}: A Multimodal Framework for Detecting Harmful Memes and Their Targets",
    author = "Pramanick, Shraman  and
      Dimitrov, Dimitar  and
      Mukherjee, Rituparna  and
      Sharma, Shivam  and
      Akhtar, Md. Shad  and
      Nakov, Preslav  and
      Chakraborty, Tanmoy",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    pages = "4439--4455",
    doi = "10.18653/v1/2021.findings-emnlp.379"
}
```
