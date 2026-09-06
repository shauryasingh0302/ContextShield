# ContextShield Dataset 🛡️

This directory houses the multimodal dataset for ContextShield, containing social media images/memes and captions annotated for risk classification and severity.

---

## Directory Structure

```text
dataset/
├── images/             # Uploaded image files
├── raw/                # Raw annotation records (annotations.jsonl)
├── train.csv           # 70% Training split
├── validation.csv      # 15% Validation split
├── test.csv            # 15% Testing split
└── README.md           # Dataset documentation and annotation guidelines
```

---

## Dataset Schema

Each entry in `train.csv`, `validation.csv`, and `test.csv` contains the following fields:

| Field | Type | Description | Allowed Values |
|---|---|---|---|
| `id` | String | Unique identifier for the sample | e.g. `sample_0001` or UUID |
| `image` | String | Filename of the image relative to `dataset/images/` | e.g. `img_001.jpg` |
| `caption` | String | Accompanying text or post caption | UTF-8 text |
| `label` | String | Risk category | `SAFE`, `OFFENSIVE`, `HATE`, `HARASSMENT` |
| `severity` | Integer | Severity level rating | `0` (None), `1` (Low), `2` (Moderate), `3` (High) |
| `language` | String | Language of caption & OCR text | `ENGLISH`, `HINDI`, `HINGLISH`, `OTHER` |

---

## Annotation Guidelines

### 1. Risk Labels
- **`SAFE`**: Benign posts, humor, everyday social media interactions with no targeted harm, hate speech, or abuse.
- **`OFFENSIVE`**: Posts containing profanity, vulgarity, or crude humor that is objectionable but does not cross into identity-based hate or targeted harassment.
- **`HATE`**: Posts promoting violence, hatred, dehumanization, or incitement against individuals or protected groups based on race, religion, caste, gender, sexual orientation, disability, or nationality.
- **`HARASSMENT`**: Targeted bullying, persistent abuse, threats, stalking behavior, or intimidation directed at a specific individual.

### 2. Severity Levels
- **`0 = None`**: No risk (typically for `SAFE` posts).
- **`1 = Low`**: Mild profanity, borderline sarcasm, or mild offensive phrasing without malicious intent.
- **`2 = Moderate`**: Explicit abusive slurs, clear hostility, or aggressive targeting.
- **`3 = High`**: Incitement to violence, direct physical threats, egregious hate speech, or severe targeted harassment.

### 3. Language Tags
- **`ENGLISH`**: English captions and text.
- **`HINDI`**: Devanagari Hindi text.
- **`HINGLISH`**: Romanized Hindi (Hindi words written in English alphabet), common in Indian social media memes.
- **`OTHER`**: Regional Indian languages or other multilingual mixtures.

---

## How to Annotate Samples

1. **Via the Web Interface**:
   - Start the backend: `uvicorn main:app --reload --port 8000` (in `backend/`)
   - Start the frontend: `npm run dev` (in `frontend/`)
   - Navigate to: **`http://localhost:3000/annotate`**
   - Upload the image, enter the caption, select label, severity, and language, then click **Save Annotation**.

2. **Generate Splits**:
   After annotating samples, run the preparation script:
   ```bash
   python scripts/prepare_dataset.py
   ```
   This will:
   - Validate each raw sample in `dataset/raw/annotations.jsonl`.
   - Ensure the image file exists in `dataset/images/`.
   - Randomly split the dataset into `train.csv` (70%), `validation.csv` (15%), and `test.csv` (15%).
