# ContextShield

Multimodal Social Media Safety & Harassment Detection System

---

## 1. Project Overview

ContextShield is an end-to-end multimodal safety classification assistant designed to analyze social-media posts—specifically combining image or meme content with an accompanying caption—before publication.

### The Problem
Modern online harm rarely occurs in a single modality:
- An innocent image paired with a hostile caption can constitute targeted harassment.
- Sarcastic or hateful memes often contain benign text whose toxic meaning only becomes apparent in the visual context of the image.
- Text-only moderation models fail on memes because the primary abusive payload is embedded in the image pixels (visual symbols) or in text rendered on the image (OCR).

### The Multimodal Approach
ContextShield evaluates the joint context of both modalities:
1. **Extracted Optical Text**: Optical Character Recognition (OCR) recovers text embedded within the meme image.
2. **Multilingual Semantic Embeddings**: Text representations are computed for both the user caption and the extracted OCR text using a frozen multilingual Transformer.
3. **Visual Embeddings**: Deep visual representations are computed using a frozen vision Transformer.
4. **Cross-Modal Fusion & Classification**: Feature projections and cross-modal attention align visual and textual features into a joint representation for classification.

### User Interaction & System Output
- **User Inputs**: An uploaded image (`.jpg`, `.jpeg`, `.png`, `.webp` up to 10 MB) and an optional caption string.
- **System Returns**:
  - Primary predicted risk label (`SAFE`, `OFFENSIVE`, `HATE`, or `HARASSMENT`)
  - Normalized risk score ($0$ to $100$)
  - Model confidence ($0.00$ to $1.00$)
  - Full 4-class probability distribution
  - Extracted OCR text
  - Deterministic, rule-based explanation of the verdict
  - Safer caption alternative or a clean "no changes needed" confirmation

---

## 2. Key Features

- **Multimodal File Upload**: Supports standard web image formats (JPG, JPEG, PNG, WEBP) up to 10 MB with client and server validation.
- **Social Media Post Composer**: Clean caption input with live character counting.
- **On-Device OCR**: Local text extraction powered by PaddleOCR (`PP-OCRv6`), operating without external cloud APIs.
- **Multilingual Text Embeddings**: 768-dimensional contextual sentence embeddings generated via `xlm-roberta-base`.
- **Vision Embeddings**: 512-dimensional visual feature vectors generated via OpenAI CLIP (`clip-vit-base-patch32`).
- **Gated Cross-Modal Fusion**: 256-dimensional joint multimodal representation aligning text, OCR, and visual features.
- **4-Class Risk Classification**: Dedicated classification head producing logits across four mutually exclusive categories: `SAFE`, `OFFENSIVE`, `HATE`, and `HARASSMENT`.
- **Calibrated Risk Score**: A bounded $0–100$ severity metric derived from the class probability distribution.
- **Full Probability Breakdown**: Granular Softmax probability scores for each of the four target classes.
- **Deterministic Rule-Based Explanation**: Truthful, concise explanations that reflect model confidence and only cite specific terms if actually detected.
- **Safer Caption Suggestion**: Recommends neutralized caption alternatives or constructive rewriting suggestions without overwriting the user's original text.
- **Built-in Dataset Annotator**: Separate ground-truth labeling tool (`/annotate`) for collecting, categorizing, and severity-tagging new samples into `dataset/raw/annotations.jsonl`.
- **Monochrome Minimalist UI**: Single-screen, no-scroll desktop workstation layout designed with a strict black/white/grayscale palette.

---

## 3. System Architecture

```text
                           ┌───────────────────────────┐
                           │   User Uploaded Post      │
                           │     (Image + Caption)     │
                           └─────────────┬─────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [ Image File ]                  [ Caption Text ]
                         │                               │
           ┌─────────────┴─────────────┐                 │
           ▼                           ▼                 ▼
   ┌───────────────┐           ┌───────────────┐ ┌───────────────┐
   │ PaddleOCR v6  │           │ CLIP ViT-B/32 │ │  XLM-RoBERTa  │
   │ Text Extract  │           │   (Frozen)    │ │   (Frozen)    │
   └───────┬───────┘           └───────┬───────┘ └───────┬───────┘
           │ (Extracted Text)          │                 │
           ▼                           │                 │
   ┌───────────────┐                   │                 │
   │  XLM-RoBERTa  │                   │                 │
   │   (Frozen)    │                   │                 │
   └───────┬───────┘                   │                 │
           │                           │                 │
           ▼                           ▼                 ▼
     OCR Embedding              Image Embedding   Caption Embedding
        (768d)                      (512d)             (768d)
           │                           │                 │
           └───────────────────┬───────┴─────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │   Multimodal Fusion Network         │
            │   - Linear Projections (-> 256d)    │
            │   - Gated Cross-Modal Attention     │
            │   - Fused Joint Representation      │
            │     (256-dimensional)               │
            └──────────────────┬──────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │   4-Class Risk Classifier Head      │
            │   Linear(256, 128) -> ReLU ->       │
            │   Dropout(0.2) -> Linear(128, 4)    │
            └──────────────────┬──────────────────┘
                               ▼
                   Raw Class Logits (4d)
                               │
                               ▼
            ┌─────────────────────────────────────┐
            │ Softmax Normalization               │
            │ - Probabilities (SAFE/OFF/HATE/HAR) │
            │ - Primary Risk Label                │
            │ - Risk Score (0-100) & Confidence   │
            └──────────────────┬──────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │ Explanation & Suggestion Service    │
            │ (Deterministic Rule-Based System)   │
            └──────────────────┬──────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │ Next.js Frontend No-Scroll Display  │
            └─────────────────────────────────────┘
```

### Component Dimensions
- **Caption Embedding**: $768$-dimensional vector (`xlm-roberta-base`, mean-pooled over hidden states).
- **OCR Text Embedding**: $768$-dimensional vector (`xlm-roberta-base`, mean-pooled over hidden states).
- **Image Embedding**: $512$-dimensional vector (`openai/clip-vit-base-patch32`, visual projection).
- **Projected Modalities**: Each mapped to $256$ dimensions via dedicated `nn.Linear` layers.
- **Joint Fusion Representation**: $256$-dimensional vector combining gated cross-modal attention with residual projection.
- **Classification Head**: $256 \to 128 \to 4$ output logits.

---

## 4. Technology Stack

| Component | Technology | Version / Specific Identifier | Purpose |
|---|---|---|---|
| **Backend API** | FastAPI | `0.115.0+` | REST API serving `/analyze`, `/health`, `/annotate` |
| **ASGI Server** | Uvicorn | `0.30.0+` | Asynchronous backend server |
| **Frontend Framework** | Next.js | `16.3.4` (React `19.2.8`) | Client interface & App Router architecture |
| **Frontend Styling** | Tailwind CSS | `v4` (`@tailwindcss/postcss`) | Minimalist monochrome responsive design |
| **Deep Learning Framework** | PyTorch | `2.6.0+` | Neural network modeling, training, and inference |
| **NLP Transformer** | Hugging Face Transformers | `xlm-roberta-base` | Multilingual text representations (768d) |
| **Vision Transformer** | Hugging Face Transformers | `openai/clip-vit-base-patch32` | Visual meme feature representations (512d) |
| **OCR Engine** | PaddleOCR / PaddlePaddle | `PP-OCRv6` | Optical character recognition on meme images |
| **Data Storage** | Flat Files / CSV / JSONL | UTF-8 encoded files | Dataset storage (`dataset/final/*.csv`, `dataset/raw/annotations.jsonl`) |
| **Python Runtime** | CPython | `3.10`–`3.12` | Backend execution environment |
| **Node.js Runtime** | Node.js / npm | Node `v18+`, npm `v9+` | Frontend execution environment |

> **Database Note**: ContextShield does not require or use PostgreSQL, MySQL, or MongoDB. Dataset annotations and metadata are persisted directly in standard JSON Lines (`dataset/raw/annotations.jsonl`) and CSV files.

---

## 5. Model Architecture

### Backbones (Frozen)
To preserve generalization across diverse linguistic and visual inputs while enabling training on modest compute, the feature backbones are frozen:
- **`xlm-roberta-base`**: Frozen. Evaluated with `torch.no_grad()`. Parameters: $\approx 125\text{M}$ (non-trainable during task training).
- **`openai/clip-vit-base-patch32`**: Frozen. Evaluated with `torch.no_grad()`. Parameters: $\approx 88\text{M}$ (non-trainable during task training).

### Trainable Layers (`ContextShieldModel`)
Only the feature projection, cross-modal attention alignment, and classification layers are trainable:

1. **Projection Layers** (`MultimodalFusionModel`):
   - `proj_caption`: `Linear(768, 256)` + `LayerNorm(256)` + `Dropout(0.1)`
   - `proj_ocr`: `Linear(768, 256)` + `LayerNorm(256)` + `Dropout(0.1)`
   - `proj_image`: `Linear(512, 256)` + `LayerNorm(256)` + `Dropout(0.1)`
2. **Cross-Modal Attention Layer**:
   - Multi-head attention (`embed_dim=256, num_heads=4`) computing joint interactions across `[caption_token, ocr_token, image_token]`.
   - Feed-Forward network (`Linear(256, 512)` $\to$ `GELU` $\to$ `Linear(512, 256)`).
3. **Risk Classifier Head** (`RiskClassifier`):
   - `Linear(256, 128)`
   - `ReLU()`
   - `Dropout(p=0.2)`
   - `Linear(128, 4)`

### Verified Parameter Counts
- **Fusion Trainable Parameters**: `721,920`
- **Classifier Trainable Parameters**: `33,412`
- **Total Trainable Parameters**: `755,332`

---

## 6. Risk Categories

ContextShield defines four mutually exclusive classification categories:

1. **`SAFE`**
   - Content that does not exhibit abusive, derogatory, hateful, or harassing patterns.
   - Includes benign humor, general social commentary, political satire that does not cross into protected-class defamation, and neutral social posts.
2. **`OFFENSIVE`**
   - Content that contains vulgarity, profanity, coarse insults, or generally offensive imagery.
   - Differs from Hate Speech in that it is not targeted at protected characteristics (e.g. race, religion, gender), and differs from Harassment in that it is not directed at a private individual.
3. **`HATE`**
   - Content directed against protected demographic groups or identity attributes (race, ethnicity, religion, nationality, sexual orientation, disability).
   - Characterized by dehumanization, incitement to discrimination or violence, and identity-targeted slurs.
4. **`HARASSMENT`**
   - Content directed at a specific individual or private person with the intent to demean, humiliate, intimidate, or bully.
   - Focuses on personal attacks, non-public figures, or persistent targeted degradation.

---

## 7. Dataset

The ContextShield unified dataset contains **2,192 verified multimodal samples** assembled from three established academic research datasets:

### Class Distribution (Total: 2,192 samples)
- **`HATE`**: 870 samples ($39.69\%$)
- **`HARASSMENT`**: 582 samples ($26.55\%$)
- **`SAFE`**: 440 samples ($20.07\%$)
- **`OFFENSIVE`**: 300 samples ($13.69\%$)

### Split Breakdown
The dataset preserves the integrity of benchmark splits:
- **Train Split** (`dataset/final/train.csv`): 1,466 samples
- **Validation Split** (`dataset/final/validation.csv`): 328 samples
- **Test Split** (`dataset/final/test.csv`): 398 samples (untouched during training)

### Source Datasets & Label Mappings

| Source Dataset | Original Labels | ContextShield Label Mapping | Language | Samples Converted |
|---|---|---|---|---|
| **MultiOFF** | `Non-offensive`<br>`offensive` | `Non-offensive` $\to$ `SAFE`<br>`offensive` $\to$ `OFFENSIVE` | English | 740 total (440 SAFE, 300 OFFENSIVE) |
| **Multi3Hate** | Cultural consensus hate annotations ($0.0$ to $1.0$) | Strict consensus score $= 1.0$ $\to$ `HATE`<br>(Ambiguous samples discarded) | English & Hindi | 870 total (all HATE) |
| **HarMeme** | Target: `individual`, `organization`, `community`<br>Harmfulness: `not harmful`, `somewhat harmful`, `very harmful` | Target $=$ `individual` AND harmfulness $\in$ {`somewhat harmful`, `very harmful`} $\to$ `HARASSMENT` | English | 582 total (all HARASSMENT) |

### Dataset Limitations & Licensing Restrictions
- **Multi3Hate**: Distributed under **CC BY-NC-ND 4.0** (Attribution-NonCommercial-NoDerivatives 4.0 International). It cannot be used for commercial purposes.
- **HarMeme**: Released for academic research and non-commercial study.
- **MultiOFF**: Released for academic research under open attribution terms.
- **Ownership**: ContextShield does not claim ownership of the original external datasets. All source data rights belong to their respective creators and institutions.

---

## 8. Training

Training is executed via `backend/train.py`.

### Configuration
- **Backbone Freezing**: Both `xlm-roberta-base` and `clip-vit-base-patch32` remain strictly frozen.
- **Trainable Components**: Multimodal projection layers, cross-modal attention, and the 4-class classification head.
- **Optimizer**: `AdamW`
  - Learning Rate: `1e-4`
  - Weight Decay: `0.01`
  - Betas: `(0.9, 0.999)`
- **Loss Function**: Class-weighted `CrossEntropyLoss` to handle class imbalance:
  $$\text{weight}_c = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_c}$$
  - Class 0 (`SAFE`): $2.0824$
  - Class 1 (`OFFENSIVE`): $3.0542$
  - Class 2 (`HATE`): $1.0516$
  - Class 3 (`HARASSMENT`): $1.5730$
- **Epochs**: `10`
- **Batch Size**: `16`
- **Checkpoint Selection**: The best checkpoint was selected based strictly on validation Macro-F1. The test set was never accessed during training or checkpoint selection.
- **Checkpoint Location**: `checkpoints/best_model.pt` (Saved at Epoch 10 with Val Macro-F1: $0.7594$, Val Loss: $0.6976$).

---

## 9. Evaluation Results

Evaluated on the held-out **398 test samples** (`dataset/final/test.csv`) using checkpoint `checkpoints/best_model.pt`. Results are recorded in `results/test_evaluation.json`:

### Overall Test Performance
- **Test Accuracy**: **81.91%**
- **Macro Precision**: **78.56%**
- **Macro Recall**: **77.86%**
- **Macro F1-Score**: **76.46%**
- **Average Test Loss**: $0.7303$

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support (Samples) |
|---|---|---|---|---|
| **SAFE** | 0.7778 | 0.4615 | 0.5793 | 91 |
| **OFFENSIVE** | 0.4444 | 0.7018 | 0.5442 | 57 |
| **HATE** | 1.0000 | 0.9847 | 0.9923 | 131 |
| **HARASSMENT** | 0.9200 | 0.9664 | 0.9426 | 119 |

### 4×4 Confusion Matrix

```text
                  Predicted SAFE  Predicted OFFENSIVE  Predicted HATE  Predicted HARASSMENT
Actual SAFE                   42                   44               0                     5
Actual OFFENSIVE              12                   40               0                     5
Actual HATE                    0                    2             129                     0
Actual HARASSMENT              0                    4               0                   115
```

### Analysis of Results
- **Hate Speech & Harassment Detection**: Extremely strong performance ($F_1 = 0.9923$ for `HATE`, $F_1 = 0.9426$ for `HARASSMENT`), demonstrating the effectiveness of joint CLIP visual features and XLM-R text embeddings on explicit harms.
- **Safe vs. Offensive Confusion**: Moderate confusion exists between `SAFE` (44 misclassified as `OFFENSIVE`) and `OFFENSIVE` (12 misclassified as `SAFE`). This reflects subjective boundary differences inherent in the original MultiOFF meme dataset.

---

## 10. API Documentation

The backend service runs on FastAPI at `http://127.0.0.1:8000`.

### Endpoints

#### 1. `GET /health`
Verifies backend service availability.
- **Response**: `{"status": "ok"}`

#### 2. `POST /analyze`
Primary endpoint for multimodal post safety analysis.
- **Request Format**: `multipart/form-data`
- **Parameters**:
  - `image`: Binary file (`.jpg`, `.jpeg`, `.png`, `.webp`, $\le 10\text{ MB}$)
  - `caption`: String (optional accompanying text)

##### Example Request (cURL)
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
     -F "image=@sample_meme.png" \
     -F "caption=When the deployment passes on the first attempt"
```

##### Example Response Body
```json
{
  "risk_label": "SAFE",
  "risk_score": 34.61,
  "confidence": 0.6539,
  "probabilities": {
    "SAFE": 0.6539,
    "OFFENSIVE": 0.3407,
    "HATE": 0.0021,
    "HARASSMENT": 0.0034
  },
  "detected_text": "DEPLOYMENT SUCCESSFUL",
  "explanation": "No significant risk category was detected in the post. The content appears safe for general audiences.",
  "suggestion": null,
  "success": true,
  "status": "success",
  "message": "Post analyzed. Classified as SAFE with 65.4% confidence.",
  "caption": "When the deployment passes on the first attempt",
  "ocr_text": "DEPLOYMENT SUCCESSFUL",
  "text_analysis": {
    "caption_embedding_size": 768,
    "ocr_embedding_size": 768
  },
  "image_analysis": {
    "embedding_size": 512
  },
  "multimodal_analysis": {
    "caption_features": 256,
    "ocr_features": 256,
    "image_features": 256,
    "fusion_features": 256
  }
}
```

#### 3. `GET /annotations`
Returns current annotation counts and recent records from `dataset/raw/annotations.jsonl`.

#### 4. `POST /annotate`
Submits a newly labeled sample into `dataset/images/` and appends to `dataset/raw/annotations.jsonl`.
- **Form Fields**: `image`, `caption`, `label` (`SAFE|OFFENSIVE|HATE|HARASSMENT`), `severity` (`0|1|2|3`), `language` (`ENGLISH|HINDI|HINGLISH|OTHER`).

---

## 11. Frontend

The web application is built with Next.js 16 and styled with Tailwind CSS v4 using a **strict monochrome aesthetic** (black, white, grayscale) and a **no-scroll workstation layout**:

### User Workflow
1. **Upload Media**: Drag and drop an image or click the upload zone. Image preview renders instantly with file metadata.
2. **Compose Caption**: Enter post text into the integrated composer textarea.
3. **Analyze**: Click *"Analyze Post Safety"*. The interface enters a minimal loading state.
4. **Examine Results**:
   - **Risk Classification**: Prominently displays the verdict (`SAFE`, `OFFENSIVE`, `HATE SPEECH`, `HARASSMENT`) using typographic hierarchy and grayscale contrast.
   - **Metrics Row**: Side-by-side Risk Score ($0–100\%$) and Model Confidence ($0–100\%$).
   - **Probability Breakdown**: Horizontal grayscale bars showing exact percentages across all 4 categories.
   - **Extracted OCR Text**: Monospace block displaying text extracted by PaddleOCR.
   - **Model Explanation**: Grounded reasoning explaining why the post was classified as such.
   - **Safer Suggestion**: Displays a constructive alternative with a one-click `"Copy"` button, or a clean `[✓] No changes needed` notice.
5. **Reset**: The `"Analyze Another"` action clears the state and readies the composer.

---

## 12. Project Structure

```text
ContextShield/
├── backend/
│   ├── main.py                     # FastAPI entry point & HTTP endpoints
│   ├── classifier_model.py         # 4-class RiskClassifier, ContextShieldModel, singleton loader
│   ├── fusion_model.py             # MultimodalFusionModel (projections + cross-modal attention)
│   ├── text_service.py             # XLM-RoBERTa multilingual text embedding service (768d)
│   ├── image_service.py            # CLIP vision feature extraction service (512d)
│   ├── ocr_service.py              # PaddleOCR text extraction wrapper
│   ├── explanation_service.py      # Rule-based explanations and safer caption suggestions
│   ├── dataset.py                  # PyTorch Dataset loader with embedding caching
│   ├── train.py                    # Training script with class weighting & checkpoint saving
│   ├── evaluate.py                 # Standalone evaluation & confusion matrix computation
│   ├── test_analyze_endpoint.py    # TestClient integration test for POST /analyze
│   ├── test_explanation_and_endpoint.py # Unit tests for explanation service
│   └── requirements.txt            # Python dependencies
├── checkpoints/
│   └── best_model.pt               # Trained model checkpoint (8.8 MB)
├── dataset/
│   ├── final/                      # Unified ContextShield dataset (2,192 samples)
│   │   ├── train.csv               # 1,466 training rows
│   │   ├── validation.csv          # 328 validation rows
│   │   ├── test.csv                # 398 test rows
│   │   └── images/                 # 2,192 image files
│   ├── processed/                  # Intermediate converted splits (MultiOFF, Multi3Hate, HarMeme)
│   ├── raw/                        # Annotator storage (annotations.jsonl)
│   └── external/                   # Preserved original external datasets
├── frontend/                       # Next.js 16 (React 19 + Tailwind CSS) client
│   ├── src/app/
│   │   ├── page.tsx                # Main no-scroll monochrome safety checker interface
│   │   ├── annotate/page.tsx       # No-scroll monochrome dataset annotator interface
│   │   ├── layout.tsx              # Root HTML wrapper with Geist typography
│   │   └── globals.css             # Tailwind CSS imports
│   ├── package.json                # Frontend dependencies and scripts
│   └── tsconfig.json               # TypeScript configuration
├── results/
│   ├── test_evaluation.json        # Genuine test metrics & confusion matrix
│   └── evaluation_results.json     # Validation / intermediate metrics
├── scripts/
│   ├── build_final_dataset.py      # Unified dataset builder from processed sources
│   ├── convert_multioff.py         # MultiOFF conversion pipeline
│   ├── convert_multi3hate.py       # Multi3Hate conversion pipeline
│   ├── convert_harmeme.py          # HarMeme conversion pipeline
│   ├── verify_system_integrity.py  # Complete 4-section verification suite
│   └── verify_frontend_api.py      # Live end-to-end API verification
└── README.md
```

---

## 13. Installation

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher
- **Git**: 2.30.0 or higher

### Step 1: Clone Repository
```bash
git clone https://github.com/shauryasingh0302/ContextShield.git
cd ContextShield
```

### Step 2: Python Environment & Dependencies
Create and activate a virtual environment:
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install backend dependencies:
```bash
pip install -r backend/requirements.txt
```

### Step 3: Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 14. Running the Project

### Running the Backend (FastAPI)
From the project root:
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Health Endpoint: `http://127.0.0.1:8000/health`

### Running the Frontend (Next.js Development)
In a separate terminal:
```bash
cd frontend
npm run dev
```
- Web Interface: `http://localhost:3000`
- Annotator: `http://localhost:3000/annotate`

### Production Frontend Build
```bash
cd frontend
npm run build
npm run start
```

---

## 15. Testing & Verification

ContextShield includes automated test and verification scripts:

### 1. Complete System Integrity Verification
Verifies backend status, input edge cases (oversized files, invalid extensions), ML parameter boundaries, dataset sample counts, and security isolations:
```bash
python scripts/verify_system_integrity.py
```

### 2. Explanation Service & Endpoint Test
Verifies rule-based explanation generation, safer caption suggestions, and absence of hallucinated terms:
```bash
python backend/test_explanation_and_endpoint.py
```

### 3. Analyze Endpoint Integration Test
Directly tests `POST /analyze` via Starlette TestClient with real images:
```bash
python backend/test_analyze_endpoint.py
```

### 4. Standalone Test Set Evaluation
Re-evaluates the saved checkpoint on the held-out test set:
```bash
python backend/evaluate.py --dataset_path dataset/final/test.csv --image_dir dataset/final/images --checkpoint_path checkpoints/best_model.pt
```

### 5. Live End-to-End API Test
Tests HTTP communication against a running backend instance:
```bash
python scripts/verify_frontend_api.py
```

---

## 16. Security & Validation

The application incorporates standard web and API safety controls:
- **Allowed Extensions**: Enforces strict extension checking (`.jpg`, `.jpeg`, `.png`, `.webp`). Executable (`.exe`), script (`.sh`), and generic binary formats are rejected with `400 Bad Request`.
- **MIME Type Validation**: Rejects mismatched headers (e.g. `text/plain` disguised with a `.png` extension).
- **Upload Size Limit**: File streaming buffers reject uploads exceeding `10 MB` ($10 \times 1024 \times 1024$ bytes) with `400 Bad Request`.
- **Path Traversal Protection**: Upload filenames are sanitized using `Path(filename).name` and prepended with random UUIDs (`uuid.uuid4().hex`), ensuring files can only be written to `backend/uploads/`.
- **Secret Isolation**: No hardcoded API keys or sensitive credentials exist in the codebase.
- **Frontend Isolation**: Model checkpoints (`checkpoints/`) and datasets (`dataset/`) reside strictly outside `frontend/public/` and cannot be downloaded directly by clients.

> **Security Note**: This system is built as a student academic prototype and is not hardened for multi-tenant production hosting.

---

## 17. Limitations

ContextShield is an academic research prototype with documented limitations:

1. **Frozen Backbones**: XLM-RoBERTa and CLIP are kept frozen due to compute constraints. Fine-tuning the backbone layers might yield higher multimodal alignment at the expense of training complexity.
2. **Dataset Scale**: The unified dataset comprises 2,192 samples. While sufficient for training the projection and classification heads, larger datasets would improve edge-case generalization.
3. **SAFE vs. OFFENSIVE Boundary**: Subjectivity in public benchmark datasets (notably MultiOFF) leads to classification overlap between mild political satire and offensive language.
4. **Rule-Based Explanation System**: The explanation and suggestion module is deterministic and keyword/rule-assisted. It does not possess generative linguistic rewriting capabilities of large language models.
5. **OCR Dependence**: Memes with low-contrast, highly stylized, or degraded typography may yield incomplete OCR text, requiring the model to rely solely on image and caption signals.
6. **Non-Production Prototype**: This system is designed for research and study, and should not be used as an autonomous, unmonitored production content moderation system.

---

## 18. Future Improvements

Future iterations of ContextShield could explore:
- **Backbone Fine-Tuning**: Unfreezing the top 2 layers of CLIP and XLM-RoBERTa with low-rank adaptation (LoRA).
- **Expanded Indian Language Support**: Incorporating regional Indian language memes (Bengali, Tamil, Marathi) beyond English and Hindi/Hinglish.
- **Improved Safe/Offensive Calibration**: Retraining on larger non-offensive humor corpora to improve discrimination between mild satire and offensive remarks.
- **Generative Neutralization**: Integrating compact on-device instruction-tuned language models (e.g. Gemma 2B) for fluent caption paraphrasing.
- **Video & Reel Analysis**: Extending the temporal dimension to analyze short-form videos (TikTok/Instagram Reels).

---

## 19. Reproducibility

To reproduce the complete pipeline from scratch:

```bash
# 1. Convert external sources to processed splits
python scripts/convert_multioff.py
python scripts/convert_multi3hate.py
python scripts/convert_harmeme.py

# 2. Build the final unified dataset (2,192 samples)
python scripts/build_final_dataset.py

# 3. Train the model (10 epochs, AdamW lr=1e-4)
python backend/train.py --epochs 10 --batch_size 16 --lr 1e-4

# 4. Evaluate on the untouched test set
python backend/evaluate.py --dataset_path dataset/final/test.csv --image_dir dataset/final/images --checkpoint_path checkpoints/best_model.pt --output_path results/test_evaluation.json
```

All split boundaries and random seeds (`42`) are fixed in code.

---

## 20. Credits & Dataset Attribution

ContextShield is developed as an academic engineering project utilizing resources from the following research teams:

- **MultiOFF**: Suryawanshi et al., *"Multimodal Offensive Language Detection in Memes"*, ComMA 2020.
- **Multi3Hate**: Bui et al., *"Multi3Hate: A Multilingual and Multimodal Dataset for Hate Speech Detection"*, Hugging Face (`MinhDucBui/Multi3Hate`).
- **HarMeme**: Dimitrov et al., *"Detecting Harmful Memes with Multimodal Feature Fusion"*, Findings of ACL 2021 (`di-dimitrov/harmeme`).
- **PaddleOCR**: PaddlePaddle Authors, *"PP-OCR: A Practical Ultra Lightweight OCR System"*.
- **CLIP**: Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"*, OpenAI, ICML 2021.
- **XLM-RoBERTa**: Conneau et al., *"Unsupervised Cross-lingual Representation Learning at Scale"*, ACL 2020.

---

## 21. License

- **ContextShield Codebase**: This project is developed for educational and academic research evaluation. If you use or build upon this work, please attribute ContextShield.
- **Dataset Licenses**:
  - The Multi3Hate dataset is licensed under **CC BY-NC-ND 4.0**.
  - HarMeme is released for academic, non-commercial research use.
  - MultiOFF is provided for research under academic attribution terms.
  - ContextShield does not grant commercial rights to external dataset components.

---

## 22. Final Project Status

ContextShield has been built, trained, evaluated, and verified:
- **Backend API**: Functional and verified across all endpoints (`/health`, `/analyze`, `/annotate`).
- **Trained Model**: Checkpoint [`checkpoints/best_model.pt`](checkpoints/best_model.pt) operational.
- **Verified Metrics**: **81.91% Test Accuracy**, **76.46% Macro-F1** on held-out 398-sample test split.
- **Frontend Client**: Minimalist monochrome, no-scroll interface verified with zero build errors.
