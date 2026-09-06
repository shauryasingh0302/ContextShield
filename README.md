# ContextShield 🛡️

ContextShield is an end-to-end multimodal social media safety assistant designed to analyze social-media posts (image/meme + caption) before publication, classify potential safety risks across 4 categories, and provide deterministic explanations and safer caption alternatives.

> **Project Level**: Academic Student Final Year Project.

---

## 1. System Architecture

```text
User Post (Image + Caption)
    │
    ├─► PaddleOCR (PP-OCRv6) ───────► Extracted Image Text
    │                                     │
    ├─► XLM-RoBERTa (Frozen) ◄────────────┘ (768d Caption + 768d OCR Embeddings)
    │
    ├─► CLIP ViT-B/32 (Frozen) ──────────► 512d Image Embedding
    │
    ▼
Multimodal Fusion Network (Trainable)
    Linear Projections (768d -> 256d, 512d -> 256d)
    + Gated Cross-Modal Alignment Layer (256d Joint Representation)
    │
    ▼
Risk Classifier Head (Trainable)
    Linear(256, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 4)
    │
    ▼
4-Class Output & Rule-Based Explanation Service
    ├── SAFE
    ├── OFFENSIVE
    ├── HATE SPEECH
    └── HARASSMENT
```

---

## 2. Directory Structure

```text
ContextShield/
├── backend/                        # FastAPI REST API & Model Serving
│   ├── main.py                     # API entry point (POST /analyze, GET /health)
│   ├── classifier_model.py         # 4-class classifier & checkpoint loader
│   ├── fusion_model.py             # 256d multimodal fusion architecture
│   ├── text_service.py             # XLM-RoBERTa multilingual text service
│   ├── image_service.py            # CLIP vision feature extraction service
│   ├── ocr_service.py              # PaddleOCR extraction service
│   ├── explanation_service.py      # Rule-based explanations and safer suggestions
│   ├── dataset.py                  # PyTorch multimodal dataset loader with caching
│   ├── train.py                    # PyTorch training pipeline with class weighting
│   ├── evaluate.py                 # Evaluation metrics & confusion matrix computation
│   └── requirements.txt            # Python dependencies
├── checkpoints/
│   └── best_model.pt               # Trained model weights (Epoch 10, Val Macro-F1: 0.7594)
├── dataset/
│   ├── final/                      # Final unified ContextShield dataset (2,192 samples)
│   │   ├── train.csv               # 1,466 training samples
│   │   ├── validation.csv          # 328 validation samples
│   │   ├── test.csv                # 398 untouched evaluation samples
│   │   └── images/                 # All 2,192 verified meme/post images
│   └── external/                   # Preserved original external datasets
│       ├── harmeme/                # HarMeme dataset (individual harassment samples)
│       ├── multi3hate/             # Multi3Hate dataset (consensus hate samples)
│       └── multioff/               # MultiOFF dataset (offensive / non-offensive samples)
├── frontend/                       # Next.js 16 (React 19 + Tailwind CSS) App
│   ├── src/app/
│   │   ├── page.tsx                # Main safety checker UI with probability breakdown
│   │   └── annotate/page.tsx       # Dataset annotation utility tool
│   └── package.json
├── results/
│   └── test_evaluation.json        # Test set evaluation results & confusion matrix
├── scripts/                        # Dataset conversion, inspection, & verification tools
└── README.md
```

---

## 3. Dataset Sources & Licensing Limitations

The unified ContextShield dataset contains **2,192 unique multimodal samples** compiled from three verified academic sources:

1. **MultiOFF** (`dataset/external/multioff`):
   - Offensiveness detection in multimodal memes.
   - Converted: `Non-offensive` → `SAFE`, `offensive` → `OFFENSIVE`.
   - License: Academic research / CC BY 4.0.
2. **Multi3Hate** (`dataset/external/multi3hate`):
   - Multilingual multimodal hate speech dataset across multiple cultural contexts.
   - Converted: Strict cultural consensus $1.0$ → `HATE`.
   - License: **CC BY-NC-ND 4.0** (Non-Commercial, No Derivatives). Must not be redistributed commercially.
3. **HarMeme** (`dataset/external/harmeme`):
   - COVID-19 and political memes targeting individuals and groups.
   - Converted: Memes targeting `individual` with `somewhat harmful` or `very harmful` severity → `HARASSMENT`.
   - License: Academic research / non-commercial use only.

---

## 4. Final Model Evaluation Metrics

Evaluated on the untouched **398 test samples** (`dataset/final/test.csv`) using checkpoint `checkpoints/best_model.pt`:

| Metric | Score |
|---|---|
| **Test Accuracy** | **81.91%** |
| **Macro Precision** | **78.56%** |
| **Macro Recall** | **77.86%** |
| **Macro F1** | **76.46%** |

### Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **SAFE** | 0.7778 | 0.4615 | 0.5793 | 91 |
| **OFFENSIVE** | 0.4444 | 0.7018 | 0.5442 | 57 |
| **HATE** | 1.0000 | 0.9847 | 0.9923 | 131 |
| **HARASSMENT** | 0.9200 | 0.9664 | 0.9426 | 119 |

### 4x4 Confusion Matrix

```text
                  Predicted SAFE  Predicted OFFENSIVE  Predicted HATE  Predicted HARASSMENT
Actual SAFE                   42                   44               0                     5
Actual OFFENSIVE              12                   40               0                     5
Actual HATE                    0                    2             129                     0
Actual HARASSMENT              0                    4               0                   115
```

---

## 5. Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm

### 1. Start the Backend API
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```
- API Health: `http://localhost:8000/health`
- Interactive Swagger Docs: `http://localhost:8000/docs`

### 2. Start the Frontend UI
```bash
cd frontend
npm install
npm run dev
```
- UI Interface: `http://localhost:3000`

---

## 6. Running Tests & Verifications

```bash
# 1. Run Complete System Integrity Verification (Backend, ML, Dataset, Security)
python scripts/verify_system_integrity.py

# 2. Run Explanation Service & Endpoint Test
python backend/test_explanation_and_endpoint.py

# 3. Run Live Endpoint Verification
python scripts/verify_frontend_api.py

# 4. Build Next.js Frontend
cd frontend
npm run build
```
