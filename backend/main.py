import json
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from ocr_service import extract_text_from_image
from text_service import generate_embedding
from image_service import generate_image_embedding
from fusion_model import fuse_multimodal_features
from classifier_model import predict_multimodal_risk
from explanation_service import generate_explanation, generate_suggestion

app = FastAPI(
    title="ContextShield API",
    description="Backend API for ContextShield - Social Media Post Safety Checker",
    version="0.1.0",
)

# Enable CORS so frontend (localhost or Render) can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://contextshield-frontend.onrender.com",
    ],
    allow_origin_regex=r"https?://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload configuration
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
DATASET_IMAGES_DIR = DATASET_DIR / "images"
DATASET_RAW_DIR = DATASET_DIR / "raw"
DATASET_RAW_FILE = DATASET_RAW_DIR / "annotations.jsonl"
DATASET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DATASET_RAW_DIR.mkdir(parents=True, exist_ok=True)

VALID_ANNOTATION_LABELS = {"SAFE", "OFFENSIVE", "HATE", "HARASSMENT"}
VALID_ANNOTATION_SEVERITIES = {0, 1, 2, 3}
VALID_ANNOTATION_LANGUAGES = {"ENGLISH", "HINDI", "HINGLISH", "OTHER"}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes


@app.get("/")
@app.head("/")
def root():
    """Root endpoint returning API status and links."""
    return {
        "status": "ok",
        "service": "contextshield-backend",
        "message": "ContextShield API is active and ready.",
        "endpoints": {
            "health": "/health",
            "cron_health": "/cron/health",
            "docs": "/docs",
            "analyze": "/analyze"
        }
    }


@app.get("/health")
@app.head("/health")
def health_check():
    """Basic health check endpoint returning system status."""
    return {"status": "ok", "service": "contextshield-backend"}


@app.get("/cron/health")
@app.head("/cron/health")
@app.get("/health/cron")
def cron_health_check():
    """
    Lightweight keep-alive and health check endpoint tailored for scheduled cron jobs,
    uptime monitors, and periodic pings to keep the free-tier service awake.
    """
    from datetime import datetime, timezone
    return {
        "status": "ok",
        "alive": True,
        "service": "contextshield-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "ContextShield backend is active and ready."
    }



@app.post("/analyze")
async def analyze_post(
    image: UploadFile = File(..., description="Uploaded image or meme (JPG, PNG, WEBP)"),
    caption: str = Form("", description="Accompanying caption for the post"),
):
    """
    Accepts an image and caption, validates file type and size (<= 10MB),
    and saves the file to local uploads/ directory.
    NOTE: ML/AI analysis is intentionally omitted for this step.
    """
    # 1. Validate filename and extension
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided for uploaded image.",
        )

    file_ext = Path(image.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension '{file_ext}'. Allowed formats: JPG, JPEG, PNG, WEBP.",
        )

    # 2. Validate MIME type
    if image.content_type and image.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MIME type '{image.content_type}'. Allowed types: image/jpeg, image/png, image/webp.",
        )

    # 3. Stream and save locally while validating file size limit (10 MB)
    unique_filename = f"{uuid.uuid4().hex}_{Path(image.filename).name}"
    save_path = UPLOAD_DIR / unique_filename

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1 MB chunks

    try:
        with open(save_path, "wb") as buffer:
            while True:
                chunk = await image.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE:
                    # Clean up partially written file
                    buffer.close()
                    if save_path.exists():
                        save_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File size exceeds the 10 MB limit.",
                    )

                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded image: {str(e)}",
        )

    # 4. Extract text from the saved image using PaddleOCR
    ocr_text = extract_text_from_image(save_path)

    # 5. Multilingual text analysis via XLM-RoBERTa
    caption_emb = generate_embedding(caption)
    ocr_emb = generate_embedding(ocr_text)
    text_analysis = {
        "caption_embedding_size": int(caption_emb.shape[0]),
        "ocr_embedding_size": int(ocr_emb.shape[0]),
    }

    # 6. Image feature extraction via CLIP
    image_emb = generate_image_embedding(save_path)
    image_analysis = {
        "embedding_size": int(image_emb.shape[0]),
    }

    # 7. Multimodal feature fusion
    multimodal_analysis = fuse_multimodal_features(
        caption_emb=caption_emb,
        ocr_emb=ocr_emb,
        image_emb=image_emb,
    )

    # 8. Trained Multimodal Risk Classification
    risk_prediction = predict_multimodal_risk(
        caption_emb=caption_emb,
        ocr_emb=ocr_emb,
        image_emb=image_emb,
    )

    # 9. Deterministic explanation and suggestion generation
    explanation = generate_explanation(
        risk_label=risk_prediction["risk_label"],
        confidence=risk_prediction["confidence"],
        caption=caption,
        ocr_text=ocr_text,
    )
    suggestion = generate_suggestion(
        risk_label=risk_prediction["risk_label"],
        caption=caption,
        ocr_text=ocr_text,
    )

    return {
        "risk_label": risk_prediction["risk_label"],
        "risk_score": risk_prediction["risk_score"],
        "confidence": risk_prediction["confidence"],
        "probabilities": risk_prediction["probabilities"],
        "detected_text": ocr_text,
        "explanation": explanation,
        "suggestion": suggestion,
        "success": True,
        "status": "success",
        "message": f"Post analyzed. Classified as {risk_prediction['risk_label']} with {risk_prediction['confidence'] * 100:.1f}% confidence.",
        "caption": caption.strip(),
        "ocr_text": ocr_text,
        "text_analysis": text_analysis,
        "image_analysis": image_analysis,
        "multimodal_analysis": multimodal_analysis,
        "data": {
            "saved_filename": unique_filename,
            "original_filename": image.filename,
            "content_type": image.content_type,
            "size_bytes": total_bytes,
            "caption": caption.strip(),
            "ocr_text": ocr_text,
            "risk_label": risk_prediction["risk_label"],
            "risk_score": risk_prediction["risk_score"],
            "confidence": risk_prediction["confidence"],
            "probabilities": risk_prediction["probabilities"],
            "explanation": explanation,
            "suggestion": suggestion,
            "text_analysis": text_analysis,
            "image_analysis": image_analysis,
            "multimodal_analysis": multimodal_analysis,
        },
    }


@app.get("/annotations")
def get_annotations():
    """Retrieve raw annotations count and list for the annotation UI."""
    if not DATASET_RAW_FILE.exists():
        return {"total": 0, "samples": []}

    samples = []
    with open(DATASET_RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    return {"total": len(samples), "samples": samples[-20:]}


@app.post("/annotate")
async def annotate_sample(
    image: UploadFile = File(..., description="Image or meme for annotation"),
    caption: str = Form(..., description="Post caption text"),
    label: str = Form(..., description="Risk category: SAFE, OFFENSIVE, HATE, HARASSMENT"),
    severity: int = Form(..., description="Severity level: 0, 1, 2, 3"),
    language: str = Form(..., description="Language: ENGLISH, HINDI, HINGLISH, OTHER"),
):
    """
    Saves an annotated dataset sample into dataset/images/ and dataset/raw/annotations.jsonl.
    """
    # 1. Validate label, severity, language
    norm_label = label.strip().upper()
    if norm_label not in VALID_ANNOTATION_LABELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid label '{label}'. Must be one of {sorted(list(VALID_ANNOTATION_LABELS))}",
        )

    if severity not in VALID_ANNOTATION_SEVERITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid severity '{severity}'. Must be 0 (None), 1 (Low), 2 (Moderate), or 3 (High).",
        )

    norm_language = language.strip().upper()
    if norm_language not in VALID_ANNOTATION_LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid language '{language}'. Must be one of {sorted(list(VALID_ANNOTATION_LANGUAGES))}",
        )

    # 2. Validate file format
    file_ext = Path(image.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS or image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image type. Only JPG, JPEG, PNG, and WEBP formats are allowed.",
        )

    # 3. Save image into dataset/images/
    sample_id = f"sample_{uuid.uuid4().hex[:8]}"
    clean_filename = Path(image.filename).name
    saved_image_name = f"{sample_id}_{clean_filename}"
    save_path = DATASET_IMAGES_DIR / saved_image_name

    total_bytes = 0
    chunk_size = 1024 * 1024
    try:
        with open(save_path, "wb") as buffer:
            while True:
                chunk = await image.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    buffer.close()
                    if save_path.exists():
                        save_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="File size exceeds the 10 MB limit.",
                    )
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save image: {str(e)}",
        )

    # 4. Record annotation in dataset/raw/annotations.jsonl
    record = {
        "id": sample_id,
        "image": saved_image_name,
        "caption": caption.strip(),
        "label": norm_label,
        "severity": severity,
        "language": norm_language,
    }

    with open(DATASET_RAW_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "success": True,
        "status": "success",
        "message": "Annotation saved successfully.",
        "id": sample_id,
        "sample": record,
    }

