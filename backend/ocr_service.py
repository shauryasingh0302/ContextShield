import logging
import os
from pathlib import Path
from typing import Optional, Union

# Disable redundant network connectivity checks to model mirrors for fast startup
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

logger = logging.getLogger("contextshield.ocr")

# Lazy singleton for PaddleOCR engine
_ocr_engine = None


def get_ocr_engine():
    """Lazily initializes and returns the optimized PaddleOCR engine singleton."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR

            # Initialize lightweight PaddleOCR engine (disabling unnecessary doc-layout pipelines)
            _ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_detection_model_name="PP-OCRv4_mobile_det",
                text_recognition_model_name="PP-OCRv4_mobile_rec",
                lang="en",
            )
            logger.info("PaddleOCR engine initialized successfully with mobile models.")
        except Exception as e:
            logger.warning(f"Initial mobile PaddleOCR setup failed ({e}), trying standard mobile fallback...")
            try:
                from paddleocr import PaddleOCR
                _ocr_engine = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception as e2:
                logger.error(f"Failed to initialize PaddleOCR engine: {e2}")
                _ocr_engine = None
    return _ocr_engine


def extract_text_from_image(image_path: Union[Path, str]) -> str:
    """
    Extracts all detected text from the provided image file using PaddleOCR.

    Returns:
        A single concatenated string of all detected text, or empty string ("") if no text is found.
    """
    path_str = str(image_path)
    try:
        engine = get_ocr_engine()
        if engine is None:
            logger.warning("OCR engine unavailable, returning empty string.")
            return ""

        # Use predict method (standard in PaddleOCR 3.x / PaddleX)
        if hasattr(engine, "predict"):
            raw_results = list(engine.predict(path_str))
        else:
            raw_results = engine.ocr(path_str)
    except Exception as e:
        logger.error(f"Error during OCR extraction on {path_str}: {e}")
        return ""

    if not raw_results:
        return ""

    extracted_lines = []

    for item in raw_results:
        if not item:
            continue

        # PaddleOCR 3.x / PaddleX dict output format
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts", [])
            for text in rec_texts:
                if text and isinstance(text, str):
                    cleaned = text.strip()
                    if cleaned:
                        extracted_lines.append(cleaned)

        # Legacy PaddleOCR list output format: [[bbox, (text, score)]]
        elif isinstance(item, list):
            for sub_item in item:
                if isinstance(sub_item, (list, tuple)) and len(sub_item) >= 2:
                    text_info = sub_item[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                        text = text_info[0]
                        if text and isinstance(text, str):
                            cleaned = text.strip()
                            if cleaned:
                                extracted_lines.append(cleaned)

    return " ".join(extracted_lines).strip()
