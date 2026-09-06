import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("contextshield.ocr")

# Lazy singleton for PaddleOCR engine
_ocr_engine = None


def get_ocr_engine():
    """Lazily initializes and returns the PaddleOCR engine singleton."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR

            # Initialize PaddleOCR engine
            _ocr_engine = PaddleOCR()
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR engine: {e}")
            raise RuntimeError(f"OCR engine initialization failed: {e}")
    return _ocr_engine


def extract_text_from_image(image_path: Union[Path, str]) -> str:
    """
    Extracts all detected text from the provided image file using PaddleOCR.

    Returns:
        A single concatenated string of all detected text, or empty string ("") if no text is found.
    """
    path_str = str(image_path)
    engine = get_ocr_engine()

    try:
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
