import logging
from typing import Any, Dict, Optional
import torch

logger = logging.getLogger("contextshield.text")

# Model identifier
MODEL_NAME = "xlm-roberta-base"

# Lazy singletons for tokenizer and model
_tokenizer = None
_model = None


def get_text_model_and_tokenizer():
    """
    Lazily loads and returns the XLM-RoBERTa tokenizer and model.
    Reuses the loaded model across requests.
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading pretrained multilingual model: {MODEL_NAME}")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModel.from_pretrained(MODEL_NAME)
            _model.eval()
            logger.info(f"Successfully loaded {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load multilingual text model {MODEL_NAME}: {e}")
            raise RuntimeError(f"Failed to initialize text analysis model: {e}")
    return _tokenizer, _model


def generate_embedding(text: Optional[str]) -> torch.Tensor:
    """
    Generates a 768-dimensional sentence embedding using XLM-RoBERTa
    via mean pooling over the token representations.
    
    Handles empty text gracefully.
    """
    tokenizer, model = get_text_model_and_tokenizer()

    clean_text = text.strip() if text else ""

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state  # [1, seq_len, 768]
        attention_mask = (
            inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        )

        sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze(0)  # [768]

    return embedding


def analyze_text(caption: str, ocr_text: str) -> Dict[str, Any]:
    """
    Processes both the post caption and OCR-detected text,
    generates their 768-dimensional multilingual embeddings,
    and returns analysis summary information.
    """
    caption_embedding = generate_embedding(caption)
    ocr_embedding = generate_embedding(ocr_text)

    return {
        "caption_embedding_size": int(caption_embedding.shape[0]),
        "ocr_embedding_size": int(ocr_embedding.shape[0]),
    }
