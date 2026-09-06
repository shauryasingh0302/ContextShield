import logging
from pathlib import Path
from typing import Any, Dict, Union
from PIL import Image
import torch

logger = logging.getLogger("contextshield.image")

# CLIP Model Identifier
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Lazy singletons
_clip_processor = None
_clip_model = None


def get_clip_model_and_processor():
    """
    Lazily loads and returns the pretrained CLIP processor and model.
    Reuses the loaded model across requests.
    """
    global _clip_processor, _clip_model
    if _clip_processor is None or _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading pretrained CLIP model: {CLIP_MODEL_NAME}")
            _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            _clip_model.eval()
            logger.info(f"Successfully loaded {CLIP_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model {CLIP_MODEL_NAME}: {e}")
            raise RuntimeError(f"Failed to initialize image analysis model: {e}")
    return _clip_processor, _clip_model


def generate_image_embedding(image_path: Union[Path, str]) -> torch.Tensor:
    """
    Extracts visual feature embeddings from an image using the pretrained CLIP model
    in inference mode (no gradients).
    
    Returns:
        1D torch.Tensor of visual features (e.g., shape [512]).
    """
    processor, model = get_clip_model_and_processor()

    with Image.open(image_path) as raw_img:
        rgb_image = raw_img.convert("RGB")

    inputs = processor(images=rgb_image, return_tensors="pt")

    with torch.no_grad():
        features_output = model.get_image_features(**inputs)

        # Handle both Transformers v5 BaseModelOutputWithPooling and legacy Tensor outputs
        if hasattr(features_output, "pooler_output") and features_output.pooler_output is not None:
            embedding = features_output.pooler_output.squeeze(0)
        elif isinstance(features_output, torch.Tensor):
            embedding = features_output.squeeze(0)
        else:
            embedding = features_output[0].squeeze(0)

    return embedding


def analyze_image(image_path: Union[Path, str]) -> Dict[str, Any]:
    """
    Extracts the image embedding and returns metadata about the image analysis,
    including the actual embedding dimension returned by the model.
    Keeps the raw embedding vector inside the backend.
    """
    embedding = generate_image_embedding(image_path)
    embedding_dim = int(embedding.shape[0])

    return {
        "embedding_size": embedding_dim,
    }
