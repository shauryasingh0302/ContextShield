import logging
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger("contextshield.fusion")

# Layer dimension constants
CAPTION_DIM = 768
OCR_DIM = 768
IMAGE_DIM = 512
PROJ_DIM = 256
FUSION_DIM = 256
DROPOUT_PROB = 0.1


class MultimodalFusionModel(nn.Module):
    """
    Multimodal feature fusion architecture:
    - Linear projection of Caption (768d -> 256d)
    - Linear projection of OCR Text (768d -> 256d)
    - Linear projection of Image (512d -> 256d)
    - Concatenation (256 + 256 + 256 = 768d)
    - Linear fusion (768d -> 256d) + ReLU + Dropout
    """

    def __init__(
        self,
        caption_dim: int = CAPTION_DIM,
        ocr_dim: int = OCR_DIM,
        image_dim: int = IMAGE_DIM,
        proj_dim: int = PROJ_DIM,
        fusion_dim: int = FUSION_DIM,
        dropout_prob: float = DROPOUT_PROB,
    ):
        super().__init__()
        # 1. Individual modality projection layers
        self.caption_proj = nn.Linear(caption_dim, proj_dim)
        self.ocr_proj = nn.Linear(ocr_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)

        # 2. Fusion layer over concatenated projections (256 * 3 = 768)
        concat_dim = proj_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

    def forward(
        self,
        caption_emb: torch.Tensor,
        ocr_emb: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass projecting each modality and fusing them.
        
        Args:
            caption_emb: [batch_size, 768] or [768]
            ocr_emb: [batch_size, 768] or [768]
            image_emb: [batch_size, 512] or [512]
            
        Returns:
            Tuple of (fusion_emb, caption_proj, ocr_proj, image_proj)
        """
        # Ensure 2D shape: [batch_size, feature_dim]
        if caption_emb.dim() == 1:
            caption_emb = caption_emb.unsqueeze(0)
        if ocr_emb.dim() == 1:
            ocr_emb = ocr_emb.unsqueeze(0)
        if image_emb.dim() == 1:
            image_emb = image_emb.unsqueeze(0)

        # Project modalities into common 256d space
        c_proj = self.caption_proj(caption_emb)  # [B, 256]
        o_proj = self.ocr_proj(ocr_emb)          # [B, 256]
        i_proj = self.image_proj(image_emb)      # [B, 256]

        # Concatenate projected features: [B, 768]
        combined = torch.cat([c_proj, o_proj, i_proj], dim=-1)

        # Apply fusion layer: [B, 256]
        fusion_emb = self.fusion(combined)

        return fusion_emb, c_proj, o_proj, i_proj


# Lazy singleton for the fusion model
_fusion_model = None


def get_fusion_model() -> MultimodalFusionModel:
    """Lazily instantiates and returns the MultimodalFusionModel in evaluation mode."""
    global _fusion_model
    if _fusion_model is None:
        _fusion_model = MultimodalFusionModel()
        _fusion_model.eval()
        logger.info("MultimodalFusionModel initialized in eval mode.")
    return _fusion_model


def fuse_multimodal_features(
    caption_emb: torch.Tensor,
    ocr_emb: torch.Tensor,
    image_emb: torch.Tensor,
) -> Dict[str, Any]:
    """
    Passes the three embeddings through the fusion network in inference mode.
    
    Returns:
        Metadata summarizing the feature dimensions (256d each).
        The raw vectors remain inside the backend.
    """
    model = get_fusion_model()

    with torch.no_grad():
        fusion_emb, c_proj, o_proj, i_proj = model(
            caption_emb=caption_emb,
            ocr_emb=ocr_emb,
            image_emb=image_emb,
        )

    return {
        "caption_features": int(c_proj.shape[-1]),
        "ocr_features": int(o_proj.shape[-1]),
        "image_features": int(i_proj.shape[-1]),
        "fusion_features": int(fusion_emb.shape[-1]),
    }
