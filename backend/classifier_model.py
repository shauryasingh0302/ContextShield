"""
Risk Classification Model for ContextShield.

Accepts the 256-dimensional multimodal fusion representation and classifies
posts into 4 risk categories:
    0: SAFE
    1: OFFENSIVE
    2: HATE
    3: HARASSMENT

Architecture:
    Fusion features (256d)
            ↓
    Linear(256, 128)
            ↓
          ReLU
            ↓
       Dropout(0.2)
            ↓
       Linear(128, 4)
            ↓
      4-class output
"""

from pathlib import Path
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_model import MultimodalFusionModel

# Class definitions
CLASS_NAMES = ["SAFE", "OFFENSIVE", "HATE", "HARASSMENT"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}


class RiskClassifier(nn.Module):
    """
    4-class risk classifier operating on 256-dimensional multimodal fusion features.
    """

    def __init__(
        self,
        in_features: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fusion_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw class logits.

        Args:
            fusion_features: Tensor of shape (batch_size, 256) or (256,)

        Returns:
            Logits tensor of shape (batch_size, 4)
        """
        if fusion_features.ndim == 1:
            fusion_features = fusion_features.unsqueeze(0)
        return self.classifier(fusion_features)


class ContextShieldModel(nn.Module):
    """
    Combined end-to-end model pairing the MultimodalFusionModel
    with the RiskClassifier.

    Allows training both projection/fusion layers and classification head
    while keeping the underlying XLM-RoBERTa and CLIP backbones frozen.
    """

    def __init__(
        self,
        fusion_model: Optional[MultimodalFusionModel] = None,
        classifier: Optional[RiskClassifier] = None,
    ):
        super().__init__()
        self.fusion = fusion_model if fusion_model is not None else MultimodalFusionModel()
        self.classifier = classifier if classifier is not None else RiskClassifier()

    def forward(
        self,
        caption_emb: torch.Tensor,
        ocr_emb: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass taking frozen embeddings and producing 4-class logits.

        Args:
            caption_emb: (batch_size, 768)
            ocr_emb: (batch_size, 768)
            image_emb: (batch_size, 512)

        Returns:
            Logits of shape (batch_size, 4)
        """
        fusion_out = self.fusion(caption_emb, ocr_emb, image_emb)
        fusion_repr = fusion_out[0] if isinstance(fusion_out, tuple) else fusion_out
        return self.classifier(fusion_repr)


# Global singleton instance for inference
_GLOBAL_CLASSIFIER: Optional[RiskClassifier] = None


def get_risk_classifier(checkpoint_path: Optional[Union[str, Path]] = None) -> RiskClassifier:
    """
    Lazy singleton loader for the RiskClassifier.
    Optionally loads weights from a saved checkpoint if available.
    """
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        model = RiskClassifier()
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                if "classifier_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["classifier_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        model.eval()
        _GLOBAL_CLASSIFIER = model
    return _GLOBAL_CLASSIFIER


def predict_risk(
    fusion_features: torch.Tensor,
    model: Optional[RiskClassifier] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Accepts the 256-dimensional multimodal fusion representation and returns
    the predicted risk category and per-class probability distribution.

    Args:
        fusion_features: Tensor of shape (256,) or (1, 256)
        model: Optional pre-initialized RiskClassifier
        checkpoint_path: Optional path to load model weights

    Returns:
        Dictionary with predicted label and class probabilities:
        {
          "label": "SAFE",
          "probabilities": {
            "SAFE": 0.25,
            "OFFENSIVE": 0.25,
            "HATE": 0.25,
            "HARASSMENT": 0.25
          }
        }
    """
    classifier = model or get_risk_classifier(checkpoint_path=checkpoint_path)
    classifier.eval()

    with torch.no_grad():
        if not isinstance(fusion_features, torch.Tensor):
            fusion_features = torch.tensor(fusion_features, dtype=torch.float32)

        if fusion_features.ndim == 1:
            fusion_features = fusion_features.unsqueeze(0)

        logits = classifier(fusion_features)  # (1, 4)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # (4,)

        predicted_idx = int(torch.argmax(probs).item())
        predicted_label = ID_TO_CLASS[predicted_idx]

        probabilities = {
            CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(CLASS_NAMES))
        }

    return {
        "label": predicted_label,
        "probabilities": probabilities,
    }


DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"

# Global singleton instance for end-to-end ContextShield model
_GLOBAL_CONTEXTSHIELD_MODEL: Optional[ContextShieldModel] = None


def get_trained_contextshield_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> ContextShieldModel:
    """
    Lazy singleton loader for the end-to-end ContextShieldModel (Fusion + Classifier).
    Loads weights from checkpoints/best_model.pt by default.
    """
    global _GLOBAL_CONTEXTSHIELD_MODEL
    if _GLOBAL_CONTEXTSHIELD_MODEL is None:
        target_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        model = ContextShieldModel()
        if target_path.exists():
            checkpoint = torch.load(target_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            print(f"[ContextShield] Loaded trained model checkpoint from {target_path}")
        else:
            print(f"[ContextShield] Warning: Checkpoint not found at {target_path}, using uninitialized model.")
        model.eval()
        _GLOBAL_CONTEXTSHIELD_MODEL = model
    return _GLOBAL_CONTEXTSHIELD_MODEL


def predict_multimodal_risk(
    caption_emb: Union[torch.Tensor, list],
    ocr_emb: Union[torch.Tensor, list],
    image_emb: Union[torch.Tensor, list],
    model: Optional[ContextShieldModel] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Takes caption embedding (768d), OCR embedding (768d), and image embedding (512d),
    runs forward pass through trained multimodal fusion + risk classifier,
    and returns risk classification results.

    Returns:
    {
      "risk_label": "SAFE | OFFENSIVE | HATE | HARASSMENT",
      "risk_score": float,
      "confidence": float,
      "probabilities": {
        "SAFE": float,
        "OFFENSIVE": float,
        "HATE": float,
        "HARASSMENT": float
      }
    }
    """
    target_model = model or get_trained_contextshield_model(checkpoint_path=checkpoint_path)
    target_model.eval()

    device = next(target_model.parameters()).device

    if not isinstance(caption_emb, torch.Tensor):
        caption_emb = torch.tensor(caption_emb, dtype=torch.float32)
    if caption_emb.ndim == 1:
        caption_emb = caption_emb.unsqueeze(0)
    caption_emb = caption_emb.to(device)

    if not isinstance(ocr_emb, torch.Tensor):
        ocr_emb = torch.tensor(ocr_emb, dtype=torch.float32)
    if ocr_emb.ndim == 1:
        ocr_emb = ocr_emb.unsqueeze(0)
    ocr_emb = ocr_emb.to(device)

    if not isinstance(image_emb, torch.Tensor):
        image_emb = torch.tensor(image_emb, dtype=torch.float32)
    if image_emb.ndim == 1:
        image_emb = image_emb.unsqueeze(0)
    image_emb = image_emb.to(device)

    with torch.no_grad():
        logits = target_model(caption_emb, ocr_emb, image_emb)  # (1, 4)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # (4,)

        predicted_idx = int(torch.argmax(probs).item())
        predicted_label = ID_TO_CLASS[predicted_idx]
        confidence = round(float(probs[predicted_idx].item()), 4)

        probabilities = {
            CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(CLASS_NAMES))
        }

        # Calculate risk score 0-100:
        # If SAFE, risk_score is the residual probability of non-safe content (1.0 - P(SAFE)) * 100
        # If non-SAFE, risk_score is the predicted risk class probability * 100
        if predicted_label == "SAFE":
            risk_score = round(float((1.0 - probs[0].item()) * 100), 2)
        else:
            risk_score = round(float(probs[predicted_idx].item() * 100), 2)

    return {
        "risk_label": predicted_label,
        "risk_score": risk_score,
        "confidence": confidence,
        "probabilities": probabilities,
    }

