"""
Dataset loader for ContextShield multimodal risk classification.

Supports CSV and JSON/JSONL dataset files with fields:
  - image / image_path: filename or path to image
  - caption: text caption string
  - label: integer (0-3) or string ('SAFE', 'OFFENSIVE', 'HATE', 'HARASSMENT')
  - ocr_text: (optional) text detected inside the image
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

from classifier_model import CLASS_TO_ID
from text_service import generate_embedding
from image_service import generate_image_embedding
from ocr_service import extract_text_from_image


class ContextShieldDataset(Dataset):
    """
    Multimodal dataset for ContextShield risk classification.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        run_ocr_if_missing: bool = False,
        cache_embeddings: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.image_dir = Path(image_dir) if image_dir is not None else self.dataset_path.parent
        self.run_ocr_if_missing = run_ocr_if_missing
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.samples: List[Dict] = []

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path}")

        self._load_dataset()

    def _load_dataset(self):
        suffix = self.dataset_path.suffix.lower()

        if suffix in [".csv", ".tsv"]:
            delimiter = "\t" if suffix == ".tsv" else ","
            with open(self.dataset_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    self._parse_row(row)

        elif suffix in [".json", ".jsonl"]:
            with open(self.dataset_path, mode="r", encoding="utf-8") as f:
                content = f.read().strip()
                if content.startswith("["):
                    data = json.loads(content)
                    for item in data:
                        self._parse_row(item)
                else:
                    for line in content.splitlines():
                        if line.strip():
                            self._parse_row(json.loads(line))
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}. Expected .csv, .tsv, .json, or .jsonl")

    def _parse_row(self, row: Dict):
        # 1. Resolve image path
        img_name = row.get("image") or row.get("image_path") or row.get("img")
        if not img_name:
            return

        img_path = Path(img_name)
        if not img_path.is_absolute():
            if not (self.image_dir / img_path).exists() and (self.image_dir / "images" / img_path).exists():
                img_path = self.image_dir / "images" / img_path
            else:
                img_path = self.image_dir / img_path

        # 2. Resolve caption
        caption = row.get("caption") or row.get("text") or ""

        # 3. Resolve label
        raw_label = row.get("label")
        if raw_label is None:
            return

        label_id: int
        if isinstance(raw_label, int) or (isinstance(raw_label, str) and raw_label.isdigit()):
            label_id = int(raw_label)
        else:
            label_str = str(raw_label).strip().upper()
            if label_str not in CLASS_TO_ID:
                return
            label_id = CLASS_TO_ID[label_str]

        if label_id < 0 or label_id > 3:
            return

        # 4. Resolve OCR text
        ocr_text = row.get("ocr_text")
        if ocr_text is None:
            ocr_text = ""

        self.samples.append({
            "image_path": img_path,
            "caption": caption,
            "ocr_text": ocr_text,
            "label": label_id,
        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_embeddings and idx in self._cache:
            return self._cache[idx]

        sample = self.samples[idx]

        # Generate frozen embeddings under no_grad
        with torch.no_grad():
            caption_emb = generate_embedding(sample["caption"])

            ocr_text = sample["ocr_text"]
            if not ocr_text and self.run_ocr_if_missing and sample["image_path"].exists():
                ocr_text = extract_text_from_image(sample["image_path"])

            ocr_emb = generate_embedding(ocr_text)
            image_emb = generate_image_embedding(sample["image_path"])

        label_tensor = torch.tensor(sample["label"], dtype=torch.long)

        result = {
            "caption_emb": caption_emb,
            "ocr_emb": ocr_emb,
            "image_emb": image_emb,
            "label": label_tensor,
        }

        if self.cache_embeddings:
            self._cache[idx] = result

        return result
