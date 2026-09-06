"""
Validation and Evaluation module for ContextShield Risk Classification.

Computes:
  - Accuracy
  - Macro Precision
  - Macro Recall
  - Macro F1
  - 4x4 Confusion Matrix
  - Per-class metrics (Precision, Recall, F1, Support)
Saves results to a JSON file and displays a formatted summary.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from classifier_model import CLASS_NAMES, ContextShieldModel


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    Computes classification metrics for 4-class risk classification.

    Classes:
      0: SAFE
      1: OFFENSIVE
      2: HATE
      3: HARASSMENT

    Args:
      y_true: List of ground truth class integer IDs (0-3)
      y_pred: List of predicted class integer IDs (0-3)

    Returns:
      Dictionary containing accuracy, macro precision/recall/f1,
      per-class breakdowns, and 4x4 confusion matrix.
    """
    num_classes = len(CLASS_NAMES)
    total_samples = len(y_true)

    if total_samples == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {name: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for name in CLASS_NAMES},
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
            "labels": CLASS_NAMES,
        }

    # Initialize 4x4 Confusion Matrix [true_idx][pred_idx]
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1

    # Calculate per-class metrics
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    correct = sum(cm[i][i] for i in range(num_classes))

    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = sum(cm[row][i] for row in range(num_classes) if row != i)
        fn = sum(cm[i][col] for col in range(num_classes) if col != i)
        support = sum(cm[i][col] for col in range(num_classes))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    accuracy = correct / total_samples if total_samples > 0 else 0.0
    macro_precision = sum(precisions) / num_classes
    macro_recall = sum(recalls) / num_classes
    macro_f1 = sum(f1s) / num_classes

    return {
        "total_samples": total_samples,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": CLASS_NAMES,
    }


def format_confusion_matrix(cm: List[List[int]], labels: List[str] = CLASS_NAMES) -> str:
    """
    Formats the 4x4 confusion matrix into an ASCII table string.
    """
    col_width = 12
    header = f"{'True \\ Pred':<{col_width}}" + "".join(f"{lbl:>{col_width}}" for lbl in labels)
    separator = "-" * len(header)
    rows = [header, separator]

    for i, row_label in enumerate(labels):
        row_str = f"{row_label:<{col_width}}" + "".join(f"{cm[i][j]:>{col_width}}" for j in range(len(labels)))
        rows.append(row_str)

    return "\n".join(rows)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Tuple[float, Dict]:
    """
    Runs evaluation on a PyTorch DataLoader.

    Returns:
      (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    loss_fn = criterion or nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            caption_emb = batch["caption_emb"].to(device)
            ocr_emb = batch["ocr_emb"].to(device)
            image_emb = batch["image_emb"].to(device)
            labels = batch["label"].to(device)

            logits = model(caption_emb, ocr_emb, image_emb)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            targets = labels.cpu().tolist()

            all_preds.extend(preds)
            all_targets.extend(targets)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = round(avg_loss, 4)

    return avg_loss, metrics


def save_evaluation_results(results: Dict, output_path: Union[str, Path]):
    """
    Saves metrics and confusion matrix to JSON.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ContextShield Risk Classifier")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to evaluation dataset (CSV or JSON)")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to images directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt", help="Model checkpoint to load")
    parser.add_argument("--output_path", type=str, default="results/evaluation_results.json", help="Path to save evaluation JSON")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()

    if not args.dataset_path or not Path(args.dataset_path).exists():
        print(f"Error: Evaluation dataset not found at '{args.dataset_path}'.")
        print("Please provide a valid dataset path using --dataset_path <path_to_csv_or_json>.")
        print("Evaluation cannot be performed without an actual dataset.")
        return

    from dataset import ContextShieldDataset
    dataset = ContextShieldDataset(dataset_path=args.dataset_path, image_dir=args.image_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = ContextShieldModel()
    ckpt_path = Path(args.checkpoint_path)
    if ckpt_path.exists():
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"Warning: No checkpoint found at '{ckpt_path}'. Evaluating with initialized weights.")

    loss, metrics = evaluate_model(model, data_loader)

    print("\n" + "=" * 60)
    print(" ContextShield Test Set Evaluation Results")
    print("=" * 60)
    print(f"Total Samples:     {metrics['total_samples']}")
    print(f"Loss:              {metrics['loss']:.4f}")
    print(f"Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Macro Precision:   {metrics['macro_precision']:.4f} ({metrics['macro_precision'] * 100:.2f}%)")
    print(f"Macro Recall:      {metrics['macro_recall']:.4f} ({metrics['macro_recall'] * 100:.2f}%)")
    print(f"Macro F1:          {metrics['macro_f1']:.4f} ({metrics['macro_f1'] * 100:.2f}%)")

    print("\nPer-Class Metrics:")
    print(f"{'Class':<14} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 60)
    for c_name in CLASS_NAMES:
        c_stats = metrics["per_class"].get(c_name, {})
        p = c_stats.get("precision", 0.0)
        r = c_stats.get("recall", 0.0)
        f = c_stats.get("f1", 0.0)
        sup = c_stats.get("support", 0)
        print(f"{c_name:<14} {p:<12.4f} {r:<12.4f} {f:<12.4f} {sup:<8}")

    print("\n4x4 Confusion Matrix:")
    print(format_confusion_matrix(metrics["confusion_matrix"]))
    print("=" * 60)

    save_evaluation_results(metrics, args.output_path)
    print(f"Evaluation results successfully saved to: {args.output_path}")


if __name__ == "__main__":
    main()
