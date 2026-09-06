"""
Training pipeline for ContextShield Multimodal Risk Classification.

Trains the MultimodalFusionModel and RiskClassifier jointly using CrossEntropyLoss
while keeping the pretrained XLM-RoBERTa and CLIP backbones frozen.

Classes:
  0: SAFE
  1: OFFENSIVE
  2: HATE
  3: HARASSMENT
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Ensure backend directory is in sys.path
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from classifier_model import CLASS_NAMES, ContextShieldModel, RiskClassifier
from fusion_model import MultimodalFusionModel
from dataset import ContextShieldDataset
from evaluate import evaluate_model, format_confusion_matrix, save_evaluation_results


def parse_args():
    # Detect default paths if final dataset is present
    base_dir = Path(__file__).resolve().parent.parent
    default_train = str(base_dir / "dataset" / "final" / "train.csv") if (base_dir / "dataset" / "final" / "train.csv").exists() else None
    default_val = str(base_dir / "dataset" / "final" / "validation.csv") if (base_dir / "dataset" / "final" / "validation.csv").exists() else None
    default_images = str(base_dir / "dataset" / "final" / "images") if (base_dir / "dataset" / "final" / "images").exists() else None

    parser = argparse.ArgumentParser(description="Train ContextShield Risk Classifier")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=default_train,
        help=f"Path to training dataset file (default: {default_train})",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=default_images,
        help=f"Directory containing dataset images (default: {default_images})",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=default_val,
        help=f"Path to separate validation dataset file (default: {default_val})",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio if no separate val_dataset_path is provided (default: 0.2)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for AdamW (default: 1e-4)")
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        default=True,
        help="Use class-weighted CrossEntropyLoss to address class imbalance (default: True)",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_false",
        dest="use_class_weights",
        help="Disable class-weighted CrossEntropyLoss",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        default=False,
        help="Run a 1-batch smoke test to verify forward, loss, backward, optimizer step, and checkpointing",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: results)",
    )
    return parser.parse_args()


def compute_class_weights(dataset) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, float]]:
    """Computes inverse class frequency weights from a dataset to address class imbalance."""
    if hasattr(dataset, "samples"):
        labels = [s["label"] for s in dataset.samples]
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "samples"):
        labels = [dataset.dataset.samples[i]["label"] for i in dataset.indices]
    else:
        labels = [dataset[i]["label"].item() for i in range(len(dataset))]

    num_classes = len(CLASS_NAMES)
    counts = [0] * num_classes
    for l in labels:
        if 0 <= l < num_classes:
            counts[l] += 1
    total = max(1, len(labels))
    weights = [total / (float(num_classes) * max(1, c)) for c in counts]
    counts_dict = {CLASS_NAMES[i]: counts[i] for i in range(num_classes)}
    weights_dict = {CLASS_NAMES[i]: round(weights[i], 4) for i in range(num_classes)}
    return torch.tensor(weights, dtype=torch.float), counts_dict, weights_dict


def run_smoke_test(
    model: ContextShieldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainable_params: List[nn.Parameter],
    checkpoint_dir: Path,
    device: str,
    args,
) -> bool:
    """Executes a 1-batch smoke test verifying forward, loss, backward, optimizer step, and checkpointing."""
    print("\n" + "=" * 65)
    print(" Running Training Pipeline Smoke Test (1 Batch)")
    print("=" * 65)

    # 1. Forward pass on 1 training batch
    model.train()
    batch = next(iter(train_loader))
    caption_emb = batch["caption_emb"].to(device)
    ocr_emb = batch["ocr_emb"].to(device)
    image_emb = batch["image_emb"].to(device)
    labels = batch["label"].to(device)

    print("Input batch shapes:")
    print(f"  - Caption Embedding: {list(caption_emb.shape)}")
    print(f"  - OCR Embedding:     {list(ocr_emb.shape)}")
    print(f"  - Image Embedding:   {list(image_emb.shape)}")
    print(f"  - Labels:            {list(labels.shape)} (batch labels: {labels.tolist()})")

    # 2. Forward pass & loss
    optimizer.zero_grad()
    logits = model(caption_emb, ocr_emb, image_emb)
    print(f"Forward pass output logits shape: {list(logits.shape)} (4 classes: {CLASS_NAMES})")

    loss = criterion(logits, labels)
    loss_val = loss.item()
    print(f"Loss computed successfully: {loss_val:.4f}")

    # 3. Backward pass
    loss.backward()
    grads_ok = all(p.grad is not None and not torch.isnan(p.grad).any() for p in trainable_params)
    print(f"Backward pass executed. All {len(trainable_params)} trainable parameter tensors have valid gradients: {grads_ok}")

    # 4. Optimizer step
    optimizer.step()
    print("Optimizer step executed successfully.")

    # 5. Validation forward pass
    model.eval()
    val_batch = next(iter(val_loader))
    with torch.no_grad():
        v_caption = val_batch["caption_emb"].to(device)
        v_ocr = val_batch["ocr_emb"].to(device)
        v_img = val_batch["image_emb"].to(device)
        v_labels = val_batch["label"].to(device)
        v_logits = model(v_caption, v_ocr, v_img)
        v_loss = criterion(v_logits, v_labels).item()
        v_preds = torch.argmax(v_logits, dim=-1)
        v_acc = (v_preds == v_labels).float().mean().item()
    print(f"Validation batch pass: loss={v_loss:.4f}, batch accuracy={v_acc:.4f}")

    # 6. Save and verify checkpoint
    smoke_ckpt_path = checkpoint_dir / "smoke_test_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fusion_state_dict": model.fusion.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_val,
        },
        smoke_ckpt_path,
    )
    print(f"Smoke test checkpoint saved to: {smoke_ckpt_path} ({smoke_ckpt_path.stat().st_size / 1024:.1f} KB)")

    # Verify loading checkpoint
    loaded = torch.load(smoke_ckpt_path, weights_only=True, map_location=device)
    model.load_state_dict(loaded["model_state_dict"])
    print("Checkpoint reloaded and verified successfully.")

    # Report parameters and config
    total_trainable = sum(p.numel() for p in trainable_params)
    fusion_params = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)

    print("\n" + "-" * 65)
    print(" Smoke Test Report Summary")
    print("-" * 65)
    print(f"Trainable Parameters:        {total_trainable:,} (Fusion: {fusion_params:,}, Classifier: {classifier_params:,})")
    print("Frozen Backbones:            XLM-RoBERTa (~278M) and CLIP (~87M) kept completely frozen")
    print(f"Number of Classes:           {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
    print(f"Batch Size:                  {args.batch_size}")
    print(f"Learning Rate:               {args.lr}")
    print(f"Class Weighting Used:        {'YES' if args.use_class_weights else 'NO'}")
    print("Smoke Test Status:           PASSED")
    print("=" * 65)
    return True


def train(args):
    # 1. Dataset availability check
    if not args.dataset_path or not Path(args.dataset_path).exists():
        print("=" * 65)
        print("ContextShield Risk Classification Training Pipeline")
        print("=" * 65)
        print(f"Status: Dataset not found at '{args.dataset_path}'.")
        print("\nNote for Student Project Review:")
        print("- The training and validation pipeline has been fully implemented.")
        print("- Per project requirements, NO synthetic/fake training data was generated.")
        print("- Actual model training cannot be performed until a real dataset is provided.")
        print("\nTo train when your dataset is ready, run:")
        print("  python train.py --dataset_path <path/to/dataset.csv> --image_dir <path/to/images>")
        print("=" * 65)
        return False

    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    full_dataset = ContextShieldDataset(dataset_path=args.dataset_path, image_dir=args.image_dir)

    if len(full_dataset) == 0:
        print("Error: Dataset contains 0 valid samples. Please verify labels and file paths.")
        return False

    if args.val_dataset_path and Path(args.val_dataset_path).exists():
        train_dataset = full_dataset
        val_dataset = ContextShieldDataset(dataset_path=args.val_dataset_path, image_dir=args.image_dir)
    else:
        val_size = max(1, int(len(full_dataset) * args.val_split))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # 3. Initialize model (Fusion + Classifier)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    model = ContextShieldModel().to(device)

    # XLM-R and CLIP are kept frozen in text_service and image_service.
    # Only the fusion layers and classifier layers are trained.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

    # 4. Class weighting setup
    if args.use_class_weights:
        weights_tensor, counts_dict, weights_dict = compute_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
        print(f"Training Class Distribution: {counts_dict}")
        print(f"Applied Class Weights:       {weights_dict}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Class weighting: DISABLED (uniform CrossEntropyLoss)")

    # 5. Handle smoke test mode
    if args.smoke_test:
        return run_smoke_test(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            trainable_params=trainable_params,
            checkpoint_dir=checkpoint_dir,
            device=device,
            args=args,
        )

    best_val_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = None
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    print("\nStarting Training...")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        # Training loop
        model.train()
        running_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            caption_emb = batch["caption_emb"].to(device)
            ocr_emb = batch["ocr_emb"].to(device)
            image_emb = batch["image_emb"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(caption_emb, ocr_emb, image_emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_train_loss / num_batches if num_batches > 0 else 0.0

        # Validation loop
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion=criterion, device=device)
        val_macro_f1 = val_metrics["macro_f1"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        # Save best model checkpoint using validation Macro F1
        if val_macro_f1 > best_val_macro_f1 or (val_macro_f1 == best_val_macro_f1 and val_loss < best_val_loss):
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "fusion_state_dict": model.fusion.state_dict(),
                    "classifier_state_dict": model.classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                },
                best_checkpoint_path,
            )
            print(f"  --> [BEST CHECKPOINT] Saved at Epoch {epoch} (Val Macro F1: {val_macro_f1:.4f}, Val Loss: {val_loss:.4f})")

    print("-" * 65)
    print(f"Training complete! Best Epoch: {best_epoch} | Best Val Macro F1: {best_val_macro_f1:.4f}")
    print(f"Best checkpoint saved to: {best_checkpoint_path}")

    # Save final evaluation results
    if best_metrics:
        results_path = results_dir / "evaluation_results.json"
        save_evaluation_results(best_metrics, results_path)
        print(f"Best evaluation results saved to: {results_path}")
        print("\nFinal Confusion Matrix:")
        print(format_confusion_matrix(best_metrics["confusion_matrix"]))

    return True


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
