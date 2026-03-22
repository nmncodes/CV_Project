import argparse
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import denormalize_image, get_dataloaders, prepare_data_splits
from model_cnn import CustomPlantCNN
from model_transfer import TransferMobileNetV2
from utils import (
    ensure_project_dirs,
    epoch_metrics_from_outputs,
    get_device,
    save_json,
    save_sample_predictions,
    save_training_curves,
    set_seed,
)

DEFAULT_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Plant Disease CNN models")
    parser.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--cnn_balanced_sampling",
        action="store_true",
        help="Enable weighted sampler for CNN (off by default to keep training fast).",
    )
    parser.add_argument(
        "--cnn_class_weights",
        action="store_true",
        help="Enable class-weighted loss for CNN (off by default).",
    )
    parser.add_argument(
        "--cnn_unfreeze_last_stage",
        action="store_true",
        help="Unfreeze ResNet layer4 for CNN mid-training for extra accuracy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=(2 if platform.system() == "Windows" else max(2, min(8, (os.cpu_count() or 4) // 2))),
    )
    parser.add_argument("--use_all_classes", action="store_true")
    parser.add_argument("--force_resplit", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Laptop fast mode: fewer epochs, smaller image size, larger batch size.",
    )
    parser.add_argument(
        "--unfreeze_last_blocks",
        action="store_true",
        help="Enable phase-2 fine-tuning by unfreezing last MobileNetV2 blocks.",
    )
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        running_total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_metrics_from_outputs(running_loss, running_correct, running_total)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_metrics_from_outputs(running_loss, running_correct, running_total)


def collect_sample_predictions(model, loader, device, class_names, max_samples=12):
    model.eval()
    all_images = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            for i in range(images.size(0)):
                all_images.append(denormalize_image(images[i].cpu()))
                true_labels.append(int(labels[i].item()))
                pred_labels.append(int(preds[i].item()))

                if len(all_images) >= max_samples:
                    return all_images, true_labels, pred_labels

    return all_images, true_labels, pred_labels


def build_model(model_name: str, num_classes: int):
    if model_name == "cnn":
        return CustomPlantCNN(num_classes=num_classes, freeze_backbone=True)
    if model_name == "mobilenet":
        return TransferMobileNetV2(num_classes=num_classes, freeze_backbone=True)
    raise ValueError(f"Unsupported model: {model_name}")


def get_class_weights(loader, num_classes: int) -> Optional[torch.Tensor]:
    targets = getattr(loader.dataset, "targets", None)
    if targets is None:
        return None

    target_tensor = torch.tensor(targets, dtype=torch.long)
    class_counts = torch.bincount(target_tensor, minlength=num_classes).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    # Soft inverse-frequency weights to improve minority recall without destabilizing accuracy.
    weights = (class_counts.sum() / (class_counts * num_classes)).pow(0.5)
    return (weights / weights.mean()).float()


def save_checkpoint(
    model,
    output_path: str,
    class_names: List[str],
    model_name: str,
    val_acc: float,
):
    payload = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "model_name": model_name,
        "val_accuracy": val_acc,
    }
    torch.save(payload, output_path)


def train_model(
    model,
    model_name: str,
    loaders: Dict,
    class_names: List[str],
    epochs: int,
    lr: float,
    device,
    project_root: Path,
    weight_decay: float,
    label_smoothing: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
    unfreeze_last_blocks: bool,
    cnn_class_weights: bool,
    cnn_unfreeze_last_stage: bool,
):
    if model_name == "cnn":
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)
    else:
        optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        scheduler = None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    model = model.to(device)
    class_weights = None
    if model_name == "cnn" and cnn_class_weights:
        class_weights = get_class_weights(loaders["train"], num_classes=len(class_names))
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"[train] Using class-balanced loss weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    best_val_acc = 0.0
    epochs_without_improvement = 0

    phase_switch_epoch = max(5, epochs // 2)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # For transfer learning: unfreeze last blocks mid-training for fine-tuning.
        if model_name == "mobilenet" and unfreeze_last_blocks and epoch == phase_switch_epoch:
            print("[train] Unfreezing last MobileNetV2 blocks for fine-tuning...")
            model.unfreeze_last_blocks(num_blocks=4)
            optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=lr / 2)

        if model_name == "cnn" and cnn_unfreeze_last_stage and epoch == phase_switch_epoch:
            print("[train] Unfreezing CNN layer4 for short fine-tuning...")
            model.unfreeze_last_stage()
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr / 2,
                weight_decay=weight_decay,
            )

        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, loaders["val"], criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc * 100:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if scheduler is not None:
            scheduler.step()

        if val_acc > (best_val_acc + early_stop_min_delta):
            best_val_acc = val_acc
            epochs_without_improvement = 0
            ckpt_path = project_root / "outputs" / "models" / f"best_{model_name}.pth"
            save_checkpoint(
                model=model,
                output_path=str(ckpt_path),
                class_names=class_names,
                model_name=model_name,
                val_acc=val_acc,
            )
            print(f"[train] Saved best checkpoint: {ckpt_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            print(f"[train] Early stopping triggered after {epoch} epochs.")
            break

    # Save training curves and history.
    curve_path = project_root / "outputs" / "plots" / f"training_curve_{model_name}.png"
    shared_curve_path = project_root / "outputs" / "plots" / "training_curve.png"
    save_training_curves(history, str(curve_path), title=f"Training Curves - {model_name}")
    save_training_curves(history, str(shared_curve_path), title=f"Training Curves - {model_name}")

    history_path = project_root / "outputs" / "reports" / f"history_{model_name}.json"
    save_json(history, str(history_path))

    # Save sample predictions from validation set for quick qualitative inspection.
    sample_images, y_true, y_pred = collect_sample_predictions(
        model, loaders["val"], device, class_names, max_samples=12
    )
    sample_plot_path = project_root / "outputs" / "plots" / f"sample_predictions_{model_name}.png"
    if sample_images:
        save_sample_predictions(sample_images, y_true, y_pred, class_names, str(sample_plot_path), max_samples=12)

    return best_val_acc


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    default_lr = 1e-3 if args.model == "cnn" else 1e-4
    lr_was_set_by_user = args.lr is not None
    if args.lr is None:
        args.lr = default_lr

    if args.quick:
        args.epochs = min(args.epochs, 8)
        args.img_size = min(args.img_size, 160)
        args.batch_size = max(args.batch_size, 64)
        if not lr_was_set_by_user:
            args.lr = default_lr
        if platform.system() == "Windows":
            args.num_workers = min(args.num_workers, 2)
        else:
            args.num_workers = max(args.num_workers, max(2, min(8, os.cpu_count() or 4)))
        if args.model == "mobilenet":
            args.epochs = min(args.epochs, 6)
        if args.model == "cnn":
            args.epochs = min(args.epochs, 6)
            args.img_size = min(args.img_size, 128)
            args.batch_size = max(args.batch_size, 96)
        print(
            "[setup] Quick mode enabled: "
            f"epochs={args.epochs}, batch_size={args.batch_size}, img_size={args.img_size}, "
            f"num_workers={args.num_workers}"
        )

    set_seed(args.seed)
    ensure_project_dirs(str(project_root))

    data_dir = project_root / args.data_dir
    selected_classes = None if args.use_all_classes else DEFAULT_CLASSES

    print("[setup] Preparing dataset splits...")
    prepare_data_splits(
        data_dir=str(data_dir),
        seed=args.seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        selected_classes=selected_classes,
        force_rebuild=args.force_resplit,
    )

    print("[setup] Creating dataloaders...")
    loaders, class_names = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        balanced_sampling=args.model == "cnn" and args.cnn_balanced_sampling,
        augment_level=("light" if args.quick else "strong"),
    )

    device = get_device()
    print(f"[setup] Using device: {device}")
    print(f"[setup] Number of classes: {len(class_names)}")

    if len(class_names) < 2:
        raise RuntimeError(
            "Detected fewer than 2 classes in train split. "
            "Run with --force_resplit to regenerate data correctly."
        )

    model = build_model(args.model, num_classes=len(class_names))

    best_val_acc = train_model(
        model=model,
        model_name=args.model,
        loaders=loaders,
        class_names=class_names,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        project_root=project_root,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        cnn_class_weights=args.cnn_class_weights,
        cnn_unfreeze_last_stage=args.cnn_unfreeze_last_stage,
    )

    print("\nTraining complete.")
    print(f"Best Validation Accuracy ({args.model}): {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
