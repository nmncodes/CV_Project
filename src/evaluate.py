import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from dataset import denormalize_image, get_dataloaders, prepare_data_splits, show_image_with_gradcam
from gradcam import GradCAM, save_gradcam_examples
from model_cnn import CustomPlantCNN
from model_transfer import TransferMobileNetV2
from utils import (
    ensure_project_dirs,
    print_model_comparison,
    save_classification_report,
    save_comparison_table,
    save_confusion_matrix,
    save_sample_predictions,
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
    parser = argparse.ArgumentParser(description="Evaluate trained plant disease models")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_all_classes", action="store_true")
    parser.add_argument("--force_resplit", action="store_true")
    # parser.add_argument("--models", nargs="+", default=["cnn", "mobilenet"], choices=["cnn", "mobilenet"])
    parser.add_argument("--models", nargs="+", default=["cnn"], choices=["cnn", "mobilenet"])
    return parser.parse_args()


def load_model_from_checkpoint(model_name: str, ckpt_path: Path, num_classes: int, device):
    if model_name == "cnn":
        model = CustomPlantCNN(num_classes=num_classes)
    elif model_name == "mobilenet":
        model = TransferMobileNetV2(num_classes=num_classes, freeze_backbone=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_model(model, data_loader, device):
    y_true, y_pred = [], []
    sample_images, sample_true, sample_pred = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Test", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            if len(sample_images) < 12:
                for i in range(images.size(0)):
                    sample_images.append(denormalize_image(images[i]))
                    sample_true.append(int(labels[i].item()))
                    sample_pred.append(int(preds[i].item()))
                    if len(sample_images) >= 12:
                        break

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sample_images": sample_images,
        "sample_true": sample_true,
        "sample_pred": sample_pred,
    }


def generate_model_gradcam(model, model_name, data_loader, class_names, device, output_path: Path):
    if not hasattr(model, "model"):
        return

    if model_name == "mobilenet":
        target_layer = model.model.features[-1]
    elif model_name == "cnn":
        target_layer = model.model.layer4[1].conv2
    else:
        return

    gradcam = GradCAM(model, target_layer=target_layer)

    original_images = []
    cam_images = []
    titles = []

    images, labels = next(iter(data_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.enable_grad():
        for i in range(min(6, images.size(0))):
            single = images[i : i + 1]
            label = int(labels[i].item())
            logits = model(single)
            pred = int(logits.argmax(dim=1).item())

            cam = gradcam(single, target_class=pred)
            denorm_tensor = denormalize_image(single[0])
            rgb_image = denorm_tensor.permute(1, 2, 0).cpu().numpy()
            overlay = show_image_with_gradcam(rgb_image, cam, alpha=0.45)

            original_images.append(rgb_image)
            cam_images.append(overlay)
            titles.append(f"P: {class_names[pred]} | T: {class_names[label]}")

    gradcam.close()
    save_gradcam_examples(original_images, cam_images, titles, str(output_path))


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    set_seed(args.seed)
    ensure_project_dirs(str(project_root))

    data_dir = project_root / args.data_dir
    selected_classes = None if args.use_all_classes else DEFAULT_CLASSES

    print("[evaluate] Ensuring dataset split exists...")
    prepare_data_splits(
        data_dir=str(data_dir),
        seed=args.seed,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        selected_classes=selected_classes,
        force_rebuild=args.force_resplit,
    )

    loaders, class_names = get_dataloaders(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )

    test_loader = loaders["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Using device: {device}")

    if len(class_names) < 2:
        raise RuntimeError(
            "Detected fewer than 2 classes in split. "
            "Run evaluate/train with --force_resplit to rebuild dataset."
        )

    results_table = []

    for model_name in args.models:
        ckpt_path = project_root / "outputs" / "models" / f"best_{model_name}.pth"
        if not ckpt_path.exists():
            print(f"[evaluate] Checkpoint missing for {model_name}: {ckpt_path}")
            continue

        print(f"\n[evaluate] Evaluating model: {model_name}")
        model, checkpoint = load_model_from_checkpoint(
            model_name=model_name,
            ckpt_path=ckpt_path,
            num_classes=len(class_names),
            device=device,
        )

        output = evaluate_model(model, test_loader, device)

        conf_model_path = project_root / "outputs" / "plots" / f"confusion_matrix_{model_name}.png"
        conf_shared_path = project_root / "outputs" / "plots" / "confusion_matrix.png"
        save_confusion_matrix(
            output["y_true"],
            output["y_pred"],
            class_names,
            str(conf_model_path),
            title=f"Confusion Matrix - {model_name}",
        )
        save_confusion_matrix(
            output["y_true"],
            output["y_pred"],
            class_names,
            str(conf_shared_path),
            title=f"Confusion Matrix - {model_name}",
        )

        report_txt_path = project_root / "outputs" / "reports" / f"classification_report_{model_name}.txt"
        report = save_classification_report(
            output["y_true"],
            output["y_pred"],
            class_names,
            str(report_txt_path),
        )

        sample_model_path = project_root / "outputs" / "plots" / f"sample_predictions_{model_name}.png"
        sample_shared_path = project_root / "outputs" / "plots" / "sample_predictions.png"
        save_sample_predictions(
            output["sample_images"],
            output["sample_true"],
            output["sample_pred"],
            class_names,
            str(sample_model_path),
            max_samples=12,
        )
        save_sample_predictions(
            output["sample_images"],
            output["sample_true"],
            output["sample_pred"],
            class_names,
            str(sample_shared_path),
            max_samples=12,
        )

        if model_name in ["mobilenet", "cnn"]:
            gradcam_path = project_root / "outputs" / "plots" / f"gradcam_{model_name}.png"
            generate_model_gradcam(model, model_name, test_loader, class_names, device, gradcam_path)

        print(f"Accuracy ({model_name}): {output['accuracy'] * 100:.2f}%")
        print(f"Precision ({model_name}): {output['precision']:.4f}")
        print(f"Recall ({model_name}): {output['recall']:.4f}")
        print(f"F1-score ({model_name}): {output['f1']:.4f}")

        results_table.append(
            {
                "model_name": model_name,
                "accuracy": output["accuracy"],
                "precision": output["precision"],
                "recall": output["recall"],
                "f1": output["f1"],
            }
        )

    if results_table:
        comparison_csv = project_root / "outputs" / "reports" / "model_comparison.csv"
        save_comparison_table(results_table, str(comparison_csv))
        print_model_comparison(results_table)
    else:
        print("[evaluate] No models were evaluated. Train at least one model first.")


if __name__ == "__main__":
    main()
