import argparse
import random
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import get_transforms, open_image
from model_cnn import CustomPlantCNN
from model_transfer import TransferMobileNetV2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-image input/output demo using a trained checkpoint"
    )
    parser.add_argument("--model", choices=["cnn", "mobilenet"], default="cnn")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(model_name: str, num_classes: int):
    if model_name == "cnn":
        return CustomPlantCNN(num_classes=num_classes, freeze_backbone=False)
    if model_name == "mobilenet":
        return TransferMobileNetV2(num_classes=num_classes, freeze_backbone=False)
    raise ValueError(f"Unsupported model: {model_name}")


def find_random_test_image(project_root: Path, data_dir: str, seed: int) -> Optional[Path]:
    test_root = project_root / data_dir / "test"
    if not test_root.exists():
        return None

    all_images: List[Path] = []
    for path in test_root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}:
            all_images.append(path)

    if not all_images:
        return None

    random.seed(seed)
    return random.choice(all_images)


def save_demo_figure(img_tensor, image_path: Path, pred_label: str, confidence: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.set_title(f"Pred: {pred_label} ({confidence * 100:.2f}%)")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else project_root / "outputs" / "models" / f"best_{args.model}.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first using src/train.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    class_names = ckpt.get("class_names")
    if not class_names:
        raise RuntimeError("Checkpoint does not contain class_names.")

    image_path = Path(args.image) if args.image else find_random_test_image(project_root, args.data_dir, args.seed)
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(
            "No image found. Pass --image path/to/leaf.jpg or ensure data/test has images."
        )

    model = build_model(args.model, num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    _, eval_transform = get_transforms(img_size=args.img_size, augment_level="light")
    pil_img = open_image(str(image_path))
    input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    topk = max(1, min(args.topk, len(class_names)))
    conf_vals, idx_vals = torch.topk(probs, k=topk)

    pred_idx = int(idx_vals[0].item())
    pred_label = class_names[pred_idx]
    confidence = float(conf_vals[0].item())

    print("\n=== Sample Input -> Output Demo ===")
    print(f"Model: {args.model}")
    print(f"Input Image: {image_path}")
    print(f"Predicted Class: {pred_label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Top Predictions:")
    for rank, (score, idx) in enumerate(zip(conf_vals.tolist(), idx_vals.tolist()), start=1):
        print(f"  {rank}. {class_names[int(idx)]} -> {score * 100:.2f}%")

    out_path = project_root / "outputs" / "plots" / "demo_single_prediction.png"
    # Reuse transformed tensor and denormalize for clean visualization.
    img_vis = input_tensor[0].detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = torch.clamp(img_vis * std + mean, 0, 1)
    save_demo_figure(img_vis, image_path, pred_label, confidence, out_path)
    print(f"Saved demo figure: {out_path}")


if __name__ == "__main__":
    main()
