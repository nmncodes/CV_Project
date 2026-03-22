import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_project_dirs(project_root: str):
    required_dirs = [
        Path(project_root) / "outputs" / "models",
        Path(project_root) / "outputs" / "plots",
        Path(project_root) / "outputs" / "reports",
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)


def epoch_metrics_from_outputs(total_loss: float, correct: int, total: int) -> Tuple[float, float]:
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_json(data: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_training_curves(history: Dict[str, List[float]], output_path: str, title: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_acc"], label="Train Accuracy")
    axes[0].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[0].set_title("Accuracy vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_loss"], label="Train Loss")
    axes[1].plot(epochs, history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(
    y_true: List[int], y_pred: List[int], class_names: List[str], output_path: str, title: str
):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_classification_report(
    y_true: List[int], y_pred: List[int], class_names: List[str], output_txt_path: str
) -> Dict:
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_txt_path.replace(".txt", ".csv"), index=True)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    return report


def save_sample_predictions(
    images,
    true_labels,
    pred_labels,
    class_names: List[str],
    output_path: str,
    max_samples: int = 12,
):
    n = min(max_samples, len(images))
    cols = 4
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        axes[i].axis("off")

    for i in range(n):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)

        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        axes[i].set_title(f"P: {pred_name}\nT: {true_name}", color=color, fontsize=9)
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_comparison_table(results: List[Dict], output_csv: str):
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)


def print_model_comparison(results: List[Dict]):
    print("\nModel Comparison")
    print("-" * 60)
    for row in results:
        print(
            f"{row['model_name']}: Accuracy={row['accuracy'] * 100:.2f}% | "
            f"Precision={row['precision']:.4f} | Recall={row['recall']:.4f} | F1={row['f1']:.4f}"
        )
