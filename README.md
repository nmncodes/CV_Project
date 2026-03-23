# Plant Disease Classification using CNNs (PyTorch)

A complete, runnable college-level deep learning project for leaf disease classification using the PlantVillage dataset.

This project compares:
- Custom CNN baseline
- MobileNetV2 transfer learning model

It is designed to run on CPU laptops, while automatically using GPU if available.

## 1. Project Overview

The goal is to classify plant leaf images into disease categories. By default, this project uses five presentation-friendly classes:
- Tomato___Early_blight
- Tomato___Late_blight
- Tomato___healthy
- Potato___Early_blight
- Potato___healthy

You can optionally train on all PlantVillage classes with `--use_all_classes`.

## 2. Folder Structure

```text
plant_disease_cnn/
  data/
    train/
    val/
    test/
  src/
    dataset.py
    model_cnn.py
    model_transfer.py
    train.py
    evaluate.py
    utils.py
    gradcam.py
  outputs/
    models/
    plots/
    reports/
  notebooks/
    eda.ipynb
  requirements.txt
  README.md
  run_train.sh
```

## 3. CNN Explanation (ResNet-18 Based Custom Model)

The custom CNN uses a **ResNet-18** backbone (pretrained or from scratch) with a customized classification head for plant disease identification.

Features include:
- **ResNet-18 Feature Extraction**: Advanced residual learning for deep feature capture.
- **Customized Head**: `Dropout(0.35) -> Linear(512-256) -> ReLU -> Dropout(0.20) -> Linear(256-classes)`.
- **Optimization**: AdamW optimizer with Label Smoothing and Cosine Annealing Learning Rate Scheduler.
- **Fine-tuning**: Selective layer unfreezing (e.g., `--cnn_unfreeze_last_stage`) to adapt the model more closely to leaf morphology.

This model is the primary choice for the current phase of the project, focusing on balancing accuracy and training time.

## 4. Transfer Learning (MobileNetV2 - Disabled by Default)

MobileNetV2 is a lightweight architecture optimized for mobile and embedded vision applications. In this project, it is available for comparative studies but currently kept inactive unless manually enabled.

Key Strategy:
1. Frozen features for initial classification head stabilization.
2. Two-phase fine-tuning by unfreezing the last 4 stage blocks.
3. Adam optimizer with Early Stopping.

This model serves as a performance benchmark against the custom ResNet-based CNN.

## 5. Dataset Description

Dataset: PlantVillage (public leaf disease image dataset).

Automatic dataset handling in this project:
1. Downloads dataset using `kagglehub` (if not already present).
2. Creates train/val/test split (70/15/15).
3. Stores split under `data/train`, `data/val`, `data/test`.

Data augmentation for train set:
- RandomResizedCrop
- RandomHorizontalFlip
- RandomRotation
- ColorJitter

## 6. Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 7. Training & CLI Commands

The training script `src/train.py` supports several flags to customize performance:

**Train CNN (Current Focus):**
```bash
python src/train.py --model cnn
```

**Custom CNN Options (advanced):**
- `--cnn_unfreeze_last_stage`: Fine-tunes the last layer group of the CNN.
- `--cnn_class_weights`: Adjusts loss for imbalanced disease categories.
- `--cnn_balanced_sampling`: Oversampling for infrequent disease classes.

**Quick Mode (CPU testing):**
```bash
python src/train.py --model cnn --quick
```
Runs fewer epochs with smaller 160x160 images for rapid validation.

**Full Dataset (KaggleHub download):**
```bash
python src/train.py --model cnn --use_all_classes
```

**MobileNet Support (inactive):**
```bash
python src/train.py --model mobilenet
```

**Run everything (bash wrapper):**
```bash
bash run_train.sh
```

## 8. Evaluation & Visualization

Evaluate and generate reports:
```bash
python src/evaluate.py
```

Evaluation outputs include:
- Training History (Loss/Accuracy plots)
- Confusion Matrix & Classification Report (`precision`, `recall`, `f1`)
- Model Comparison Summary (in CLI and `outputs/reports/model_comparison.csv`)
- Sample Predictions Grid (visualize predictions vs grounds truth)
- Grad-CAM heatmaps (Explainability - see where the model "looks")

## 9. Live Demo (Professor-Friendly Input -> Output)

For a quick in-class demonstration:

1. Train quickly on the existing split (small, fast run):

```bash
python src/train.py --model cnn --quick --epochs 1
```

2. Run one-image inference and print prediction output:

```bash
python src/demo_sample_io.py --model cnn
```

Optional: use a specific image path for a manual demo:

```bash
python src/demo_sample_io.py --model cnn --image data/test/Tomato_healthy/<your_image>.jpg
```

This prints:
- Input image path
- Predicted class label
- Confidence score
- Top-k prediction list

and saves a visual output to:
- `outputs/plots/demo_single_prediction.png`

## 10. Expected Terminal Logs

During training you will see logs like:

```text
Epoch 1/20
Train Loss: ...
Train Accuracy: ...
Validation Loss: ...
Validation Accuracy: ...
```

A model summary section is shown during evaluation:

```text
Model Comparison
------------------------------------------------------------
cnn: Accuracy=XX.XX% | Precision=... | Recall=... | F1=...
mobilenet: Accuracy=XX.XX% | Precision=... | Recall=... | F1=...
```

## 11. Output Artifacts

Saved files:
- `outputs/models/best_cnn.pth`
- `outputs/models/best_mobilenet.pth`
- `outputs/plots/training_curve.png`
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/sample_predictions.png`
- `outputs/plots/gradcam_mobilenet.png`
- `outputs/plots/demo_single_prediction.png`
- `outputs/reports/classification_report_*.txt`
- `outputs/reports/model_comparison.csv`

## 12. Training Procedure Summary (for Viva)

1. Prepare and split dataset (70/15/15)
2. Apply normalization + augmentation
3. Train baseline Custom CNN
4. Train transfer model (MobileNetV2)
5. Evaluate on unseen test set
6. Compare models using accuracy, precision, recall, f1
7. Use Grad-CAM to explain model attention regions

## 13. Future Improvements

- Hyperparameter tuning (learning rate schedulers, weight decay)
- Class balancing for skewed classes
- More robust augmentations (CutMix, MixUp)
- Test-time augmentation
- Model quantization for faster edge deployment
- Web app deployment (Streamlit/Flask)
