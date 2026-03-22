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

## 3. CNN Explanation (Custom Model)

The custom CNN follows this flow:
1. Convolution -> ReLU -> MaxPool
2. Convolution -> ReLU -> MaxPool
3. Convolution -> ReLU -> MaxPool
4. Flatten
5. Dense(512) + Dropout
6. Dense(num_classes)

It learns spatial disease patterns from scratch and acts as a baseline model.

## 4. Transfer Learning Explanation (MobileNetV2)

MobileNetV2 is pretrained on ImageNet and provides efficient feature extraction.

Training strategy:
1. Freeze backbone and train classification head first.
2. Unfreeze last MobileNet blocks and fine-tune with lower learning rate.

This usually converges faster and performs better than training from scratch.

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

## 7. Training Commands

Run individual model training:

```bash
python src/train.py --model cnn
python src/train.py --model mobilenet
```

Run with all PlantVillage classes:

```bash
python src/train.py --model mobilenet --use_all_classes
```

Run complete pipeline:

```bash
bash run_train.sh
```

## 8. Evaluation Command

```bash
python src/evaluate.py
```

Evaluation outputs include:
- Accuracy vs Epoch plot
- Loss vs Epoch plot
- Confusion Matrix
- Classification Report (precision, recall, f1)
- Model comparison CSV table
- Sample predictions grid (12 images)
- Grad-CAM visualizations (MobileNetV2)

## 9. Expected Terminal Logs

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

## 10. Output Artifacts

Saved files:
- `outputs/models/best_cnn.pth`
- `outputs/models/best_mobilenet.pth`
- `outputs/plots/training_curve.png`
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/sample_predictions.png`
- `outputs/plots/gradcam_mobilenet.png`
- `outputs/reports/classification_report_*.txt`
- `outputs/reports/model_comparison.csv`

## 11. Training Procedure Summary (for Viva)

1. Prepare and split dataset (70/15/15)
2. Apply normalization + augmentation
3. Train baseline Custom CNN
4. Train transfer model (MobileNetV2)
5. Evaluate on unseen test set
6. Compare models using accuracy, precision, recall, f1
7. Use Grad-CAM to explain model attention regions

## 12. Future Improvements

- Hyperparameter tuning (learning rate schedulers, weight decay)
- Class balancing for skewed classes
- More robust augmentations (CutMix, MixUp)
- Test-time augmentation
- Model quantization for faster edge deployment
- Web app deployment (Streamlit/Flask)
