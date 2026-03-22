#!/usr/bin/env bash
set -e

echo "Training Custom CNN..."
python src/train.py --model cnn --epochs 20 --batch_size 32 --lr 0.0003

# echo "Training MobileNetV2..."
# python src/train.py --model mobilenet --epochs 20 --batch_size 32 --lr 0.0001

echo "Running evaluation..."
python src/evaluate.py --models cnn

echo "Done. Check outputs/ for models, plots, and reports."
