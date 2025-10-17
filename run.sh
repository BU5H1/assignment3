#!/usr/bin/env bash
# ==============================================================
# Assignment 3 Reproduction Script (Linux/Mac/Windows Git Bash)
# Usage:
#   bash run.sh             # full run (GPU if available)
#   bash run.sh --small     # lightweight CPU-friendly run
# ==============================================================

set -e  # stop if any command fails

SMALL=false
if [[ "$1" == "--small" ]]; then
  SMALL=true
fi

echo ">>> Starting Assignment 3 pipeline (small run = $SMALL)"
echo

# 1. Full fine-tuning
if [ "$SMALL" = true ]; then
  python train_full.py --model_name distilbert-base-uncased --out results/distilbert_full \
    --epochs 1 --batch 8 --max_len 96 --lr 3e-5
else
  python train_full.py --model_name distilbert-base-uncased --out results/distilbert_full
fi

# 2. Linear probe (frozen encoder)
if [ "$SMALL" = true ]; then
  python train_head.py --model_name distilbert-base-uncased --out results/distilbert_head \
    --epochs 1 --batch 16 --max_len 96 --lr 1e-3
else
  python train_head.py --model_name distilbert-base-uncased --out results/distilbert_head
fi

# 3. Evaluate both models
python evaluate.py --ckpt results/distilbert_full --split test
python evaluate.py --ckpt results/distilbert_head --split test

# 4. Plot comparison
python scripts/plot_compare.py --roots results/distilbert_full results/distilbert_head \
  --out results/compare.png

echo
echo ">>> All steps completed!"
echo "Metrics: results/*/test_metrics.json"
echo "Plot:    results/compare.png"
