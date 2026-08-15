#!/usr/bin/env bash
# 润色后训练：造好的对子 → 训练 → 按判据验收。
# 判据写死在 scripts/measure_polish.py 顶部，训练之前就定好了。
#
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/run_polish_training.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
PAIRS=${PAIRS:-artifacts/polish_train/pairs.jsonl}
BASE=${BASE:-D:/environment/models/mindsurf-local/sft_merge_768.pth}
ROOT=${ROOT:-D:/environment/models/minimind-o-repo}
OUT=${OUT:-D:/environment/models/mindsurf-local/sft_polish_768.pth}
STAMP=${STAMP:-2026-08-15}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-5}
DEVICE=${DEVICE:-cuda}

echo "== 1/3 训练 =="
"$PYTHON" scripts/train_polish.py --checkpoint "$BASE" --pairs "$PAIRS" \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer \
    --epochs "$EPOCHS" --learning-rate "$LR" --device "$DEVICE" \
    --output "$OUT" --report "artifacts/polish-train-${STAMP}.json"

echo "== 2/3 验收（留出集，判据在脚本顶部）=="
"$PYTHON" scripts/measure_polish.py --checkpoint "$OUT" --pairs "$PAIRS" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --output "artifacts/polish_train/val_polished.jsonl" \
    --report "artifacts/polish-eval-${STAMP}.json"

echo "== 3/3 对照：训练前的同一批留出集 =="
# 没有这一臂，「润色后 CER 0.0x」读不出是训练买来的还是模型本来就会。
"$PYTHON" scripts/measure_polish.py --checkpoint "$BASE" --pairs "$PAIRS" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --output "artifacts/polish_train/val_baseline.jsonl" \
    --report "artifacts/polish-eval-baseline-${STAMP}.json"
