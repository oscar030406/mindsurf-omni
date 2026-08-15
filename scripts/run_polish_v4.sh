#!/usr/bin/env bash
# 训练池里去掉不是口述的那 7.5%，重训一版。
#
# 残余的数字删除追到这里：注入器的重复臂拿从句头两个字，从句是「1. 准备好…」时
# 就复制成「1.1. 准备好…」，转写回来「1.1」，目标只留一个，于是数字被标成该删。
# 这类文本（编号列表、多行提示）本来就没人口述，训练池里占 7.5%。
#
# 过滤是在已有对子上取子集，不重新合成——只花训练时间。
#
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/run_polish_v4.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
POOL=${POOL:-artifacts/polish_train/pool.jsonl}
PAIRS=${PAIRS:-artifacts/polish_train/pairs_v4.jsonl}
HOLDOUT=${HOLDOUT:-artifacts/polish_train/pairs_holdout.jsonl}
BASE=${BASE:-D:/environment/models/mindsurf-local/sft_merge_768.pth}
OUT=${OUT:-D:/environment/models/mindsurf-local/sft_polish7_768.pth}
ROOT=${ROOT:-D:/environment/models/minimind-o-repo}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-5}

echo "== 1/3 过滤训练对子（按池子原文判语域）=="
"$PYTHON" -u scripts/filter_dictation_register.py --rows artifacts/polish_train/pairs_v3.jsonl \
    --pool "$POOL" --by-pool-text --output "$PAIRS"

echo "== 2/3 训练 =="
"$PYTHON" -u scripts/train_polish.py --checkpoint "$BASE" --pairs "$PAIRS" \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer \
    --epochs "$EPOCHS" --device "$DEVICE" \
    --output "$OUT" --report artifacts/polish-train-v4.json

echo "== 3/3 在大留出集上验收（复制约束窗口 6）=="
"$PYTHON" -u scripts/measure_polish.py --checkpoint "$OUT" --pairs "$HOLDOUT" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --copy-only --copy-lookahead 6 \
    --output artifacts/polish_train/val_bigholdout_polish7.jsonl \
    --report artifacts/polish-eval-bigholdout-polish7.json

echo "== 完 =="
