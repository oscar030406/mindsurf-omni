#!/usr/bin/env bash
# 第二轮数据：口语词不再只落在从句边界。
#
# 第一轮的注入器把每个口语词放在从句前面，模型学到的是「边界」而不是「词」——
# 留出集上句首残留 0.054、句中残留 0.214，差 4 倍。这一轮的注入器按 jieba 的
# 词边界往从句内部也放，判据不动，产物记为 sft_polish4。
#
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/run_polish_round2.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
POOL=${POOL:-artifacts/polish_train/pool.jsonl}
PAIRS=${PAIRS:-artifacts/polish_train/pairs_v2.jsonl}
BASE=${BASE:-D:/environment/models/mindsurf-local/sft_merge_768.pth}
OUT=${OUT:-D:/environment/models/mindsurf-local/sft_polish4_768.pth}
ROOT=${ROOT:-D:/environment/models/minimind-o-repo}
ASR_DIR=${ASR_DIR:-D:/environment/models/mindsurf-local/SenseVoiceSmall}
DEVICE=${DEVICE:-cuda}
STAMP=${STAMP:-2026-08-15-r2}
EPOCHS=${EPOCHS:-5}

echo "== 1/4 造对子（新注入器：从句内部也放）=="
"$PYTHON" scripts/build_polish_pairs.py --texts "$POOL" --asr-dir "$ASR_DIR" \
    --device "$DEVICE" --output "$PAIRS" --batch 64 --concurrency 8

echo "== 2/4 训练 =="
"$PYTHON" scripts/train_polish.py --checkpoint "$BASE" --pairs "$PAIRS" \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer \
    --epochs "$EPOCHS" --device "$DEVICE" \
    --output "$OUT" --report "artifacts/polish-train-${STAMP}.json"

echo "== 3/4 验收：复制约束（窗口 6）=="
"$PYTHON" scripts/measure_polish.py --checkpoint "$OUT" --pairs "$PAIRS" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --copy-only --copy-lookahead 6 \
    --output "artifacts/polish_train/val_${STAMP}_copy6.jsonl" \
    --report "artifacts/polish-eval-${STAMP}-copy6.json"

echo "== 4/4 验收：自由解码 + 投影修复 =="
"$PYTHON" scripts/measure_polish.py --checkpoint "$OUT" --pairs "$PAIRS" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --repetition-penalty 1.0 --project \
    --output "artifacts/polish_train/val_${STAMP}_project.jsonl" \
    --report "artifacts/polish-eval-${STAMP}-project.json"
