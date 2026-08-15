#!/usr/bin/env bash
# 第三轮：不动合成，只换训练目标。
#
# 目标从「语料原文」换成「转写 - 注入段」，于是润色器只被要求删口语词，不再被要求
# 改识别错、也不再被要求把标点改回原文那一套。后两件都不在判据里
# （normalise_for_cer 两边都剥标点），但它们在留出集上供了 659 个删除标签里的 191 个，
# 而且每一个都歧义——逗号删 98 次留 253 次。
#
# 打分仍然对 pairs_v2 的 val（同 id、同 split、同 source、原目标），
# 所以这一轮的四个数和前九个配置那张表可以并排读。
#
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/run_polish_retarget.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
POOL=${POOL:-artifacts/polish_train/pool.jsonl}
BASE_PAIRS=${BASE_PAIRS:-artifacts/polish_train/pairs_v2.jsonl}
PAIRS=${PAIRS:-artifacts/polish_train/pairs_v3.jsonl}
BASE=${BASE:-D:/environment/models/mindsurf-local/sft_merge_768.pth}
OUT=${OUT:-D:/environment/models/mindsurf-local/sft_polish6_768.pth}
ROOT=${ROOT:-D:/environment/models/minimind-o-repo}
DEVICE=${DEVICE:-cuda}
STAMP=${STAMP:-retarget}
EPOCHS=${EPOCHS:-5}

echo "== 1/4 换目标（不重新合成）=="
"$PYTHON" -u scripts/retarget_polish_pairs.py --pairs "$BASE_PAIRS" --pool "$POOL" \
    --output "$PAIRS"

echo "== 2/4 训练 =="
"$PYTHON" -u scripts/train_polish.py --checkpoint "$BASE" --pairs "$PAIRS" \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer \
    --epochs "$EPOCHS" --device "$DEVICE" \
    --output "$OUT" --report "artifacts/polish-train-${STAMP}.json"

# 打分对 BASE_PAIRS，不对 PAIRS：判据和前九个配置必须是同一把尺子。
echo "== 3/4 验收：复制约束（窗口 6）=="
"$PYTHON" -u scripts/measure_polish.py --checkpoint "$OUT" --pairs "$BASE_PAIRS" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --copy-only --copy-lookahead 6 \
    --output "artifacts/polish_train/val_${STAMP}_copy6.jsonl" \
    --report "artifacts/polish-eval-${STAMP}-copy6.json"

echo "== 4/4 验收：自由解码 + 投影修复 =="
"$PYTHON" -u scripts/measure_polish.py --checkpoint "$OUT" --pairs "$BASE_PAIRS" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --repetition-penalty 1.0 --project \
    --output "artifacts/polish_train/val_${STAMP}_project.jsonl" \
    --report "artifacts/polish-eval-${STAMP}-project.json"

echo "== 完 =="
