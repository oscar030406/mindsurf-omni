#!/usr/bin/env bash
# 标注器换成新标签重训，然后在大留出集上扫阈值。
#
# 上一版标注器学的是 pairs_v2 的标签，那里面 191 个删除标签是标点、
# 17.7% 的数字字符被标成该删。两样都是判据看不见、产品不想要的。
# pairs_v3 把它们去掉了，查表判官在召回 0.70 处从 0.664 涨到 0.863——
# 这一轮看真模型跟不跟得上。
#
# 打分在 986 句大留出集上（没有任何 checkpoint 见过），不在 156 句上：
# 156 句分不开 0.003–0.006，而标注器几条阈值臂之间正是这个量级。
#
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/run_polish_tagger_v3.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
PAIRS=${PAIRS:-artifacts/polish_train/pairs_v3.jsonl}
HOLDOUT=${HOLDOUT:-artifacts/polish_train/pairs_holdout.jsonl}
BASE=${BASE:-D:/environment/models/mindsurf-local/sft_merge_768.pth}
ROOT=${ROOT:-D:/environment/models/minimind-o-repo}
DEVICE=${DEVICE:-cuda}
HEAD=${HEAD:-D:/environment/models/mindsurf-local/polish_tagger_v3.pt}
BACKBONE=${BACKBONE:-D:/environment/models/mindsurf-local/polish_tagger_v3_backbone.pth}

echo "== 1/2 训练标注器（解冻 3 层，和上一版同配置）=="
"$PYTHON" -u scripts/train_polish_tagger.py --checkpoint "$BASE" --pairs "$PAIRS" \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --unfreeze 3 --epochs 3 \
    --output "$HEAD" --backbone-output "$BACKBONE" \
    --report artifacts/polish-tagger-v3-2026-08-15.json

echo "== 2/2 在大留出集上扫阈值 =="
for THRESHOLD in 0.5 0.8 0.9 0.95 0.99; do
    echo "-- t=${THRESHOLD}"
    "$PYTHON" -u scripts/measure_polish.py --checkpoint "$BACKBONE" \
        --pairs "$HOLDOUT" --split val \
        --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
        --tagger "$HEAD" --tagger-threshold "$THRESHOLD" \
        --output "artifacts/polish_train/val_bigholdout_taggerv3_t${THRESHOLD}.jsonl" \
        --report "artifacts/polish-eval-bigholdout-taggerv3-t${THRESHOLD}.json"
done

echo "== 完 =="
