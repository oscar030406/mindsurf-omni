#!/usr/bin/env bash
# 数据量曲线：25 / 50 / 75 / 100% 的训练对子，同一批留出集上读四个点。
#
# 上一轮的结论是「加数据量买到的是 0.005 量级，买不到那 0.006」——那是在 156 句上量的，
# 而 156 句分不开 0.005。在 986 句上，砍掉 7.5% 的对子要付 0.027 的口语词清除。
# 所以这条曲线得重画：它决定了「再找一批语料」值不值。
#
# 100% 那个点就是 sft_polish6，已经在磁盘上，不重跑。
#
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/run_polish_data_curve.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
HOLDOUT=${HOLDOUT:-artifacts/polish_train/pairs_holdout.jsonl}
BASE=${BASE:-D:/environment/models/mindsurf-local/sft_merge_768.pth}
ROOT=${ROOT:-D:/environment/models/minimind-o-repo}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-5}

for FRACTION in 0.25 0.5 0.75; do
    TAG="frac${FRACTION}"
    OUT="D:/environment/models/mindsurf-local/sft_polish_${TAG}_768.pth"
    echo "== ${FRACTION} 训练 =="
    "$PYTHON" -u scripts/train_polish.py --checkpoint "$BASE" \
        --pairs "artifacts/polish_train/pairs_v3_${FRACTION}.jsonl" \
        --minimind-root "$ROOT" --tokenizer assets/tokenizer \
        --epochs "$EPOCHS" --device "$DEVICE" \
        --output "$OUT" --report "artifacts/polish-train-${TAG}.json"

    echo "== ${FRACTION} 验收 =="
    "$PYTHON" -u scripts/measure_polish.py --checkpoint "$OUT" --pairs "$HOLDOUT" --split val \
        --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
        --copy-only --copy-lookahead 6 \
        --output "artifacts/polish_train/val_bigholdout_${TAG}.jsonl" \
        --report "artifacts/polish-eval-bigholdout-${TAG}.json"
    rm -f "$OUT"   # 曲线要的是读数不是权重，四份 456 MB 不留
done

echo "== 完 =="
