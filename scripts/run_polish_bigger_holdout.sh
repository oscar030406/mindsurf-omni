#!/usr/bin/env bash
# 把留出集做大，然后拿它重读已经在磁盘上的两条臂。
#
# 四个数一直读在 156 句上，而 156 句的内容保留 95% 区间半宽是 ±0.009——
# 配置表却在按 0.003–0.006 排名次。这一轮不训模型，只换尺子。
#
# 关键是这 988 句**没有任何一个 checkpoint 见过**（按正文对训练池去重），
# 所以 sft_polish5 和 sft_polish6 可以直接在上面重读，不用为了扩留出集重训——
# 重训会把被测的东西一起改掉。
#
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/run_polish_bigger_holdout.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
POOL=${POOL:-artifacts/polish_train/pool.jsonl}
HOLDOUT_POOL=${HOLDOUT_POOL:-artifacts/polish_train/pool_holdout.jsonl}
PAIRS=${PAIRS:-artifacts/polish_train/pairs_holdout.jsonl}
ROOT=${ROOT:-D:/environment/models/minimind-o-repo}
ASR_DIR=${ASR_DIR:-D:/environment/models/mindsurf-local/SenseVoiceSmall}
DEVICE=${DEVICE:-cuda}
OLD=${OLD:-D:/environment/models/mindsurf-local/sft_polish5_768.pth}
NEW=${NEW:-D:/environment/models/mindsurf-local/sft_polish6_768.pth}

echo "== 1/4 攒没训练过的句子 =="
"$PYTHON" -u scripts/build_polish_holdout_pool.py --pool "$POOL" --output "$HOLDOUT_POOL"

echo "== 2/4 造对子（全部标 val）=="
"$PYTHON" -u scripts/build_polish_pairs.py --texts "$HOLDOUT_POOL" --asr-dir "$ASR_DIR" \
    --device "$DEVICE" --output "$PAIRS" --batch 64 --concurrency 8 --holdout 1.0

# 两条臂都用复制约束窗口 6——那是前九轮里最好的解码设置，两边一样才比得了。
echo "== 3/4 重读旧目标那条臂（sft_polish5）=="
"$PYTHON" -u scripts/measure_polish.py --checkpoint "$OLD" --pairs "$PAIRS" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --copy-only --copy-lookahead 6 \
    --output artifacts/polish_train/val_bigholdout_polish5.jsonl \
    --report artifacts/polish-eval-bigholdout-polish5.json

echo "== 4/4 重读新目标那条臂（sft_polish6）=="
"$PYTHON" -u scripts/measure_polish.py --checkpoint "$NEW" --pairs "$PAIRS" --split val \
    --minimind-root "$ROOT" --tokenizer assets/tokenizer --device "$DEVICE" \
    --copy-only --copy-lookahead 6 \
    --output artifacts/polish_train/val_bigholdout_polish6.jsonl \
    --report artifacts/polish-eval-bigholdout-polish6.json

echo "== 配对比一比 =="
"$PYTHON" -u scripts/compare_polish_arms.py \
    --before artifacts/polish_train/val_bigholdout_polish5.jsonl \
    --after artifacts/polish_train/val_bigholdout_polish6.jsonl

echo "== 完 =="
