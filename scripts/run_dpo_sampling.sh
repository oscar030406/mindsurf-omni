#!/usr/bin/env bash
# Four on-policy draft sets from the deliverable, for the quality DPO round.
#
# Detached with setsid: this runs for about an hour and the launching ssh
# connection will not survive it. That is not a guess, it already happened
# once tonight -- the first attempt died with the connection and left nothing,
# not even a log.
#
# PYTHONUNBUFFERED because the log is the only way to see progress, and block
# buffering makes a live run look hung for thousands of steps.
set -u
cd ~/omni/mindsurf-omni || exit 1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
R=$HOME/omni/minimind-o
OUT=$HOME/omni/dpo3
mkdir -p "$OUT"
rm -f "$OUT/DONE" "$OUT/FAILED"

for seed in 11 22 33 44; do
  target="$OUT/samples_s${seed}.json"
  if [ -s "$target" ]; then
    echo "seed $seed already on disk, skipping"
    continue
  fi
  echo "=== seed $seed starting $(date -Is)"
  ~/.venvs/omni/bin/python scripts/measure_chat_loss.py \
    --checkpoint "$R/out/sft_merge_768.pth" \
    --probes configs/preference_prompts_zh_all.jsonl \
    --tokenizer assets/tokenizer \
    --minimind-root "$R" \
    --generate --seed "$seed" \
    --temperature 0.7 --top-p 0.9 --max-tokens 512 \
    --output "$target"
  status=$?
  echo "=== seed $seed exit $status $(date -Is)"
  if [ "$status" -ne 0 ]; then
    echo "$seed" > "$OUT/FAILED"
    exit "$status"
  fi
done

touch "$OUT/DONE"
echo "ALL DONE $(date -Is)"
