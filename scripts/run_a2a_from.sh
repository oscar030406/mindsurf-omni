#!/usr/bin/env bash
# The two A2A stages, starting from a T2A product that already speaks.
#
# The retrain's chain is not re-runnable as a whole once its first stage is
# replaced: the probe produces a different T2A product, and everything after it
# is the same recipe. So this is run_full_recipe.sh with the first stage cut
# out and the starting weight made an argument.
#
# The two A2A learning rates are the report's, unscaled, on purpose. Our base's
# weights are 5.2x larger per parameter than upstream's, so the same AdamW rate
# moves them a fifth as far in relative terms -- but the round that produced
# speech ran a2a_full at 2e-5 on this same base and got there, so A2A is not the
# stage the scale argument indicts. T2A is. See
# docs/experiments/2026-07-26-weight-scale.md.
#
# Only to be run when the probe's stage product has passed its own gate --
# CER out of 1.0, silence out of 0.6 on the fixed 160. Starting the seven-hour
# tail from a Talker that does not speak is what this round already cost.
#
#   setsid nohup bash ~/omni/mindsurf-omni/scripts/run_a2a_from.sh t2a_lr5e5 \
#     >~/omni/a2a_from.log 2>&1 </dev/null &
set -u

FROM="${1:?give the starting weight name, e.g. t2a_lr5e5}"
SAVE="${2:-sft_${FROM}}"
ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
LIB="${MINDSURF_LIB:-$HOME/omni/lib}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
LOG="${A2A_LOG:-$HOME/omni/a2a_from.log}"
DATA_A2A="${A2A_DATA:-../dataset/sft_a2a.parquet}"

cd "$ROOT/trainer" || exit 1

for required in "$ROOT/out/${FROM}_768.pth" "$ROOT/${DATA_A2A#../}" "$LIB/train_omni.py"; do
  if [ ! -f "$required" ]; then
    echo "missing: $required" >&2
    exit 1
  fi
done

running=$(pgrep -f "python.*train_omni\.py --data_path" 2>/dev/null | wc -l)
if [ "$running" -gt 0 ]; then
  echo "a training run is already going ($running processes); stop it first" >&2
  exit 1
fi

# Keep the starting point. The A2A stages write over their own product, and
# without a copy the next failure cannot be attributed to a stage -- the
# lesson this round was archived for.
cp "$ROOT/out/${FROM}_768.pth" "$ROOT/out/${FROM}_768.keep.pth" 2>/dev/null \
  && echo "kept ${FROM}" >>"$LOG"

run() {
  local stage=$1
  shift
  echo "===== $stage $(date -Is) =====" >>"$LOG"
  setsid nohup env \
    MINIMIND_O_ROOT="$ROOT" PYTHONPATH="$LIB" PYTHONUNBUFFERED=1 \
    "$PY" "$LIB/train_omni.py" "$@" >>"$LOG" 2>&1 </dev/null &
  local pid=$!
  echo "$stage pid $pid" | tee -a "$LOG"
  wait "$pid"
  local status=$?
  echo "$stage exited $status" | tee -a "$LOG"
  [ "$status" -eq 0 ] || exit "$status"
}

run "a2a_proj" \
  --data_path "$DATA_A2A" --epochs 1 --batch_size 24 --max_seq_len 640 \
  --learning_rate 5e-4 --from_weight "$FROM" --save_weight "$SAVE" \
  --mode audio_proj --num_workers 8 --use_moe 0 --log_interval 50

run "a2a_full" \
  --data_path "$DATA_A2A" --epochs 3 --batch_size 16 --max_seq_len 768 \
  --learning_rate 5e-5 --from_weight "$SAVE" --save_weight "$SAVE" \
  --num_workers 8 --use_moe 0 --log_interval 50

echo "===== a2a from $FROM done $(date -Is) =====" >>"$LOG"

# Read it here, same as the probe does, so the answer does not wait for a human.
REPO="${MINDSURF_REPO:-$HOME/omni/mindsurf-omni}"
OUT="${A2A_OUT:-$HOME/omni/a2a_from_eval}"
cd "$REPO" || exit 1
mkdir -p "$OUT"
"$PY" scripts/evaluate_talker.py \
  --checkpoint "$ROOT/out/${SAVE}_768.pth" --shape mindsurf \
  --minimind-root "$ROOT" --audio-encoder "$ROOT/model/SenseVoiceSmall" \
  --codec "$ROOT/model/mimi" --tokenizer assets/tokenizer \
  --texts configs/talker_texts_zh_v1.jsonl \
  --output "$OUT/$SAVE" >>"$LOG" 2>&1 || exit 1
"$PY" scripts/transcribe_samples.py --manifest "$OUT/$SAVE/manifest.json" \
  --output "$OUT/$SAVE.jsonl" --judge paraformer >>"$LOG" 2>&1 || exit 1
echo "rows at $OUT/$SAVE.jsonl -- score against artifacts/codebook_baseline_mos.jsonl" | tee -a "$LOG"
