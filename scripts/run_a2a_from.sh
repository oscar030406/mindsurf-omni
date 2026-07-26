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
# The third argument is the shape, and it is one variable rather than two on
# purpose: a grafted checkpoint needs the Talker left at upstream's shape both
# while training and while being read, and two separate knobs is two chances to
# set only one of them. train_omni.py now refuses a checkpoint it can only half
# load, so getting this wrong costs a minute rather than seven hours.
#
# Three stages now, not two, and at upstream's sequence length. Its train.sh
# runs A2A at --max_seq_len 1024 and we were passing 640 and 768, which drops
# the target Mimi codes past the cap on 16.9% and 7.5% of samples -- the ends
# of the longest utterances, never supervised, no warning in the log. And its
# line 6 is a third A2A pass at 5e-6 that we had never run at all. Neither is
# expected to fix anything: upstream's own T2A truncates twice as hard as ours
# and scores four times better. They are known deviations being removed. See
# docs/experiments/2026-07-26-sequence-truncation.md.
#
# A2A_SEQ and the batch sizes are variables because 1024 has not been run on
# this card. If the first stage dies on memory, lower PROJ_BS / FULL_BS rather
# than the sequence length -- the batch is the axis we already differ from
# upstream on (one GPU against four), the sequence length is the one we are
# aligning.
#
# SKIP_PROJ=1 starts at the full stage. The projector pass belongs at the top
# of a chain; a model that has already had one gets its projector moved a long
# way by a second 5e-4 epoch.
#
#   setsid nohup bash ~/omni/mindsurf-omni/scripts/run_a2a_from.sh t2a_lr5e5 \
#     >~/omni/a2a_from.log 2>&1 </dev/null &
#   setsid nohup bash ~/omni/mindsurf-omni/scripts/run_a2a_from.sh \
#     t2a_graft sft_graft graft >~/omni/a2a_from.log 2>&1 </dev/null &
#   SKIP_PROJ=1 setsid nohup bash ~/omni/mindsurf-omni/scripts/run_a2a_from.sh \
#     sft_mindsurf sft_mindsurf_tail >~/omni/a2a_from.log 2>&1 </dev/null &
set -u

FROM="${1:?give the starting weight name, e.g. t2a_lr5e5}"
SAVE="${2:-sft_${FROM}}"
SHAPE="${3:-mindsurf}"
case "$SHAPE" in
  mindsurf) TALKER_SHAPE="" ;;
  graft) TALKER_SHAPE="upstream" ;;
  *) echo "shape must be 'mindsurf' or 'graft', not '$SHAPE'" >&2; exit 1 ;;
esac
ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
LIB="${MINDSURF_LIB:-$HOME/omni/lib}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
LOG="${A2A_LOG:-$HOME/omni/a2a_from.log}"
DATA_A2A="${A2A_DATA:-../dataset/sft_a2a.parquet}"
A2A_SEQ="${A2A_SEQ:-1024}"   # upstream's value; ours were 640 and 768
PROJ_BS="${PROJ_BS:-16}"
FULL_BS="${FULL_BS:-10}"
SKIP_PROJ="${SKIP_PROJ:-0}"

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
    MINDSURF_TALKER_SHAPE="$TALKER_SHAPE" \
    "$PY" "$LIB/train_omni.py" "$@" >>"$LOG" 2>&1 </dev/null &
  local pid=$!
  echo "$stage pid $pid" | tee -a "$LOG"
  wait "$pid"
  local status=$?
  echo "$stage exited $status" | tee -a "$LOG"
  [ "$status" -eq 0 ] || exit "$status"
}

echo "seq $A2A_SEQ, proj bs $PROJ_BS, full bs $FULL_BS, skip_proj $SKIP_PROJ" | tee -a "$LOG"

# Upstream line 2.
if [ "$SKIP_PROJ" -eq 0 ]; then
  run "a2a_proj" \
    --data_path "$DATA_A2A" --epochs 1 --batch_size "$PROJ_BS" --max_seq_len "$A2A_SEQ" \
    --learning_rate 5e-4 --from_weight "$FROM" --save_weight "$SAVE" \
    --mode audio_proj --num_workers 8 --use_moe 0 --log_interval 50
  NEXT_FROM="$SAVE"
else
  echo "skipping a2a_proj: $FROM has had one" | tee -a "$LOG"
  NEXT_FROM="$FROM"
fi

# Upstream line 3.
run "a2a_full" \
  --data_path "$DATA_A2A" --epochs 3 --batch_size "$FULL_BS" --max_seq_len "$A2A_SEQ" \
  --learning_rate 5e-5 --from_weight "$NEXT_FROM" --save_weight "$SAVE" \
  --num_workers 8 --use_moe 0 --log_interval 50

# Upstream line 6, which we had never run. In its pipeline the two I2T passes
# sit between this and the one above; we skip vision, so this follows directly.
# That ordering difference is itself a deviation and is recorded as one.
run "a2a_tail" \
  --data_path "$DATA_A2A" --epochs 1 --batch_size "$FULL_BS" --max_seq_len "$A2A_SEQ" \
  --learning_rate 5e-6 --from_weight "$SAVE" --save_weight "$SAVE" \
  --num_workers 8 --use_moe 0 --log_interval 50

echo "===== a2a from $FROM done $(date -Is) =====" >>"$LOG"

# Read it here, same as the probe does, so the answer does not wait for a human.
REPO="${MINDSURF_REPO:-$HOME/omni/mindsurf-omni}"
OUT="${A2A_OUT:-$HOME/omni/a2a_from_eval}"
cd "$REPO" || exit 1
mkdir -p "$OUT"
"$PY" scripts/evaluate_talker.py \
  --checkpoint "$ROOT/out/${SAVE}_768.pth" --shape "$SHAPE" \
  --minimind-root "$ROOT" --audio-encoder "$ROOT/model/SenseVoiceSmall" \
  --codec "$ROOT/model/mimi" --tokenizer assets/tokenizer \
  --texts configs/talker_texts_zh_v1.jsonl \
  --output "$OUT/$SAVE" >>"$LOG" 2>&1 || exit 1
"$PY" scripts/transcribe_samples.py --manifest "$OUT/$SAVE/manifest.json" \
  --output "$OUT/$SAVE.jsonl" --judge paraformer >>"$LOG" 2>&1 || exit 1
echo "rows at $OUT/$SAVE.jsonl -- score against artifacts/codebook_baseline_mos.jsonl" | tee -a "$LOG"
