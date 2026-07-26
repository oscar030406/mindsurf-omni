#!/usr/bin/env bash
# T2A starting from a Talker that already speaks.
#
# Two rounds have failed to teach our Talker to speak from a copy of the
# Thinker's text layers. Upstream's released Talker does speak -- CER 0.0763,
# with fewer parameters than ours. Grafting it onto our Thinker and running
# T2A changes the problem from "learn to produce speech" to "learn to read this
# Thinker's hidden states", which is a much smaller job.
#
# The graft alone is silent (7 of 8 smoke samples transcribe to nothing): the
# bridge talker.embed_proj was fitted to upstream's hidden states. That is what
# this run is for.
#
# MINDSURF_TALKER_SHAPE=upstream is load-bearing. Without it the wrapper widens
# both halves to our base's shape, upstream's 20 Talker tensors mismatch, and
# the loader skips mismatched tensors in silence -- four hours training the very
# thing this run exists to replace.
#
#   setsid nohup bash ~/omni/mindsurf-omni/scripts/run_t2a_graft.sh \
#     >~/omni/t2a_graft.log 2>&1 </dev/null &
set -u

ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
LIB="${MINDSURF_LIB:-$HOME/omni/lib}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
LOG="${GRAFT_LOG:-$HOME/omni/t2a_graft.log}"
DATA_T2A="${T2A_DATA:-../dataset/sft_t2a.parquet}"
FROM="${FROM_WEIGHT:-graft_ours_thinker_up_talker}"
LR="${GRAFT_LR:-5e-5}"

cd "$ROOT/trainer" || exit 1

for required in "$ROOT/out/${FROM}_768.pth" "$ROOT/${DATA_T2A#../}" "$LIB/train_omni.py"; do
  [ -f "$required" ] || { echo "missing: $required" >&2; exit 1; }
done

running=$(pgrep -f "python.*train_omni\.py --data_path" 2>/dev/null | wc -l)
if [ "$running" -gt 0 ]; then
  echo "a training run is already going ($running processes); stop it first" >&2
  exit 1
fi

echo "===== t2a_graft lr$LR $(date -Is) =====" >>"$LOG"
setsid nohup env \
  MINIMIND_O_ROOT="$ROOT" PYTHONPATH="$LIB" PYTHONUNBUFFERED=1 \
  MINDSURF_TALKER_SHAPE=upstream \
  "$PY" "$LIB/train_omni.py" \
  --data_path "$DATA_T2A" --epochs 1 --batch_size 32 --max_seq_len 640 \
  --learning_rate "$LR" --from_weight "$FROM" --save_weight t2a_graft \
  --num_workers 8 --use_moe 0 --log_interval 50 \
  >>"$LOG" 2>&1 </dev/null &
pid=$!
echo "t2a_graft pid $pid" | tee -a "$LOG"
wait "$pid"
status=$?
echo "t2a_graft exited $status" | tee -a "$LOG"
[ "$status" -eq 0 ] || exit "$status"

# Read it here, on the same fixed 160, same judge. The gate is unchanged and was
# written before any of this: does the stage product speak -- CER out of 1.0,
# silence out of 0.6.
REPO="${MINDSURF_REPO:-$HOME/omni/mindsurf-omni}"
OUT="${GRAFT_OUT:-$HOME/omni/graft_eval}"
cd "$REPO" || exit 1
mkdir -p "$OUT"
"$PY" scripts/evaluate_talker.py \
  --checkpoint "$ROOT/out/t2a_graft_768.pth" --shape graft \
  --minimind-root "$ROOT" --audio-encoder "$ROOT/model/SenseVoiceSmall" \
  --codec "$ROOT/model/mimi" --tokenizer assets/tokenizer \
  --texts configs/talker_texts_zh_v1.jsonl \
  --output "$OUT/t2a_graft" >>"$LOG" 2>&1 || exit 1
"$PY" scripts/transcribe_samples.py --manifest "$OUT/t2a_graft/manifest.json" \
  --output "$OUT/t2a_graft.jsonl" --judge paraformer >>"$LOG" 2>&1 || exit 1
echo "rows at $OUT/t2a_graft.jsonl -- score against artifacts/codebook_baseline_mos.jsonl" | tee -a "$LOG"
