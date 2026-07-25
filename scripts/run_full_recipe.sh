#!/usr/bin/env bash
# The whole curriculum, at the hyperparameters upstream's release was actually
# trained with -- every one explicit, because the last run lost 4x speech
# accuracy to an implicit default.
#
# What happened: the T2A pass ran `--epochs 6 --from_weight llm` with no
# --learning_rate, so it silently took the trainer's default of 5e-4. The
# technical report trains T2A at 5e-6 for one epoch -- a hundred times lower.
# Six epochs at that rate drove the Talker into a basin the A2A stage could not
# pull it out of: on the same 160 spoken probes with the same paraformer judge,
# our checkpoint spoke at CER 0.2172 while upstream's release -- same
# architecture, same data, a *smaller* Thinker -- spoke at 0.0579. The loss
# curve never showed it. It plateaued smoothly, at a bad optimum.
#
# Recipe per the MiniMind-O technical report (arXiv:2605.03937):
#   T2A       full model   1 epoch   lr 5e-6
#   A2A proj  projector    1 epoch   lr 5e-4
#   A2A full  full model   3 epochs  lr 5e-5
# One variable differs from upstream: the base checkpoint is ours.
#
# Launch it detached, because the chain must outlive the ssh session that
# started it -- a dropped connection has already killed a run here:
#
#   setsid nohup bash /tmp/run_full_recipe.sh >/dev/null 2>&1 </dev/null &
set -u

ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
LIB="${MINDSURF_LIB:-$HOME/omni/lib}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
LOG="${RETRAIN_LOG:-$HOME/omni/retrain.log}"
DATA_T2A="${T2A_DATA:-../dataset/sft_t2a.parquet}"
DATA_A2A="${A2A_DATA:-../dataset/sft_a2a.parquet}"
BASE="${BASE_WEIGHT:-llm_mindsurf}"

cd "$ROOT/trainer" || exit 1

# Every input named and checked before anything occupies the GPU for ten hours.
# `--from_weight X` resolves to out/X_768.pth inside the trainer; a name that
# does not resolve is loaded as nothing and training proceeds from random
# weights, which is the same class of silent fault this rerun exists to fix.
for required in "$ROOT/out/${BASE}_768.pth" "$ROOT/${DATA_T2A#../}" "$ROOT/${DATA_A2A#../}" "$LIB/train_omni.py"; do
  if [ ! -f "$required" ]; then
    echo "missing: $required" >&2
    exit 1
  fi
done

# Match a real python process running the launcher, not any command line that
# merely mentions it. A bare `pgrep -f` substring match has produced a false
# positive here twice.
running=$(pgrep -f "python.*train_omni\.py --data_path" 2>/dev/null | wc -l)
if [ "$running" -gt 0 ]; then
  echo "a training run is already going ($running processes); stop it first" >&2
  ps -eo pid,etime,cmd | grep "train_omni\.py --data_path" | grep -v grep >&2
  exit 1
fi

# PYTHONUNBUFFERED so the log is readable while it runs. Without it, print()
# block-buffers into a file and the log lags thousands of steps behind.
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
  # A failed stage must not silently roll into the next: the second would train
  # from whatever the first left behind.
  [ "$status" -eq 0 ] || exit "$status"
}

# Stage one: T2A, the report's single epoch at 5e-6. Stated, not defaulted.
run "t2a_full" \
  --data_path "$DATA_T2A" --epochs 1 --batch_size 32 --max_seq_len 640 \
  --learning_rate 5e-6 --from_weight "$BASE" --save_weight t2a_mindsurf \
  --num_workers 8 --use_moe 0 --log_interval 50

# Keep it. Last time the A2A stages wrote over the T2A product, so when the
# result came out wrong there was nothing left to attribute the damage to a
# stage. A copy costs 300 MB and buys the next diagnosis.
cp "$ROOT/out/t2a_mindsurf_768.pth" "$ROOT/out/t2a_mindsurf_768.keep.pth" \
  && echo "kept t2a checkpoint" >>"$LOG"

# Stage two: align the audio projector alone, so the encoder's output lands in
# a space the Thinker already understands before anything else moves.
run "a2a_proj" \
  --data_path "$DATA_A2A" --epochs 1 --batch_size 24 --max_seq_len 640 \
  --learning_rate 5e-4 --from_weight t2a_mindsurf --save_weight sft_mindsurf2 \
  --mode audio_proj --num_workers 8 --use_moe 0 --log_interval 50

# Stage three: everything unfrozen at the report's 5e-5 -- not the 2e-5 the
# last run used, which was a guess in the safe direction and 2.5x too low.
run "a2a_full" \
  --data_path "$DATA_A2A" --epochs 3 --batch_size 16 --max_seq_len 768 \
  --learning_rate 5e-5 --from_weight sft_mindsurf2 --save_weight sft_mindsurf2 \
  --num_workers 8 --use_moe 0 --log_interval 50

echo "===== retrain done $(date -Is) =====" >>"$LOG"
