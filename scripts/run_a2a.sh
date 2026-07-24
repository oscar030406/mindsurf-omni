#!/usr/bin/env bash
# A2A after T2A: bring speech input into the same Thinker-Talker reply path.
#
# Two stages on purpose. The projector aligns first, on its own, so the audio
# encoder's output lands in a space the Thinker already understands; only then
# is everything unfrozen. Doing it in one step lets a badly-aligned projector
# drag the language weights with it, and the loss curve looks fine while it
# happens.
#
# Detached with setsid, because a dropped ssh must not take the run with it --
# that has already happened once here.
set -u

ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
LIB="${MINDSURF_LIB:-$HOME/omni/lib}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
LOG="${A2A_LOG:-$HOME/omni/a2a_full.log}"
DATA="${A2A_DATA:-../dataset/sft_a2a.parquet}"
FROM="${FROM_WEIGHT:-sft_mindsurf}"

cd "$ROOT/trainer" || exit 1

# Match a real python process running the launcher, not any command line that
# merely mentions it. A bare `pgrep -f` substring match has already produced a
# false positive here twice: once matching an ssh command that carried the
# path as an argument, once matching a dry run still winding down.
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

# Stage one: align the audio projector alone.
run "a2a_proj" \
  --data_path "$DATA" --epochs 1 --batch_size 24 --max_seq_len 640 \
  --learning_rate 5e-4 --from_weight "$FROM" --save_weight sft_mindsurf \
  --mode audio_proj --num_workers 8 --use_moe 0 --log_interval 50

# Stage two: unfreeze, at a much lower rate. 2e-5 rather than 5e-4 because the
# Thinker arrives already trained and a large step here unlearns it.
run "a2a_full" \
  --data_path "$DATA" --epochs 3 --batch_size 16 --max_seq_len 768 \
  --learning_rate 2e-5 --from_weight sft_mindsurf --save_weight sft_mindsurf \
  --num_workers 8 --use_moe 0 --log_interval 50

echo "===== a2a done $(date -Is) =====" >>"$LOG"
