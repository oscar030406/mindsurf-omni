#!/usr/bin/env bash
# The complete curriculum, at the hyperparameters upstream's release was
# actually trained with -- every one explicit, because the last run lost 4x
# speech accuracy to an implicit default.
#
# What happened: our T2A pass ran `--epochs 6 --from_weight llm` with no
# --learning_rate, which silently took the trainer's default of 5e-4. The
# technical report trains T2A at 5e-6 -- one hundred times lower -- for one
# epoch. Six epochs at a hundred times the learning rate drove the Talker into
# a basin the A2A stage could not pull it out of: measured on the same 160
# spoken probes with the same paraformer judge, our checkpoint spoke at CER
# 0.2172 while upstream's release, same architecture and data, spoke at 0.0579.
# The loss curve never showed it -- it plateaued smoothly, at a bad optimum.
#
# Recipe per the MiniMind-O technical report (arXiv:2605.03937):
#   T2A       full model   1 epoch   lr 5e-6
#   A2A proj  projector    1 epoch   lr 5e-4
#   A2A full  full model   3 epochs  lr 5e-5
# One variable differs from upstream: the base checkpoint is ours.

set -euo pipefail

LOG="$HOME/omni/retrain.log"
DATA_T2A="../dataset/sft_t2a.parquet"
DATA_A2A="../dataset/sft_a2a.parquet"

run() {
  local stage="$1"; shift
  echo "===== $stage $(date -Is) =====" >>"$LOG"
  PYTHONUNBUFFERED=1 PYTHONPATH="$HOME/omni/lib:$HOME/omni/minimind-o" \
    "$HOME/.venvs/omni/bin/python" "$HOME/omni/lib/train_omni.py" "$@" >>"$LOG" 2>&1
  echo "$stage exited $?" >>"$LOG"
}

# Stage one: T2A, the report's one epoch at 5e-6. Explicit, not defaulted.
run "t2a_full" \
  --data_path "$DATA_T2A" --epochs 1 --batch_size 32 --max_seq_len 640 \
  --learning_rate 5e-6 --from_weight llm_mindsurf --save_weight t2a_mindsurf \
  --num_workers 8 --use_moe 0 --log_interval 50

# Keep the T2A checkpoint: last run overwrote it with the A2A stages, which
# left nothing to evaluate the stages separately when the result went wrong.
cp "$HOME/omni/minimind-o/out/t2a_mindsurf_768.pth" \
   "$HOME/omni/minimind-o/out/t2a_mindsurf_768.keep.pth"

# Stage two: align the audio projector alone (matches the report).
run "a2a_proj" \
  --data_path "$DATA_A2A" --epochs 1 --batch_size 24 --max_seq_len 640 \
  --learning_rate 5e-4 --from_weight t2a_mindsurf --save_weight sft_mindsurf2 \
  --mode audio_proj --num_workers 8 --use_moe 0 --log_interval 50

# Stage three: full A2A at the report's 5e-5, not last run's 2e-5.
run "a2a_full" \
  --data_path "$DATA_A2A" --epochs 3 --batch_size 16 --max_seq_len 768 \
  --learning_rate 5e-5 --from_weight sft_mindsurf2 --save_weight sft_mindsurf2 \
  --num_workers 8 --use_moe 0 --log_interval 50

echo "===== retrain done $(date -Is) =====" >>"$LOG"
