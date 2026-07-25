#!/usr/bin/env bash
# One arm, one variable: the T2A stage at ten times the report's learning rate.
#
# Why this and not "train it longer": the mid-run reads say the T2A stage
# product does not speak (CER 1.0087, 100 of 160 clips silent), that a full
# epoch of A2A afterwards does not fix it (0.9856), and that the Talker has
# moved 3% from an initialisation that is a copy of thinker layers -- where the
# round that did speak, badly, had moved it by its own norm. Three readings,
# one hypothesis: the report's recipe is a fine-tune for a base that has seen
# audio, and ours has not.
#
# So this changes exactly one thing against run_full_recipe.sh's first stage --
# --learning_rate 5e-5 instead of 5e-6, still a tenth of the 5e-4 the broken
# round took by accident -- and stops after that stage. The stage product is
# the verdict; there is no need to spend a full chain to read it.
#
# Prediction, written before the run: the product speaks. CER leaves 1.0 and
# the silent rate leaves 0.6. If it does not, the hypothesis is dead and the
# next question is data and architecture, not hyperparameters.
#
#   setsid nohup bash ~/omni/mindsurf-omni/scripts/run_t2a_probe.sh \
#     >~/omni/t2a_probe.log 2>&1 </dev/null &
set -u

ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
LIB="${MINDSURF_LIB:-$HOME/omni/lib}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
LOG="${PROBE_LOG:-$HOME/omni/t2a_probe.log}"
DATA_T2A="${T2A_DATA:-../dataset/sft_t2a.parquet}"
BASE="${BASE_WEIGHT:-llm_mindsurf}"

cd "$ROOT/trainer" || exit 1

for required in "$ROOT/out/${BASE}_768.pth" "$ROOT/${DATA_T2A#../}" "$LIB/train_omni.py"; do
  if [ ! -f "$required" ]; then
    echo "missing: $required" >&2
    exit 1
  fi
done

# Never beside another run: two jobs on one card make both of them slower and
# neither of them measurable.
running=$(pgrep -f "python.*train_omni\.py --data_path" 2>/dev/null | wc -l)
if [ "$running" -gt 0 ]; then
  echo "a training run is already going ($running processes); this probe waits its turn" >&2
  exit 1
fi

echo "===== t2a_probe lr5e-5 $(date -Is) =====" >>"$LOG"
setsid nohup env \
  MINIMIND_O_ROOT="$ROOT" PYTHONPATH="$LIB" PYTHONUNBUFFERED=1 \
  "$PY" "$LIB/train_omni.py" \
  --data_path "$DATA_T2A" --epochs 1 --batch_size 32 --max_seq_len 640 \
  --learning_rate 5e-5 --from_weight "$BASE" --save_weight t2a_lr5e5 \
  --num_workers 8 --use_moe 0 --log_interval 50 \
  >>"$LOG" 2>&1 </dev/null &
pid=$!
echo "t2a_probe pid $pid" | tee -a "$LOG"
wait "$pid"
status=$?
echo "t2a_probe exited $status" | tee -a "$LOG"
[ "$status" -eq 0 ] || exit "$status"

# Read it here rather than leaving a checkpoint for someone to discover. The
# verdict is one number -- does the stage product speak -- and waiting for a
# human to run three commands is how a four-hour answer becomes a next-day one.
REPO="${MINDSURF_REPO:-$HOME/omni/mindsurf-omni}"
OUT="${PROBE_OUT:-$HOME/omni/probe_eval}"
cd "$REPO" || exit 1
mkdir -p "$OUT"

echo "===== reading the probe's stage product $(date -Is) =====" | tee -a "$LOG"
"$PY" scripts/evaluate_talker.py \
  --checkpoint "$ROOT/out/t2a_lr5e5_768.pth" --shape mindsurf \
  --minimind-root "$ROOT" --audio-encoder "$ROOT/model/SenseVoiceSmall" \
  --codec "$ROOT/model/mimi" --tokenizer assets/tokenizer \
  --texts configs/talker_texts_zh_v1.jsonl \
  --output "$OUT/t2a_lr5e5" >>"$LOG" 2>&1 || exit 1
"$PY" scripts/transcribe_samples.py --manifest "$OUT/t2a_lr5e5/manifest.json" \
  --output "$OUT/t2a_lr5e5.jsonl" --judge paraformer >>"$LOG" 2>&1 || exit 1

# Scoring needs zhconv, which this venv does not carry, and installing into a
# training environment is not something to do casually. The rows are the
# deliverable: copy $OUT/t2a_lr5e5.jsonl and run evaluate_speech.py against
# artifacts/codebook_baseline_mos.jsonl wherever zhconv is installed.
{
  echo "===== probe read done $(date -Is) ====="
  echo "rows at $OUT/t2a_lr5e5.jsonl -- score them against artifacts/codebook_baseline_mos.jsonl"
  echo "prediction on record: the stage product speaks, CER leaves 1.0 and silence leaves 0.6"
} | tee -a "$LOG"
