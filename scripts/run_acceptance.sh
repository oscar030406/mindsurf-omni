#!/usr/bin/env bash
# The retrain's acceptance test, as a single command, because it runs whenever
# the eleven-hour chain happens to finish and a procedure that needs five
# correct commands at three in the morning is a procedure that gets one wrong.
#
# The criteria were fixed before the run started and are not in here: this
# script produces the numbers, docs/experiments/2026-07-25-retrain-plan.md says
# what passes. Paired CER and UTMOS against upstream's release must come back
# `indistinguishable` or `improved`, and the *median* CER must collapse toward
# 0.05 -- a mean that improves while the median stays above 0.2 means fewer
# catastrophes, not better speech.
#
# Both arms are generated here, in one run, on one machine. The reference arm
# is not optional and cannot be reused from the last round: the pairing needs
# per-sample rows and only the summary was ever kept, so upstream's checkpoint
# speaks the same 160 sentences again. Two arms, same harness, same judge --
# that is the whole reason the verdict is allowed to gate.
#
#   ssh <host> "setsid nohup bash ~/omni/mindsurf-omni/scripts/run_acceptance.sh \
#                 >~/omni/acceptance.log 2>&1 </dev/null &"
set -u

REPO="${MINDSURF_REPO:-$HOME/omni/mindsurf-omni}"
ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
OUT="${ACCEPTANCE_OUT:-$HOME/omni/acceptance}"
OURS="${OURS_CHECKPOINT:-$ROOT/out/sft_mindsurf2_768.pth}"
THEIRS="${THEIRS_CHECKPOINT:-$ROOT/out/sft_omni_768.pth}"

# Never while the card is training. The measurement would be taken against a
# checkpoint still being written, and the training would be slowed by an
# evaluation nobody asked to run beside it.
running=$(pgrep -f "python.*train_omni\.py --data_path" 2>/dev/null | wc -l)
if [ "$running" -gt 0 ]; then
  echo "training is still running ($running processes); acceptance would measure a half-written checkpoint" >&2
  exit 1
fi

for required in "$OURS" "$THEIRS" "$REPO/scripts/evaluate_talker.py" \
                "$ROOT/model/SenseVoiceSmall" "$ROOT/model/mimi"; do
  if [ ! -e "$required" ]; then
    echo "missing: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT"
cd "$REPO" || exit 1

# One arm: speak the fixed 160, transcribe with the judge every earlier verdict
# used, then score naturalness on the same clips.
arm() {
  local name=$1 checkpoint=$2 shape=$3
  echo "===== $name $(date -Is) ====="
  "$PY" scripts/evaluate_talker.py \
    --checkpoint "$checkpoint" --shape "$shape" \
    --minimind-root "$ROOT" --audio-encoder "$ROOT/model/SenseVoiceSmall" \
    --codec "$ROOT/model/mimi" --tokenizer assets/tokenizer \
    --texts configs/talker_texts_zh_v1.jsonl \
    --output "$OUT/$name" || exit 1
  "$PY" scripts/transcribe_samples.py \
    --manifest "$OUT/$name/manifest.json" \
    --output "$OUT/$name.jsonl" --judge paraformer || exit 1
  "$PY" scripts/measure_naturalness.py \
    --scored "$OUT/$name.jsonl" --output "$OUT/${name}_mos.jsonl" || exit 1
}

arm ours "$OURS" mindsurf
arm official "$THEIRS" upstream-default

# Scoring needs zhconv, which this venv does not carry -- the fold from
# traditional to simplified is four fifths of the CER on this data, so a run
# without it is not a smaller measurement, it is a wrong one. If it is absent
# the two _mos.jsonl files are the deliverable and the comparison runs
# wherever zhconv is installed.
if "$PY" -c "import zhconv" 2>/dev/null; then
  "$PY" scripts/evaluate_speech.py \
    --candidate "$OUT/ours_mos.jsonl" --reference "$OUT/official_mos.jsonl" \
    --output "$OUT/acceptance-report.json"
else
  echo "zhconv absent: copy $OUT/ours_mos.jsonl and $OUT/official_mos.jsonl and score them elsewhere"
fi

echo "===== acceptance done $(date -Is) ====="
