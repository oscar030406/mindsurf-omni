#!/usr/bin/env bash
# Everything the frozen-Thinker A2A product has to pass, in one detached run.
#
# The gates were written before the run started
# (docs/experiments/2026-07-29-frozen-thinker-a2a.md section 3) and this only
# collects the evidence for them:
#
#   1. the Thinker is bit-identical to t2a_graft -- if it is not, the freeze
#      leaked somewhere and the conversation numbers cannot be cited
#   2. cloning survives: 12 voices x 20 texts, paired against sft_graft
#   3. speech does not regress: the fixed 160, paired against sft_graft
#
# Order matters. The bit-identity check is seconds and settles whether the rest
# is even interpretable, so it runs first and stops the script if it fails.
set -u
ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
REPO="${MINDSURF_REPO:-$HOME/omni/mindsurf-omni}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
SAVE="${1:-sft_graft_frozen}"
OUT="${2:-$HOME/omni/frozen_eval}"
LOG="$OUT/acceptance.log"
mkdir -p "$OUT"
cd "$REPO" || exit 1

echo "===== gate 1: Thinker bit-identity $(date -Is) =====" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" "$PY" - <<PY | tee -a "$LOG"
import sys
import torch

start = torch.load("$ROOT/out/t2a_graft_768.keep.pth", map_location="cpu", weights_only=True)
final = torch.load("$ROOT/out/${SAVE}_768.pth", map_location="cpu", weights_only=True)
keys = [k for k in start if k.startswith(("model.", "lm_head."))]
same = sum(1 for k in keys if k in final and torch.equal(start[k], final[k]))
print(f"Thinker tensors identical to t2a_graft: {same}/{len(keys)}")
if same != len(keys):
    print("FREEZE LEAKED -- the conversation numbers of t2a_graft cannot be cited for this product")
    sys.exit(1)
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "gate 1 failed; stopping" | tee -a "$LOG"; exit 1; }

echo "===== gate 2: cloning $(date -Is) =====" | tee -a "$LOG"
CLONE_DEVICE=cuda CLONE_CUDA=0 bash "$HOME/omni/run_clone_eval.sh" \
  "$ROOT/out/${SAVE}_768.pth" graft "$OUT/clone" >>"$LOG" 2>&1 || exit 1
"$PY" scripts/measure_voice_clone.py --score "$OUT"/clone/*/ \
  --minimind-root "$ROOT" --output "$OUT/clone-${SAVE}.json" >>"$LOG" 2>&1 || exit 1
echo "clone report at $OUT/clone-${SAVE}.json" | tee -a "$LOG"

echo "===== gate 3: speech on the fixed 160 $(date -Is) =====" | tee -a "$LOG"
"$PY" scripts/evaluate_talker.py --checkpoint "$ROOT/out/${SAVE}_768.pth" --shape graft \
  --minimind-root "$ROOT" --audio-encoder "$ROOT/model/SenseVoiceSmall" \
  --codec "$ROOT/model/mimi" --tokenizer assets/tokenizer \
  --texts configs/talker_texts_zh_v1.jsonl --output "$OUT/talker" >>"$LOG" 2>&1 || exit 1
"$PY" scripts/transcribe_samples.py --manifest "$OUT/talker/manifest.json" \
  --output "$OUT/${SAVE}.jsonl" --judge paraformer >>"$LOG" 2>&1 || exit 1
"$PY" scripts/measure_naturalness.py --scored "$OUT/${SAVE}.jsonl" \
  --output "$OUT/${SAVE}_mos.jsonl" --device cuda >>"$LOG" 2>&1 || exit 1

echo "===== done $(date -Is) =====" | tee -a "$LOG"
echo "score locally: evaluate_speech.py --candidate ${SAVE}_mos.jsonl --reference sft_graft_mos.jsonl" | tee -a "$LOG"
