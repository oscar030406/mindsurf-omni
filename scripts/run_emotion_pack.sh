#!/usr/bin/env bash
# Generate the emotional-voice arm at the same protocol the clone metric uses.
#
# The deliverable is not a processed reference clip -- signal processing was
# ruled out twice, and the reference code strip turned out to carry no prosody
# at all. It is arithmetic on the speaker vector:
#
#     spk_emb = the voice's own + alpha * delta,   ref_codes = the voice's own
#
# where delta is the difference between two references of one speaker read in
# two prosodies. Criteria are written before this runs, in
# docs/experiments/2026-07-30-emotion-harvest-gate.md section 13.
#
# The baseline arm is not regenerated: the clone acceptance already produced
# the same 12 voices over the same 20 texts under the same seed, and pairing
# against a rerun would add sampling noise for nothing.
#
#   bash scripts/run_emotion_pack.sh sft_graft_frozen 0.50
set -u
ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
REPO="${MINDSURF_REPO:-$HOME/omni/mindsurf-omni}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
SAVE="${1:-sft_graft_frozen}"
ALPHA="${2:-0.50}"
# The conditioned arm puts the emotion in the user turn instead of in the
# speaker vector, so it wants the untouched vectors and a probe file per
# instruction -- same generation protocol, same guard, same resumability, and
# no alpha in the path. EMOTION_PACK names the pack outright for that case.
PACK="${EMOTION_PACK:-${EMOTION_PACK_ROOT:-$HOME/omni/emo_vec_full}/alpha$ALPHA}"
OUT="${3:-$HOME/omni/emo_pack_eval/$SAVE/alpha$ALPHA}"
TEXTS="${EMOTION_TEXTS:-configs/talker_texts_zh_v1.jsonl}"
LIMIT="${EMOTION_LIMIT:-20}"
SEED="${EMOTION_SEED:-20260725}"
# Twelve is the protocol. The override is for the positive control, which only
# has to show the instrument can see identity being destroyed -- four voices do
# that, and the other eight would be twenty-six minutes of card proving it again.
VOICES="${EMOTION_VOICES:-dylan eric serena uncle_fu vivian arthur chelsie cherry ethan jennifer momo moon}"

# Never beside training. Anchored on the interpreter and on an argument only a
# real run carries -- a bare pgrep -f has produced a false positive here twice.
running=$(pgrep -f "python.*train_omni\.py --data_path" 2>/dev/null | wc -l)
if [ "$running" -gt 0 ]; then
  echo "training is still running ($running processes); not taking the card" >&2
  exit 1
fi

[ -f "$PACK/voices.pt" ] || { echo "no emotional pack at $PACK/voices.pt" >&2; exit 1; }
[ -f "$ROOT/out/${SAVE}_768.pth" ] || { echo "no checkpoint $ROOT/out/${SAVE}_768.pth" >&2; exit 1; }

cd "$REPO" || exit 1
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "===== emotion pack $SAVE alpha=$ALPHA $(date -Is) =====" | tee -a "$LOG"

for voice in $VOICES; do
  # Resumable: an interrupted run continues rather than regenerating an hour.
  if [ -f "$OUT/$voice/manifest.json" ]; then
    echo "skip $voice (already done)" | tee -a "$LOG"; continue
  fi
  echo "----- $voice $(date -Is)" | tee -a "$LOG"
  "$PY" scripts/evaluate_talker.py --checkpoint "$ROOT/out/${SAVE}_768.pth" --shape graft \
    --minimind-root "$ROOT" --audio-encoder "$ROOT/model/SenseVoiceSmall" \
    --codec "$ROOT/model/mimi" --tokenizer assets/tokenizer \
    --texts "$TEXTS" --limit "$LIMIT" --seed "$SEED" \
    --voice "$voice" --voice-pack "$PACK" --output "$OUT/$voice" >>"$LOG" 2>&1 || exit 1
done

echo "===== done $(date -Is) =====" | tee -a "$LOG"
echo "score: measure_voice_clone.py --score $OUT/*/ --voice-pack <original vectors>" | tee -a "$LOG"
echo "       measure_prosody.py --arm base=<clone acceptance dir> --arm emo=$OUT ..." | tee -a "$LOG"
