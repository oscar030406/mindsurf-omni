#!/usr/bin/env bash
# The three arms of the harvested-emotion gate, one pass per speaker.
#
# Criteria were written before this ran, in
# docs/experiments/2026-07-30-emotion-harvest-gate.md section 3, and this only
# produces the audio they judge:
#
#   calm      the speaker's lower-pitched reference
#   lively    the same speaker's higher-pitched reference   <- the effect
#   control   calm again under a second seed                <- the noise floor
#
# The control arm is not optional. Generation is sampled, so two runs from one
# reference already differ; without knowing by how much, a separation between
# two different references has nothing to be larger than.
#
#   bash scripts/run_emotion_gate.sh sft_graft
#   bash scripts/run_emotion_gate.sh sft_graft_frozen
set -u
ROOT="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
REPO="${MINDSURF_REPO:-$HOME/omni/mindsurf-omni}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
SAVE="${1:-sft_graft}"
PACK="${2:-$HOME/omni/emo_harvest}"
OUT="${3:-$HOME/omni/emo_gate/$SAVE}"
TEXTS="${EMOTION_TEXTS:-configs/talker_texts_zh_v1.jsonl}"
LIMIT="${EMOTION_LIMIT:-12}"
# Written here rather than passed, so a rerun is the same run. The control arm
# differs from calm in the seed and in nothing else.
SEED_MAIN=20260725
SEED_CONTROL=20260730

# Never beside training: this wants the card, and the run it would slow down is
# measured in hours. Anchored on the interpreter and on an argument only a real
# run carries -- a bare pgrep -f has produced a false positive here twice.
running=$(pgrep -f "python.*train_omni\.py --data_path" 2>/dev/null | wc -l)
if [ "$running" -gt 0 ]; then
  echo "training is still running ($running processes); not taking the card" >&2
  exit 1
fi

[ -f "$PACK/voices.pt" ] || { echo "no voice pack at $PACK/voices.pt" >&2; exit 1; }
[ -f "$ROOT/out/${SAVE}_768.pth" ] || { echo "no checkpoint $ROOT/out/${SAVE}_768.pth" >&2; exit 1; }

cd "$REPO" || exit 1
mkdir -p "$OUT"
LOG="$OUT/gate.log"

# The speakers are whatever the harvest found, read off the pack rather than
# listed here: a list would go stale the first time the harvest is rerun, and
# quietly measure fewer speakers than the pack holds.
speakers=$(ls "$PACK" | sed -n 's/^\(speaker[0-9]*\)_calm\.wav$/\1/p' | sort -u)
[ -n "$speakers" ] || { echo "no speakerN_calm.wav in $PACK" >&2; exit 1; }
echo "===== emotion gate $SAVE $(date -Is) =====" | tee -a "$LOG"
echo "speakers: $(echo "$speakers" | tr '\n' ' ')" | tee -a "$LOG"

for speaker in $speakers; do
  for arm in calm lively control; do
    case "$arm" in
      calm)    voice="${speaker}_calm";   seed="$SEED_MAIN" ;;
      lively)  voice="${speaker}_lively"; seed="$SEED_MAIN" ;;
      control) voice="${speaker}_calm";   seed="$SEED_CONTROL" ;;
    esac
    echo "----- $speaker $arm (voice $voice, seed $seed) $(date -Is)" | tee -a "$LOG"
    "$PY" scripts/evaluate_talker.py --checkpoint "$ROOT/out/${SAVE}_768.pth" --shape graft \
      --minimind-root "$ROOT" --audio-encoder "$ROOT/model/SenseVoiceSmall" \
      --codec "$ROOT/model/mimi" --tokenizer assets/tokenizer \
      --texts "$TEXTS" --limit "$LIMIT" \
      --voice "$voice" --voice-pack "$PACK" --seed "$seed" \
      --output "$OUT/$speaker/$arm" >>"$LOG" 2>&1 || exit 1
  done
done

echo "===== done $(date -Is) =====" | tee -a "$LOG"
echo "score locally, per speaker:" | tee -a "$LOG"
echo "  measure_prosody.py --arm calm=<dir>/calm/manifest.json --arm lively=<dir>/lively/manifest.json --arm control=<dir>/control/manifest.json --baseline calm" | tee -a "$LOG"
