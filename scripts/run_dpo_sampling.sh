#!/usr/bin/env bash
# On-policy sampling for DPO: one checkpoint, several seeds, the preference set.
#
# Never the evaluation 160 -- training on those would contaminate the instrument
# that judges the result. The prompt file is an argument so the second round can
# point at the enlarged set without editing this file.
#
# Resumable, and that is not decoration: the first attempt at the enlarged set
# died with its ssh session after two seeds because it ran in the foreground.
# A seed already on disk is skipped, so re-running costs nothing.
#
#   setsid nohup bash scripts/run_dpo_sampling.sh sft_graft_frozen \
#       configs/preference_prompts_zh_all.jsonl ~/omni/dpo2 > ~/omni/dpo2.log 2>&1 &
set -u
R="${MINIMIND_O_ROOT:-$HOME/omni/minimind-o}"
PY="${OMNI_PYTHON:-$HOME/.venvs/omni/bin/python}"
CKPT="${1:-sft_graft_frozen}"
PROMPTS="${2:-configs/preference_prompts_zh_all.jsonl}"
OUT="${3:-$HOME/omni/dpo2}"
SEEDS="${DPO_SEEDS:-1 2 3 4}"

# Never beside training. Anchored on an argument only a real run carries -- a
# bare pgrep -f has produced a false positive here twice.
running=$(pgrep -f "python.*train_omni\.py --data_path" 2>/dev/null | wc -l)
if [ "$running" -gt 0 ]; then
  echo "training is still running ($running processes); not taking the card" >&2
  exit 1
fi

[ -f "$R/out/${CKPT}_768.pth" ] || { echo "no checkpoint $R/out/${CKPT}_768.pth" >&2; exit 1; }
[ -f "$PROMPTS" ] || { echo "no prompt set $PROMPTS" >&2; exit 1; }

mkdir -p "$OUT"
echo "===== dpo sampling $CKPT over $(wc -l < "$PROMPTS") prompts $(date -Is) ====="
for seed in $SEEDS; do
  if [ -s "$OUT/samples-$seed.json" ]; then
    echo "skip seed $seed (already on disk)"; continue
  fi
  echo "----- seed $seed $(date -Is)"
  "$PY" scripts/measure_chat_loss.py \
    --checkpoint "$R/out/${CKPT}_768.pth" --minimind-root "$R" \
    --probes "$PROMPTS" \
    --generate --device cuda --seed "$seed" \
    --temperature 0.7 --top-p 0.9 --max-tokens 512 \
    --output "$OUT/samples-$seed.json" || exit 1
done
echo "===== done $(date -Is) ====="
