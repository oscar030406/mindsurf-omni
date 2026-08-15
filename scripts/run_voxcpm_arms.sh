#!/usr/bin/env bash
# 本地合成器的两条臂，同一批句子、同一张卡，只差一条参考音频。
#
# 无参考那臂是现状（factory 建 VoxCPMSynthesiser 时 prompt_wav/prompt_text 全空，
# 模型每次调用随机抽说话人）；带参考那臂是配置级修复之后。
# 判据写死在 scripts/measure_voice_consistency.py 顶部。
#
#   PROMPT_TEXT='您好，请问您有什么需要帮忙的问题吗？' bash scripts/run_voxcpm_arms.sh
set -euo pipefail

PYTHON=${PYTHON:-$HOME/.venvs/voxcpm/bin/python}
REPO=${REPO:-$HOME/omni/mindsurf-omni}
OUT=${OUT:-$HOME/omni/voxcpm_arms}
PROMPT_WAV=${PROMPT_WAV:-$HOME/omni/voice_prompt/reference_zh_5s.wav}
: "${PROMPT_TEXT:?参考音频说的话，一个字都不能错——它是告诉模型哪段声音对应哪个字的}"
DEVICE=${DEVICE:-cuda}
LIMIT=${LIMIT:-160}

cd "$REPO"

echo "== 无参考臂（现状）=="
"$PYTHON" scripts/speak_texts.py --synthesiser voxcpm --device "$DEVICE" \
    --limit "$LIMIT" --output "$OUT/noref"

echo "== 带参考臂（修复后）=="
"$PYTHON" scripts/speak_texts.py --synthesiser voxcpm --device "$DEVICE" \
    --limit "$LIMIT" --output "$OUT/ref" \
    --prompt-wav "$PROMPT_WAV" --prompt-text "$PROMPT_TEXT"

echo "两臂写在 $OUT，拉回本机量音色一致性与粗大缺陷"
