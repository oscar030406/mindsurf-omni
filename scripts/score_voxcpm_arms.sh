#!/usr/bin/env bash
# 三臂打分（本机跑：campplus 权重、paraformer 判官、UTMOS 缓存都在这）。
# 音频从服务器拉回来之后跑这一条。
#
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/score_voxcpm_arms.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
ARMS=${ARMS:-artifacts/voxcpm_arms}
EDGE=${EDGE:-artifacts/tts_edge}
ROOT=${ROOT:-D:/environment/models/minimind-o-repo}
DEVICE=${DEVICE:-cuda}
STAMP=${STAMP:-2026-08-15}

echo "== 1/4 manifest 里的路径改成本机的（合成在服务器，打分在本机）=="
"$PYTHON" - "$ARMS" <<'PY'
import json
import pathlib
import sys

for directory in sorted(pathlib.Path(sys.argv[1]).iterdir()):
    path = directory / "manifest.json"
    if not path.is_file():
        continue
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        if sample.get("audio_path"):
            sample["audio_path"] = str(directory / f"{sample['id']}.wav")
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  {path} 改了 {len(manifest['samples'])} 条")
PY

echo "== 2/4 独立判官转写（paraformer-zh，不是 SenseVoice：判官不能是我们自己的件）=="
for arm in noref ref; do
    "$PYTHON" scripts/transcribe_samples.py --manifest "$ARMS/$arm/manifest.json" \
        --output "$ARMS/$arm.jsonl" --judge paraformer
done

echo "== 3/4 UTMOS =="
for arm in noref ref; do
    "$PYTHON" scripts/measure_naturalness.py --scored "$ARMS/$arm.jsonl" \
        --output "$ARMS/${arm}_mos.jsonl" --device "$DEVICE"
done

echo "== 4/4 音色一致性 + 语速离群 + 念标点 =="
"$PYTHON" scripts/measure_voice_consistency.py \
    --minimind-root "$ROOT" --device "$DEVICE" \
    --arm "edge=$EDGE" \
    --arm "voxcpm_noref=$ARMS/noref" \
    --arm "voxcpm_ref=$ARMS/ref" \
    --transcripts "voxcpm_noref=$ARMS/noref.jsonl" \
    --transcripts "voxcpm_ref=$ARMS/ref.jsonl" \
    --report "artifacts/voice-consistency-${STAMP}.json"

echo "== 附：贴顶分布（爆音那把尺子，同一批目录）=="
"$PYTHON" scripts/measure_clipping.py --clips "$EDGE" "$ARMS/noref" "$ARMS/ref" \
    --output "artifacts/clipping-voxcpm-arms-${STAMP}.json"
