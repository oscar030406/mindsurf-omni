#!/usr/bin/env bash
# 润色底数：同一批文本走两遍生产链路（edge-tts 念 → SenseVoice 听回），
# 一遍干净、一遍注入口语词，量纠错底数、口语词保留率、断句缺失率。
#
# 判据写死在 scripts/measure_polish_floor.py 顶部，跑完不许改。
#
#   ASR_DIR=D:/environment/models/mindsurf-local/SenseVoiceSmall \
#   PYTHON=D:/environment/tools/python/python.exe bash scripts/run_polish_floor.sh
set -euo pipefail

PYTHON=${PYTHON:-python}
TEXTS=${TEXTS:-configs/talker_texts_zh_v1.jsonl}
OUT=${OUT:-artifacts/polish}
STAMP=${STAMP:-2026-08-15}
DEVICE=${DEVICE:-cuda}
: "${ASR_DIR:?需要 SenseVoiceSmall 目录，识别用的是产品自己的识别器（这里量的就是它）}"

mkdir -p "$OUT"

echo "== 1/5 注入口语词 =="
"$PYTHON" scripts/inject_fillers.py --texts "$TEXTS" --output "$OUT/texts_filler.jsonl"

echo "== 2/5 合成两臂（edge-tts，同一把嗓子，同一批句子）=="
"$PYTHON" scripts/speak_texts.py --synthesiser edge --texts "$TEXTS" --output "$OUT/tts_clean"
"$PYTHON" scripts/speak_texts.py --synthesiser edge --texts "$OUT/texts_filler.jsonl" \
    --output "$OUT/tts_filler"

echo "== 3/5 摊平成打分行（口语词臂把干净原文和注入记录带回来）=="
"$PYTHON" - "$OUT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
for arm, source in (("clean", None), ("filler", out / "texts_filler.jsonl")):
    manifest = json.loads((out / f"tts_{arm}" / "manifest.json").read_text(encoding="utf-8"))
    side = {}
    if source is not None:
        side = {
            row["id"]: row
            for row in map(json.loads, source.read_text(encoding="utf-8").splitlines())
        }
    rows = []
    for sample in manifest["samples"]:
        if not sample.get("audio_path"):
            continue
        row = dict(sample)
        if sample["id"] in side:
            row["clean_text"] = side[sample["id"]]["clean_text"]
            row["injections"] = side[sample["id"]]["injections"]
        rows.append(row)
    path = out / f"{arm}_rows.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    print(f"  {path} {len(rows)} 行")
PY

echo "== 4/5 SenseVoice 听回两臂 =="
for arm in clean filler; do
    "$PYTHON" scripts/measure_asr.py --rows "$OUT/${arm}_rows.jsonl" \
        --model-dir "$ASR_DIR" --device "$DEVICE" \
        --output "$OUT/${arm}_scored.jsonl" --report "$OUT/asr-${arm}-${STAMP}.json"
done

echo "== 5/5 出底数 =="
"$PYTHON" scripts/measure_polish_floor.py \
    --clean "$OUT/clean_scored.jsonl" --filler "$OUT/filler_scored.jsonl" \
    --report "artifacts/polish-floor-${STAMP}.json"
