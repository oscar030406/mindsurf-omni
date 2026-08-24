"""Test a streaming recogniser, which is what the lead asked for.

The task is narrow: does this thing work well enough to go on the model list.
It only ever drives the display -- at release the authoritative transcript is
still SenseVoice's, because that is what polish was trained against. So the
readings that matter are latency and how often a word appears, and accuracy is
a floor check rather than a target.

chunk_size=[0,10,5] is FunASR's standard streaming config: 600 ms chunk with
300 ms of lookahead, which is the number the lead remembered.
"""

import json
import os
import re
import statistics
import sys
import time
import wave

import numpy as np

CORP = "/home/oscar/omni/corpora/MagicData-RAMC/MDT2021S003"
RATE = 16_000
MARK = re.compile(r"\[[^\]]*\]|<[^>]*>")
TAG = re.compile(r"<\|[^|]*\|>")
CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

sys.path.insert(0, "/home/oscar/omni/mindsurf-omni/src")
from mindsurf_omni.evaluation.metrics import character_error_rate  # noqa: E402


def turns(path):
    for line in open(path, encoding="utf-8", errors="ignore"):  # noqa: SIM115
        parts = line.split("\t")
        if len(parts) < 4 or not parts[0].startswith("["):
            continue
        try:
            a, b = (float(x) for x in parts[0].strip("[]").split(","))
        except ValueError:
            continue
        text = MARK.sub("", parts[3]).strip()
        if text:
            yield a, b, parts[1], text


def dictations(limit):
    out = []
    for name in sorted(os.listdir(f"{CORP}/WAV"))[300:]:
        txt = f"{CORP}/TXT/{name[:-4]}.txt"
        if not os.path.isfile(txt):
            continue
        with wave.open(f"{CORP}/WAV/{name}") as fh:
            pcm = np.frombuffer(fh.readframes(fh.getnframes()), np.int16)
        run, who = [], None
        for a, b, speaker, text in turns(txt):
            if speaker != who:
                if run and 15 <= run[-1][1] - run[0][0] <= 60:
                    i, j = int(run[0][0] * RATE), int(run[-1][1] * RATE)
                    out.append((pcm[i:j], "".join(t for _, _, _, t in run)))
                    if len(out) >= limit:
                        return out
                run, who = [], speaker
            run.append((a, b, speaker, text))
    return out


clips = dictations(CLIPS)
print(
    f"{len(clips)} 段真人口述，中位 {statistics.median(len(p) / RATE for p, _ in clips):.0f} 秒",
    flush=True,
)  # noqa: E501

from funasr import AutoModel  # noqa: E402

# --- the streaming one ---
print("\n拉 Paraformer-online…", flush=True)
online = AutoModel(
    model="paraformer-zh-streaming",
    device="cuda:0",
    disable_update=True,
    disable_pbar=True,
    disable_log=True,
)

CHUNK = [0, 10, 5]  # 600 ms chunk, 300 ms lookahead
STRIDE = CHUNK[1] * 960  # 10 * 960 samples = 600 ms
BACK, AHEAD = CHUNK[0], CHUNK[2]

offline = AutoModel(
    model="/home/oscar/omni/minimind-o/model/SenseVoiceSmall",
    device="cuda:0",
    disable_update=True,
    disable_pbar=True,
    disable_log=True,
)

rows = []
for pcm, human in clips:
    audio = pcm.astype(np.float32) / 32768.0

    # streaming: feed it 600 ms at a time and time each step
    cache, said, steps, first_at = {}, [], [], None
    started = time.perf_counter()
    total = int(len(audio) - 1) // STRIDE + 1
    for i in range(total):
        piece = audio[i * STRIDE : (i + 1) * STRIDE]
        t = time.perf_counter()
        out = online.generate(
            input=piece,
            cache=cache,
            is_final=(i == total - 1),
            chunk_size=CHUNK,
            encoder_chunk_look_back=BACK,
            decoder_chunk_look_back=AHEAD,
        )
        steps.append((time.perf_counter() - t) * 1000)
        text = out[0]["text"] if out else ""
        if text:
            said.append(text)
            if first_at is None:
                first_at = (i + 1) * STRIDE / RATE
    stream_text = "".join(said)

    t = time.perf_counter()
    o = offline.generate(input=audio, cache={}, language="zh", use_itn=True, batch_size=1)
    offline_ms = (time.perf_counter() - t) * 1000
    offline_text = TAG.sub("", o[0]["text"]).strip()

    rows.append(
        {
            "seconds": round(len(pcm) / RATE, 1),
            "stream_cer": character_error_rate(human, stream_text, fold_numbers=True),
            "offline_cer": character_error_rate(human, offline_text, fold_numbers=True),
            "step_ms_median": statistics.median(steps),
            "step_ms_max": max(steps),
            "first_word_at_s": first_at,
            "tail_ms": steps[-1],
            "offline_ms": offline_ms,
            "stream_chars": len(stream_text),
            "offline_chars": len(offline_text),
        }
    )
    _ = human

print(f"\n{len(rows)} 段跑完\n")


def med(k):
    vals = [r[k] for r in rows if r[k] is not None]
    return statistics.median(vals) if vals else 0.0


print(f"{'':<26}{'流式 Paraformer':>18}{'SenseVoice 整段':>18}")
print(f"{'字错率（对人工转写）':<26}{med('stream_cer'):>18.4f}{med('offline_cer'):>18.4f}")
print(f"{'写出的字数':<26}{med('stream_chars'):>18.0f}{med('offline_chars'):>18.0f}")
print(
    f"\n每 600 ms 一步的耗时：中位 {med('step_ms_median'):.0f} ms，最坏 {max(r['step_ms_max'] for r in rows):.0f} ms"  # noqa: E501
)  # noqa: E501
print(f"第一个字出现在：{med('first_word_at_s'):.1f} 秒")
print(f"松手后最后一步：{med('tail_ms'):.0f} ms")
print(f"（对比）SenseVoice 整段一次：{med('offline_ms'):.0f} ms")
print(
    f"\n实时率：一步 {med('step_ms_median'):.0f} ms 处理 600 ms 音频，"
    f"RTF {med('step_ms_median') / 600:.3f}"
)

json.dump(rows, open("/home/oscar/omni/streamasr.json", "w"), ensure_ascii=False, indent=1)  # noqa: SIM115
