"""Why we read 6.14% where upstream reports 2.96%.

Before swapping the recogniser, find out whether we are using the one we have
properly. Four knobs are in play and we have never varied any of them:
inverse text normalisation (which writes 40000 where the reference says 四万),
the FSMN VAD front end the official recipe uses, the language token, and how
the reference is normalised before scoring.
"""

import glob
import json
import os
import re
import statistics
import time

CLIPS = 400
root = os.path.expanduser("~/omni/corpora/data_aishell/data_aishell")
trans = {}
for line in open(f"{root}/transcript/aishell_transcript_v0.8.txt", encoding="utf-8"):  # noqa: SIM115
    key, _, text = line.strip().partition(" ")
    trans[key] = text.replace(" ", "")

wavs = sorted(glob.glob(f"{root}/wav/test/*/*.wav"))[:CLIPS]
wavs = [w for w in wavs if os.path.basename(w)[:-4] in trans]
print(f"{len(wavs)} clips with references", flush=True)

import torch  # noqa: E402
from funasr import AutoModel  # noqa: E402

MODEL = os.path.expanduser("~/omni/minimind-o/model/SenseVoiceSmall")

TAG = re.compile(r"<\|[^|]*\|>")
PUNCT = re.compile(r"[，。！？、；：" "''（）《》,.!?;:\"'()\[\]…—\-·\s]+")

DIGITS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


def bare(text):
    return PUNCT.sub("", TAG.sub("", text)).strip()


def digits_to_cn(text):
    """A blunt fallback so 2019 and 二零一九 stop counting as four errors."""
    return "".join(DIGITS.get(c, c) for c in text)


def cer(ref, hyp):
    if not ref:
        return 0.0, 0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1], len(ref)


def run(name, itn, use_vad, language):
    kw = dict(
        model=MODEL, device="cuda:0", disable_update=True, disable_pbar=True, disable_log=True
    )
    if use_vad:
        kw["vad_model"] = "fsmn-vad"
        kw["vad_kwargs"] = {"max_single_segment_time": 30000}
    model = AutoModel(**kw)

    plain_e = plain_n = folded_e = folded_n = 0
    ms = []
    for wav in wavs:
        t = time.perf_counter()
        out = model.generate(input=wav, cache={}, language=language, use_itn=itn, batch_size=1)
        ms.append((time.perf_counter() - t) * 1000)
        hyp = bare(out[0]["text"])
        ref = bare(trans[os.path.basename(wav)[:-4]])
        e, n = cer(ref, hyp)
        plain_e += e
        plain_n += n
        e, n = cer(digits_to_cn(ref), digits_to_cn(hyp))
        folded_e += e
        folded_n += n

    del model
    torch.cuda.empty_cache()
    row = {
        "arm": name,
        "itn": itn,
        "vad": use_vad,
        "language": language,
        "cer": round(plain_e / plain_n, 4),
        "cer_digits_folded": round(folded_e / folded_n, 4),
        "p50_ms": round(statistics.median(ms[1:]), 1),
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


rows = [
    run("现在部署的（ITN 开，无 VAD，zh）", True, False, "zh"),
    run("关掉 ITN", False, False, "zh"),
    run("关 ITN + 接 FSMN-VAD", False, True, "zh"),
    run("关 ITN + language=auto", False, False, "auto"),
]
json.dump(rows, open(os.path.expanduser("~/omni/asrgap.json"), "w"), ensure_ascii=False, indent=1)  # noqa: SIM115
print("\n=== 汇总 ===", flush=True)
for r in rows:
    print(f"  {r['cer']:.4f} / 折数字 {r['cer_digits_folded']:.4f}  {r['arm']}", flush=True)
