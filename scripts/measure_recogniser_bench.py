"""对标重跑：把数字折上，并且加一条同族的 Paraformer。

bench.py 调 character_error_rate 时没传 fold_numbers，所以报的是没折数字的数。
这正是 DECISIONS §39 抓到过的那个病——「声明了、论证了、算过了，就是没接到 CER 上」——
润色那一侧修了，识别这一侧还带着。AISHELL 上单这一项就是 0.0638 对 0.0278 的差别。

同时加 Paraformer-large：同一个 FunASR，换过去几乎零集成成本，
所以它是「换识别器」这条路上唯一不用先付集成代价就能量的候选。
"""

from __future__ import annotations

import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/oscar/omni/segcheck")
CORP = Path("/home/oscar/omni/corpora/MagicData-RAMC/MDT2021S003")
sys.path.insert(0, str(ROOT / "src"))

from mindsurf_omni.evaluation.metrics import character_error_rate  # noqa: E402

RATE = 16_000
MARK = re.compile(r"\[[^\]]*\]|<[^>]*>")
TAG = re.compile(r"<\|[^|]*\|>")
CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def turns(txt: Path):
    for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or not parts[0].startswith("["):
            continue
        try:
            start, stop = (float(x) for x in parts[0].strip("[]").split(","))
        except ValueError:
            continue
        text = MARK.sub("", parts[3]).strip()
        if text:
            yield start, stop, parts[1], text


def dictations(limit: int):
    out = []
    for wav in sorted((CORP / "WAV").glob("*.wav"))[:8]:
        txt = CORP / "TXT" / (wav.stem + ".txt")
        if not txt.is_file():
            continue
        with wave.open(str(wav)) as handle:
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), np.int16)
        run, who = [], None
        for start, stop, speaker, text in turns(txt):
            if speaker != who or (run and stop - run[0][0] > 40):
                if run and run[-1][1] - run[0][0] >= 10:
                    a, b = int(run[0][0] * RATE), int(run[-1][1] * RATE)
                    out.append((pcm[a:b].tobytes(), "".join(t for _, _, t in run)))
                    if len(out) >= limit:
                        return out
                run, who = [], speaker
            run.append((start, stop, text))
    return out


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 2**20


def main() -> None:
    clips = dictations(CLIPS)
    seconds = sum(len(p) for p, _ in clips) / 2 / RATE
    print(
        f"{len(clips)} 段单人口述，{seconds / 60:.1f} 分钟，"
        f"中位 {sorted(len(p) for p, _ in clips)[len(clips) // 2] / 2 / RATE:.1f} 秒\n",
        flush=True,
    )

    rows = {}

    # --- Whisper small ---
    import whisper

    torch.cuda.reset_peak_memory_stats()
    model = whisper.load_model("small", download_root="/home/oscar/omni/whisper", device="cuda")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    got, spent = [], []
    for pcm, want in clips:
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        t = time.perf_counter()
        text = model.transcribe(audio, language="zh", fp16=True)["text"]
        spent.append((time.perf_counter() - t) * 1000)
        got.append((want, str(text)))
    rows["Whisper small"] = (params, got, spent, peak_mb())
    del model
    torch.cuda.empty_cache()
    print("whisper done", flush=True)

    # --- SenseVoice 裸的 ---
    from funasr import AutoModel

    torch.cuda.reset_peak_memory_stats()
    sense = AutoModel(
        model="/home/oscar/omni/minimind-o/model/SenseVoiceSmall",
        device="cuda:0",
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
    )
    params = sum(p.numel() for p in sense.model.parameters()) / 1e6
    got, spent = [], []
    for pcm, want in clips:
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        t = time.perf_counter()
        out = sense.generate(input=audio, cache={}, language="zh", use_itn=True, batch_size=1)
        spent.append((time.perf_counter() - t) * 1000)
        got.append((want, TAG.sub("", out[0]["text"])))
    rows["SenseVoice-Small"] = (params, got, spent, peak_mb())
    del sense
    torch.cuda.empty_cache()
    print("sensevoice done", flush=True)

    # --- Paraformer-large，同一个 FunASR ---
    try:
        torch.cuda.reset_peak_memory_stats()
        para = AutoModel(
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            punc_model="ct-punc",
            device="cuda:0",
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )
        params = sum(p.numel() for p in para.model.parameters()) / 1e6
        got, spent = [], []
        for pcm, want in clips:
            audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
            t = time.perf_counter()
            out = para.generate(input=audio, cache={}, batch_size=1)
            spent.append((time.perf_counter() - t) * 1000)
            got.append((want, out[0]["text"]))
        rows["Paraformer-large + ct-punc"] = (params, got, spent, peak_mb())
        del para
        torch.cuda.empty_cache()
        print("paraformer done", flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"paraformer 跑不起来：{type(error).__name__}: {error}", flush=True)

    print(f"\n{'系统':<28}{'参数M':<9}{'折数字':<10}{'不折':<10}{'P50':<8}{'P95':<8}{'显存MB'}")
    out = []
    for name, (params, got, spent, peak) in rows.items():
        folded = sum(character_error_rate(w, g, fold_numbers=True) for w, g in got) / len(got)
        plain = sum(character_error_rate(w, g, fold_numbers=False) for w, g in got) / len(got)
        spent.sort()
        p50, p95 = spent[len(spent) // 2], spent[int(len(spent) * 0.95) - 1]
        out.append(
            {
                "name": name,
                "params_m": round(params, 1),
                "cer_folded": round(folded, 4),
                "cer_unfolded": round(plain, 4),
                "p50_ms": round(p50),
                "p95_ms": round(p95),
                "peak_mb": round(peak),
            }
        )
        print(
            f"{name:<28}{params:<9.1f}{folded:<10.4f}{plain:<10.4f}{p50:<8.0f}{p95:<8.0f}{peak:.0f}"
        )
    (ROOT / "bench2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
