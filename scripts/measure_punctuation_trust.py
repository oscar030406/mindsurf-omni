"""§18 的推翻条件，拿真人语音去压。

§18 决定「照抄识别器的标点」，并写下推翻条件：识别器的标点变得不可信时——
换识别器，或者输入换成真人之后断句召回从 0.939 掉下来——照抄就不再是对的默认。
那个 0.939 是在 TTS→ASR 合成回环上量的。这里量真人。

对齐方式：两边都剥掉标点得到裸字串，用 SequenceMatcher 把识别结果映射回参考的
字位置，再看参考的每一个断点识别器有没有在同一处（容差 2 字）也打了标点。
"""

from __future__ import annotations

import difflib
import json
import random
import re
import statistics
import sys
import wave
from pathlib import Path

import numpy as np

CORP = Path("<工作目录>/corpora/MagicData-RAMC/MDT2021S003")
RATE = 16_000
MARK = re.compile(r"\[[^\]]*\]|<[^>]*>")
TAG = re.compile(r"<\|[^|]*\|>")
PUNC = set("，。！？；：、,.!?;:")
TOL = 2
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


def dictations(limit):
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


def split(text):
    """Bare characters, plus the bare-index each mark sits after."""
    bare, marks = [], []
    for ch in text:
        if ch in PUNC:
            if bare:
                marks.append(len(bare) - 1)
        elif not ch.isspace():
            bare.append(ch)
    return "".join(bare), sorted(set(marks))


def longest_run(bare_len, marks):
    prev, longest = -1, 0
    for m in [*marks, bare_len - 1]:
        longest = max(longest, m - prev)
        prev = m
    return longest


def main():
    clips = dictations(CLIPS)
    print(f"{len(clips)} 段真人口述", flush=True)

    from funasr import AutoModel

    sense = AutoModel(
        model="<权重目录>/SenseVoiceSmall",
        device="cuda:0",
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
    )

    hit = ref_total = hyp_total = 0
    chance_hit = self_hit = 0
    rng = random.Random(20260824)
    ref_density, hyp_density, ref_run, hyp_run = [], [], [], []

    for pcm, want in clips:
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        out = sense.generate(input=audio, cache={}, language="zh", use_itn=True, batch_size=1)
        got = TAG.sub("", out[0]["text"])

        ref_bare, ref_marks = split(want)
        hyp_bare, hyp_marks = split(got)
        if not ref_bare or not hyp_bare:
            continue

        # Where each reference character landed in the hypothesis.
        where = {}
        for block in difflib.SequenceMatcher(
            None, ref_bare, hyp_bare, autojunk=False
        ).get_matching_blocks():
            for k in range(block.size):
                where[block.a + k] = block.b + k

        hyp_set = set(hyp_marks)
        for m in ref_marks:
            target = where.get(m)
            if target is None:
                continue  # the recogniser lost that character entirely
            if any((target + d) in hyp_set for d in range(-TOL, TOL + 1)):
                hit += 1
        # Chance: the same number of marks, thrown at random positions.
        if len(hyp_bare) > len(hyp_marks):
            spray = set(rng.sample(range(len(hyp_bare)), len(hyp_marks)))
            for m in ref_marks:
                target = where.get(m)
                if target is not None and any((target + d) in spray for d in range(-TOL, TOL + 1)):
                    chance_hit += 1
        # Ceiling: score the reference against itself through the same path.
        for m in ref_marks:
            if any((m + d) in set(ref_marks) for d in range(-TOL, TOL + 1)):
                self_hit += 1
        ref_total += len(ref_marks)
        hyp_total += len(hyp_marks)

        ref_density.append(len(ref_marks) / len(ref_bare) * 100)
        hyp_density.append(len(hyp_marks) / len(hyp_bare) * 100)
        ref_run.append(longest_run(len(ref_bare), ref_marks))
        hyp_run.append(longest_run(len(hyp_bare), hyp_marks))

    recall = hit / ref_total if ref_total else 0.0
    # Precision the other way round would need the reverse alignment; the
    # density pair below says the same thing without pretending to more.
    row = {
        "clips": len(clips),
        "boundary_recall_real": round(recall, 4),
        "chance_same_density": round(chance_hit / ref_total, 4) if ref_total else 0.0,
        "ruler_ceiling_self": round(self_hit / ref_total, 4) if ref_total else 0.0,
        "synthetic_baseline_from_s18": 0.939,
        "ref_marks": ref_total,
        "hyp_marks": hyp_total,
        "marks_per_100_chars": {
            "human": round(statistics.mean(ref_density), 2),
            "sensevoice": round(statistics.mean(hyp_density), 2),
        },
        "longest_run_without_a_mark": {
            "human": round(statistics.mean(ref_run), 1),
            "sensevoice": round(statistics.mean(hyp_run), 1),
            "sensevoice_worst": max(hyp_run),
        },
    }
    print(json.dumps(row, ensure_ascii=False, indent=1), flush=True)
    Path("<工作目录>/punc.json").write_text(json.dumps(row, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
