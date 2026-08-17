"""Drive the dictation product end to end and read what comes back out loud.

The transcription half has been used as a user twice. The read-aloud half never
had been: the four synthesis arms were compared on timbre and clarity, and
nobody had walked 口述 -> 转写 -> 润色 -> 朗读 as one thing. This is that walk,
scripted, because every valuable finding on this project came from driving the
running service rather than from a held-out set.

Five readings, each answering a question the four acceptance criteria cannot:

``notes``
    The loop itself. Speak a note, transcribe it, polish it, speak the polished
    text, transcribe that back, and compare. The readback CER is the closest
    thing to an ear this project has -- it is not one, and 0.0000 means the
    recogniser heard the same characters, not that the audio was good.

``pronunciation``
    How numbers, abbreviations and units are read. Readback cannot answer this:
    synthesise ``B203`` or ``B二零三`` and the recogniser returns ``B203`` for
    both. Duration can -- the candidate spellings differ in length, so whichever
    one matches is the one being spoken. Measured bare and inside a carrier
    sentence, because an isolated token gets its own reading.

``length``
    Punctuation density against utterance length, and what that does downstream.
    ``asr.py`` hands the whole buffer to one ``generate`` call, and SenseVoice
    punctuates less as it gets longer. Downstream that matters because
    ``group_sentences`` splits on 。！？； alone: fewer full stops means longer
    sentences, and a sentence past ``TRAINED_LENGTH`` is what made 164 seconds
    of dictation come back as 18 characters. So this reports the chain, not just
    the density -- longest sentence, group sizes, ``consumed``, and whether the
    filler actually left.

``prosody``
    A question mark in the wrong place. The recogniser puts one after 我们要不要
    and a full stop after 别的方案, and polish then removes the filler that stood
    between them. CER normalises punctuation away, so all four criteria price
    this at zero. The pause length and the pitch slope before it do not.

``language``
    ``asr.py`` passes ``language="auto"``. A short utterance that is nothing but
    English letters comes back as another language, and for two of them as kana:
    a user who dictates the single word ``demo`` is handed でも.

Needs a running service with a synthesiser wired::

    MINDSURF_TTS=edge  (plus the transcription configuration; see docs)
    python scripts/measure_read_aloud.py --base http://127.0.0.1:8099 \\
        --out artifacts/read-aloud-2026-08-16.json

Audio is written next to the report only when ``--audio DIR`` is given, and that
directory belongs outside the repository -- the readings are the deliverable and
the wav files are forty megabytes of scratch.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.evaluation.metrics import character_error_rate  # noqa: E402
from mindsurf_omni.service.polish import (  # noqa: E402
    TRAINED_LENGTH,
    consumed,
    group_sentences,
    split_sentences,
)

PUNCTUATION = "，。？！、；：,.?!;:"
SAMPLE_RATE = 16_000
SECTIONS = ("notes", "pronunciation", "length", "prosody", "language")

# Written to carry what a real speaker carries: a filler to open with, a
# mid-sentence restart, a topic raised again rather than repeated (表，表在 --
# see DECISIONS 35), and the numbers, letters and units that only the read-aloud
# half has to deal with.
NOTES = [
    "嗯，这个，我想说一下明天的安排啊。就是我们那个，那个会议改到下午三点了，"
    "因为，因为上午张老师有课。然后地点还是，还是在B203。",
    "那个报销的事情，我今天问了一下，就是说，需要发票原件，然后还要填一个表，"
    "表在，那个OA系统上能下载。反正，就是月底之前交上去就行。",
    "我觉得吧，这个方案还行，但是，但是有一个问题就是，成本有点高。"
    "我们要不要，嗯，再考虑一下别的方案？比如说，其实我觉得第二个方案也不错。",
    "然后就是那个，客户那边催了两次了，说是，说是希望下周能看到demo。"
    "我们这边，这边进度大概完成了百分之六十吧。应该，应该来得及。",
]

# The written form on the left is what polish actually emits; the candidates are
# the ways it could be read. Duration picks between them.
PRONUNCIATION = {
    "B203": ["B二零三", "B两百零三"],
    "60%": ["百分之六十", "六十百分号"],
    "3点": ["三点", "三点钟"],
    "demo": ["地谋", "demo演示"],
}
CARRIER = "地点是{}。"

# How close a candidate has to be before the nearest one counts as the answer.
# Without it ``read_as`` always names something: an English word read in English
# has no Chinese candidate within 200 ms, and the minimum would still report the
# closest one as though it matched. The confident cases here land within 5 ms and
# the misses at 77 ms and up, so this sits between them with room on both sides.
MATCH_SECONDS = 0.05

# Paragraphs of one dictated meeting note, spoken cumulatively. Nested rather
# than independent so the length section adds material without changing register.
PARAGRAPHS = [
    "嗯，那个，我把今天周会的纪要说一下啊。就是，第一个是进度，"
    "客户端那边这个礼拜合了三个PR，然后，然后剩下两个还在评审里，"
    "预计，预计周五之前能合完。",
    "服务端这边，那个压测跑完了，单机QPS大概是一千二百左右，"
    "比上个季度高了大概百分之三十，但是，但是P99还是有点高，"
    "在八百毫秒上下，这块下周要专门看一下。",
    "第二个是那个招聘，就是说，我们那个后端还差两个人，"
    "HR那边推了五份简历，我看了一下，有两份还行，就是，下周约时间面一下。",
    "然后第三个，就是那个预算的事情，财务说，说是Q3的预算要在这个月底之前提交，"
    "所以，所以大家把自己那部分的数字，嗯，周三之前发给我，我来汇总。",
    "最后就是，那个下周三下午两点半，我们在B203开一个复盘会，"
    "主要是聊一下上个版本那个线上事故，就是那个，数据库连接池打满那次。",
    "对了，还有一个事，那个测试环境的机器，就是，运维说下周要迁一次，"
    "迁的时候大概停两个小时，所以，所以大家把要跑的东西提前跑完。",
    "还有就是，那个文档，我看了一下，就是接口那部分好久没更新了，"
    "然后新来的同学看着挺费劲的，这个，谁有空补一下。",
    "嗯，最后就是，下个月那个季度总结，就是，大家提前想一下自己那块要讲什么，"
    "反正，到时候一个人五分钟左右。",
]

FILLERS = ("嗯", "那个", "就是", "然后", "反正", "其实", "我觉得", "比如说", "这个")

# One word of Chinese in front is enough to hold the language decision; these
# are the shapes a dictation user actually produces on their own.
LANGUAGE_CASES = ["demo", "OA", "PR", "QPS", "OK", "看demo", "我们看一下demo", "好的", "知道了"]

PROSODY_CASES = {
    "misplaced": "我们要不要？再考虑一下别的方案。",
    "comma_instead": "我们要不要，再考虑一下别的方案。",
    "question_at_end": "我们要不要再考虑一下别的方案？",
    "statement_at_end": "我们要不要再考虑一下别的方案。",
}


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------


def speak(base: str, text: str, timeout: float = 900) -> bytes:
    body = json.dumps({"input": text, "response_format": "wav"}).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/v1/audio/speech", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def transcribe(base: str, samples: np.ndarray, timeout: float = 900) -> dict[str, Any]:
    pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()
    request = urllib.request.Request(
        f"{base}/v1/audio/transcriptions",
        data=pcm,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def decode(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes)) as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, rate


def to_input_rate(samples: np.ndarray, rate: int) -> np.ndarray:
    """Linear interpolation, which is what the synthesiser's 24k needs to reach 16k."""
    if rate == SAMPLE_RATE:
        return samples
    target = int(round(len(samples) * SAMPLE_RATE / rate))
    return np.interp(np.linspace(0.0, len(samples) - 1, target), np.arange(len(samples)), samples)


def gaps(
    samples: np.ndarray, rate: int, floor: float = 0.02, least_ms: int = 120
) -> list[tuple[float, float]]:
    """Interior silences as (start, length) in seconds, at 10 ms resolution.

    Interior on purpose. The leading silence is not a pause the speaker took,
    and counting it as one puts the first "pause" at offset zero -- which leaves
    nothing in front of it to measure a pitch slope over. That is exactly how the
    first version of this returned None for every case.
    """
    window = max(1, rate // 100)
    usable = len(samples) - len(samples) % window
    if usable <= 0:
        return []
    energy = np.abs(samples[:usable].reshape(-1, window)).max(axis=1)
    found, start = [], None
    for index, quiet in enumerate(energy < floor):
        if quiet and start is None:
            start = index
        elif not quiet and start is not None:
            if start > 0 and (index - start) * 10 >= least_ms:
                found.append((start / 100, round((index - start) / 100, 2)))
            start = None
    return found


def voiced_seconds(samples: np.ndarray, rate: int, floor: float = 0.02) -> float:
    """Length with the leading and trailing silence taken off.

    The carrier sentence puts its own silence at both ends, and comparing raw
    file lengths would measure that rather than the token under test.
    """
    loud = np.flatnonzero(np.abs(samples) >= floor)
    return float(loud[-1] - loud[0] + 1) / rate if len(loud) else 0.0


def f0_track(samples: np.ndarray, rate: int, low: int = 70, high: int = 400) -> np.ndarray:
    """Autocorrelation pitch, 40 ms frames every 10 ms, zero where unvoiced.

    Enough to answer "did it go up or come down", which is the whole question.
    Not enough to be a pitch tracker; do not read absolute values off it.
    """
    frame, hop = int(rate * 0.04), int(rate * 0.01)
    lag_low, lag_high = rate // high, rate // low
    track = []
    for start in range(0, max(0, len(samples) - frame), hop):
        chunk = samples[start : start + frame]
        if np.abs(chunk).max() < 0.015:
            track.append(0.0)
            continue
        chunk = chunk - chunk.mean()
        correlation = np.correlate(chunk, chunk, mode="full")[len(chunk) - 1 :]
        window = correlation[lag_low:lag_high]
        if len(window) == 0 or correlation[0] <= 0:
            track.append(0.0)
            continue
        lag = int(np.argmax(window)) + lag_low
        track.append(rate / lag if window.max() / correlation[0] > 0.25 else 0.0)
    return np.array(track)


def slope_before_gap(samples: np.ndarray, rate: int, milliseconds: int = 400) -> dict[str, Any]:
    """Pitch slope over the run-up to the first interior silence, in Hz per second."""
    found = gaps(samples, rate)
    if not found:
        return {"pause_at": None, "slope_hz_per_s": None, "voiced_frames": 0}
    at, _ = found[0]
    track = f0_track(samples, rate)
    end = min(len(track), int(at * 100))
    segment = track[max(0, end - milliseconds // 10) : end]
    index = np.flatnonzero(segment > 0)
    if len(index) < 5:
        return {"pause_at": at, "slope_hz_per_s": None, "voiced_frames": int(len(index))}
    return {
        "pause_at": at,
        "slope_hz_per_s": round(float(np.polyfit(index / 100.0, segment[index], 1)[0]), 1),
        "first_f0": round(float(segment[index[0]]), 1),
        "last_f0": round(float(segment[index[-1]]), 1),
        "voiced_frames": int(len(index)),
    }


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


def punctuation_profile(text: str) -> dict[str, Any]:
    """Marks per hundred characters, and the longest stretch without one.

    The rate alone hides the defect: a transcript can hold its average while
    concentrating the loss into one fifty-character run that a reader meets as a
    single unbreathing clause.
    """
    marks = sum(1 for character in text if character in PUNCTUATION)
    letters = len(text) - marks
    runs = [len(piece) for piece in re.split(f"[{re.escape(PUNCTUATION)}]", text) if piece]
    return {
        "characters": len(text),
        "marks": marks,
        "marks_per_100": round(marks / letters * 100, 2) if letters else 0.0,
        "longest_run": max(runs) if runs else 0,
        "runs_over_20": sum(1 for run in runs if run > 20),
    }


def filler_count(text: str) -> int:
    return sum(text.count(filler) for filler in FILLERS)


# ---------------------------------------------------------------------------
# the readings
# ---------------------------------------------------------------------------


def measure_notes(base: str, audio: Path | None) -> list[dict[str, Any]]:
    rows = []
    for index, note in enumerate(NOTES, start=1):
        spoken = speak(base, note)
        samples, rate = decode(spoken)
        heard = transcribe(base, to_input_rate(samples, rate))
        polished = heard["polished"] or heard["text"]

        read = speak(base, polished)
        read_samples, read_rate = decode(read)
        readback = transcribe(base, to_input_rate(read_samples, read_rate))
        if audio is not None:
            (audio / f"note{index}_spoken.wav").write_bytes(spoken)
            (audio / f"note{index}_read.wav").write_bytes(read)

        rows.append(
            {
                "note": index,
                "written": note,
                "transcribed": heard["text"],
                "language": heard["language"],
                "polished": polished,
                "readback": readback["text"],
                "cer_written_vs_transcribed": round(
                    character_error_rate(note, heard["text"], fold_numbers=True), 4
                ),
                "cer_polished_vs_readback": round(
                    character_error_rate(polished, readback["text"], fold_numbers=True), 4
                ),
                "spoken_seconds": round(len(samples) / rate, 2),
                "read_seconds": round(len(read_samples) / read_rate, 2),
                "read_peak": round(float(np.abs(read_samples).max()), 3),
                "read_pauses": [length for _, length in gaps(read_samples, read_rate)],
            }
        )
    return rows


def measure_pronunciation(base: str) -> dict[str, Any]:
    report = {}
    for written, candidates in PRONUNCIATION.items():
        rows = []
        for spelling in [written, *candidates]:
            bare, bare_rate = decode(speak(base, spelling))
            carried, carried_rate = decode(speak(base, CARRIER.format(spelling)))
            rows.append(
                {
                    "spelling": spelling,
                    "bare_seconds": round(voiced_seconds(bare, bare_rate), 3),
                    "carried_seconds": round(voiced_seconds(carried, carried_rate), 3),
                    "readback": transcribe(base, to_input_rate(bare, bare_rate))["text"],
                }
            )
        report[written] = pick_reading(rows)
    return report


def pick_reading(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which candidate spelling the written form was read as, or none of them.

    ``read_as`` is deliberately allowed to be None. Taking the minimum without a
    cutoff always names something, and an English word read in English has no
    Chinese candidate within two hundred milliseconds -- the instrument would
    then report 地谋 for ``demo`` with the same confidence it reports B二零三 for
    ``B203``, which is one of them being right and one of them being an artefact
    of asking for the nearest.
    """
    written = rows[0]
    for row in rows:
        row["carried_delta"] = round(row["carried_seconds"] - written["carried_seconds"], 3)
    nearest = min(rows[1:], key=lambda row: abs(row["carried_delta"]))
    return {
        "candidates": rows,
        "read_as": nearest["spelling"] if abs(nearest["carried_delta"]) <= MATCH_SECONDS else None,
        "nearest": nearest["spelling"],
        "nearest_delta": nearest["carried_delta"],
    }


def measure_length(base: str, counts: tuple[int, ...], audio: Path | None) -> list[dict[str, Any]]:
    rows = []
    for count in counts:
        said = "".join(PARAGRAPHS[:count])
        spoken = speak(base, said)
        samples, rate = decode(spoken)
        if audio is not None:
            (audio / f"length_{count}p.wav").write_bytes(spoken)
        heard = transcribe(base, to_input_rate(samples, rate))
        text = heard["text"]
        polished = heard["polished"] or text
        before, after = filler_count(text), filler_count(polished)
        rows.append(
            {
                "paragraphs": count,
                "spoken_seconds": round(len(samples) / rate, 1),
                "spoken": punctuation_profile(said),
                "transcribed": punctuation_profile(text),
                "longest_sentence": max((len(piece) for piece in split_sentences(text)), default=0),
                "trained_length": TRAINED_LENGTH,
                "groups": [len(group) for group in group_sentences(text)],
                "groups_over_trained_length": [
                    len(group) for group in group_sentences(text) if len(group) > TRAINED_LENGTH
                ],
                "consumed": round(consumed(text, polished), 4),
                "fillers_before": before,
                "fillers_after": after,
                "filler_clearance": round(1 - after / before, 4) if before else None,
                "text": text,
                "polished": polished,
            }
        )
    return rows


def measure_prosody(base: str) -> dict[str, Any]:
    report = {}
    for label, text in PROSODY_CASES.items():
        samples, rate = decode(speak(base, text))
        measured = slope_before_gap(samples, rate)
        measured["text"] = text
        measured["pauses"] = [length for _, length in gaps(samples, rate)]
        measured["readback"] = transcribe(base, to_input_rate(samples, rate))["text"]
        report[label] = measured
    return report


def measure_language(base: str) -> list[dict[str, Any]]:
    rows = []
    for said in LANGUAGE_CASES:
        samples, rate = decode(speak(base, said))
        heard = transcribe(base, to_input_rate(samples, rate))
        rows.append(
            {
                "said": said,
                "seconds": round(len(samples) / rate, 2),
                "language": heard["language"],
                "text": heard["text"],
                "flipped": heard["language"] not in (None, "zh"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8099")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="write the wav files here; leave unset to keep only the readings",
    )
    parser.add_argument(
        "--paragraphs",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="cumulative paragraph counts for the length section (8 is about two minutes)",
    )
    parser.add_argument("--skip", nargs="*", default=[], choices=list(SECTIONS))
    arguments = parser.parse_args()

    if arguments.audio is not None:
        arguments.audio.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {}
    for name in SECTIONS:
        if name in arguments.skip:
            continue
        print(f"--- {name}", flush=True)
        if name == "notes":
            report[name] = measure_notes(arguments.base, arguments.audio)
        elif name == "pronunciation":
            report[name] = measure_pronunciation(arguments.base)
        elif name == "length":
            report[name] = measure_length(
                arguments.base, tuple(arguments.paragraphs), arguments.audio
            )
        elif name == "prosody":
            report[name] = measure_prosody(arguments.base)
        elif name == "language":
            report[name] = measure_language(arguments.base)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {arguments.out}")


SECTIONS = ("notes", "pronunciation", "length", "prosody", "language")


if __name__ == "__main__":
    main()
