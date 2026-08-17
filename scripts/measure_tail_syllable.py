"""Does the last character get spoken, or swallowed.

Reported by ear on the local synthesiser's output: "the last character is not
finished" -- 慢慢看 losing its 看, 一碗面就成了 losing its 了. Nothing already
here sees it. Read-back CER does not: the recogniser reconstructs the character
from a partial syllable, so the transcript is correct and the criterion reads
0.0080, which the round that produced it explained away as "these texts are
harder". The waveform does not either: the file ends in silence, not a hard cut,
so there is no discontinuity to find. The timbre and level instruments are
about other things entirely.

So this asks the question directly: **how much audio does the last character
occupy**. Trim T milliseconds off the end and hand what is left to the
recogniser. A character that was properly spoken survives a trim of its own
length; one that was swallowed disappears almost immediately. Sweeping T and
reading where it stops surviving is a duration, in the one unit that matters --
the audio the listener actually gets.

Two things this is not:

* Not absolute. The recogniser reconstructs, so the number is "how far this
  recogniser can still find it", which is longer than a person would call
  intelligible. Compare arms with it; do not read a single arm's value as
  milliseconds of speech.
* Not a listening test. It says a syllable is short, not that a listener minds.
  The report that started this was a person, and it stays that way.

    python scripts/measure_tail_syllable.py \\
        --arm t10=<dir> --arm t6=<dir> --asr <SenseVoiceSmall> \\
        --texts artifacts/polish_train/val_service_words.jsonl --limit 8 \\
        --report artifacts/voxcpm-tail-2026-08-17.json

Each arm directory is one produced by ``measure_voxcpm_cost.py``: ``note<N>.wav``
plus the manifest naming what each was asked to say.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import statistics
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

CUTS_MS = (0, 50, 100, 150, 200, 300, 400)
TRAILING = "。，！？、；：,.!?;: "


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(path.read_bytes())) as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, rate


def drop_trailing_silence(samples: np.ndarray, floor: float = 0.01) -> np.ndarray:
    """Trim from where the speech ends, not where the file does.

    Otherwise the sweep spends its first steps on however much padding the
    synthesiser happened to append, and two arms with different padding are not
    being asked the same question.
    """
    loud = np.flatnonzero(np.abs(samples) >= floor)
    return samples[: loud[-1] + 1] if len(loud) else samples


def survives_up_to(present: list[bool], cuts: tuple[int, ...]) -> int:
    """The largest trim the character survived before the first time it did not."""
    lasted = 0
    for cut, ok in zip(cuts, present, strict=True):
        if not ok:
            break
        lasted = cut
    return lasted


def measure_arm(
    directory: Path, hear: Any, cuts: tuple[int, ...], limit: int | None
) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for sample in manifest["samples"][:limit]:
        wanted = sample["reference_text"].rstrip(TRAILING)
        if not wanted:
            continue
        last = wanted[-1]
        samples, rate = read_wav(directory / sample["audio_path"])
        samples = drop_trailing_silence(samples)
        present = []
        for cut in cuts:
            keep = len(samples) - int(rate * cut / 1000)
            piece = samples[: max(keep, 1)]
            present.append(hear(piece, rate).rstrip(TRAILING).endswith(last))
        rows.append(
            {
                "id": sample["id"],
                "last_character": last,
                "present": present,
                "survives_up_to_ms": survives_up_to(present, cuts),
            }
        )
    lasted = [row["survives_up_to_ms"] for row in rows]
    return {
        "n": len(rows),
        "median_ms": statistics.median(lasted) if lasted else None,
        "mean_ms": round(statistics.fmean(lasted), 1) if lasted else None,
        "gone_at_50ms": sum(1 for value in lasted if value < 50),
        "clips": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()

    from mindsurf_omni.service.asr import SenseVoiceRecogniser

    recogniser = SenseVoiceRecogniser(model_dir=arguments.asr, device=arguments.device)
    recogniser.load()

    def hear(samples: np.ndarray, rate: int) -> str:
        pcm = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16).tobytes()
        text, _ = asyncio.run(recogniser.transcribe(pcm, rate))
        return text

    report: dict[str, Any] = {"cuts_ms": list(CUTS_MS), "arms": {}}
    for entry in arguments.arm:
        name, _, directory = entry.partition("=")
        block = measure_arm(Path(directory), hear, CUTS_MS, arguments.limit)
        report["arms"][name] = block
        marks = " ".join(
            "".join("o" if ok else "." for ok in row["present"]) for row in block["clips"]
        )
        print(
            f"{name:<10} n={block['n']}  末字撑到 中位 {block['median_ms']} ms"
            f"  均值 {block['mean_ms']} ms  50 ms 内就没的 {block['gone_at_50ms']}/{block['n']}"
        )
        print(f"           {marks}")

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {arguments.report}")


if __name__ == "__main__":
    main()
