"""若把流式换成「每隔几秒把整段重转一次」，屏幕会不会一直在闪。

现在说话期间出字走的是 Paraformer 流式：只追加、不改写，字不跳，但**一块听错
就一直错到松手**。在 RAMC 上它 0.2796，同一批 SenseVoice 整段 0.1094，2.56 倍。

另一条路是把流式识别器整个拿掉：SenseVoice 本来就是整段非自回归的，说话期间
每隔几秒拿当前 buffer 重跑一次、把显示替换掉。这样预览的质量就是 0.1094，还少
维护一个模型、少一份权重。代价是**已经显示出来的字会被改写**。

这个脚本量的就是那个代价，而且不需要真值——它是同一段音频自己跟自己比：

* 把 buffer 从 1 秒起按步长喂给 SenseVoice，每次拿到一份完整转写；
* 相邻两次求最长公共前缀，前缀之后的部分就是「被改写的」；
* 报**改写发生在离末尾多远的地方**。如果改写都挤在末尾几个字里，那这条路成立
  ——把末尾那一小段标成未定稿，前面的冻住，屏幕就只有尾巴在动；如果一次重跑
  能把三十个字前的内容改掉，用户会看见整段文字重排，那就得退回流式顶前沿。

判据名字用业界的：**Edit Overhead**（改写掉的字数 ÷ 最终字数，Niehues 等人在
流式翻译里用的那个）和稳定滞后（离末尾多少字之后就不再变）。

这批音频没有人工转写，也不需要——问的是「它自己稳不稳」，不是「它对不对」。

    python scripts/measure_retranscribe_stability.py \\
        --audio /home/oscar/omni/dictation_test --step 2.0 \\
        --model-dir ~/omni/minimind-o/model/SenseVoiceSmall \\
        --report artifacts/retranscribe-stability.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

RATE = 16_000


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes()), handle.getframerate()


# A pass that moves only a comma is not the same event as one that changes a
# word, and a reader does not react to them the same way. Both get counted.
MARKS = "，。！？；：、,.!?;: " + chr(10)


def bare(text: str) -> str:
    return "".join(c for c in text if c not in MARKS)


def common_prefix(before: str, after: str) -> int:
    limit = min(len(before), len(after))
    index = 0
    while index < limit and before[index] == after[index]:
        index += 1
    return index


async def one_clip(recogniser, pcm: bytes, rate: int, step: float, start: float) -> dict:
    """Every pass over a growing buffer, and what each pass moved."""
    width = 2  # 16-bit mono
    seconds = len(pcm) / (rate * width)
    marks = []
    at = start
    while at < seconds:
        marks.append(at)
        at += step
    marks.append(seconds)

    passes: list[str] = []
    costs: list[float] = []
    for mark in marks:
        cut = int(mark * rate) * width
        began = time.perf_counter()
        text, _ = await recogniser.transcribe(pcm[:cut], rate)
        costs.append(1000.0 * (time.perf_counter() - began))
        passes.append(text or "")

    moves = []
    for before, after in zip(passes, passes[1:], strict=False):
        keep = common_prefix(before, after)
        moves.append(
            {
                # How much of what the reader had already seen got taken back.
                "rewritten": len(before) - keep,
                # Where the rewrite began, counted back from the end of what was
                # on screen. Small means only the tail moved.
                "from_end": len(before) - keep,
                # The same question with punctuation taken out, so a display
                # that only re-punctuates is not booked as one that rewrites.
                "rewritten_words": len(bare(before)) - common_prefix(bare(before), bare(after)),
                "was": len(before),
                "now": len(after),
            }
        )

    final = passes[-1]
    rewritten = sum(move["rewritten"] for move in moves)
    return {
        "seconds": seconds,
        "passes": len(passes),
        "final_chars": len(final),
        "rewritten_chars": rewritten,
        # Niehues' Edit Overhead: what the display took back, over what it
        # finally showed. 0.0 is append-only.
        "edit_overhead": rewritten / len(final) if final else 0.0,
        "rewrite_depth": [move["from_end"] for move in moves],
        "rewrite_depth_words": [move["rewritten_words"] for move in moves],
        "pass_ms_median": statistics.median(costs),
        "pass_ms_max": max(costs),
        # Cost against how much audio the pass had to read. Re-transcribing the
        # whole buffer gets more expensive the longer somebody talks, and how
        # fast it grows decides whether the whole buffer can be re-read at all
        # or only a window of it.
        "pass_cost": [
            {"buffer_s": round(mark, 2), "ms": round(cost, 1)}
            for mark, cost in zip(marks, costs, strict=True)
        ],
        "final": final,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="wav 目录")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--step", type=float, default=2.0, help="每隔几秒重跑一次")
    parser.add_argument("--start", type=float, default=1.0, help="第一次重跑在第几秒")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    from mindsurf_omni.service.asr import SenseVoiceRecogniser

    recogniser = SenseVoiceRecogniser(
        model_dir=args.model_dir, device=args.device, language=args.language
    )

    clips = sorted(args.audio.glob("*.wav"))[: args.limit]
    if not clips:
        raise SystemExit(f"{args.audio} 里没有 wav")
    print(f"{len(clips)} 段，每 {args.step} 秒重跑一次", flush=True)

    report = {}
    depths: list[int] = []
    word_depths: list[int] = []
    print(f"\n{'音频':>22}{'秒':>7}{'趟数':>6}{'最终字':>8}{'改写字':>8}{'EO':>8}{'一趟ms':>9}")
    for clip in clips:
        pcm, rate = read_wav(clip)
        if rate != RATE:
            print(f"  跳过 {clip.name}：{rate} Hz，不是 16 kHz")
            continue
        got = await one_clip(recogniser, pcm, rate, args.step, args.start)
        report[clip.name] = got
        depths.extend(got["rewrite_depth"])
        word_depths.extend(got["rewrite_depth_words"])
        print(
            f"{clip.name:>22}{got['seconds']:>7.1f}{got['passes']:>6}"
            f"{got['final_chars']:>8}{got['rewritten_chars']:>8}"
            f"{got['edit_overhead']:>8.3f}{got['pass_ms_median']:>9.0f}",
            flush=True,
        )

    if depths:
        ordered = sorted(depths)
        print(f"\n改写深度（离末尾几个字）共 {len(depths)} 次重跑：")
        for share in (0.5, 0.9, 0.95, 1.0):
            index = min(int(share * (len(ordered) - 1)), len(ordered) - 1)
            print(f"  p{int(share * 100):<3} {ordered[index]:>4} 字")
        print(f"  一个字都没改的重跑：{sum(1 for d in depths if d == 0)}/{len(depths)}")
        bare_sorted = sorted(word_depths)
        print("  只看正文（标点不算）：")
        for share in (0.5, 0.9, 0.95, 1.0):
            index = min(int(share * (len(bare_sorted) - 1)), len(bare_sorted) - 1)
            print(f"    p{int(share * 100):<3} {bare_sorted[index]:>4} 字")
        no_move = sum(1 for d in word_depths if d == 0)
        print(f"    正文一个字没动的重跑：{no_move}/{len(word_depths)}")
        report["_depth"] = {
            "n": len(depths),
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[min(int(0.9 * (len(ordered) - 1)), len(ordered) - 1)],
            "max": ordered[-1],
            "unchanged": sum(1 for d in depths if d == 0),
            "words_p50": sorted(word_depths)[len(word_depths) // 2],
            "words_p90": sorted(word_depths)[
                min(int(0.9 * (len(word_depths) - 1)), len(word_depths) - 1)
            ],
            "words_unchanged": sum(1 for d in word_depths if d == 0),
        }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n写到 {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
