"""Score the path the service actually runs, not the one the measurement script does.

``measure_polish.py`` calls the model once on the whole transcript. The service
does not: since driving it by hand showed 164 seconds of dictation coming back
as 18 characters, ``Polisher.polish`` splits on sentence marks, polishes each
piece, and keeps the input for any piece whose output stopped early. That is a
different function and it had never been scored.

Also carries the skip filter, which is off unless asked for. A sentence holding
no filler word and no adjacent repetition has nothing this stage can remove, so
calling a 140 ms model on it buys nothing -- and measured over the held-out
transcripts the model *does* edit some of them, which by construction is
over-deletion. The filter is therefore a candidate for being faster and better
at the same time, which is why it is measured rather than assumed.

    python scripts/measure_polish_service.py --pairs artifacts/polish_train/pairs_holdout.jsonl \
        --checkpoint out/sft_polish6_768.pth --minimind-root ~/omni/minimind-o \
        --output artifacts/polish_train/val_service.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.service.polish import (  # noqa: E402
    BRIDGING_FILLERS,
    LEADING_FILLERS,
    RECOGNISED_FILLERS,
    Polisher,
    group_sentences,
)
from scripts.measure_polish import (  # noqa: E402
    DOUBLE_DUTY,
    content_kept,
    filler_removed,
    invented,
    polished_cer,
)

# Wider than the decoder's door on purpose. The door decides whether a span may
# be deleted, so a wrong entry costs content; this decides whether the model is
# called at all, so a wrong entry costs 140 ms. The costs are not symmetric and
# neither should the lists be -- 饿 恶 啊 温 are here and not in the door.
WORTH_A_LOOK = (
    *LEADING_FILLERS,
    *BRIDGING_FILLERS,
    *RECOGNISED_FILLERS,
    "饿",
    "恶",
    "啊",
    "恩",
    "扼",
    "温",
    "反证",
    "其是",
    "奇实",
)


def has_repetition(text: str, longest: int = 5) -> bool:
    return any(
        text[start : start + size] == text[start + size : start + 2 * size]
        for size in range(2, longest + 1)
        for start in range(len(text) - 2 * size + 1)
    )


def worth_polishing(piece: str) -> bool:
    """Whether this sentence holds anything the stage could remove."""
    return any(word in piece for word in WORTH_A_LOOK) or has_repetition(piece)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--minimind-root", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--tagger",
        type=Path,
        help="the second arm, merged by veto. Measured here rather than offline "
        "because the service merges per grouped piece, not over the whole "
        "transcript -- the piece boundary is part of the product",
    )
    parser.add_argument("--tagger-backbone", type=Path)
    parser.add_argument("--tagger-threshold", type=float, default=0.5)
    parser.add_argument(
        "--punctuation-insertable",
        action="store_true",
        help="let the decode emit marks the transcript did not; content is still copied",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == args.split][: args.limit]

    polisher = Polisher(
        checkpoint=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        minimind_root=args.minimind_root,
        device=args.device,
        tagger=args.tagger,
        tagger_backbone=args.tagger_backbone,
        tagger_threshold=args.tagger_threshold,
        punctuation_insertable=args.punctuation_insertable,
    )
    polisher.load()

    written, latencies, calls, pieces_seen = [], [], 0, 0

    async def run() -> None:
        """Through ``Polisher.polish``, not through a copy of it.

        This loop used to rebuild the grouping, the skip filter and the floor
        here and call the private ``_polish`` underneath -- a faithful copy of
        the method on the day it was written, and silently not one after that.
        It stopped being faithful the moment the merge went into ``polish``:
        every number came back identical to the arm without a tagger, because
        the tagger was in the product and not in what this script ran.

        One event loop for all of them: the batching queue's futures belong to
        the loop that made them, and a fresh ``asyncio.run`` per row hands the
        second call a queue bound to a loop that has closed.
        """
        nonlocal calls, pieces_seen
        for index, row in enumerate(rows, start=1):
            started = time.perf_counter()
            for piece in group_sentences(row["source"]):
                pieces_seen += 1
                if piece.strip() and worth_polishing(piece):
                    calls += 1
            text = await polisher.polish(row["source"])
            latencies.append((time.perf_counter() - started) * 1000)
            _record(row, text, index)

    def _record(row: dict[str, Any], text: str, index: int) -> None:
        arrived, removed = filler_removed(row, text)
        # Split as well as combined, because the combined figure answers two
        # different questions at once and the answers moved apart. The injector
        # plants nine leading fillers uniformly and seven of them are
        # double-duty: the 这个 it planted is labelled "delete this" while the
        # 这个 in 这个模块 is not, same spelling, same slot. Every fix that
        # stopped the stage deleting a double-duty word in a content position
        # -- and a run of them came out of driving the service -- pushes the
        # combined number down while making the product better. Only the gated
        # figure, which leaves those words out, is a number to hold a line
        # against.
        gated_arrived, gated_removed = filler_removed(row, text, exclude=DOUBLE_DUTY)
        double_arrived, double_removed = filler_removed(row, text, DOUBLE_DUTY)
        written.append(
            {
                **row,
                "polished": text,
                "cer_before": polished_cer(row["target"], row["source"]),
                "cer_after": polished_cer(row["target"], text),
                "content_kept": content_kept(row["target"], text),
                "invented": invented(row["target"], text),
                "filler_arrived": arrived,
                "filler_removed": removed,
                "gated_arrived": gated_arrived,
                "gated_removed": gated_removed,
                "double_arrived": double_arrived,
                "double_removed": double_removed,
                # CER, invented and filler_removed are three views of one
                # question -- did the injected characters go away -- and this
                # row's target is the text with all nine words taken out. So a
                # 这个 correctly left in a content position is charged three
                # times, and on the rows that got one the three numbers move
                # together and against the fix. Split on it so the half where
                # the target is trustworthy can be read on its own.
                "got_double_duty": double_arrived > 0,
                "elapsed_ms": latencies[-1],
            }
        )
        if index % 100 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    import asyncio

    asyncio.run(run())

    arrived = sum(row["filler_arrived"] for row in written)

    def _rate(removed_key: str, arrived_key: str) -> float | None:
        """None rather than 0.0 when nothing of that kind arrived.

        Zero would read as "removed none of it" and go into a table next to
        numbers that mean that.
        """
        seen = sum(row[arrived_key] for row in written)
        return sum(row[removed_key] for row in written) / seen if seen else None

    def _half(got_double: bool) -> dict[str, Any]:
        here = [row for row in written if row["got_double_duty"] is got_double]
        if not here:
            return {"rows": 0}
        return {
            "rows": len(here),
            "cer_after": statistics.fmean(row["cer_after"] for row in here),
            "cer_before": statistics.fmean(row["cer_before"] for row in here),
            "content_kept": statistics.fmean(row["content_kept"] for row in here),
            "invented": statistics.fmean(row["invented"] for row in here),
        }

    summary = {
        "n": len(written),
        # Always true now: the skip lives inside Polisher.polish, so a run
        # without it would be measuring something the service does not do. The
        # ablation it existed for is recorded (46.7% fewer calls, the four
        # numbers not worse) and does not need re-running.
        "skip_clean": True,
        "tagger": str(args.tagger) if args.tagger else None,
        "tagger_threshold": args.tagger_threshold if args.tagger else None,
        "sentences": pieces_seen,
        "model_calls": calls,
        "calls_per_sentence": calls / max(pieces_seen, 1),
        "cer_after": statistics.fmean(row["cer_after"] for row in written),
        "filler_removed_rate": (
            sum(row["filler_removed"] for row in written) / arrived if arrived else 0.0
        ),
        # 嗯 呃 你知道吧 怎么说呢 对吧 -- the words whose label and whose
        # judgement agree. This is the one to read.
        "filler_removed_rate_gated": _rate("gated_removed", "gated_arrived"),
        # 我觉得 那个 这个 就是 然后 反正 其实 那种 那些 那 就 对 -- obedience to
        # the injector, not correctness. Reported so the combined figure can be
        # taken apart rather than argued about.
        "filler_removed_rate_double_duty": _rate("double_removed", "double_arrived"),
        # The same split applied to the other two. Named by what the target is
        # worth on that half rather than by the injection, because that is what
        # the reader has to decide about.
        "rows_with_a_trustworthy_target": _half(False),
        "rows_whose_target_wants_a_content_word_gone": _half(True),
        "content_kept": statistics.fmean(row["content_kept"] for row in written),
        "invented": statistics.fmean(row["invented"] for row in written),
        "empty": sum(1 for row in written if not row["polished"].strip()),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": sorted(latencies)[int(0.95 * len(latencies))],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in written)
            + "\n",
            encoding="utf-8",
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
