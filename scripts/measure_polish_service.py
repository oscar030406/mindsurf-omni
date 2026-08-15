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

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.evaluation.metrics import character_error_rate  # noqa: E402
from mindsurf_omni.service.polish import (  # noqa: E402
    BRIDGING_FILLERS,
    LEADING_FILLERS,
    RECOGNISED_FILLERS,
    Polisher,
    group_sentences,
)
from scripts.measure_polish import content_kept, filler_removed, invented  # noqa: E402

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
        "--skip-clean",
        action="store_true",
        help="do not call the model on a sentence with no filler and no repetition",
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
    )
    polisher.load()

    written, latencies, calls, pieces_seen = [], [], 0, 0
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        out = []
        for piece in group_sentences(row["source"]):
            pieces_seen += 1
            if not piece.strip() or (args.skip_clean and not worth_polishing(piece)):
                out.append(piece)
                continue
            calls += 1
            polished = polisher._polish(piece)  # noqa: SLF001
            from mindsurf_omni.service.polish import FLOOR, consumed

            out.append(polished if consumed(piece, polished) >= FLOOR else piece)
        text = "".join(out)
        latencies.append((time.perf_counter() - started) * 1000)
        arrived, removed = filler_removed(row, text)
        written.append(
            {
                **row,
                "polished": text,
                "cer_before": character_error_rate(row["target"], row["source"]),
                "cer_after": character_error_rate(row["target"], text),
                "content_kept": content_kept(row["target"], text),
                "invented": invented(row["target"], text),
                "filler_arrived": arrived,
                "filler_removed": removed,
                "elapsed_ms": latencies[-1],
            }
        )
        if index % 100 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    arrived = sum(row["filler_arrived"] for row in written)
    summary = {
        "n": len(written),
        "skip_clean": args.skip_clean,
        "sentences": pieces_seen,
        "model_calls": calls,
        "calls_per_sentence": calls / max(pieces_seen, 1),
        "cer_after": statistics.fmean(row["cer_after"] for row in written),
        "filler_removed_rate": (
            sum(row["filler_removed"] for row in written) / arrived if arrived else 0.0
        ),
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
