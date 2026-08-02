"""The first ruler this project has for whose turn it is.

Every quality number here answers "does it sound good". None of them answers
"does it hold the floor too long", and a practitioner running a voice agent in
production for eight months reports that half of what users call "robotic" is
turn-taking rather than voice -- the agent stepping on people or leaving dead
air (docs/experiments/2026-08-01-field-comparison.md section 6.1).

We are turn-based by contract, so the stepping-on-people half is the client's
endpoint decision and not ours (docs/DECISIONS.md section 11). What IS ours is
the other half: once we start talking, how long do we hold the floor, and what
does a user who runs out of patience lose by cutting us off.

Three numbers, all computed from replies already on disk:

* **occupancy** -- spoken seconds of the reply over spoken seconds of the
  prompt. Natural conversation sits near parity; a ratio of ten means the user
  asked for three seconds and got half a minute.
* **unfinished at T** -- the share of replies still going at T seconds. This is
  what a barge-in actually costs: if most replies are unfinished when patience
  runs out, most of what we generate is never heard.
* **front-loading** -- how much of the reply arrives in the first T seconds.

Seconds come from characters at a measured rate, so they inherit that
estimate's limits: characters only proxy duration, and the rate belongs to one
synthesiser. See SPOKEN_CHARS_PER_SECOND in measure_chat_loss.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scripts.measure_chat_loss import SPOKEN_CHARS_PER_SECOND

# What a listener will sit through before wanting to cut in. Not measured here
# and not a threshold anything passes or fails -- these are the points the
# curve is reported at, chosen to bracket the corpus's own 11.3 second median
# assistant turn.
PATIENCE_SECONDS = (3.0, 5.0, 10.0, 15.0, 30.0)


def seconds(text: str) -> float:
    return len(text) / SPOKEN_CHARS_PER_SECOND


def measure(replies: list[dict[str, Any]]) -> dict[str, Any]:
    spoken = [seconds(row["reply"]) for row in replies]
    asked = [seconds(row.get("prompt", "")) for row in replies]
    # A prompt of zero length would make the ratio infinite, and a probe set
    # without prompts is a different kind of file than this reads.
    ratios = [reply / ask for reply, ask in zip(spoken, asked, strict=True) if ask > 0]

    return {
        "replies": len(replies),
        "spoken_seconds": {
            "median": statistics.median(spoken),
            "p90": sorted(spoken)[int(0.9 * len(spoken))],
            "max": max(spoken),
        },
        "prompt_seconds_median": statistics.median(asked) if asked else None,
        "occupancy": {
            "median": statistics.median(ratios) if ratios else None,
            "note": "reply spoken seconds per second the user spoke",
        },
        "unfinished_at": {
            str(limit): sum(1 for value in spoken if value > limit) / len(spoken)
            for limit in PATIENCE_SECONDS
        },
        "heard_fraction_by": {
            str(limit): statistics.median(min(1.0, limit / value) for value in spoken)
            for limit in PATIENCE_SECONDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replies",
        type=Path,
        nargs="+",
        required=True,
        help="reports from measure_chat_loss --generate",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report: dict[str, Any] = {"rate_chars_per_second": SPOKEN_CHARS_PER_SECOND, "arms": {}}
    for path in args.replies:
        payload = json.loads(path.read_text(encoding="utf-8"))
        replies = payload.get("replies") or []
        if not replies:
            raise SystemExit(f"{path} has no replies; run measure_chat_loss with --generate")
        entry = measure(replies)
        entry["checkpoint"] = payload.get("checkpoint")
        report["arms"][path.stem] = entry

        print(f"\n{path.stem}  n={entry['replies']}  checkpoint={entry['checkpoint']}")
        print(
            f"  说话 中位 {entry['spoken_seconds']['median']:.1f}s  "
            f"P90 {entry['spoken_seconds']['p90']:.1f}s  最长 {entry['spoken_seconds']['max']:.1f}s"
        )
        if entry["occupancy"]["median"]:
            print(
                f"  占用比 中位 {entry['occupancy']['median']:.1f}×"
                f"（用户说 {entry['prompt_seconds_median']:.1f}s）"
            )
        unfinished = "  ".join(
            f"{limit}s {share:.0%}" for limit, share in entry["unfinished_at"].items()
        )
        print(f"  还没说完的比例: {unfinished}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n报告 {args.output}")


if __name__ == "__main__":
    main()
