"""Blind pairwise preference between two arms, with the judge on the record.

A verdict is worth only as much as the record of who produced it, and a judge's
identity cannot be recovered after the fact. So every result this writes carries
the judge's identity, the endpoint, the seed that chose the sides, and a hash of
the prompt that was sent.

What it does not fix: a re-judgement by a different judge that disagrees does
not mean the old number was wrong. It means the verdict depends on the judge,
which is a finding about the instrument and belongs in the write-up as one.

Blinding is on our side of the call. Each probe's two replies are assigned to
left and right by a seeded coin, the judge never sees which arm is which, and
the share of wins that landed on the right is reported so a judge with a side
preference is visible rather than assumed away.

Ties are counted, never split. A pair the judge could not separate carries no
preference, and folding it in as half a win invents one.

    JUDGE_API_KEY=... python scripts/blind_preference.py \
        --arm base=artifacts/chat-sft_graft_frozen-ext.json \
        --arm length=artifacts/chat-sft_len-ext.json \
        --output artifacts/blind-length-rejudged.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

PROMPT = """你在做一次盲评。下面是同一个问题的两个回答，来自两个不同的系统。

请判断哪个回答更好。判断依据是**实质**：是否答到了问题、信息是否准确、是否可用。
**简短不是缺点，冗长也不是优点。** 不要因为一个回答更长就认为它更用心，
也不要因为一个更短就认为它敷衍。

问题：{prompt}

回答 A：
{left}

回答 B：
{right}

只回一个词：A、B 或 平局。"""


def load_credentials(path: Path) -> dict[str, str]:
    """The local file first, then the environment.

    A key belongs in none of three places. Not the source, which is tracked and
    which the gate rejects on the way out. Not a machine-wide variable, which
    hands it to every process on the box. And not inside the project directory
    either, even gitignored: verify_delivery scans the working tree rather than
    the index, because packaging the project copies what is on disk and not what
    git would ship -- it caught exactly this and was right to. So the default
    sits under the user's home, one file, one place to delete it.
    """
    settings: dict[str, str] = {}
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        settings = {k: v for k, v in blob.items() if isinstance(v, str) and v.strip()}
    for field, variable in (
        ("api_key", "JUDGE_API_KEY"),
        ("base_url", "JUDGE_BASE_URL"),
        ("model", "JUDGE_MODEL"),
    ):
        settings.setdefault(field, os.environ.get(variable, ""))
    placeholder = settings.get("api_key", "").startswith("<")
    return {k: v for k, v in settings.items() if v and not (k == "api_key" and placeholder)}


def load_replies(path: Path) -> dict[str, dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    replies = blob.get("replies")
    if not replies:
        raise SystemExit(f"{path} 没有 replies 字段，它记的不是这个臂自己的生成")
    return {row["id"]: row for row in replies}


def build_pairs(
    first: tuple[str, dict[str, Any]], second: tuple[str, dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """One pair per shared probe, sides chosen by a seeded coin."""
    (left_label, left_rows), (right_label, right_rows) = first, second
    rng = random.Random(seed)
    pairs = []
    for probe in sorted(set(left_rows) & set(right_rows)):
        a, b = left_rows[probe], right_rows[probe]
        flip = rng.random() < 0.5
        pairs.append(
            {
                "id": probe,
                "prompt": a.get("prompt") or b.get("prompt", ""),
                "left": b["reply"] if flip else a["reply"],
                "right": a["reply"] if flip else b["reply"],
                # Which side the second arm landed on, kept out of the prompt.
                "second_side": "left" if flip else "right",
                "first_arm": left_label,
                "second_arm": right_label,
            }
        )
    return pairs


def load_prebuilt(path: Path) -> list[dict[str, Any]]:
    """Pairs that ``build_preference_pairs.py pairs`` already formed.

    That step draws several drafts of one checkpoint against each other, which
    is a shape this file cannot build: it pairs two arms. Judging them had no
    entry point here, so the previous rounds were judged outside the repository
    and the judge's identity did not survive -- the failure this module's
    docstring is about. It has one now, and it writes the same record.

    Sides were shuffled when the pairs were formed and the key kept separate,
    so nothing here re-randomises them; doing so would detach the judgement
    from the pair the judge saw.
    """
    blob = json.loads(path.read_text(encoding="utf-8"))
    pairs = blob["pairs"] if isinstance(blob, dict) else blob
    missing = [p for p in pairs if not p.get("key")]
    if missing:
        raise SystemExit(
            f"{path} 里有 {len(missing)} 个对没有 key。判决是按 key 贴回去的，"
            "没有 key 的话半数标签会落到判官没看过的那一对上"
        )
    return list(pairs)


def judge_one(client: Any, model: str, pair: dict[str, Any]) -> str:
    text = PROMPT.format(prompt=pair["prompt"], left=pair["left"], right=pair["right"])
    for attempt in range(4):
        try:
            reply = client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": text}],
                    "temperature": 0,
                    "max_tokens": 8,
                },
            )
            reply.raise_for_status()
            answer = reply.json()["choices"][0]["message"]["content"].strip()
        except Exception:  # transient endpoint failure should not kill 158 calls
            if attempt == 3:
                raise
            continue
        if answer.startswith("A"):
            return "left"
        if answer.startswith("B"):
            return "right"
        return "tie"
    return "tie"


def tally(pairs: list[dict[str, Any]], second_arm: str) -> dict[str, Any]:
    decided = [p for p in pairs if p.get("winner") in ("left", "right")]
    ties = len(pairs) - len(decided)
    second_wins = sum(1 for p in decided if p["winner"] == p["second_side"])
    right_wins = sum(1 for p in decided if p["winner"] == "right")
    if not decided:
        return {"n": len(pairs), "tie": ties, "verdict": "全部平局，判不了"}
    share = second_wins / len(decided)
    noise = 1.96 * math.sqrt(0.25 / len(decided))

    # A side preference and a distorted estimate are different things, and the
    # first version of this conflated them. Reporting only the share of wins
    # landing on the right flags a property of the judge -- real, worth knowing
    # -- but says nothing about whether the headline number is contaminated,
    # because sides are assigned by a seeded coin. Randomisation cancels a side
    # bias out of the pooled estimate to the extent the seats came out even.
    #
    # So both are reported. ``side_bias`` is how much the judge favours the left
    # seat, estimated from the two seatings; ``residual_bias`` is what actually
    # leaks into the share, which is the side bias times the seat imbalance.
    # Measured once at 608 probes: a side bias of +0.0726 with seats at 0.4775
    # left moved the estimate by -0.0033, three tenths of a point, while the
    # share-of-right test read as a failure. Gate on the residual.
    left = [p for p in decided if p["second_side"] == "left"]
    right = [p for p in decided if p["second_side"] == "right"]
    if left and right:
        won_left = sum(1 for p in left if p["winner"] == "left") / len(left)
        won_right = sum(1 for p in right if p["winner"] == "right") / len(right)
        side_bias = (won_left - won_right) / 2
        left_share = len(left) / len(decided)
        residual = side_bias * (2 * left_share - 1)
        seating = {
            "side_bias": side_bias,
            "left_seat_share": left_share,
            "residual_bias": residual,
            "seat_balanced_share": (won_left + won_right) / 2,
        }
    else:
        # Every pair on one seat: the coin cannot cancel what it never varied,
        # so the share carries the whole side bias and none of it is estimable.
        seating = {
            "side_bias": None,
            "left_seat_share": 1.0 if left else 0.0,
            "residual_bias": None,
            "seat_balanced_share": None,
            "note": "所有对都在同一边，位置偏好无法与胜率分开",
        }
    return {
        "n_pairs": len(pairs),
        "tie": ties,
        "n_decided": len(decided),
        f"{second_arm}_wins": second_wins,
        f"{second_arm}_share": share,
        "noise": noise,
        # Kept because it is the raw observation, but it is not the gate.
        "position_right_share": right_wins / len(decided),
        "seating": seating,
        "verdict": "赢" if share - 0.5 > noise else ("输" if 0.5 - share > noise else "未胜出"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", metavar="LABEL=PATH")
    parser.add_argument(
        "--judge-pairs",
        type=Path,
        help="judge a pairs file from build_preference_pairs instead of two arms. "
        "Writes the [{key, winner}] list that its resolve step reads, and a "
        "provenance sidecar beside it",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="write every judged pair here too")
    parser.add_argument(
        "--credentials", type=Path, default=Path.home() / ".mindsurf" / "judge.json"
    )
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="first N pairs, for a smoke run")
    args = parser.parse_args()

    if bool(args.arm) == bool(args.judge_pairs):
        raise SystemExit("要么给两个 --arm，要么给一个 --judge-pairs，不能都给也不能都不给")
    if args.arm and len(args.arm) != 2:
        raise SystemExit("要正好两个 --arm")
    settings = load_credentials(args.credentials)
    key = settings.get("api_key")
    if not key:
        raise SystemExit(
            f"没有判官凭据。把 key 填进 {args.credentials} 的 api_key 字段"
            "（.gitignore 已经挡住那个文件），或者设 JUDGE_API_KEY 环境变量。"
        )
    base = settings.get("base_url") or "https://api.deepseek.com/v1"
    model = settings.get("model") or "deepseek-chat"

    arms = []
    if args.judge_pairs:
        pairs = load_prebuilt(args.judge_pairs)
    else:
        for spec in args.arm:
            label, _, path = spec.partition("=")
            arms.append((label, load_replies(Path(path))))
        pairs = build_pairs(arms[0], arms[1], args.seed)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} 对，判官 {model} @ {base}")

    import httpx

    with (
        httpx.Client(
            base_url=base, timeout=120.0, headers={"Authorization": f"Bearer {key}"}
        ) as client,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {pool.submit(judge_one, client, model, p): p for p in pairs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            futures[future]["winner"] = future.result()
            if done % 25 == 0:
                print(f"  判了 {done}/{len(pairs)}", flush=True)

    if args.judge_pairs:
        # The shape resolve reads, and nothing else: a winner keyed to the pair
        # the judge was shown.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                [{"key": p["key"], "winner": p["winner"]} for p in pairs],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        record = {
            "model": model,
            "endpoint": base,
            "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
            "pairs_from": str(args.judge_pairs),
            "judged": len(pairs),
            "ties": sum(1 for p in pairs if p["winner"] == "tie"),
            "note": "判官是外部模型，不是被比较的任何一臂。key 本身从不进记录，只记有过一个。",
        }
        args.output.with_suffix(".provenance.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"判了 {len(pairs)} 对，平局 {record['ties']}")
        print(f"写入 {args.output}")
        print(f"出处 {args.output.with_suffix('.provenance.json')}")
        if args.pairs:
            args.pairs.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    summary = tally(pairs, arms[1][0])
    summary["judge"] = {
        "model": model,
        "endpoint": base,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "side_seed": args.seed,
        # The key itself is never read into the record, only the fact of one.
        "note": "判官是外部模型，不是被比较的任何一臂",
    }
    summary["arms"] = {"first": arms[0][0], "second": arms[1][0]}
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.pairs:
        args.pairs.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    seat = summary.get("seating", {})
    if seat.get("residual_bias") is not None:
        print(
            f"  座位: 左偏 {seat['side_bias']:+.4f}，左座占比 {seat['left_seat_share']:.4f}"
            f" -> 残余偏倚 {seat['residual_bias']:+.5f}"
            f"（座位平衡估计 {seat['seat_balanced_share']:.4f}）"
        )
    elif seat:
        print(f"  座位: {seat.get('note', '无法估计')}")
    for field in ("n_pairs", "tie", "n_decided", "position_right_share", "noise", "verdict"):
        if field in summary:
            print(f"  {field}: {summary[field]}")
    print(f"  {arms[1][0]} 胜率: {summary.get(arms[1][0] + '_share')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
