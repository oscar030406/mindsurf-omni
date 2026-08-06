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


def pair_key(pair: dict[str, Any]) -> str:
    """What a judgement is filed under, for both shapes of input."""
    return str(pair.get("key") or pair["id"])


def load_partial(path: Path) -> dict[str, str]:
    """Judgements from a run that was killed before it could write its output.

    A run holds every verdict in memory and writes once at the end, so losing
    the process loses the whole spend -- 575 of 900 calls, the first time this
    bit. The handover already carries the ssh version of this lesson; the
    session-scoped background job is the same failure with a different rope.

    Nothing here is a checkpoint format. It is the same {key, winner} rows the
    output carries, one per line, appended as they land, deleted once the real
    output exists. A truncated last line is dropped rather than repaired: one
    re-judged pair costs a fraction of a cent, and a half-parsed one is a wrong
    verdict attached to a real key.
    """
    if not path.exists():
        return {}
    done: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("winner") in ("left", "right", "tie"):
            done[str(row["key"])] = row["winner"]
    return done


def bin_tally(pairs: list[dict[str, Any]], sigma: float = 3.0) -> dict[str, Any]:
    """How often the longer draft won, per character-gap bin.

    This measures **sensitivity, not bias**. The share of wins landing on the
    longer draft mixes a judge that rewards length with a longer draft that
    genuinely answered more, and this corpus cannot separate them: every
    operation that makes a reply longer also changes what it says. The number
    below is the first quantity, and the write-up is only allowed to claim it.

    Pairs whose two drafts are the same length are dropped, not counted as a
    loss for the longer one -- there is no longer one. They are reported so the
    drop is visible rather than silently shrinking a bin.

    The noise floor is ``sigma`` binomial standard deviations, defaulting to the
    3 the criteria were registered against rather than the 1.96 ``tally`` uses
    for a pass/fail on one comparison. Two different questions, two floors; the
    one that was written down before the run is the one that counts here.
    """
    report: dict[str, Any] = {"sigma": sigma, "bins": {}}
    for name in sorted({p["bin"] for p in pairs}):
        rows = [p for p in pairs if p["bin"] == name]
        even = [p for p in rows if p.get("gap", 0) == 0]
        rows = [p for p in rows if p.get("gap", 0) != 0]
        decided = [p for p in rows if p.get("winner") in ("left", "right")]
        entry: dict[str, Any] = {
            "n": len(rows),
            "same_length_dropped": len(even),
            "tie": len(rows) - len(decided),
            "tie_rate": (len(rows) - len(decided)) / len(rows) if rows else None,
            "n_decided": len(decided),
            "gap_median": sorted(p["gap"] for p in rows)[len(rows) // 2] if rows else None,
        }
        if decided:
            longer = sum(1 for p in decided if p["winner"] == p["longer_side"])
            entry["longer_wins"] = longer
            entry["longer_share"] = longer / len(decided)
            entry["noise"] = sigma * math.sqrt(0.25 / len(decided))
            # The floor a criterion gets registered against is the bin size,
            # because that is the number you can plan. The floor the estimate
            # actually has is over the pairs the judge separated, and ties took
            # half of one bin here. The second is the right one -- the share is
            # a share of decided pairs -- but it is also the wider one, which
            # makes "inside the noise" easier to reach than it was registered to
            # be. Both are recorded so that gap is read rather than discovered.
            entry["noise_at_bin_size"] = sigma * math.sqrt(0.25 / len(rows))
            # Sides were assigned when the pairs were formed; if the longer draft
            # sat on one seat far more often, a seat preference would read as a
            # length effect. Reported so that confusion is checkable, not assumed.
            entry["longer_on_left_share"] = sum(
                1 for p in rows if p["longer_side"] == "left"
            ) / len(rows)
        report["bins"][name] = entry
    return report


def bin_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Land L1 / L2 / L3, the three lines registered before the run."""
    bins = report["bins"]
    near, far = bins.get("near"), bins.get("far")
    if not (near and far and "longer_share" in near and "longer_share" in far):
        return {"line": "L3", "why": "缺箱或某箱全平局，三条线都判不了"}

    within = [
        name
        for name, entry in bins.items()
        if "longer_share" in entry and abs(entry["longer_share"] - 0.5) <= entry["noise"]
    ]
    spread = far["longer_share"] - near["longer_share"]
    if far["longer_share"] >= 0.60 and spread > far["noise"] + near["noise"]:
        return {
            "line": "L1",
            "why": f"远箱 {far['longer_share']:.4f} ≥ 0.60，且高出近箱 {spread:+.4f}"
            f"（两者噪声之和 {far['noise'] + near['noise']:.4f}）",
        }
    # The same three lines read against the floor the criteria were planned on.
    # It is not the right floor for the estimate, but if the answer changes
    # between the two, that is the finding, and it does not get to be silent.
    planned = [
        name
        for name, entry in bins.items()
        if "longer_share" in entry
        and abs(entry["longer_share"] - 0.5) <= entry["noise_at_bin_size"]
    ]
    fragile = (
        None
        if sorted(planned) == sorted(within)
        else f"按注册时那个「每箱 300 对」的噪声底重读，落在 0.50 内的只有 "
        f"{sorted(planned)}——这一条的判词依赖用哪个 n 算底"
    )

    if len(within) == len(bins):
        return {
            "line": "L2",
            "why": "三个箱都落在 0.50 ± 各自噪声内，我事前那个推论在这条上被证伪",
            "floor_sensitive": fragile,
        }
    return {
        "line": "L3",
        "why": f"既不满足 L1 也不满足 L2：落在 0.50 噪声内的箱 {sorted(within)}，"
        f"远箱 {far['longer_share']:.4f}，远近之差 {spread:+.4f}"
        f"（噪声之和 {far['noise'] + near['noise']:.4f}）",
        "floor_sensitive": fragile,
    }


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
    parser.add_argument(
        "--tally-bins",
        type=Path,
        metavar="JUDGED_PAIRS",
        help="no judging: read a judged --pairs file whose pairs carry bin/gap/"
        "longer_side and report how often the longer draft won, per bin",
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

    if args.tally_bins:
        # A reading, not a judging: no key, no endpoint, no spend.
        judged = json.loads(args.tally_bins.read_text(encoding="utf-8"))
        judged = judged["pairs"] if isinstance(judged, dict) else judged
        report = bin_tally(judged)
        report["verdict"] = bin_verdict(report)
        report["judged_pairs_from"] = str(args.tally_bins)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for name, entry in report["bins"].items():
            if "longer_share" in entry:
                print(
                    f"  {name:5s} n={entry['n_decided']:3d}(判) 较长者胜 "
                    f"{entry['longer_share']:.4f} ± {entry['noise']:.4f}"
                    f"  平局率 {entry['tie_rate']:.4f}"
                    f"  字数差中位 {entry['gap_median']}"
                )
            else:
                print(f"  {name:5s} 全部平局，判不了")
        print(f"  {report['verdict']['line']}: {report['verdict']['why']}")
        print(f"写入 {args.output}")
        return 0

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.with_suffix(".partial.jsonl")
    already = load_partial(partial_path)
    todo = []
    for pair in pairs:
        cached = already.get(pair_key(pair))
        if cached is None:
            todo.append(pair)
        else:
            pair["winner"] = cached
    if already:
        print(f"  续上 {len(pairs) - len(todo)} 对已判的，还剩 {len(todo)} 对")

    import httpx

    with (
        httpx.Client(
            base_url=base, timeout=120.0, headers={"Authorization": f"Bearer {key}"}
        ) as client,
        concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool,
        partial_path.open("a", encoding="utf-8") as partial,
    ):
        futures = {pool.submit(judge_one, client, model, p): p for p in todo}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            pair = futures[future]
            pair["winner"] = future.result()
            partial.write(
                json.dumps({"key": pair_key(pair), "winner": pair["winner"]}, ensure_ascii=False)
                + "\n"
            )
            partial.flush()
            if done % 25 == 0:
                print(f"  判了 {done}/{len(todo)}", flush=True)

    if args.judge_pairs:
        # The shape resolve reads, and nothing else: a winner keyed to the pair
        # the judge was shown.
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
        partial_path.unlink(missing_ok=True)
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
    partial_path.unlink(missing_ok=True)

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
