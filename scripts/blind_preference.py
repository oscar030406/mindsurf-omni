"""Blind pairwise preference between two arms, with the judge on the record.

This comparison decided two DPO rounds and it was run outside the repository.
The pairs survive on disk, the verdicts survive, and who judged them does not --
searching both machines for anything that produced those numbers returns the
output files and nothing else. PROJECT_RULES section 6 requires a reference
set's author to be declared because authorship cannot be recovered afterwards;
the same argument applies to a judge, and this had no such record. So the
measurement is a script now, and its result carries the judge's identity, the
endpoint, the seed that chose the sides and a hash of the prompt that was sent.

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

    A key belongs in neither the source nor a global environment variable: the
    source is tracked and the gate rejects it on the way out, and a machine-wide
    variable hands it to every process on the box. A file next to the project
    that .gitignore already excludes is the narrow option -- one place to paste
    it, one place to delete it, and it cannot be committed by accident.
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
    return {
        "n_pairs": len(pairs),
        "tie": ties,
        "n_decided": len(decided),
        f"{second_arm}_wins": second_wins,
        f"{second_arm}_share": share,
        "noise": noise,
        # A judge that prefers whichever reply sits second is a side preference,
        # not a preference between the arms. Near 0.5 is what a clean run looks like.
        "position_right_share": right_wins / len(decided),
        "verdict": "赢" if share - 0.5 > noise else ("输" if 0.5 - share > noise else "未胜出"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="write every judged pair here too")
    parser.add_argument("--credentials", type=Path, default=Path("configs/judge.local.json"))
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="first N pairs, for a smoke run")
    args = parser.parse_args()

    if len(args.arm) != 2:
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

    for field in ("n_pairs", "tie", "n_decided", "position_right_share", "noise", "verdict"):
        if field in summary:
            print(f"  {field}: {summary[field]}")
    print(f"  {arms[1][0]} 胜率: {summary.get(arms[1][0] + '_share')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
