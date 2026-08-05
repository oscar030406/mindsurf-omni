"""Author a short-register reference set for chat_nll, one external model at a time.

The fourth ruler (2026-08-03-merged-thinker.md section 10) keeps chat_nll's
engine and changes its fuel. The instrument died of a measured mechanism --
reference length drives the two authors' per-probe disagreement, r = +0.62
within-set -- so the repair is references written in the register the candidates
actually speak: short, spoken, straight to the point.

The register is enforced, not requested. A model asked for sixty characters
pads a third of the time; anything over the cap gets one rewrite pass with the
overlong draft shown back, and rows that still miss are dropped and counted.
A dropped row is a probe both authors must drop, or the paired comparison
would pair different probes across authors -- the caller handles alignment,
this script just refuses to ship an overlong reply as if it were short.

    python scripts/build_short_refs.py --model deepseek-ai/DeepSeek-V4-Flash \
        --output configs/chat_refs_short_a_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.judge import Judge  # noqa: E402

WRITE = """用口语直接回答下面的问题，**不超过 60 个字**。

要求：
1. 像人说话，一两句就答到点上，不铺垫、不列条目、不客套。
2. 答不了的（要实时信息、要个人情况）就老实说一句「这个我查不了」加一句建议。
3. 只输出回答本身。

问题：{prompt}"""

REWRITE = """你上一版回答太长了（{n} 字）。压到 **60 个字以内**，意思不变，只输出回答：

问题：{prompt}
上一版：{draft}"""

MAX_CHARS = 60


def clean(text: str) -> str:
    return text.strip().strip("「」\"'").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, default=Path("configs/chat_refs_external_v1.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    judge = Judge(credentials=args.credentials, model=args.model, timeout=300.0)
    print(f"作者 {judge.model} @ {judge.base_url}")

    rows = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    drafts = judge.run(
        rows, lambda row: WRITE.format(prompt=row["prompt"]), label="写", max_tokens=120
    )
    drafts = [clean(d) for d in drafts]

    over = [i for i, d in enumerate(drafts) if len(d) > MAX_CHARS or not d]
    if over:
        fixes = judge.run(
            over,
            lambda i: REWRITE.format(prompt=rows[i]["prompt"], draft=drafts[i], n=len(drafts[i])),
            label="压缩",
            max_tokens=120,
        )
        for index, fix in zip(over, fixes, strict=True):
            drafts[index] = clean(fix)

    kept, dropped = [], []
    for row, draft in zip(rows, drafts, strict=True):
        if draft and len(draft) <= MAX_CHARS:
            kept.append({"id": row["id"], "prompt": row["prompt"], "text": draft})
        else:
            dropped.append(row["id"])

    args.output.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8"
    )
    lengths = sorted(len(r["text"]) for r in kept)
    sidecar = args.output.with_suffix(args.output.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "author": judge.model,
                "purpose": (
                    "short-register chat_nll reference set, the fourth ruler's fuel: "
                    "the instrument died of reference length, so the references match "
                    "the register the candidates speak"
                ),
                "probes": str(args.probes),
                "max_chars": MAX_CHARS,
                "dropped_overlong": dropped,
                "median_chars": lengths[len(lengths) // 2] if lengths else None,
                "note": (
                    "Author is neither a compared arm nor the blind-preference judge. "
                    "Authorship is declared because it cannot be inferred afterwards."
                ),
                **judge.provenance(WRITE),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    median = lengths[len(lengths) // 2] if lengths else 0
    print(f"写出 {len(kept)} 条（丢 {len(dropped)}），中位 {median} 字 -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
