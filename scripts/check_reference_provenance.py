"""Who wrote this reference set, and did anyone train on it.

A reference set is the ruler. If one of the arms being compared wrote it, that
arm is scoring the likelihood of its own samples and wins by construction --
which is what happened here: `talker_texts_zh_v1.jsonl` was adopted as a chat
holdout, produced a certified regression, and turned out to have been generated
by `sft_mindsurf_768` itself. The four scores were monotone in each
checkpoint's weight distance from the author.

A membership check was run at the time and passed. It was the wrong question.
Contamination is not only "is this text in the training data" -- it is also
"did one of the things I am comparing produce this text", and a model's own
samples leave no trace in any dataset.

**Authorship cannot be recovered after the fact, and this file exists partly to
say so.** The obvious detector -- ask each candidate to answer the same prompts
and see whose output the reference resembles -- was built and measured against
the known case. It failed. Sampling at temperature 0.7 means a model does not
reproduce its own earlier reply, so the true author scored mean similarity
0.291 and would have been cleared. The only hits above 0.8 were replies short
enough to be degenerate: a twelve-character echo of the prompt, and "米饭更顶饱",
which several unrelated models produce identically. Fitting a threshold to
those is fitting to noise.

So provenance is **declared, not inferred**. A reference set needs a sidecar
`<name>.provenance.json` naming its author, and this refuses to bless a set
without one. The two checks that follow corroborate; they cannot substitute.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Only verbatim reuse is detectable, so the bar is set where verbatim lives.
# This does NOT catch a model re-sampling its own prompts -- see the module
# docstring for the measurement that established it does not.
VERBATIM = 0.95
# Below this many characters a reply is short enough that unrelated models
# collide on it: prompt echoes, "米饭更顶饱", one-word answers.
TOO_SHORT_TO_MEAN_ANYTHING = 20


def provenance_path(reference: Path) -> Path:
    return reference.with_suffix(reference.suffix + ".provenance.json")


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    report = json.loads(path.read_text(encoding="utf-8"))
    return [{"id": row["id"], "text": row["reply"]} for row in report.get("replies", [])]


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def verbatim_reuse(
    reference: list[dict[str, str]], candidates: dict[str, list[dict[str, str]]]
) -> dict[str, Any]:
    """Long replies reproduced word for word, which means the file was copied.

    Short replies are excluded rather than weighted down. Two unrelated models
    answering "米饭和面条哪个更顶饱" with "米饭更顶饱" is not evidence of anything,
    and it is most of what a similarity search returns.
    """
    findings: dict[str, Any] = {}
    for name, rows in candidates.items():
        theirs = {row["id"]: row.get("text", "") for row in rows}
        pairs = [
            (row["id"], similarity(row["text"], theirs[row["id"]]))
            for row in reference
            if row["id"] in theirs
            and theirs[row["id"]]
            and len(row["text"]) >= TOO_SHORT_TO_MEAN_ANYTHING
        ]
        if not pairs:
            findings[name] = {"compared": 0, "verdict": "nothing long enough to compare"}
            continue
        hits = [name for name, value in pairs if value >= VERBATIM]
        findings[name] = {
            "compared": len(pairs),
            "max_similarity": max(value for _, value in pairs),
            "verbatim": hits,
            "verdict": "COPIED" if hits else "no verbatim reuse",
        }
    return findings


def membership(reference: list[dict[str, str]], parquets: list[Path]) -> dict[str, Any]:
    import pyarrow.parquet as pq

    texts = {row["text"]: row["id"] for row in reference}
    hits: set[str] = set()
    scanned = 0
    for path in parquets:
        handle = pq.ParquetFile(path)
        for batch in handle.iter_batches(batch_size=4096, columns=["conversations"]):
            for conversation in batch.column("conversations").to_pylist():
                scanned += 1
                turns = conversation if isinstance(conversation, list) else json.loads(conversation)
                for turn in turns:
                    if not isinstance(turn, dict):
                        continue
                    found = texts.get(turn.get("content"))
                    if found:
                        hits.add(found)
    return {"rows_scanned": scanned, "reference_texts_in_training": sorted(hits)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="*",
        default=[],
        help="checkpoints this set will be used to compare; the declared author "
        "must not be one of them",
    )
    parser.add_argument(
        "--generated",
        type=Path,
        nargs="*",
        default=[],
        help="measure_chat_loss --generate reports, for the verbatim-reuse screen",
    )
    parser.add_argument("--parquet", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_rows(args.reference)
    report: dict[str, Any] = {"reference": str(args.reference), "rows": len(reference)}
    problems: list[str] = []

    declared = provenance_path(args.reference)
    if not declared.is_file():
        problems.append(
            f"no {declared.name}: a reference set without a recorded author cannot be "
            "cleared, because authorship is not recoverable from the text"
        )
        report["provenance"] = None
    else:
        record = json.loads(declared.read_text(encoding="utf-8"))
        report["provenance"] = record
        author = record.get("author")
        print(f"声明的作者: {author}")
        if not author:
            problems.append(f"{declared.name} has no 'author' field")
        for arm in args.arms:
            if author and (author in arm or arm in author):
                problems.append(
                    f"the declared author {author!r} is one of the arms being compared "
                    f"({arm!r}) -- it would be scoring the likelihood of its own samples"
                )

    if args.generated:
        candidates = {path.stem: load_rows(path) for path in args.generated}
        report["verbatim_reuse"] = verbatim_reuse(reference, candidates)
        print("逐字复用筛查（只能查复制，查不出重新采样）：")
        for name, finding in report["verbatim_reuse"].items():
            if finding.get("compared"):
                print(
                    f"  {name:<28} 最大相似度 {finding['max_similarity']:.3f}  "
                    f"逐字命中 {len(finding['verbatim'])}  -> {finding['verdict']}"
                )
                if finding["verdict"] == "COPIED":
                    count = len(finding["verbatim"])
                    problems.append(f"{name} reproduces {count} replies verbatim")

    if args.parquet:
        report["membership"] = membership(reference, args.parquet)
        found = report["membership"]["reference_texts_in_training"]
        print(
            f"训练集成员检查：扫过 {report['membership']['rows_scanned']} 行，"
            f"命中 {len(found)} 条 {found[:5]}"
        )
        if len(found) > len(reference) // 20:
            problems.append(
                f"{len(found)} of {len(reference)} reference texts are in the training data"
            )

    report["problems"] = problems
    report["usable"] = not problems
    print(f"\n可用作参照集: {report['usable']}")
    for problem in problems:
        print(f"  - {problem}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
