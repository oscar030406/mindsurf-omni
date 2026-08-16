"""读一把不是我们自己造的尺子：真人标注的中文不流利。

我们所有的数都读在自己造的留出集上，而那批数据的口语词是我们自己插进去的——
删除标签等价于「这个字是不是我们插的」。CS2W 不一样：7237 句取自转写下来的真实对话，
逐字的 1（口语词）/ 2（重复改口）/ 0（留）是人标的
（EMNLP 2023，github.com/guozishan/CS2W，仓库无 LICENSE 文件，不商用照用并署来源）。

两处分布差，读数的人要知道：

* **输入是人工转写，不是 SenseVoice 输出。** 标点和断句是人写的。我们量过标点约定
  的差别能占掉 28% 的删除标签，所以绝对值和自造留出集不可比。
* **它的「重复／改口」比我们注入的野得多。** 我们插的是从句头两个字的整齐复制，
  它记的是「值得大家应，值得大家注意的」这种真人半途改口。

所以这一份读的不是「过不过线」，是**在真人标的不流利上，这个润色器删对了多少、
删错了多少**——按 CS2W 的标签当真值算逐字的精确率和召回率。

    python scripts/measure_polish_on_cs2w.py --data <cs2w>/test.jsonl \
        --checkpoint out/sft_polish6_768.pth --minimind-root ~/omni/minimind-o
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
# The repository root too, so `scripts.train_polish_tagger` imports when this
# runs as a file rather than as a module -- the holdout scorer does the same.
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.service.polish import Polisher, group_sentences, worth_polishing  # noqa: E402


def deleted_positions(source: str, output: str) -> set[int]:
    """Which characters of ``source`` the polisher dropped."""
    kept: set[int] = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(
        None, source, output, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            kept.update(range(i1, i2))
    return set(range(len(source))) - kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="CS2W 的 jsonl")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--minimind-root", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--tagger",
        type=Path,
        help="score a tagger head instead of the generative polisher. Without "
        "this the second ruler could only read the arm we were replacing -- "
        "which is how a tagger gets judged entirely on the data we injected "
        "ourselves. --checkpoint is then the tuned backbone, not the polisher",
    )
    parser.add_argument("--tagger-threshold", type=float, default=0.9)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = []
    for line in args.data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # One row in 724 has a label shorter than its text; scoring it would
        # silently misalign every position after the gap.
        if len(row["content"]) == len(row["disfluency_label"]):
            rows.append(row)
    rows = rows[: args.limit]
    print(f"{len(rows)} 句（标签和正文对得上的）", flush=True)

    tagger = None
    if args.tagger:
        # The same loading the holdout scorer does, so the two rulers read one
        # head the same way. --checkpoint is the tuned backbone here.
        import torch

        from mindsurf_omni.service.thinker import ThinkerGenerator
        from scripts.train_polish_tagger import features as tag_features
        from scripts.train_polish_tagger import token_spans

        generator = ThinkerGenerator(
            checkpoint=args.checkpoint,
            tokenizer_dir=args.tokenizer,
            minimind_root=args.minimind_root,
            device=args.device,
        )
        generator.load()
        backbone, tokeniser = generator._model, generator._tokenizer  # noqa: SLF001
        backbone.eval()
        saved = torch.load(str(args.tagger), map_location=args.device, weights_only=False)
        tagger = torch.nn.Linear(saved["hidden"], 2).to(args.device)
        tagger.load_state_dict(saved["state_dict"])
        tagger.eval()
        tag_lookahead, tag_repetition = saved["lookahead"], saved.get("repetition", 0)
        print(f"标注器 lookahead={tag_lookahead} repetition={tag_repetition}", flush=True)
    else:
        polisher = Polisher(
            checkpoint=args.checkpoint,
            tokenizer_dir=args.tokenizer,
            minimind_root=args.minimind_root,
            device=args.device,
        )
        polisher.load()

    written = []
    hit = predicted = actual = 0
    hit_filler = actual_filler = 0
    hit_repeat = actual_repeat = 0
    for index, row in enumerate(rows, start=1):
        source = row["content"]
        if tagger is not None:
            ids, spans = token_spans(tokeniser, source)
            with torch.no_grad():
                matrix = tag_features(
                    backbone, ids, torch, args.device, tag_lookahead, tag_repetition
                )
                probability = torch.softmax(tagger(matrix), dim=-1)[:, 1]
            # Read straight off the tags rather than through a diff: the tagger
            # says which characters it drops, so there is nothing to infer.
            dropped = {
                position
                for (start, end), keep in zip(spans, probability.tolist(), strict=True)
                if keep >= args.tagger_threshold
                for position in range(start, min(end, len(source)))
            }
            output = "".join(c for i, c in enumerate(source) if i not in dropped)
        else:
            pieces = group_sentences(source)
            wanted = [piece for piece in pieces if piece.strip() and worth_polishing(piece)]
            answers = (
                dict(zip(wanted, polisher._polish_batch(wanted), strict=True)) if wanted else {}
            )
            output = "".join(answers.get(piece, piece) for piece in pieces)
            dropped = deleted_positions(source, output)
        labels = row["disfluency_label"]
        truth = {i for i, mark in enumerate(labels) if mark in "12"}
        predicted += len(dropped)
        actual += len(truth)
        hit += len(dropped & truth)
        filler = {i for i, mark in enumerate(labels) if mark == "1"}
        repeat = {i for i, mark in enumerate(labels) if mark == "2"}
        actual_filler += len(filler)
        hit_filler += len(dropped & filler)
        actual_repeat += len(repeat)
        hit_repeat += len(dropped & repeat)
        written.append({**row, "polished": output})
        if index % 200 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    summary = {
        # Which arm produced these, because a report that does not say cannot
        # be compared to the next one.
        "arm": f"tagger t={args.tagger_threshold}" if args.tagger else "generative",
        "tagger": str(args.tagger) if args.tagger else None,
        "checkpoint": args.checkpoint.name,
        "n": len(rows),
        "deleted": predicted,
        "should_delete": actual,
        "precision": hit / predicted if predicted else 0.0,
        "recall": hit / actual if actual else 0.0,
        "recall_filler": hit_filler / actual_filler if actual_filler else 0.0,
        "recall_repetition": hit_repeat / actual_repeat if actual_repeat else 0.0,
        "chars": sum(len(row["content"]) for row in rows),
        "unchanged": sum(1 for row in written if row["polished"] == row["content"]),
        "median_length": statistics.median(len(row["content"]) for row in rows),
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
