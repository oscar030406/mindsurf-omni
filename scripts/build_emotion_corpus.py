"""Write the training corpus back out with an emotion instruction in each user turn.

The labels exist now (scripts/label_emotion.py over all 414,024 rows). This is
what turns them into something the model can be trained on. The shape was
decided before the labelling ran: a natural-language instruction on the USER
side, over a vocabulary token or an assistant-side prefix, for three reasons:

* it changes no vocabulary, so the frozen-Thinker recipe applies unchanged --
  and freezing is what stopped A2A costing 1.52 nat of prose;
* at inference the caller supplies the emotion, which is the same shape as
  choosing a voice, and nothing leaks into the model's text output;
* it trains instruction-to-prosody, which is what the product needs, rather
  than label-to-prosody.

**Every row gets an instruction, neutral included.** Labelling only the
emotional rows would teach "an instruction means be expressive", so any
instruction would push the model off neutral regardless of which one it was.
With neutral spelled out, the mapping is from a specific instruction to a
specific prosody.

This writes a new parquet rather than patching the loader on the way past. The
training code stays byte-identical to the run this is compared against, so the
only thing that differs between the two is the data -- which is the whole point
of running it.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# The tail was 0.9% of the corpus across four labels. Folding it into neutral is
# the conservative direction: it dilutes neutral slightly rather than creating a
# control the model has too few examples to learn and too few to certify.
INSTRUCTIONS = {
    "happy": "请用开心的语气回答。",
    "angry": "请用生气的语气回答。",
    "surprised": "请用惊讶的语气回答。",
    "sad": "请用难过的语气回答。",
    "neutral": "请用平静的语气回答。",
}
FOLD_TO_NEUTRAL = {"other", "unk", "<unk>", "disgusted", "fearful"}


def instruction_for(label: str) -> str:
    """The instruction a row's label maps to, folding the rare tail to neutral."""
    english = label.split("/")[-1].strip("<>")
    if english in FOLD_TO_NEUTRAL:
        english = "neutral"
    return INSTRUCTIONS.get(english, INSTRUCTIONS["neutral"])


def condition(conversations: str, instruction: str) -> str:
    """Prepend the instruction to the LAST user turn.

    The last user turn is the one the assistant audio in this row answers, and
    that audio is what carries the emotion being labelled. Putting it on the
    first turn of a multi-turn row would attach the instruction to a question
    whose answer is not the clip that was labelled.
    """
    turns = json.loads(conversations)
    for turn in reversed(turns):
        if turn.get("role") == "user":
            turn["content"] = f"{instruction}{turn.get('content', '')}"
            break
    else:
        return conversations
    return json.dumps(turns, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="+", required=True, help="shard jsonl files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    labels: dict[tuple[int, int], str] = {}
    for path in args.labels:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                labels[(row["row_group"], row["index"])] = row["label"]
    print(f"读到 {len(labels)} 条标签")

    handle_in = pq.ParquetFile(str(args.parquet))
    used: collections.Counter[str] = collections.Counter()
    missing = 0
    writer = None
    try:
        for group in range(handle_in.metadata.num_row_groups):
            table = handle_in.read_row_group(group)
            conversations = table.column("conversations").to_pylist()
            rewritten = []
            for index, value in enumerate(conversations):
                label = labels.get((group, index))
                if label is None:
                    # A row the labeller skipped -- it had no assistant audio, so
                    # it carries no emotion to condition on. Neutral keeps it in
                    # the corpus without inventing a label for it.
                    missing += 1
                    label = "中立/neutral"
                instruction = instruction_for(label)
                used[instruction] += 1
                rewritten.append(condition(value, instruction))
            table = table.set_column(
                table.schema.get_field_index("conversations"),
                "conversations",
                pa.array(rewritten, type=pa.string()),
            )
            if writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(args.output), table.schema)
            writer.write_table(table)
            if (group + 1) % 10 == 0:
                print(f"  行组 {group + 1}/{handle_in.metadata.num_row_groups}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    total = sum(used.values())
    report = {
        "source": args.parquet.name,
        "rows": total,
        "labels_found": len(labels),
        "rows_without_a_label": missing,
        "instructions": {text: count for text, count in used.most_common()},
        "folded_to_neutral": sorted(FOLD_TO_NEUTRAL),
        "note": (
            "every row carries an instruction, neutral included, so the model "
            "learns instruction-to-prosody rather than presence-of-instruction"
        ),
    }
    print(f"\n写出 {total} 行 -> {args.output}")
    for text, count in used.most_common():
        print(f"  {text:<16} {count:>7} ({count / total:.4f})")
    if missing:
        print(f"  没有标签因而按中立处理: {missing}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
