"""Keep the sentences a person would dictate, drop the ones only a screen holds.

The 986-sentence held-out set is big enough to separate two arms, and it is
built from whatever the corpus had spare -- which turned out to include a lot
of text nobody speaks aloud. Both arms fail all four criteria on it, and that
is not a fact about the polisher.

Three things go, each for a reason visible in the text itself:

* **Instructions written to a model.** The probe file is almost entirely
  "生成一个…", "根据以下文本…", "对以下段落…". Its one empty output in the whole
  run was "回答问题并给出详细的推理过程：在下面的句子中…". Nobody dictates that
  into a text box.
* **Numbered lists.** 118 rows carry 一、 or 2、 at a clause boundary. A list
  marker is a thing you type, not a thing you say, and the polisher deleting
  one is not the failure the criteria are about.
* **Anything with a line break in it.** Those rows are a prompt plus a
  structured payload, and the second half is data rather than speech.

What survives is still synthetic -- filler injected into corpus text, read by
edge-tts, heard by SenseVoice -- so it is not a recording of a person
dictating. It is only the part of this set whose *register* is dictation.
Said plainly because the number it produces will be quoted against the
acceptance lines and it should carry that limit with it.

    python scripts/filter_dictation_register.py --rows artifacts/polish_train/val_x.jsonl \
        --pool artifacts/polish_train/pool_holdout.jsonl \
        --output artifacts/polish_train/val_x_dictation.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Sources that are prompts written at a model rather than speech. Named rather
# than detected: the file is what makes them instructions, and a keyword rule
# over "生成/根据/对以下" would also catch things people really do say.
INSTRUCTION_SOURCES = frozenset({"asr_probe_scored"})

# 一、 二． 3) 4. at the start of the text or right after a clause break.
LIST_MARKER = re.compile(r"(?:^|[，。！？；\s])[一二三四五六七八九十0-9]+[、．.)）]")


def is_dictation(text: str, source: str) -> bool:
    if source in INSTRUCTION_SOURCES:
        return False
    if "\n" in text or "\\n" in text:
        return False
    return not LIST_MARKER.search(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path, help="a val_*.jsonl of per-row output")
    parser.add_argument("--pool", required=True, type=Path, help="the pool, for the source field")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_of = {}
    for line in args.pool.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            source_of[row["id"]] = row.get("source", "")

    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kept = [
        row
        for row in rows
        # The target rather than the transcript: what the person meant is what
        # decides the register, and the transcript carries injected filler.
        if is_dictation(row["target"], source_of.get(row["id"], ""))
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in kept) + "\n",
        encoding="utf-8",
    )
    print(f"{len(rows)} 条留下 {len(kept)} 条，丢掉 {len(rows) - len(kept)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
