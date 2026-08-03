"""Build one probe file per emotion instruction, for the conditioned arm.

The conditioned route puts the emotion in the user turn as a sentence, so at
evaluation time the instruction has to reach the model the same way it reached
it during training: prepended to the last user turn with no separator. That
join and those five sentences live in ``build_emotion_corpus``, and they are
imported rather than retyped -- a probe set that words the instruction slightly
differently from the corpus is measuring a prompt the model never saw, and the
difference would not show up as an error anywhere.

``evaluate_talker.py`` renders ``prompt`` as the user turn and teacher-forces
``text``, so only ``prompt`` is touched. The spoken text stays identical across
arms, which is what makes the comparison paired: the same sentence, the same
voice, the same seed, one instruction apart.

    python scripts/build_emotion_probes.py --texts configs/talker_texts_zh_v1.jsonl \
        --output-dir configs --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_emotion_corpus import INSTRUCTIONS


def probe_rows(rows: list[dict[str, str]], instruction: str) -> list[dict[str, str]]:
    """Same ids, same spoken text, instruction prepended to the user turn."""
    return [{**row, "prompt": f"{instruction}{row['prompt']}"} for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, default=Path("configs/talker_texts_zh_v1.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs"))
    parser.add_argument("--limit", type=int, help="first N texts, to match the pack protocol")
    parser.add_argument("--prefix", default="emotion_probes_zh")
    args = parser.parse_args()

    lines = args.texts.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    if args.limit:
        rows = rows[: args.limit]
    missing = [row["id"] for row in rows if "prompt" not in row or "text" not in row]
    if missing:
        raise SystemExit(f"rows without prompt/text: {missing[:5]}")

    for label, instruction in sorted(INSTRUCTIONS.items()):
        out = args.output_dir / f"{args.prefix}_{label}.jsonl"
        body = "\n".join(
            json.dumps(row, ensure_ascii=False) for row in probe_rows(rows, instruction)
        )
        out.write_text(body + "\n", encoding="utf-8")
        print(f"{out}: {len(rows)} 条，指令 {instruction!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
