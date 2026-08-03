"""Extend the chat probe set, matching the register of the one it extends.

The main criterion resolves +/-9.6 points at 158 probes and needs about 5, which
is a sample size problem and nothing else -- blind preference has a binomial
floor, so it falls as 1/sqrt(n) with no argument. About 600 probes gets there.
The plan and its review are in docs/experiments/2026-08-03-probe-budget.md.

Three things about the new probes matter more than how many there are.

**Register.** The existing probes run 6 to 13 characters, median 9: short spoken
questions. The DPO training prompts are longer and more specific. Generating in
the training set's register would quietly change what the measurement measures,
and the two halves could not be read as one set.

**Authorship.** The training prompts have two authors, and the blind evaluation
has a judge. New probes share none of them. A probe written in the same hand as
the training prompts hands the tuned arm a familiarity advantage, and a probe
written by the judge invites a second circularity for no benefit. So the writer
is a third model, named in the sidecar.

**Overlap.** Exact duplicates against both the existing probes and all 838
training prompts are dropped, and so are near-duplicates -- character bigram
Jaccard, because "怎么煮面好吃" and "面怎么煮好吃" are the same probe and neither
exact matching nor a token diff will say so.

    python scripts/build_chat_probes.py --classify configs/chat_refs_external_v1.jsonl \
        --output artifacts/probe-themes.json
    python scripts/build_chat_probes.py --generate 450 --themes artifacts/probe-themes.json \
        --output configs/chat_probes_zh_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.judge import Judge  # noqa: E402

# The taxonomy the second DPO round already used, so the two sets can be
# compared by theme without inventing a second vocabulary for the same idea.
THEMES = (
    "人际与家常沟通", "养宠与花草", "出行与交通", "季节与生活应对",
    "安全与应急", "情绪与状态", "文娱与创作", "消费与办事",
    "生活法律与办事常识", "算术与逻辑", "育儿与教育", "语言文字",
)

CLASSIFY = """把下面这个口语提问归到一个类别里。

类别（只能选一个，一字不差地回答）：
{themes}

提问：{prompt}

只回类别名，不要解释。"""

GENERATE = """写 {count} 条中文口语提问，主题是「{theme}」。

严格照下面的样子写——这是要点，不是建议：

{examples}

硬性要求：
1. **每条 6 到 13 个字**，中位 9 字左右。超过 13 字的不要。
2. 口语，像人对着手机随口问的，**不要书面语、不要场景铺垫**。
3. 一条一行，**不要编号、不要标点收尾、不要引号**。
4. 彼此不重复，也不要和上面的样例重复。
5. 允许出现「需要实时信息或个人情况才能答」的问题（比如问天气、问附近），
   那一类是故意保留的。

只输出 {count} 行提问，别的都不要。"""

# Two signals, because one of them misses the case it was picked for. Character
# bigrams catch rewording, but on a six-character question a reordering breaks
# most of the bigrams too: "怎么煮面好吃" and "面怎么煮好吃" share three of seven
# and score 0.43, under any threshold that also keeps "怎么挑西瓜" and
# "怎么挑螃蟹" apart at 0.33. The margin is too thin to sit a rule on.
#
# A reordering is an anagram, so the character set survives it intact. Scoring
# that separately separates the two cases cleanly -- 1.00 against 0.43 -- and
# costs nothing.
NEAR_DUPLICATE = 0.6
SAME_CHARACTERS = 0.9


def bigrams(text: str) -> set[str]:
    stripped = "".join(text.split())
    return {stripped[i : i + 2] for i in range(len(stripped) - 1)} or {stripped}


def characters(text: str) -> set[str]:
    return set("".join(text.split()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def too_similar(candidate: str, seen: list[set[str]]) -> bool:
    grams, chars = bigrams(candidate), characters(candidate)
    for other in seen:
        if jaccard(grams, other) >= NEAR_DUPLICATE:
            return True
        # ``seen`` holds bigram sets; rebuilding the character set from them is
        # cheaper than carrying a second list and cannot drift out of step.
        if jaccard(chars, {ch for gram in other for ch in gram}) >= SAME_CHARACTERS:
            return True
    return False


def load_prompts(path: Path) -> list[str]:
    return [
        json.loads(line)["prompt"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def classify(judge: Judge, prompts: list[str]) -> dict[str, str]:
    listed = "\n".join(f"- {theme}" for theme in THEMES)
    answers = judge.run(
        prompts,
        lambda prompt: CLASSIFY.format(themes=listed, prompt=prompt),
        label="分类",
    )
    labelled = {}
    for prompt, answer in zip(prompts, answers, strict=True):
        clean = answer.strip().strip("「」\"' 。")
        labelled[prompt] = clean if clean in THEMES else "未归类"
    return labelled


def targets(labelled: dict[str, str], total: int) -> dict[str, int]:
    """The same mix as the set being extended, rounded to hit the total."""
    counts = Counter(theme for theme in labelled.values() if theme in THEMES)
    known = sum(counts.values())
    wanted = {theme: round(total * count / known) for theme, count in counts.items()}
    # Rounding rarely lands on the total; put the difference on the largest
    # theme rather than spreading a fudge across all of them.
    drift = total - sum(wanted.values())
    if wanted:
        biggest = max(wanted, key=lambda theme: wanted[theme])
        wanted[biggest] += drift
    return wanted


def parse_lines(reply: str) -> list[str]:
    out = []
    for raw in reply.splitlines():
        line = raw.strip().strip("-·•*").strip()
        for prefix in range(1, 40):
            for mark in (f"{prefix}.", f"{prefix}、", f"{prefix})", f"{prefix}）"):
                if line.startswith(mark):
                    line = line[len(mark) :].strip()
        line = line.strip("「」\"'。？?！! ")
        # A model asked for twenty lines pads with a heading, and a heading ends
        # in a colon. Length alone lets those through -- "好的，以下是提问：" is
        # nine characters, right in the middle of the range a probe occupies.
        if "：" in line or ":" in line:
            continue
        if 4 <= len(line) <= 20:
            out.append(line)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classify", type=Path)
    parser.add_argument("--generate", type=int)
    parser.add_argument("--themes", type=Path, help="output of --classify")
    parser.add_argument(
        "--existing", type=Path, default=Path("configs/chat_refs_external_v1.jsonl")
    )
    parser.add_argument(
        "--exclude", type=Path, nargs="*", default=[Path("configs/preference_prompts_zh_all.jsonl")]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--model", help="who writes or classifies; not the blind judge")
    parser.add_argument("--max-chars", type=int, default=13)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()

    judge = Judge(credentials=args.credentials, model=args.model)
    print(f"模型 {judge.model} @ {judge.base_url}")

    if args.classify:
        labelled = classify(judge, load_prompts(args.classify))
        counts = Counter(labelled.values())
        for theme, count in counts.most_common():
            print(f"  {theme}: {count}")
        args.output.write_text(
            json.dumps(
                {"labels": labelled, "counts": dict(counts), "by": judge.provenance(CLASSIFY)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    if not args.generate or not args.themes:
        raise SystemExit("要么 --classify，要么 --generate N --themes <分类结果>")

    labelled = json.loads(args.themes.read_text(encoding="utf-8"))["labels"]
    wanted = targets(labelled, args.generate)
    existing = load_prompts(args.existing)
    banned = set(existing)
    for path in args.exclude:
        banned |= set(load_prompts(path))
    print(f"排除 {len(banned)} 条（现有 + 训练用过的），目标 {sum(wanted.values())} 条")

    kept: dict[str, list[str]] = {theme: [] for theme in wanted}
    seen_grams = [bigrams(prompt) for prompt in banned]
    for theme, need in sorted(wanted.items(), key=lambda item: -item[1]):
        samples = [p for p, t in labelled.items() if t == theme][:8]
        examples = "\n".join(samples) or "\n".join(existing[:8])
        for _ in range(args.rounds):
            if len(kept[theme]) >= need:
                break
            # Built here rather than inside the lambda: a closure over the loop
            # variables reads whatever they hold when it runs, which is correct
            # only because run() happens to consume within the iteration. That
            # is a coincidence, not a design.
            question = GENERATE.format(count=max(need * 2, 20), theme=theme, examples=examples)
            reply = judge.run([None], lambda _, q=question: q, label=f"造 {theme}")[0]
            for line in parse_lines(reply):
                if len(line) > args.max_chars or line in banned:
                    continue
                if too_similar(line, seen_grams):
                    continue
                kept[theme].append(line)
                banned.add(line)
                seen_grams.append(bigrams(line))
                if len(kept[theme]) >= need:
                    break
        print(f"  {theme}: {len(kept[theme])}/{need}")

    rows = []
    for theme in sorted(kept):
        for prompt in kept[theme]:
            rows.append({"id": f"zx{len(rows):03d}", "prompt": prompt, "theme": theme})
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    sidecar = args.output.with_suffix(args.output.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "author": judge.model,
                "purpose": "extend the chat probe set so blind preference resolves 5 points",
                "written_to_match": str(args.existing),
                "excluded": [str(p) for p in args.exclude],
                "note": (
                    "Not the author of either training prompt set and not the blind judge. "
                    "A probe in the training prompts' hand would give a tuned arm a "
                    "familiarity advantage; one in the judge's hand invites a second "
                    "circularity for no benefit."
                ),
                **judge.provenance(GENERATE),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"写出 {len(rows)} 条 -> {args.output}，出处 -> {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
