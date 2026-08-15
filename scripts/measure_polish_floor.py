"""How much work the polisher actually has, before anyone trains one.

The second-phase product is dictation: speech -> SenseVoice -> polish -> text
box. The polish stage was specified as two jobs -- fix what the recogniser got
wrong, and drop the spoken filler. Neither had a number. This measures both on
the production loop (edge-tts reads it, SenseVoice hears it) so the training
target comes from the floor rather than from an assumption about what ASR gets
wrong.

Three questions, and each one can retire a piece of the training plan:

1. **Is there anything to correct?** The recogniser reads 0.0315 on corpus
   audio, and the folded number on this text is lower still. Errors that rare
   put most of the polisher's value somewhere else.
2. **Is there any filler left to drop?** SenseVoice is trained on speech with
   filler in it, but a recogniser can also quietly tidy. If it already drops
   嗯 and 那个, the second half of the training target does not exist.
3. **Does the transcript come back punctuated?** If not, the polisher's real
   job is a third thing nobody put on the list -- restoring sentence breaks.

Both arms read the same sentences: one clean, one with spoken filler injected
by ``inject_fillers.py``. The difference between their error rates is what the
disfluency itself costs recognition; the filler that survives into the
transcript is what a polisher would have to remove.

    python scripts/measure_polish_floor.py \
        --clean artifacts/polish/clean_scored.jsonl \
        --filler artifacts/polish/filler_scored.jsonl \
        --report artifacts/polish-floor-2026-08-15.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import (  # noqa: E402
    assess,
    character_error_rate,
    normalise_for_cer,
)

# 判据，跑之前写死。跑完不许调线、不许添臂。
#
# 口语词保留率：注入的口语词有多少还留在转写里。
RETENTION_REAL = 0.70  # >= 这个数：ASR 忠实转写，「剔除口语词」是一件真活
RETENTION_MOOT = 0.30  # <= 这个数：ASR 自己吃掉了，这项训练目标基本不存在
# 折叠数字口径的 CER。低于这个数意味着每五十个字才错一个，纠错买不到多少东西。
CORRECTION_SMALL = 0.02
# 转写里带标点的比例。低于这个数说明断句是润色的主要收益，而它不在原计划里。
PUNCTUATION_MISSING = 0.10

_PUNCTUATION = "。，、？！；：,.?!;:"


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    missing = [row.get("id") for row in rows if row.get("asr_transcript") is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} 行没有 asr_transcript，先跑 measure_asr.py：{missing[:3]}"
        )
    return rows


def error_shapes(rows: list[dict[str, Any]], reference_key: str) -> dict[str, Any]:
    """What the errors are, not just how many.

    difflib rather than a hand-written alignment: the opcodes are the same
    substitution / insertion / deletion split, and a DP backtrace written here
    would be one more thing to get right for no extra information.
    """
    counts: Counter[str] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    for row in rows:
        reference = normalise_for_cer(row[reference_key], fold_numbers=True)
        heard = normalise_for_cer(row["asr_transcript"], fold_numbers=True)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, reference, heard, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                continue
            counts[tag] += max(i2 - i1, j2 - j1)
            confusions[(reference[i1:i2], heard[j1:j2])] += 1
    return {
        "substituted": counts["replace"],
        "inserted": counts["insert"],
        "deleted": counts["delete"],
        "top_confusions": [
            {"reference": pair[0], "heard": pair[1], "n": n}
            for pair, n in confusions.most_common(15)
        ],
    }


def digit_split(rows: list[dict[str, Any]], reference_key: str) -> dict[str, Any]:
    """Rows with digits scored apart from rows without.

    A synthesiser reading 2021 as 二零二一 is doing its job, and the unfolded
    metric charges a substitution for every character of it. Folding fixes most
    of that; this says how much of what is left is still the number front end.
    """
    split: dict[str, list[float]] = {"with_digits": [], "without_digits": []}
    for row in rows:
        key = "with_digits" if any(c.isdigit() for c in row[reference_key]) else "without_digits"
        split[key].append(
            character_error_rate(row[reference_key], row["asr_transcript"], fold_numbers=True)
        )
    return {
        name: {"n": len(values), "cer": statistics.mean(values) if values else None}
        for name, values in split.items()
    }


def filler_retention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How much of the injected filler survived into the transcript.

    Counted as an excess over the clean sentence rather than as a plain
    occurrence: 就是 and 然后 are ordinary words that the written text already
    contains, and counting those would report filler the injection never added.
    """
    kept: Counter[str] = Counter()
    total: Counter[str] = Counter()
    per_token: dict[str, dict[str, int]] = {}
    for row in rows:
        heard = normalise_for_cer(row["asr_transcript"])
        clean = normalise_for_cer(row["clean_text"])
        wanted: Counter[tuple[str, str]] = Counter(
            (item["kind"], normalise_for_cer(item["token"])) for item in row["injections"]
        )
        for (kind, token), count in wanted.items():
            if not token:
                continue
            added = max(0, heard.count(token) - clean.count(token))
            survived = min(added, count)
            kept[kind] += survived
            total[kind] += count
            entry = per_token.setdefault(token, {"injected": 0, "kept": 0})
            entry["injected"] += count
            entry["kept"] += survived
    injected = sum(total.values())
    return {
        "injected": injected,
        "kept": sum(kept.values()),
        "retention": (sum(kept.values()) / injected) if injected else None,
        "by_kind": {
            kind: {
                "injected": total[kind],
                "kept": kept[kind],
                "retention": kept[kind] / total[kind] if total[kind] else None,
            }
            for kind in sorted(total)
        },
        "by_token": dict(sorted(per_token.items(), key=lambda item: -item[1]["injected"])),
    }


def punctuated(rows: list[dict[str, Any]]) -> float:
    """The share of transcripts that came back with any sentence punctuation."""
    return sum(
        1 for row in rows if any(mark in row["asr_transcript"] for mark in _PUNCTUATION)
    ) / len(rows)


_SENTENCE_END = "。？！?!"


def _breaks(text: str) -> tuple[str, dict[int, str]]:
    """The characters without punctuation, and where the marks sat between them."""
    characters: list[str] = []
    marks: dict[int, str] = {}
    for character in text:
        if character in _PUNCTUATION:
            marks.setdefault(
                len(characters), "sentence" if character in _SENTENCE_END else "clause"
            )
        elif not character.isspace():
            characters.append(character)
    return "".join(characters), marks


def punctuation_agreement(rows: list[dict[str, Any]], reference_key: str) -> dict[str, Any]:
    """Not whether the transcript has punctuation, but whether it breaks in the same places.

    Reported only -- the criterion fixed before the run is presence, and adding
    a second line after seeing the first would be moving the ruler. It is here
    because "comes back punctuated" and "comes back punctuated correctly" are
    different claims, and dropping sentence-splitting from the training target
    rests on the second one.

    Positions are matched through the same alignment the error split uses, so a
    transcript that misheard a character earlier in the sentence does not count
    every later break as misplaced.
    """
    matched = reference_total = heard_total = 0
    for row in rows:
        reference, wanted = _breaks(row[reference_key])
        heard, got = _breaks(row["asr_transcript"])
        mapping: dict[int, int] = {len(reference): len(heard)}
        for tag, i1, i2, j1, _ in difflib.SequenceMatcher(
            None, reference, heard, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                mapping.update({i1 + offset: j1 + offset for offset in range(i2 - i1)})
        reference_total += len(wanted)
        heard_total += len(got)
        matched += sum(1 for position in wanted if mapping.get(position, -1) in got)
    return {
        "reference_breaks": reference_total,
        "heard_breaks": heard_total,
        "matched": matched,
        "recall": matched / reference_total if reference_total else None,
        "precision": matched / heard_total if heard_total else None,
    }


def arm(rows: list[dict[str, Any]], reference_key: str, name: str) -> dict[str, Any]:
    rates = [character_error_rate(row[reference_key], row["asr_transcript"]) for row in rows]
    folded = [
        character_error_rate(row[reference_key], row["asr_transcript"], fold_numbers=True)
        for row in rows
    ]
    measurement = assess(f"{name}_cer_folded", folded, effect_of_interest=CORRECTION_SMALL)
    return {
        "n": len(rows),
        "reference": reference_key,
        "cer": statistics.mean(rates),
        "cer_folded": measurement.value,
        "noise_floor": measurement.noise_floor,
        "median_folded": statistics.median(folded),
        "exact": sum(1 for rate in folded if rate == 0.0),
        "worst": sorted(
            ({"id": row["id"], "cer": rate} for row, rate in zip(rows, folded, strict=True)),
            key=lambda item: -item["cer"],
        )[:5],
        "punctuated": punctuated(rows),
        "punctuation_agreement": punctuation_agreement(rows, reference_key),
        "digits": digit_split(rows, reference_key),
        "errors": error_shapes(rows, reference_key),
    }


def verdicts(clean: dict[str, Any], retention: dict[str, Any]) -> dict[str, str]:
    """The three judgements, against the lines written above rather than the numbers."""
    rate = retention["retention"]
    if rate is None:
        filler = "无法判定（没有注入）"
    elif rate >= RETENTION_REAL:
        filler = f"口语词剔除是真活（保留率 {rate:.3f} ≥ {RETENTION_REAL}）"
    elif rate <= RETENTION_MOOT:
        filler = f"这项训练目标基本不存在，ASR 自己吃掉了（保留率 {rate:.3f} ≤ {RETENTION_MOOT}）"
    else:
        filler = f"部分保留（{rate:.3f}），收益按保留的那部分算"
    correction = (
        f"纠错收益小（折叠 CER {clean['cer_folded']:.4f} ≤ {CORRECTION_SMALL}）"
        if clean["cer_folded"] <= CORRECTION_SMALL
        else f"纠错是真活（折叠 CER {clean['cer_folded']:.4f} > {CORRECTION_SMALL}）"
    )
    breaks = (
        f"断句是主要收益，转写基本不带标点（{clean['punctuated']:.3f} < {PUNCTUATION_MISSING}）"
        if clean["punctuated"] < PUNCTUATION_MISSING
        else f"转写自带标点（{clean['punctuated']:.3f}），断句不是主要缺口"
    )
    return {"口语词": filler, "纠错": correction, "断句": breaks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True, type=Path, help="干净文本那一臂的打分行")
    parser.add_argument("--filler", required=True, type=Path, help="注入口语词那一臂的打分行")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    clean_rows = load_rows(args.clean)
    filler_rows = load_rows(args.filler)

    clean = arm(clean_rows, "reference_text", "clean")
    # Two references, two different questions. Against the sentence that was
    # read aloud, this is what disfluency costs recognition; against the clean
    # original, it is the whole job the polisher would be trained to do.
    heard_as_read = arm(filler_rows, "reference_text", "filler_vs_spoken")
    heard_vs_clean = arm(filler_rows, "clean_text", "filler_vs_clean")
    retention = filler_retention(filler_rows)

    report = {
        "arms": {
            "clean": clean,
            "filler_vs_spoken": heard_as_read,
            "filler_vs_clean": heard_vs_clean,
        },
        "retention": retention,
        "thresholds": {
            "retention_real": RETENTION_REAL,
            "retention_moot": RETENTION_MOOT,
            "correction_small": CORRECTION_SMALL,
            "punctuation_missing": PUNCTUATION_MISSING,
        },
        "verdicts": verdicts(clean, retention),
        "caveat": (
            "合成语音、注入式口语词：真人的重启和拖音比这难，所以纠错这一侧是下界，"
            "而口语词保留率是上界（读得清清楚楚的 那个 是识别器最容易留住的情况）"
        ),
    }

    lines = [
        f"干净臂 n={clean['n']}：CER {clean['cer']:.4f}，折叠 {clean['cer_folded']:.4f}"
        f" ± {clean['noise_floor']:.4f}，全对 {clean['exact']}/{clean['n']}",
        f"  含数字 {clean['digits']['with_digits']['n']} 句折叠 "
        f"{clean['digits']['with_digits']['cer']:.4f}，"
        f"不含 {clean['digits']['without_digits']['n']} 句 "
        f"{clean['digits']['without_digits']['cer']:.4f}",
        f"  替换 {clean['errors']['substituted']}、插入 {clean['errors']['inserted']}、"
        f"删除 {clean['errors']['deleted']}",
        f"口语词臂 n={heard_as_read['n']}：对着读出去的那句 {heard_as_read['cer_folded']:.4f}，"
        f"对着干净原文 {heard_vs_clean['cer_folded']:.4f}",
        f"  注入 {retention['injected']} 处，留下 {retention['kept']} 处"
        + (f"，保留率 {retention['retention']:.3f}" if retention["retention"] is not None else ""),
        f"  转写带标点：干净臂 {clean['punctuated']:.3f}，"
        f"口语词臂 {heard_as_read['punctuated']:.3f}",
    ]
    for kind, verdict in report["verdicts"].items():
        lines.append(f"判据 {kind}：{verdict}")
    lines.append(f"  {report['caveat']}")
    print("\n".join(lines))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
