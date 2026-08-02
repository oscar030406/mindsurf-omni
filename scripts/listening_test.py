"""A blind listening panel, scoped to what a panel that size can actually answer.

The plan asks for "UTMOS + 20 clips of human blind listening". The plan also
warns, in the same section, that a MOS panel's noise floor may be larger than
the difference it is meant to detect, and that the arithmetic should be done
before anyone spends an afternoon listening. Doing that arithmetic first is
what this file is for.

The arithmetic, on measured numbers rather than assumed ones. The clip-by-clip
difference between our Talker and upstream's has sd 0.397 -- the systems do not
differ by a constant, they differ by a wildly varying amount, so pairing barely
helps. Adding literature-typical single-rating MOS noise (sd 0.8 to 1.0) and
solving for the n that resolves the 0.29 difference at three times the floor:

    3 raters   241 - 339 clips each
    5 raters   171 - 230
   10 raters   118 - 148

A 20-clip panel is off by roughly a factor of twelve. It cannot certify this
effect, and running it as though it could would produce a number that looks
like evidence and is not -- the exact failure the evaluation rules exist to
prevent.

So this panel is not a gate, and --score refuses to emit a verdict. What twenty
clips genuinely buy, both of which the automatic metrics cannot:

* Gross defects. A sister project chased intermittent pops through 38 clips of
  acoustic screening and clean spectrograms; ASR cannot hear them because the
  words are right, and a MOS predictor trained on clean synthesis may not
  either. A human hears one immediately. Every sheet has a defect flag.
* Whether UTMOS ranks clips the way people do. Agreeing on an *ordering* needs
  far less power than certifying a *magnitude*, so twenty clips can check the
  instrument that does gate -- which matters, because UTMOS is calibrated on
  English and this is Chinese.

Naturalness gating stays with UTMOS, which is eligible at n=160.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.evaluation.metrics import bootstrap_noise_floor  # noqa: E402

MOS_SCALE = "1=完全听不懂/难以忍受  2=费力  3=可以接受  4=自然  5=与真人无异"
# Measured, not assumed: the clip-by-clip spread of the difference between two
# systems on this material. Used to tell a panel how big it would need to be.
OBSERVED_PAIRED_SD = 0.397
# What a single human MOS rating scatters by, from the listening-test
# literature. Two values because the honest answer is a range.
RATER_SD_RANGE = (0.8, 1.0)


def required_clips(effect: float, raters: int, rater_sd: float) -> int:
    """Clips per system for three times the floor to fit inside `effect`."""
    sd = math.sqrt(OBSERVED_PAIRED_SD**2 + 2 * rater_sd**2 / raters)
    return math.ceil((3 * 1.96 * sd / effect) ** 2)


def spearman(left: list[float], right: list[float]) -> float:
    """Rank correlation, written out to avoid a dependency for twelve lines."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            shared = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                out[order[position]] = shared
            index = stop + 1
        return out

    a, b = ranks(left), ranks(right)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    spread = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return covariance / spread if spread else float("nan")


def load_system(entry: str) -> tuple[str, dict[str, dict[str, Any]]]:
    label, _, path = entry.rpartition("=")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = {
        sample["id"]: sample
        for sample in manifest["samples"]
        if sample.get("audio_path") and Path(sample["audio_path"]).is_file()
    }
    return label, samples


def stratified_ids(shared: list[str], scores: dict[str, float], count: int, seed: int) -> list[str]:
    """Spread the picks across the quality range instead of sampling flat.

    A uniform draw of twenty from a hundred and sixty can miss the bad tail
    entirely, and the bad tail is where a listener has something to say.
    """
    rng = random.Random(seed)
    if not scores:
        return rng.sample(shared, min(count, len(shared)))
    ordered = sorted(shared, key=lambda item: scores.get(item, 0.0))
    buckets = max(1, min(count, 5))
    per_bucket = math.ceil(count / buckets)
    picks: list[str] = []
    size = math.ceil(len(ordered) / buckets)
    for index in range(buckets):
        chunk = ordered[index * size : (index + 1) * size]
        if chunk:
            picks.extend(rng.sample(chunk, min(per_bucket, len(chunk))))
    rng.shuffle(picks)
    return picks[:count]


def prepare(args: argparse.Namespace) -> None:
    systems = dict(load_system(entry) for entry in args.system)
    if len(systems) < 2:
        raise SystemExit("give at least two --system LABEL=manifest.json to compare")

    shared = sorted(set.intersection(*(set(samples) for samples in systems.values())))
    if len(shared) < args.clips:
        raise SystemExit(f"only {len(shared)} ids are in every system; asked for {args.clips}")

    first = next(iter(systems.values()))
    utmos = {i: first[i]["utmos"] for i in shared if "utmos" in first[i]}
    chosen = stratified_ids(shared, utmos, args.clips, args.seed)

    if args.anchor:
        # Known-good audio mixed in under its own label. A rater who scores it
        # the same as everything else is not discriminating, and that shows up
        # in the report before their numbers reach a conclusion.
        label, samples = load_system(args.anchor)
        systems[label] = samples

    args.output.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output / "audio"
    audio_dir.mkdir(exist_ok=True)

    key: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    for identifier in chosen:
        for label, samples in systems.items():
            if identifier not in samples:
                continue
            token = hashlib.sha256(f"{args.seed}:{label}:{identifier}".encode()).hexdigest()[:12]
            shutil.copyfile(samples[identifier]["audio_path"], audio_dir / f"{token}.wav")
            key.append(
                {
                    "token": token,
                    "system": label,
                    "id": identifier,
                    "text": samples[identifier].get("reference_text", ""),
                    "utmos": samples[identifier].get("utmos"),
                }
            )

    # A few clips asked twice, to measure whether a rater agrees with themselves.
    repeats = rng.sample(key, min(args.repeats, len(key)))

    (args.output / "key.json").write_text(
        json.dumps(
            {"entries": key, "repeats": [r["token"] for r in repeats]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for rater in range(1, args.raters + 1):
        # A fresh order per rater, so a drift in attention over the session
        # does not land on the same system every time.
        sheet = [*key, *repeats]
        random.Random(args.seed + rater).shuffle(sheet)
        path = args.output / f"rater{rater}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["token", "mos_1_to_5", "defect", "note"])
            for index, entry in enumerate(sheet):
                writer.writerow([f"{entry['token']}#{index}", "", "", ""])
        print(f"  {path.name}: {len(sheet)} clips ({len(repeats)} repeated)")

    (args.output / "README.txt").write_text(
        "盲听评分\n\n"
        f"打分标准: {MOS_SCALE}\n\n"
        "1. 用耳机，安静环境，音量固定，中途不要调。\n"
        "2. 打开 raterN.csv，按顺序听 audio/<token>.wav（token 取 # 前面那段）。\n"
        "3. mos_1_to_5 填 1-5 整数。听起来有爆音、断裂、忽大忽小、\n"
        "   机械杂音这类问题，defect 列填 1 并在 note 里说一句是什么。\n"
        "   **这一列比分数重要**：自动指标看不见非语音的毛病。\n"
        "4. 不要打开 key.json——那是揭盲用的，看了这次评分就作废。\n"
        "5. 有的片段会重复出现，这是故意的，照常打分不要回头找。\n",
        encoding="utf-8",
    )

    print(f"\n{len(chosen)} clips x {len(systems)} systems, {args.raters} raters")
    print(f"打包在 {args.output}（key.json 是揭盲用的，评分员不要看）")
    print("\n这个规模能回答什么，不能回答什么:")
    print(f"  不能：认证 {args.effect} 的 MOS 差——需要每人")
    for rater_sd in RATER_SD_RANGE:
        need = required_clips(args.effect, args.raters, rater_sd)
        print(f"        {need} 条（单次评分 sd {rater_sd}），现在是 {args.clips} 条")
    print("  能：查自动指标看不见的粗大缺陷；验 UTMOS 的排序是否与人耳一致")


def score(args: argparse.Namespace) -> None:
    key_blob = json.loads((args.pack / "key.json").read_text(encoding="utf-8"))
    by_token = {entry["token"]: entry for entry in key_blob["entries"]}

    # Sheets live one per rater folder, and the first column is that rater's
    # play order rather than the clip's identity -- the clips are numbered so a
    # rater can play them straight through instead of hunting a hash. The key
    # holds the position-to-token map, per rater, because the three orders
    # differ on purpose.
    layout = key_blob.get("raters") or {}
    ratings: dict[str, list[tuple[int, float, bool, str]]] = {}
    sheets = sorted(args.pack.glob("rater*/rater*.csv")) or sorted(args.pack.glob("rater*.csv"))
    for path in sheets:
        rater = path.stem.replace("rater", "")
        positions = {row["position"]: row["token"] for row in layout.get(rater, [])}
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("mos_1_to_5") or "").strip()
                if not raw:
                    continue
                key_cell = (row.get("音频文件") or next(iter(row.values())) or "").strip()
                position = key_cell.removesuffix(".wav")
                token = positions.get(position, "") or position.split("#")[0]
                if token not in by_token:
                    raise SystemExit(
                        f"{path.name}: {key_cell} does not resolve to a clip in the key"
                    )
                defect = (row.get("defect") or "").strip() not in ("", "0")
                ratings.setdefault(token, []).append(
                    (int(rater), float(raw), defect, (row.get("note") or "").strip())
                )
    if not ratings:
        raise SystemExit(f"no filled sheets in {args.pack}; the mos_1_to_5 column is empty")

    systems: dict[str, list[float]] = {}
    per_clip: dict[str, float] = {}
    defects: list[str] = []
    for token, entries in ratings.items():
        info = by_token[token]
        mean = statistics.fmean(value for _, value, _, _ in entries)
        per_clip[token] = mean
        systems.setdefault(info["system"], []).append(mean)
        for rater, _, flagged, note in entries:
            if flagged:
                defects.append(f"{info['system']} {info['id']} (rater {rater}): {note or '未描述'}")

    print(
        f"评分员 {len({r for e in ratings.values() for r, _, _, _ in e})} 名，"
        f"片段 {len(ratings)} 条\n"
    )
    print("各系统平均 MOS:")
    for label, values in sorted(systems.items()):
        floor = bootstrap_noise_floor(values) if len(values) > 1 else float("inf")
        print(f"  {label:28} {statistics.fmean(values):.2f} ± {floor:.2f}  n={len(values)}")

    # Self-consistency: the same clip asked twice inside one sheet.
    spreads = [
        abs(a - b)
        for token in key_blob["repeats"]
        for rater in {r for r, _, _, _ in ratings.get(token, [])}
        for a, b in [
            tuple(v for r, v, _, _ in ratings[token] if r == rater)[:2]
        ]
        if len([v for r, v, _, _ in ratings[token] if r == rater]) >= 2
    ]
    if spreads:
        print(f"\n评分员自洽性: 重复片段两次打分平均差 {statistics.fmean(spreads):.2f} 分")
        if statistics.fmean(spreads) > 1.0:
            print("  ⚠ 超过 1 分——这批评分本身噪声太大，结论都不要用")

    # The question twenty clips can actually answer.
    paired = [(per_clip[t], by_token[t]["utmos"]) for t in per_clip if by_token[t].get("utmos")]
    if len(paired) >= 8:
        rho = spearman([h for h, _ in paired], [u for _, u in paired])
        signal = statistics.stdev([u for _, u in paired])
        rater_count = len({r for entries in ratings.values() for r, _, _, _ in entries}) or 1
        # The standard error of one clip's human mean. Rank correlation is
        # attenuated by measurement noise: when this exceeds the spread UTMOS
        # shows across the same clips, rho is crushed toward zero no matter how
        # well the two actually agree. Reporting that as "UTMOS disagrees with
        # listeners" would be an instrument judging what it cannot resolve --
        # the failure this project checks for everywhere else.
        noise = max(RATER_SD_RANGE) / math.sqrt(rater_count)
        print(f"\n与 UTMOS 的排序一致性: Spearman ρ = {rho:.2f}（n={len(paired)}）")
        if signal < noise:
            print(
                f"  UTMOS 在这批片段上的离散度 {signal:.2f} 小于人耳每条的标准误 "
                f"{noise:.2f}（{rater_count} 人平均）——**这个检查没有区分力，不作结论**"
            )
            print(
                "  要让它有意义：混入 UTMOS 明显不同的系统（--anchor 且给它打上 utmos），"
                "或者加评分员"
            )
        elif rho >= 0.5:
            print("  UTMOS 的排序与人耳一致——它作为门控指标的效度得到支持")
        else:
            print("  ⚠ 排序不一致——UTMOS 在中文上的门控资格要重新审")

    print(f"\n缺陷标记 {len(defects)} 条:" if defects else "\n没有标出缺陷")
    for line in defects[:10]:
        print(f"  {line}")

    print("\n判定资格:")
    for rater_sd in RATER_SD_RANGE:
        raters = len({r for e in ratings.values() for r, _, _, _ in e}) or 1
        need = required_clips(args.effect, raters, rater_sd)
        print(f"  要认证 {args.effect} 的差，{raters} 名评分员需要每人 {need} 条")
    print(f"  本轮 {len(ratings) // max(1, len(systems))} 条/系统 —— **不作门控判定**")
    print("  自然度门控继续用 UTMOS（n=160 已有资格）；本轮只用于查缺陷与验排序")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    build = sub.add_parser("prepare", help="build a blinded listening pack")
    build.add_argument("--system", action="append", required=True, metavar="LABEL=manifest.json")
    build.add_argument(
        "--anchor",
        metavar="LABEL=manifest.json",
        help="known-good audio mixed in to catch a non-discriminating rater",
    )
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--clips", type=int, default=20)
    build.add_argument("--raters", type=int, default=3)
    build.add_argument("--repeats", type=int, default=4)
    build.add_argument("--effect", type=float, default=0.29)
    build.add_argument("--seed", type=int, default=20260725)
    build.set_defaults(func=prepare)

    read = sub.add_parser("score", help="unblind filled sheets and report")
    read.add_argument("--pack", required=True, type=Path)
    read.add_argument("--effect", type=float, default=0.29)
    read.set_defaults(func=score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
