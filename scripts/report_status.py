"""One command that says where this project actually stands.

Written because the honest answer is spread across a training log, a git
history, a test suite and a licence record, and assembling it by hand invites
rounding in a favourable direction. This reads each source and prints what it
finds, including the absences.

Run it before telling anyone how it is going.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# The repository root too, because this imports scripts.watch_training. Without
# it the --log path -- the one the handover documents -- died on ModuleNotFound
# while the flagless invocation worked, so the crash only appeared to whoever
# followed the instructions.
sys.path.insert(0, str(ROOT))


def git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a status report must not fail on git
        return ""


def repository() -> list[str]:
    commits = git("rev-list", "--count", "HEAD")
    tests = len(list((ROOT / "tests").glob("test_*.py")))
    modules = len([p for p in (ROOT / "src").rglob("*.py") if p.name != "__init__.py"])
    return [
        f"提交 {commits or '?'}，测试文件 {tests}，源码模块 {modules}",
        f"最近提交: {git('log', '-1', '--format=%s') or '?'}",
    ]


def licence() -> list[str]:
    path = ROOT / "configs" / "release" / "licence.json"
    if not path.is_file():
        # Reported rather than raised. A status command that dies on a missing
        # file tells you less than one that names the file, and this one exists
        # precisely to surface absences.
        return [f"许可记录不在 {path}——无法判断产出物能否使用"]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        record["conclusion"], record["assets"]
    except (json.JSONDecodeError, KeyError) as error:
        return [f"许可记录无法解析（{type(error).__name__}）——按不可用处理"]
    unverified = [asset["name"] for asset in record["assets"] if not asset["verified"]]
    lines = [
        f"可商用: {record['conclusion']['commercial_use_permitted']}"
        f"（受 {record['conclusion']['binding_constraint']} 约束）"
    ]
    if unverified:
        # Stated as a count and a list, because "mostly verified" is the
        # rounding this report exists to prevent.
        lines.append(f"条款未读: {len(unverified)}/{len(record['assets'])} — {unverified}")
    return lines


def measured_results() -> list[str]:
    """The section most likely to be over-read, so it names what is absent."""
    artifacts = ROOT / "artifacts"
    reports = sorted(artifacts.glob("*report*.json")) if artifacts.exists() else []
    if not reports:
        return [
            "无。评测管线完整但只跑过桩数据。",
            "任何 CER / MOS / TTF-Audio 的说法目前都没有依据。",
        ]

    lines = []
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = payload.get("candidate", {})
        # A run where the model never spoke produces a CER that belongs to the
        # synthesiser and the judge. Read as a model score it is the most
        # flattering wrong number this project can print, and this list is
        # exactly where someone skims it out of context.
        instrument = "（仪器底噪，模型未参与）" if payload.get("instrument_only") else ""
        for name, measurement in sorted(candidate.get("measurements", {}).items()):
            mark = "" if measurement["gating_eligible"] else "（仅报告）"
            lines.append(
                f"{path.name}: {name} {measurement['value']:.4f} "
                f"± {measurement['noise_floor']:.4f} "
                f"n={measurement['sample_size']}{mark}{instrument}"
            )
        regression = payload.get("text_regression")
        if regression is None:
            lines.append(f"{path.name}: 文本能力未测")
        elif isinstance(regression, dict) and "delta_over_base" in regression:
            # Surfaced, not buried in the file: this is the one measurement the
            # audio numbers structurally cannot show, and a large delta is the
            # thing most worth seeing at a glance -- with its caveat, because
            # base vs SFT is not a like-for-like regression.
            base = regression["base_strict_val"]
            a2a = regression["a2a_strict_val"]
            delta = regression["delta_over_base"]
            lines.append(
                f"{path.name}: 文本 strict_val {a2a:.4f}（base {base:.4f}, 差 {delta:+.4f}）"
                "——base 对 SFT 非同类比较，大不等于坏"
            )
    return lines


def training(log: Path | None) -> list[str]:
    if log is None or not log.exists():
        return ["未提供训练日志（--log）"]

    from scripts.watch_training import compare_epochs, parse, runs

    points = parse(log)
    if not points:
        return [f"{log} 里没有训练记录"]

    latest = points[-1]
    # A2A writes two stages into one log and both number their epochs from
    # one, so comparing across the whole file compares two different training
    # configurations and calls the difference an epoch effect.
    staged = [point for point in points if point.run == latest.run]
    lines = [
        f"epoch {latest.epoch}  step {latest.step}/{latest.steps}  "
        f"loss {latest.loss:.4f}  text {latest.text:.4f}  audio {latest.audio:.4f}"
    ]
    if len(runs(points)) > 1:
        lines.append(f"  日志含多个阶段 {runs(points)}，只看 {latest.run!r}")
    # More than one window, because a single one has already given the wrong
    # answer about convergence here. Placed as fractions of the run rather than
    # at fixed steps: the fixed ones were chosen for T2A's 31224 steps, and
    # A2A's 25877 left the last window hanging off the end of the run, where it
    # is never full and is therefore always dropped.
    total = latest.steps
    for fraction in (0.2, 0.5, 0.8):
        low = int(total * fraction)
        high = min(low + 3000, total)
        rows = compare_epochs(staged, "audio", low, high)
        verdicts = [str(row.get("verdict", "-")) for row in rows if "verdict" in row]
        if verdicts:
            lines.append(f"  audio {low}-{high}: {' -> '.join(verdicts)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, help="training log to summarise")
    args = parser.parse_args()

    sections = [
        ("仓库", repository()),
        ("许可", licence()),
        ("训练", training(args.log)),
        ("已测得的结果", measured_results()),
    ]
    for title, lines in sections:
        print(f"\n== {title} ==")
        for line in lines:
            print(f"  {line}")
    print()


if __name__ == "__main__":
    main()
