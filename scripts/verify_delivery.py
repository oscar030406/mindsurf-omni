"""Check that what we are about to hand over is complete and consistent.

Run before giving the service to anyone. It reads only what is on disk and in
the repository, so it can be run without a GPU, a model, or a network.

The point is not to re-run the test suite. It is to catch the class of gap the
test suite cannot see: a document that promises an endpoint nobody built, a
licence conclusion that the code contradicts, a script the runbook cites that
was renamed. Each of those passes every unit test and fails the person who
receives the work.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class Findings:
    ok: list[str] = field(default_factory=list)
    gaps: list[tuple[str, str]] = field(default_factory=list)

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        (self.ok if condition else self.gaps).append(name if condition else (name, detail))  # type: ignore[arg-type]

    def report(self) -> int:
        for name in self.ok:
            print(f"  齐  {name}")
        for name, detail in self.gaps:
            print(f"  缺  {name}: {detail}")
        print(f"\n{len(self.ok)} 项齐备, {len(self.gaps)} 项缺失")
        return 1 if self.gaps else 0


def check_deliverables(findings: Findings) -> None:
    """Everything the handover promises must exist."""
    for name, path in [
        ("接口契约", "src/mindsurf_omni/contract.py"),
        ("推理服务", "src/mindsurf_omni/service/app.py"),
        ("容器定义", "Dockerfile"),
        ("编排文件", "docker-compose.yml"),
        ("说明", "README.md"),
        ("模型卡", "configs/release/MODEL_CARD.md"),
        ("对外数字真源", "configs/release/headline_numbers.json"),
        ("许可记录", "configs/release/licence.json"),
        ("探针集", "configs/speech_probes_zh_v1.jsonl"),
        ("冒烟脚本", "scripts/smoke_service.py"),
        ("延迟脚本", "scripts/measure_latency.py"),
    ]:
        findings.check(name, (ROOT / path).exists(), f"{path} 不存在")


def check_documents_match_code(findings: Findings) -> None:
    """A guide that names a missing endpoint sends the reader to a 404."""
    app = (ROOT / "src" / "mindsurf_omni" / "service" / "app.py").read_text(encoding="utf-8")
    # The endpoint table moved out of the README on 2026-08-06. It was a second
    # copy of the integration guide's, and the copy that is not the one people
    # wire against is the one that goes stale.
    guide = (ROOT / "docs" / "INTEGRATION.md").read_text(encoding="utf-8")

    implemented = set(re.findall(r'@app\.(?:get|post|websocket)\("(/v1/[^"]+)"', app))
    documented = set(re.findall(r"(?:POST|GET|WS) (/v1/[a-z/-]+)", guide))

    findings.check(
        "文档端点与实现一致",
        implemented == documented,
        f"仅文档有 {sorted(documented - implemented)}，仅实现有 {sorted(implemented - documented)}",
    )


def check_licence_is_consistent(findings: Findings) -> None:
    """The conclusion is stated in several places; they must not disagree."""
    record = json.loads((ROOT / "configs" / "release" / "licence.json").read_text(encoding="utf-8"))
    engine = (ROOT / "src" / "mindsurf_omni" / "service" / "engine.py").read_text(encoding="utf-8")

    permitted = record["conclusion"]["commercial_use_permitted"]
    findings.check(
        "许可结论与代码一致",
        (permitted is False) and ("commercial_use_permitted: bool = False" in engine),
        f"记录说 {permitted}，代码未匹配",
    )

    unverified = [asset["name"] for asset in record["assets"] if not asset["verified"]]
    # Not a failure -- it is the honest state -- but it must be visible in the
    # handover rather than discovered by whoever ships something.
    if unverified:
        print(f"  注意  {len(unverified)} 项资产条款未读: {unverified}")


def check_nothing_claims_more_than_it_measured(findings: Findings) -> None:
    """Capability wording must trace to an instrument that may gate."""
    forbidden = [
        ("README.md", "达到"),
        ("configs/release/MODEL_CARD.md", "达到"),
        ("configs/release/LISTENING_DATASET_CARD.md", "达到"),
    ]
    for name, phrase in forbidden:
        text = (ROOT / name).read_text(encoding="utf-8")
        offenders = [
            line.strip()
            for line in text.splitlines()
            if phrase in line and "未" not in line and "尚" not in line
        ]
        findings.check(
            f"{name} 无未经测量的能力表述",
            not offenders,
            f"疑似断言: {offenders[:2]}",
        )


def check_headline_numbers_are_traceable(findings: Findings) -> None:
    """A number whose evidence file is missing is a number nobody can check.

    Every figure we publish carries a ``source``. This walks them and confirms
    the paths are actually in the repository, because the failure we care about
    is silent: a file gets moved or left out of the release and the citation
    still reads as if it points somewhere.
    """
    record = json.loads(
        (ROOT / "configs" / "release" / "headline_numbers.json").read_text(encoding="utf-8")
    )

    missing: list[str] = []

    def visit(node: object, where: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key != "source":
                visit(value, f"{where}.{key}")
                continue
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                for candidate in str(entry).split(","):
                    # Sources may carry a section marker (" §8-§11") or say in
                    # words that no reading exists; only bare paths are checked.
                    path = candidate.strip().split(" ")[0].rstrip("/")
                    if not path or "/" not in path:
                        continue
                    if not (ROOT / path).exists():
                        missing.append(f"{where}: {path}")

    visit(record, "headline_numbers")
    findings.check("对外数字的证据都在", not missing, "; ".join(missing))


def check_secrets_are_absent(findings: Findings) -> None:
    """The publication gate covers commits; this covers the working tree."""
    patterns = [
        (r"ghp_[A-Za-z0-9]{30,}", "GitHub token"),
        (r"sk-[A-Za-z0-9]{20,}", "API key"),
        (r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY", "private key"),
    ]
    hits = []
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").rglob("*.json")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, label in patterns:
            if re.search(pattern, text):
                hits.append(f"{path.relative_to(ROOT)}: {label}")
    findings.check("无凭据泄漏", not hits, "; ".join(hits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    print("交付前检查\n")
    findings = Findings()
    check_deliverables(findings)
    check_documents_match_code(findings)
    check_licence_is_consistent(findings)
    check_nothing_claims_more_than_it_measured(findings)
    check_headline_numbers_are_traceable(findings)
    check_secrets_are_absent(findings)
    sys.exit(findings.report())


if __name__ == "__main__":
    main()
