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
        ("接入指南", "docs/INTEGRATION.md"),
        ("运行手册", "docs/RUNBOOK.md"),
        ("评测说明", "docs/EVALUATION.md"),
        ("决策记录", "docs/DECISIONS.md"),
        ("许可记录", "configs/release/licence.json"),
        ("探针集", "configs/speech_probes_zh_v1.jsonl"),
        ("冒烟脚本", "scripts/smoke_service.py"),
        ("延迟脚本", "scripts/measure_latency.py"),
    ]:
        findings.check(name, (ROOT / path).exists(), f"{path} 不存在")


def check_documents_match_code(findings: Findings) -> None:
    """A guide that names a missing endpoint sends the reader to a 404."""
    app = (ROOT / "src" / "mindsurf_omni" / "service" / "app.py").read_text(encoding="utf-8")
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
        ("docs/INTEGRATION.md", "达到"),
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
    check_secrets_are_absent(findings)
    sys.exit(findings.report())


if __name__ == "__main__":
    main()
