"""Publication gate rules."""

from __future__ import annotations

import re

import pytest
from scripts.release_gate import CONTENT_RULES, EXEMPT, PATH_RULES

CONTENT_BY_NAME = {name: pattern for name, pattern, _ in CONTENT_RULES}
PATH_BY_NAME = {name: pattern for name, pattern, _ in PATH_RULES}


def _hits(rule: str, line: str) -> bool:
    return bool(re.search(CONTENT_BY_NAME[rule], line, re.IGNORECASE))


@pytest.mark.parametrize(
    "rule,line",
    [
        ("agent_process_record", "Co-Authored-By: Claude Opus 4.8"),
        ("agent_process_record", "generated with codex"),
        ("personal_path", '"/home/oscar/omni/minimind-o"'),
        ("personal_path", "D:\\\\UserData\\\\Desktop\\\\minimind-o"),
        ("personal_path", "C:/Users/oscar/model"),
        ("internal_address", "HostName 192.168.6.3"),
        ("credential", "-----BEGIN OPENSSH PRIVATE KEY-----"),
        ("credential", "token = ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
    ],
)
def test_forbidden_content_is_caught(rule: str, line: str) -> None:
    assert _hits(rule, line)


@pytest.mark.parametrize(
    "rule,line",
    [
        # The rules run case-insensitively, so a bare drive-letter pattern
        # fires on ordinary text. This exact string sits in the vendored
        # tokenizer's chat template and is not a path.
        ("personal_path", "within <tools></tools> XML tags:\\\\n<tools>"),
        ("personal_path", "响应时间 P95 1.93s，来源 eval/voice_latency_report.json"),
        ("personal_path", "ratio 12.5:1 measured over 1636 hours"),
    ],
)
def test_compliant_content_is_not_flagged(rule: str, line: str) -> None:
    assert not _hits(rule, line)


def test_version_quads_in_lock_files_are_exempted_by_path() -> None:
    """A dotted quad in a lock file is a package version, not a host.

    The rule itself still matches -- the exemption is per file, so this checks
    the exemption rather than the pattern.
    """
    from scripts.release_gate import RULE_EXEMPT_PATHS

    assert _hits("internal_address", "nvidia-curand-cu12 10.3.5.147")
    assert RULE_EXEMPT_PATHS["internal_address"].search("uv.lock")
    assert not RULE_EXEMPT_PATHS["internal_address"].search("docs/ACTION_PLAN.md")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("out/llm_768.pth", True),
        ("dataset/sft_t2a.parquet", True),
        (".codex/state.json", True),
        ("src/mindsurf_omni/contract.py", False),
        ("artifacts/reference-logits.manifest.json", False),
    ],
)
def test_path_rules(name: str, expected: bool) -> None:
    matched = any(re.search(pattern, name, re.IGNORECASE) for pattern in PATH_BY_NAME.values())

    assert matched is expected


def test_every_exempt_file_exists() -> None:
    """An exemption for a path that no longer exists is a silent hole."""
    from pathlib import Path

    from scripts.release_gate import ROOT

    missing = sorted(name for name in EXEMPT if not (ROOT / Path(name)).is_file())

    assert missing == []


def test_a_provenance_record_may_name_a_model_as_the_author() -> None:
    """Two project rules collide here, and this is which one wins.

    Section 3 forbids shipping agent traces. Section 6 requires every reference
    set to record its author, because authorship cannot be recovered later --
    at sampling temperature a model does not reproduce its own old replies, so
    the record is a declaration or there is none. That requirement was bought
    with a whole round: a set written by one of the compared arms was used as a
    holdout, and every arm then scored best on its own text.

    Removing the name to satisfy the trace rule would restore that failure, so
    the trace rule steps aside for these files and only these files.
    """
    from scripts.release_gate import RULE_EXEMPT_PATHS

    exempt = RULE_EXEMPT_PATHS["agent_process_record"]
    assert exempt.search("configs/chat_refs_external_v1.jsonl.provenance.json")
    # And nowhere else: a plan or a transcript is still a trace.
    assert not exempt.search("docs/PLAN.md")
    assert not exempt.search("configs/chat_refs_external_v1.jsonl")
