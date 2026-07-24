"""Documentation that names things must name things that exist.

A runbook citing a script that was renamed, or an error message that was
reworded, is worse than no runbook: it is consulted under pressure and sends
the reader somewhere wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_every_script_a_document_cites_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")

    missing = [
        script
        for script in re.findall(r"(scripts/[a-z_]+\.py)", text)
        if not (ROOT / script).is_file()
    ]

    assert not missing, f"{doc.name} cites scripts that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda path: path.name)
def test_every_repository_path_a_document_links_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    base = doc.parent

    missing = []
    for target in re.findall(r"\]\((\.\./[^)#]+|[a-z][a-z_/]*\.(?:md|py|json|toml|yml))\)", text):
        if not (base / target).resolve().exists():
            missing.append(target)

    assert not missing, f"{doc.name} links to missing files: {missing}"


def test_the_runbook_quotes_error_messages_the_code_actually_emits() -> None:
    """A reworded message makes the runbook unsearchable at the moment it matters."""
    runbook = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    app = (ROOT / "src" / "mindsurf_omni" / "service" / "app.py").read_text(encoding="utf-8")
    config = (ROOT / "src" / "mindsurf_omni" / "service" / "config.py").read_text(encoding="utf-8")

    assert "no speech engine is configured" in runbook
    assert "no speech engine is configured" in app

    assert "mount the weights directory" in runbook
    assert "mount the weights directory" in config


def test_the_runbook_covers_the_failures_that_have_actually_happened() -> None:
    """Each of these cost real time; a runbook that omits them is decorative."""
    runbook = (ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")

    for symptom in [
        "OOM",  # the full dataset does not fit in memory
        "113.13M",  # silently skipped weights left at random init
        "缓冲",  # the log lags because stdout is block-buffered
        "采样率",  # a wrong rate is a changed pitch, not silence
        "循环论证",  # scoring a model with its own component
    ]:
        assert symptom in runbook, f"the runbook does not cover {symptom!r}"


def test_the_evaluation_guide_states_the_three_rules_it_enforces() -> None:
    """Each is enforced in code; the guide must not omit one and imply choice."""
    guide = (ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")

    assert "循环论证" in guide  # the judge must be independent
    assert "仅报告" in guide  # an instrument that cannot resolve may not judge
    assert "无法区分" in guide  # a difference inside the noise has no direction


def test_the_evaluation_guide_chains_scripts_that_exist_in_that_order() -> None:
    """A guide whose steps do not chain sends the reader in a circle."""
    guide = (ROOT / "docs" / "EVALUATION.md").read_text(encoding="utf-8")

    for step in (
        "generate_speech_samples.py",
        "transcribe_samples.py",
        "evaluate_speech.py",
    ):
        assert step in guide
        assert (ROOT / "scripts" / step).is_file()

    # The order matters: transcription consumes what generation writes.
    assert guide.index("generate_speech_samples.py") < guide.index("transcribe_samples.py")
    assert guide.index("transcribe_samples.py") < guide.index("evaluate_speech.py")


def test_every_decision_says_what_would_overturn_it() -> None:
    """A decision without a reversal condition is a decision nobody can revisit.

    It reads as permanent, so the next person either follows it blindly or
    ignores it entirely -- and neither is what the record is for.
    """
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")

    headings = [line for line in text.splitlines() if line.startswith("## ")]
    reversals = text.count("什么情况下推翻")

    assert len(headings) >= 5
    assert reversals == len(headings), (
        f"{len(headings)} decisions but {reversals} reversal conditions"
    )


def test_the_decisions_that_contradict_the_brief_give_their_evidence() -> None:
    """Departing from the brief is fine; departing without a reason is not."""
    text = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")

    # The parameter-count decision rests on arithmetic that must be shown.
    assert "20 tokens/参数" in text or "tokens/参数" in text
    assert "23.0" in text  # our actual ratio
    # The architecture decision rests on a measurement.
    assert "logit 差 0.0" in text


def test_the_handover_states_that_no_quality_number_exists_yet() -> None:
    """The most likely misreading of this repository is that it has results.

    It has a pipeline that would produce them. Someone skimming the test count
    and the CI badge could easily conclude otherwise.
    """
    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")

    assert "没有任何质量数字" in text
    assert "CER" in text


def test_the_handover_names_the_trap_that_silently_ruins_a_run() -> None:
    """Calling the upstream trainer directly loses the base without saying so."""
    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")

    assert "train_omni.py" in text
    assert "113.13M" in text  # the wrong parameter count that signals it
    assert "152.06M" in text  # the right one


def test_the_handover_tells_the_reader_how_to_audit_the_work() -> None:
    """A handover that only says "trust this" gives the reader nothing to check."""
    text = (ROOT / "docs" / "HANDOVER.md").read_text(encoding="utf-8")

    assert "仅报告" in text  # instruments without gating eligibility
    assert "无法区分" in text  # differences inside the noise
    assert "永远不会失败" in text  # checks that cannot fail


def test_the_a2a_script_stages_the_projector_before_unfreezing() -> None:
    """One-step training lets a badly-aligned projector drag the language weights.

    The loss curve looks healthy while it happens, so the ordering is the only
    thing preventing it.
    """
    script = (ROOT / "scripts" / "run_a2a.sh").read_text(encoding="utf-8")

    # Match the invocations, not any occurrence of the names: the log file is
    # called a2a_full.log and appears near the top.
    projector = script.index('run "a2a_proj"')
    unfrozen = script.index('run "a2a_full"')

    assert projector < unfrozen
    assert "--mode audio_proj" in script[projector:unfrozen]
    assert "--mode" not in script[unfrozen:]  # the second stage trains everything
    # The second stage uses a much smaller rate: the Thinker arrives trained,
    # and a large step here unlearns it.
    assert "--learning_rate 5e-4" in script[projector:unfrozen]
    assert "--learning_rate 2e-5" in script[unfrozen:]


def test_the_a2a_script_does_not_roll_a_failed_stage_into_the_next() -> None:
    """The second stage would train from whatever the first left behind."""
    script = (ROOT / "scripts" / "run_a2a.sh").read_text(encoding="utf-8")

    assert 'exit "$status"' in script


def test_the_a2a_script_survives_a_dropped_connection() -> None:
    """This has already cost a run here."""
    script = (ROOT / "scripts" / "run_a2a.sh").read_text(encoding="utf-8")

    assert "setsid" in script
    assert "PYTHONUNBUFFERED=1" in script  # or the log lags thousands of steps


def test_the_plan_states_its_own_status_rather_than_reading_as_a_forecast() -> None:
    """A plan written before the work, left unmarked, misdirects whoever reads it.

    It is the largest document here and the one most likely to be opened first.
    """
    plan = (ROOT / "docs" / "ACTION_PLAN.md").read_text(encoding="utf-8")
    banner = plan[: plan.index("本文件是")]

    assert "状态" in banner
    # The most important thing it must not let a reader assume.
    assert "尚无任何真实数字" in banner or "没有的" in banner
    # And it must point at where departures were recorded rather than hiding them.
    assert "DECISIONS.md" in banner


def test_no_document_claims_a_measured_result_that_does_not_exist() -> None:
    """The pipeline can produce CER and MOS; nothing has run it on a real model.

    A number quoted anywhere would be from a stub, and quoting a stub as a
    result is the fabrication this project is built to avoid.
    """
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for line in text.splitlines():
            # A CER figure would look like "CER 0.12" or "CER 为 12%".
            if re.search(r"CER\s*[:：为=]?\s*\d+(\.\d+)?\s*%?", line):
                assert any(
                    marker in line for marker in ("±", "极限", "阈值", "分辨", "假设", "例")
                ), f"{doc.name} quotes a CER without qualifying it: {line.strip()}"


def test_the_architecture_lists_every_service_module() -> None:
    """A module absent from the map is one nobody knows they can change safely."""
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    modules = {
        path.name
        for path in (ROOT / "src" / "mindsurf_omni" / "service").glob("*.py")
        if path.name != "__init__.py"
    }
    missing = [name for name in modules if name not in architecture]

    assert not missing, f"the architecture does not mention: {sorted(missing)}"


def test_the_architecture_states_the_cost_of_row_group_shuffling() -> None:
    """Stating only the benefit would make it look free, and it is not."""
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    assert "不是全局洗牌" in architecture
    assert "偏斜" in architecture


def test_the_example_client_only_uses_endpoints_that_exist() -> None:
    """A copyable example that calls a missing route is worse than no example."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")
    app = (ROOT / "src" / "mindsurf_omni" / "service" / "app.py").read_text(encoding="utf-8")

    called = set(re.findall(r"/v1/[a-z/-]+", example))
    implemented = set(re.findall(r'@app\.(?:get|post|websocket)\("(/v1/[^"]+)"', app))

    assert called <= implemented, f"the example calls: {sorted(called - implemented)}"


def test_the_example_does_not_hardcode_the_sample_rate() -> None:
    """Assuming it is how a client ends up playing 24 kHz audio at 16 kHz."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")

    assert "x-sample-rate" in example
    assert "input_sample_rate" in example  # read from session.created too


def test_the_example_checks_the_licence_before_using_the_output() -> None:
    """It is the first thing a backend author should see."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")

    assert "commercial_use_permitted" in example


def test_the_a2a_guard_matches_a_process_not_a_substring() -> None:
    """A bare `pgrep -f` has produced a false positive here twice.

    Once it matched an ssh command carrying the path as an argument, once a
    dry run still winding down. Both times the guard refused to start a run
    that nothing was actually blocking.
    """
    script = (ROOT / "scripts" / "run_a2a.sh").read_text(encoding="utf-8")

    assert "python.*train_omni" in script  # anchored on the interpreter
    assert "--data_path" in script  # and on an argument only a real run has
