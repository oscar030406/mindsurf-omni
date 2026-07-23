"""The harness must report weak instruments, not hide them."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_speech import Sample, load, score


def _samples(count: int, cer_gap: str = "") -> list[Sample]:
    return [
        Sample(
            prompt=f"问题{index}",
            reference_text="今天天气真好",
            transcript="今天天气真好" + cer_gap,
        )
        for index in range(count)
    ]


def test_a_perfect_transcript_scores_zero_cer() -> None:
    report = score("candidate", _samples(200), {"cer": 0.05})

    assert report.measurements["cer"].value == 0.0


def test_a_small_sample_is_reported_but_not_allowed_to_judge() -> None:
    """Twenty samples cannot resolve a five-point CER difference."""
    report = score("candidate", _samples(20), {"cer": 0.05})

    assert "cer" in report.measurements  # still reported
    assert not report.measurements["cer"].gating_eligible
    assert "cer" in report.to_json()["not_claimed"]


def test_silence_is_counted_as_its_own_failure() -> None:
    """A model that says nothing has nothing to score wrong, so CER alone misses it."""
    samples = _samples(150)
    for sample in samples[:30]:
        sample.transcript = ""

    report = score("candidate", samples, {"cer": 0.05})

    assert report.measurements["silent_rate"].value == 0.2
    assert "30 of 150 produced no speech" in report.measurements["silent_rate"].note


def test_utmos_is_scored_only_when_present() -> None:
    without = score("candidate", _samples(150), {})
    assert "utmos" not in without.measurements

    samples = _samples(150)
    for index, sample in enumerate(samples):
        sample.utmos = 3.5 + (index % 5) * 0.1
    assert "utmos" in score("candidate", samples, {}).measurements


def test_the_report_names_what_it_does_not_claim() -> None:
    """A reader should not have to infer which numbers are load-bearing."""
    report = score("candidate", _samples(20), {"cer": 0.05})

    payload = report.to_json()

    assert payload["sample_size"] == 20
    assert set(payload["not_claimed"]) <= set(payload["measurements"])
    assert all(
        not payload["measurements"][key]["gating_eligible"] for key in payload["not_claimed"]
    )


def test_samples_round_trip_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {"prompt": "你好", "reference_text": "你好呀", "transcript": "你好哇"},
                {"prompt": "再见", "reference_text": "再见", "transcript": "再见", "utmos": 4.1},
            ]
        ),
        encoding="utf-8",
    )

    samples = load(path)

    assert len(samples) == 2
    assert samples[0].transcript == "你好哇"
    assert samples[1].utmos == 4.1


def test_blank_lines_in_the_file_are_skipped_not_scored_as_empty(tmp_path: Path) -> None:
    """A trailing newline should not become a silent sample."""
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps({"prompt": "a", "reference_text": "b", "transcript": "b"}) + "\n\n",
        encoding="utf-8",
    )

    assert len(load(path)) == 1


def test_the_probe_set_has_no_template_breeding() -> None:
    """The pretraining phase shipped 32 "long context" items that were
    4 templates x 8 fills -- nominally 32 samples, effectively 4.

    Duplicates and near-duplicates inflate the apparent sample size, which
    inflates every noise floor computed from it.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    rows = [
        json.loads(line)
        for line in (root / "configs" / "speech_probes_zh_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    prompts = [row["prompt"] for row in rows]

    assert len(prompts) >= 100, "generation health needs at least 100 probes"
    assert len(set(prompts)) == len(prompts), "duplicate prompts inflate the sample size"
    assert len({row["id"] for row in rows}) == len(rows)

    # No prompt should share a long prefix with another: that is what template
    # breeding looks like when the fills differ only at the end.
    for index, first in enumerate(prompts):
        for second in prompts[index + 1 :]:
            shared = 0
            for a, b in zip(first, second, strict=False):
                if a != b:
                    break
                shared += 1
            assert shared < 6, f"{first!r} and {second!r} share a {shared}-character prefix"


def test_a_report_without_the_text_check_says_so_rather_than_omitting_it() -> None:
    """Silence would read as "text ability intact" when it means "not looked at"."""
    import subprocess
    import sys
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as directory:
        candidate = _Path(directory) / "samples.jsonl"
        candidate.write_text(
            "\n".join(
                json.dumps({"prompt": "a", "reference_text": "b", "transcript": "b"})
                for _ in range(120)
            ),
            encoding="utf-8",
        )
        output = _Path(directory) / "report.json"
        root = _Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "evaluate_speech.py"),
                "--candidate",
                str(candidate),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            cwd=root,
        )

        assert result.returncode == 0, result.stderr
        assert "未测" in result.stdout
        assert json.loads(output.read_text(encoding="utf-8"))["candidate"]
        assert json.loads(output.read_text(encoding="utf-8"))["text_regression"] is None
