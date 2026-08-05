"""The harness must report weak instruments, not hide them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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


def test_paired_deltas_refuse_mismatched_texts() -> None:
    """A pair over different texts subtracts two unrelated numbers.

    Free-run output has different texts per run, so pairing it silently would
    produce a smaller noise floor that certifies nonsense. None means "use the
    unpaired comparison", not "something went wrong".
    """
    from scripts.evaluate_speech import Sample, paired_deltas

    mine = [Sample(prompt="p", reference_text="文本甲", transcript="文本甲", id="a")]
    theirs = [Sample(prompt="p", reference_text="文本乙", transcript="文本乙", id="a")]

    assert paired_deltas(mine, theirs) is None


def test_paired_deltas_refuse_thin_overlap() -> None:
    """Pairing a fifth of the runs would report the pairs as if they were the runs."""
    from scripts.evaluate_speech import Sample, paired_deltas

    mine = [
        Sample(prompt="p", reference_text=f"t{i}", transcript=f"t{i}", id=f"s{i}")
        for i in range(10)
    ]
    theirs = [Sample(prompt="p", reference_text="t0", transcript="t0", id="s0")]

    assert paired_deltas(mine, theirs) is None


def test_paired_deltas_cancel_shared_item_difficulty() -> None:
    """The whole point of pairing: per-item difficulty subtracts out.

    Two systems that differ by a constant on wildly different items should
    show that constant with a tiny floor, not the items' spread.
    """
    from scripts.evaluate_speech import Sample, paired_deltas

    mine, theirs = [], []
    for index in range(40):
        text = "字" * (10 + index * 3)  # very different difficulties
        wrong = 2 + index % 5
        mine.append(
            Sample(
                prompt="p",
                reference_text=text,
                transcript="错" * wrong + text[wrong:],
                id=f"s{index}",
            )
        )
        theirs.append(Sample(prompt="p", reference_text=text, transcript=text, id=f"s{index}"))

    deltas = paired_deltas(mine, theirs)

    assert deltas is not None
    assert all(delta > 0 for delta in deltas["cer"])  # ours strictly worse per item


def test_two_arms_judged_differently_are_refused() -> None:
    """The 0.08 between whisper-small and paraformer-zh is bigger than most
    effects this harness is asked to certify, so it would arrive as a result."""
    from scripts.evaluate_speech import check_same_judge

    mine = [Sample(prompt="p", reference_text="t", transcript="t", judge="paraformer-zh")]
    theirs = [Sample(prompt="p", reference_text="t", transcript="t", judge="whisper-small")]

    reason = check_same_judge(mine, theirs)

    assert reason and "paraformer-zh" in reason and "whisper-small" in reason


def test_one_arm_judged_by_two_recognisers_is_refused() -> None:
    """Halfway through a rerun is exactly how this happens."""
    from scripts.evaluate_speech import check_same_judge

    mixed = [
        Sample(prompt="p", reference_text="t", transcript="t", judge="paraformer-zh"),
        Sample(prompt="p", reference_text="t", transcript="t", judge="whisper-small"),
    ]

    assert check_same_judge(mixed, mixed)


def test_the_same_judge_on_both_arms_passes() -> None:
    from scripts.evaluate_speech import check_same_judge

    arm = [Sample(prompt="p", reference_text="t", transcript="t", judge="paraformer-zh")]

    assert check_same_judge(arm, list(arm)) is None


def test_rows_without_a_judge_are_unknown_rather_than_matching() -> None:
    """Artifacts predate the field; silence must not read as agreement."""
    from scripts.evaluate_speech import check_same_judge, judges_of

    old = [Sample(prompt="p", reference_text="t", transcript="t")]
    new = [Sample(prompt="p", reference_text="t", transcript="t", judge="paraformer-zh")]

    assert judges_of(old) == set()
    assert check_same_judge(new, old) is None  # not refused, but not confirmed either


def test_compare_paired_gives_only_three_answers() -> None:
    from mindsurf_omni.evaluation.metrics import compare_paired

    clear = compare_paired("cer", [0.2] * 50)
    assert "regressed" in clear and "paired n=50" in clear

    noisy = compare_paired("cer", [0.5, -0.5] * 25)
    assert "indistinguishable" in noisy

    better = compare_paired("utmos", [0.4] * 50, lower_is_better=False)
    assert "improved" in better

    thin = compare_paired("cer", [0.9])
    assert "reported only" in thin


def test_the_median_is_reported_beside_the_mean() -> None:
    """A mean improving while the median holds means fewer catastrophes, not better speech.

    Both shapes below average near 0.15; only the median tells them apart, and
    they want different fixes.
    """
    from scripts.evaluate_speech import score

    uniform = [
        Sample(prompt="p", reference_text="字" * 10, transcript="错" + "字" * 9) for _ in range(120)
    ]
    occasional = [
        Sample(
            prompt="p",
            reference_text="字" * 10,
            transcript=("错" * 10 if index < 18 else "字" * 10),
        )
        for index in range(120)
    ]

    flat = score("uniform", uniform, {"cer": 0.05})
    spiky = score("occasional", occasional, {"cer": 0.05})

    assert flat.shape["cer_median"] == pytest.approx(0.1)
    assert spiky.shape["cer_median"] == 0.0
    assert spiky.shape["cer_over_0_3"] == 18.0
    assert flat.to_json()["shape"]["cer_median"] == pytest.approx(0.1)


def test_the_shape_line_names_the_failure_pattern() -> None:
    from scripts.evaluate_speech import score, shape_lines

    widespread = score(
        "ours",
        [Sample(prompt="p", reference_text="字" * 10, transcript="错" * 10) for _ in range(120)],
        {"cer": 0.05},
    )

    assert "普遍念不准" in shape_lines(widespread)[0]


def test_a_judge_that_never_ran_refuses_rather_than_scoring_silence() -> None:
    """Zero transcripts is the judge failing, not the model failing to speak.

    Counting failures as silence is right for a few clips and wrong for all of
    them: a recogniser that could not load produces a perfect CER of 1.0 and a
    report that looks like a measurement. This happened -- whisper could not
    find ffmpeg on the server, wrote 158 empty transcripts, and only set
    transcribed:false on each row.
    """
    import pytest
    from scripts.transcribe_samples import refuse_if_the_judge_never_ran

    with pytest.raises(SystemExit, match="the judge failed to run"):
        refuse_if_the_judge_never_ran(
            [
                {"id": "a", "transcribed": False, "error": "[Errno 2] ... 'ffmpeg'"},
                {"id": "b", "transcribed": False, "error": "[Errno 2] ... 'ffmpeg'"},
            ]
        )


def test_some_failures_are_still_counted_as_silence() -> None:
    """The all-or-nothing distinction is the whole point of the guard.

    A model that goes silent on a few hard prompts is a real result and must
    keep scoring; only a clean sweep means the instrument, not the subject.
    """
    from scripts.transcribe_samples import refuse_if_the_judge_never_ran

    refuse_if_the_judge_never_ran(
        [{"id": "a", "transcribed": True}, {"id": "b", "transcribed": False}]
    )
    refuse_if_the_judge_never_ran([])


def test_the_digit_split_separates_notation_from_a_model_that_cannot_speak() -> None:
    """Most of what Arabic numerals cost this metric is correct speech.

    A speaker reading 2021 aloud as 二零二一 is right, and the judge transcribes
    what it hears, so the comparison charges a substitution per digit. Without
    the split beside the mean, a reader comparing our number to a cascade's
    concludes the model is several times worse when most of the gap is notation.
    """
    from scripts.evaluate_speech import score, shape_lines

    report = score(
        "ours",
        [Sample(prompt="p", reference_text="2021年", transcript="二零二一年") for _ in range(8)]
        + [Sample(prompt="p", reference_text="今天天气", transcript="今天天气") for _ in range(8)],
        {"cer": 0.05},
    )

    assert report.shape["n_with_digits"] == 8.0
    assert report.shape["cer_with_digits"] > report.shape["cer_without_digits"]
    assert report.shape["cer_without_digits"] == 0.0
    assert "含阿拉伯数字 8/16 条" in shape_lines(report)[1]


def test_the_split_is_absent_when_every_sample_looks_the_same() -> None:
    """Reporting a split of one group against nothing invites reading a zero."""
    from scripts.evaluate_speech import score

    report = score(
        "ours",
        [Sample(prompt="p", reference_text="今天天气", transcript="今天天气") for _ in range(4)],
        {"cer": 0.05},
    )

    assert "cer_with_digits" not in report.shape
