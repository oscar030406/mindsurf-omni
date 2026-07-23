"""The log reader, and the comparison rule that stopped a wrong conclusion."""

from __future__ import annotations

from pathlib import Path

from scripts.watch_training import compare_epochs, parse, window


def _line(step: int, loss: float, text: float, audio: float, lr: float) -> str:
    return (
        f"Epoch:[1/6]({step}/31224), loss: {loss}, text: {text}, "
        f"audio: {audio}, lr: {lr}, epoch_time: 120.0min"
    )


LOG = "\n".join(
    [
        "[mindsurf] row-group shuffle, epoch 0",
        _line(50, 9.0840, 2.1921, 6.8918, 0.00050000),
        _line(5050, 6.9586, 1.9141, 5.0445, 0.00049800),
        _line(6000, 6.9000, 1.9000, 5.0000, 0.00049700),
        _line(7000, 6.9300, 1.9200, 5.0100, 0.00049600),
    ]
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "train.log"
    path.write_text(text, encoding="utf-8")
    return path


def test_only_training_lines_are_parsed(tmp_path: Path) -> None:
    """The log carries banners and progress noise the parser must ignore."""
    points = parse(_write(tmp_path, LOG))

    assert len(points) == 4
    assert points[0].epoch == 1
    assert points[0].audio == 6.8918
    assert points[-1].lr == 0.000496


def test_an_empty_or_unrelated_log_yields_nothing_rather_than_guessing(tmp_path: Path) -> None:
    assert parse(_write(tmp_path, "starting up\nloading weights\n")) == []


def test_windows_select_by_step_not_by_order(tmp_path: Path) -> None:
    points = parse(_write(tmp_path, LOG))

    selected = window(points, epoch=1, low=5000, high=7000)

    assert [p.step for p in selected] == [5050, 6000, 7000]


def _synthetic(epochs: dict[int, list[float]]) -> str:
    lines = []
    for epoch, values in epochs.items():
        for index, audio in enumerate(values):
            step = 5000 + index * 50
            lines.append(
                f"Epoch:[{epoch}/6]({step}/31224), loss: {audio + 1.6:.4f}, "
                f"text: 1.6000, audio: {audio:.4f}, lr: 0.00040000, epoch_time: 100.0min"
            )
    return "\n".join(lines)


def test_a_difference_larger_than_the_noise_is_named(tmp_path: Path) -> None:
    """Epoch 1 to 2 really did improve, by three times the threshold."""
    log = _synthetic(
        {
            1: [5.05 + (index % 5) * 0.02 for index in range(60)],
            2: [4.35 + (index % 5) * 0.02 for index in range(60)],
        }
    )

    rows = compare_epochs(parse(_write(tmp_path, log)), "audio", 5000, 8000)

    assert rows[1]["verdict"] == "improved"
    assert abs(float(rows[1]["difference"])) > float(rows[1]["threshold"])  # type: ignore[arg-type]


def test_a_difference_inside_the_noise_is_refused(tmp_path: Path) -> None:
    """The case that fooled me by eye, with the run's own numbers.

    Epoch 2 measured 4.3479 +/- 0.0476 over 61 points and epoch 3 measured
    4.4434 +/- 0.1095 over 59, so the rise of 0.0954 sits well inside the
    combined threshold of 0.3582. The spreads here are chosen to reproduce
    those noise floors -- a tidy periodic sequence would bootstrap far tighter
    than real training noise and would not exercise the rule at all.
    """
    import random

    rng = random.Random(0)
    log = _synthetic(
        {
            2: [rng.gauss(4.3479, 0.19) for _ in range(61)],
            3: [rng.gauss(4.4434, 0.43) for _ in range(59)],
        }
    )

    rows = compare_epochs(parse(_write(tmp_path, log)), "audio", 5000, 8000)

    assert rows[1]["verdict"] == "indistinguishable"
    assert abs(float(rows[1]["difference"])) < float(rows[1]["threshold"])  # type: ignore[arg-type]


def test_an_epoch_that_has_not_reached_the_window_is_skipped(tmp_path: Path) -> None:
    """A truncated sample would give a noise floor computed from too few points."""
    log = _synthetic({1: [5.0] * 60, 2: [4.4] * 60})
    log += "\n" + (
        "Epoch:[3/6](100/31224), loss: 6.0, text: 1.6, audio: 4.4, lr: 0.0004, epoch_time: 1.0min"
    )

    rows = compare_epochs(parse(_write(tmp_path, log)), "audio", 5000, 8000)

    assert [row["epoch"] for row in rows] == [1, 2]


def test_the_first_epoch_gets_no_verdict_because_there_is_nothing_to_compare(
    tmp_path: Path,
) -> None:
    rows = compare_epochs(parse(_write(tmp_path, _synthetic({1: [5.0] * 60}))), "audio", 5000, 8000)

    assert len(rows) == 1
    assert "verdict" not in rows[0]


def test_a_partly_filled_window_is_skipped_not_compared(tmp_path: Path) -> None:
    """A quarter-filled window is biased, not merely noisy.

    Its points are the *first* part of the window, and during annealing that
    part differs systematically from the rest -- so its mean is wrong in a
    direction, which a noise floor cannot express.
    """
    log = _synthetic({1: [5.0 + (i % 5) * 0.02 for i in range(60)]})
    log += "\n" + _synthetic({2: [4.4 + (i % 5) * 0.02 for i in range(60)]})
    # Epoch 3 has crossed only a quarter of the window.
    log += "\n" + _synthetic({3: [4.2 + (i % 5) * 0.02 for i in range(15)]})

    rows = compare_epochs(parse(_write(tmp_path, log)), "audio", 5000, 8000)

    assert [row["epoch"] for row in rows] == [1, 2]


def test_a_window_that_is_merely_sparse_for_everyone_still_compares(tmp_path: Path) -> None:
    """Coverage is relative: a wide log interval is not a partial epoch."""
    log = _synthetic({1: [5.0 + (i % 5) * 0.02 for i in range(12)]})
    log += "\n" + _synthetic({2: [4.4 + (i % 5) * 0.02 for i in range(12)]})

    rows = compare_epochs(parse(_write(tmp_path, log)), "audio", 5000, 8000)

    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[1]["verdict"] == "improved"
