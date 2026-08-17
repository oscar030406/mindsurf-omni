"""末字那把尺子：读的是「第一次消失之前」，不是「总共出现过几次」。"""

from __future__ import annotations

import numpy as np
from scripts.measure_tail_syllable import CUTS_MS, drop_trailing_silence, survives_up_to


def test_the_reading_stops_at_the_first_disappearance() -> None:
    """识别器会补字，所以切得更狠反而又「认出来」是会发生的。
    取最大值会把那次回光返照读成「撑得更久」，读的必须是第一次消失之前。"""
    flickering = [True, True, False, False, True, False, False]

    assert survives_up_to(flickering, CUTS_MS) == 50


def test_a_character_that_was_never_there_reads_zero() -> None:
    assert survives_up_to([False] * len(CUTS_MS), CUTS_MS) == 0


def test_a_character_that_survives_everything_reads_the_last_cut() -> None:
    assert survives_up_to([True] * len(CUTS_MS), CUTS_MS) == CUTS_MS[-1]


def test_the_sweep_starts_where_the_speech_ends_not_where_the_file_does() -> None:
    """两条臂尾巴上补的静音不一样长，不先削掉的话，前几档切的是各自的填充，
    问的就不是同一个问题了。"""
    speech = 0.5 * np.sin(np.linspace(0, 40, 8000))
    padded = np.concatenate([speech, np.zeros(16000)])

    trimmed = drop_trailing_silence(padded)

    assert len(trimmed) < len(padded)
    assert abs(len(trimmed) - len(speech)) < 200
