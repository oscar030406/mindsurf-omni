"""Endpoint detection, including the ways it can end a turn wrongly."""

from __future__ import annotations

import math
import struct

from mindsurf_omni.service.vad import FRAME_MS, EndpointDetector, frame_energy

SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000


def tone(frames: int, amplitude: float = 0.3) -> bytes:
    """A sine, which is what speech looks like to an energy detector."""
    samples = []
    for index in range(frames * SAMPLES_PER_FRAME):
        value = amplitude * math.sin(2 * math.pi * 220 * index / SAMPLE_RATE)
        samples.append(int(value * 32767))
    return struct.pack(f"<{len(samples)}h", *samples)


def quiet(frames: int, amplitude: float = 0.0005) -> bytes:
    samples = []
    for index in range(frames * SAMPLES_PER_FRAME):
        samples.append(int(amplitude * 32767 * ((index % 7) - 3) / 3))
    return struct.pack(f"<{len(samples)}h", *samples)


def test_energy_separates_speech_from_room_tone() -> None:
    assert frame_energy(tone(1)) > frame_energy(quiet(1)) * 20


def test_energy_of_an_empty_frame_is_zero_not_an_error() -> None:
    assert frame_energy(b"") == 0.0
    assert frame_energy(b"\x00") == 0.0  # odd byte count, truncated cleanly


def test_a_turn_ends_after_the_configured_silence() -> None:
    detector = EndpointDetector(silence_ms=300)

    assert detector.push_stream(tone(30)) is None  # still talking
    ended = detector.push_stream(quiet(20))  # 400 ms of quiet

    assert ended is not None


def test_silence_alone_never_ends_a_turn_that_never_began() -> None:
    """Otherwise an empty room ends a turn and the model answers nothing."""
    detector = EndpointDetector(silence_ms=300)

    assert detector.push_stream(quiet(200)) is None
    assert detector.speech_started is False


def test_a_pause_shorter_than_the_threshold_does_not_cut_the_speaker_off() -> None:
    """Mid-sentence breath must not be read as the end of the turn."""
    detector = EndpointDetector(silence_ms=300)

    detector.push_stream(tone(20))
    assert detector.push_stream(quiet(10)) is None  # 200 ms pause
    assert detector.push_stream(tone(20)) is None


def test_a_click_too_short_to_be_speech_does_not_arm_the_detector() -> None:
    """A door closing should not start and then end a turn."""
    detector = EndpointDetector(silence_ms=300, min_speech_ms=200)

    detector.push_stream(tone(5))  # 100 ms, below the minimum

    assert detector.speech_started is False
    assert detector.push_stream(quiet(30)) is None


def test_the_floor_adapts_so_a_quiet_microphone_is_not_read_as_silence() -> None:
    """A fixed absolute threshold makes the detector deaf on quiet hardware."""
    detector = EndpointDetector(initial_noise_floor=0.02)
    detector.push_stream(quiet(100, amplitude=0.0002))

    # Speech well below the initial threshold is still detected once the floor
    # has settled to the room.
    detector.push_stream(tone(20, amplitude=0.01))

    assert detector.speech_started is True


def test_loud_speech_does_not_drag_the_floor_up_and_deafen_the_detector() -> None:
    """If the floor tracked during speech, everything after it would be silence."""
    detector = EndpointDetector()
    before = detector._noise_floor

    detector.push_stream(tone(50, amplitude=0.8))

    assert detector._noise_floor == before


def test_reset_clears_the_turn_but_the_object_stays_usable() -> None:
    detector = EndpointDetector(silence_ms=300)
    detector.push_stream(tone(20))
    detector.reset()

    assert detector.speech_started is False
    assert detector.push_stream(quiet(30)) is None
    detector.push_stream(tone(20))
    assert detector.speech_started is True


def test_the_endpoint_offset_points_past_the_frame_that_ended_the_turn() -> None:
    """The caller slices at this offset; an off-by-one would clip or duplicate."""
    detector = EndpointDetector(silence_ms=100)
    speech = tone(20)
    detector.push_stream(speech)

    offset = detector.push_stream(quiet(20))

    assert offset is not None
    assert offset % detector.frame_bytes == 0
    assert offset <= len(quiet(20))


# --- 分段：长录音按停顿切开 -------------------------------------------------


def _speech(seconds: float, level: int = 3000) -> bytes:
    """噪声乘一层音节包络。

    平的白噪不是语音的替身：这一级读的是「这段录音自己的安静档位」，
    而真语音每个音节之间都弱下去一次，平的信号量不出档位来。
    """
    import numpy as np

    count = int(seconds * 16_000)
    rng = np.random.default_rng(int(seconds * 1000))
    syllables = (0.5 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(count) / 16_000)) ** 4
    return (rng.standard_normal(count) * level * syllables).astype(np.int16).tobytes()


def _quiet(seconds: float) -> bytes:
    import numpy as np

    return np.zeros(int(seconds * 16_000), dtype=np.int16).tobytes()


def test_a_recording_with_pauses_is_cut_in_the_pauses() -> None:
    from mindsurf_omni.service.vad import SEGMENT_HARD_SECONDS, segments

    pcm = _quiet(0.3) + _speech(20) + _quiet(0.5) + _speech(20) + _quiet(0.5) + _speech(20)
    cuts = segments(pcm)

    assert len(cuts) > 1
    assert all((end - start) / 32_000 <= SEGMENT_HARD_SECONDS + 0.01 for start, end in cuts)
    # Every seam falls in a gap, so no piece starts or ends inside speech.
    for (_, end), (start, _) in zip(cuts, cuts[1:], strict=False):
        assert start > end


def test_speech_with_no_pause_at_all_is_still_cut() -> None:
    """An open microphone next to a fan has no gaps, and it is the case that
    reached the allocator for 44 GiB."""
    from mindsurf_omni.service.vad import SEGMENT_HARD_SECONDS, segments

    cuts = segments(_speech(100))

    assert len(cuts) == 3
    assert all((end - start) / 32_000 <= SEGMENT_HARD_SECONDS + 0.01 for start, end in cuts)


def test_nothing_to_hear_is_no_pieces() -> None:
    from mindsurf_omni.service.vad import segments

    assert segments(_quiet(3)) == []


def test_the_last_piece_stops_at_the_last_word() -> None:
    """Holding the key down after finishing must not hand the last piece a
    stretch of nothing: the guard that reads a speaking rate divides by the
    length of the piece, and it would throw the words away for being too slow."""
    from mindsurf_omni.service.vad import segments

    pcm = _speech(3) + _quiet(10)
    ((start, end),) = segments(pcm)

    assert (end - start) / 32_000 < 3.5


def test_the_quiet_level_comes_from_the_recording_not_from_a_constant() -> None:
    """噪底的下限如果写死，那就是对麦克风电平写死一个假设。

    实测（80 段真语料拼成 26 分钟，逐级衰减）：写死 0.0007 时，录得轻十倍的那份
    58 个切口有 22 个落在语音里；按录音自己的安静档位取，是 3 个。
    合成信号复现不出这个差别（真语料每段都有自己的电平和首尾余音），
    所以这里守的是这一级的性质，对比读数在 headline_numbers.json。
    """
    import numpy as np

    from mindsurf_omni.service.vad import (
        EndpointDetector,
        _quiet_level,
        frame_energies,
    )

    # 恰好为零的帧是文件的静音，不是房间的，不参与取值。
    room = 0.001
    energies = [0.0] * 90 + [room] * 30 + [0.05] * 100
    assert abs(_quiet_level(energies) - room) < 1e-9

    # 录得轻十倍，取出来的档位也跟着轻十倍。
    rng = np.random.default_rng(5)
    talk = np.frombuffer(_speech(3), dtype=np.int16).astype(np.float32)
    loud = np.clip(talk + rng.standard_normal(len(talk)) * 80, -32768, 32767).astype(np.int16)
    faint = np.clip(talk * 0.1 + rng.standard_normal(len(talk)) * 8, -32768, 32767).astype(np.int16)
    louder = _quiet_level(frame_energies(loud.tobytes(), 640))
    fainter = _quiet_level(frame_energies(faint.tobytes(), 640))
    assert fainter < louder / 3

    # 起始值也来自这段录音，不是绝对先验。先验的毛病是：噪底只在判成静音的帧上
    # 更新，房间一旦高过 先验 × speech_ratio，第一帧就判成语音、噪底从此不动，
    # 整段变成一个 span。
    room = 0.02
    noisy = np.clip(talk + rng.standard_normal(len(talk)) * room * 32768, -32768, 32767)

    assert _quiet_level(frame_energies(noisy.astype(np.int16).tobytes(), 640)) > (
        EndpointDetector().initial_noise_floor
    ), "这个房间正是先验够不着的那一档"


def test_leading_silence_does_not_make_the_room_tone_into_speech() -> None:
    """噪底跟着数字静音一路衰减到零，之后任何声音都是它的三倍。

    实测的形状：片段开头是数字静音，后面接 0.0008 的室内底噪，
    结果底噪被标成语音、前面真的语音被标成静音——检测器整个反过来。
    """
    import numpy as np

    from mindsurf_omni.service.vad import segments, voiced_seconds

    rng = np.random.default_rng(3)
    room = (rng.standard_normal(8 * 16_000) * 26).astype(np.int16).tobytes()
    pcm = _quiet(1.5) + _speech(1.8) + room

    # 只有那 1.8 秒算说话，八秒底噪不算。
    assert 0.2 < voiced_seconds(pcm) < 3.0
    assert len(segments(pcm)) == 1


def test_the_speaking_rate_denominator_survives_a_room() -> None:
    """尾巴不许被算成说话，房间吵起来也不许。

    这一条和 `segments` 要的是相反的错误：切录音要把停顿找准，
    这里宁可少算——分母小、速率高、话留下，而误杀一句真话是这一格最贵的错。
    共用一个自适应噪底服务不了两件事，两个方向都实测坏过。

    信号按真的短指令标定：edge 合成的 12 条，最大帧 RMS 中位 0.226、最小 0.183。
    **够不着的那一档写在这里而不是断言里**：信噪比低到 14 dB 左右
    （房间 0.02 对最大帧 0.09），能量分不开语音和房间，这道闸会误杀。
    真语音在 0.02 的房间里是 21 dB，够得着。
    """
    import numpy as np

    from mindsurf_omni.service.vad import voiced_seconds

    rng = np.random.default_rng(11)
    raw = np.frombuffer(_speech(1.8), dtype=np.int16).astype(np.float32)
    said = raw / max(float(np.abs(raw).max()), 1.0) * 0.226 * 32768 * 2.5
    spoken = len(said) / 16_000

    for room in (0.0008, 0.02):
        for pad in (0.0, 3.0, 8.0):
            tail = rng.standard_normal(int(pad * 16_000)) * room * 32768
            pcm = np.clip(np.concatenate([said, tail]), -32768, 32767).astype(np.int16).tobytes()

            # 三个字的短指令，闸线 0.5 字每秒——不许因为尾巴长就掉下去。
            voiced = voiced_seconds(pcm)
            assert voiced < 1.0 or 3 / voiced >= 0.5, f"room={room} pad={pad} 有声 {voiced:.2f}s"
            assert voiced <= spoken * 1.5, f"room={room} pad={pad} 把尾巴算进去了"
