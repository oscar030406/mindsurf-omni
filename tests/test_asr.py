"""Tag stripping, and the rule that keeps the judge independent."""

from __future__ import annotations

from pathlib import Path

import pytest

from mindsurf_omni.service.asr import (
    SenseVoiceRecogniser,
    WhisperRecogniser,
    require_independent_judge,
    strip_tags,
)


def test_inline_tags_are_not_counted_as_speech() -> None:
    """They are metadata; scoring them charges the model for the recogniser's notes."""
    text, language = strip_tags("<|zh|><|NEUTRAL|><|Speech|><|woitn|>今天天气真好")

    assert text == "今天天气真好"
    assert language == "zh"


def test_the_reported_language_survives_stripping() -> None:
    """Chinese audio read as English is a failure worth seeing before the CER."""
    assert strip_tags("<|en|><|HAPPY|>hello there")[1] == "en"
    assert strip_tags("<|yue|>食咗飯未")[1] == "yue"


def test_text_without_tags_passes_through() -> None:
    assert strip_tags("今天天气真好") == ("今天天气真好", None)


def test_a_transcript_of_nothing_is_empty_not_a_tag_soup() -> None:
    assert strip_tags("<|nospeech|><|Event_UNK|>") == ("", "nospeech")


def test_the_service_recogniser_is_refused_as_a_judge() -> None:
    """It is the native path's own audio encoder; scoring with it is circular."""
    service = SenseVoiceRecogniser(model_dir="/nonexistent")

    with pytest.raises(ValueError, match="not an independent judge"):
        require_independent_judge(service, model_lineage="mindsurf-omni")


def test_an_independent_recogniser_is_accepted() -> None:
    require_independent_judge(WhisperRecogniser(), model_lineage="mindsurf-omni")


def test_a_judge_sharing_lineage_with_the_model_is_refused() -> None:
    """The rule is about shared failure modes, not about which vendor it is."""
    judge = WhisperRecogniser()
    judge.lineage = "mindsurf-omni"

    with pytest.raises(ValueError, match="share lineage"):
        require_independent_judge(judge, model_lineage="mindsurf-omni")


def test_the_rule_is_enforced_not_merely_documented() -> None:
    """A convention only written down is one that gets skipped under deadline."""
    assert SenseVoiceRecogniser(model_dir="/x").eligible_as_judge is False
    assert WhisperRecogniser().eligible_as_judge is True


def _tone(seconds: float = 1.0, amplitude: float = 0.2, rate: int = 16_000) -> bytes:
    """Something with energy in it, without needing an audio fixture."""
    import math
    import struct

    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * 220 * index / rate))
        for index in range(int(rate * seconds))
    ]
    return struct.pack(f"<{len(samples)}h", *samples)


@pytest.mark.asyncio
async def test_silence_is_not_sent_to_the_recogniser() -> None:
    """Asked to read an empty room, SenseVoice invents rather than declines.

    Measured live on three seconds of digital silence: it returned the Korean
    "그.", the model answered that in English, and the caller heard a
    9.6-second reply to a button they had pressed by accident.
    """
    from typing import Any

    calls: list[Any] = []

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            calls.append(kwargs)
            return [{"text": "그."}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    text, language = await recogniser.transcribe(b"\x00" * 16_000 * 2 * 3, 16_000)

    assert (text, language) == ("", None)
    assert calls == [], "the model was asked to read silence"


@pytest.mark.asyncio
async def test_audio_with_speech_in_it_still_reaches_the_recogniser() -> None:
    """The guard has to be quiet enough not to eat someone speaking softly."""
    from typing import Any

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            return [{"text": "<|zh|>你好"}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    text, _ = await recogniser.transcribe(_tone(), 16_000)

    assert text == "你好"


def test_the_silence_floor_sits_well_below_speech() -> None:
    """A threshold nobody can read is a threshold nobody keeps honest."""
    from mindsurf_omni.service.asr import SILENCE_RMS
    from mindsurf_omni.service.vad import frame_energy

    assert frame_energy(b"\x00" * 3200) < SILENCE_RMS
    # A quarter of full scale is ordinary speech: two orders of magnitude clear.
    assert frame_energy(_tone(0.1, amplitude=0.25)) > SILENCE_RMS * 50


def test_dither_is_off_so_the_same_recording_gives_the_same_text(tmp_path: Path) -> None:
    """Kaldi feature extraction adds gaussian noise to the waveform -- there so
    a digitally silent frame does not take log(0), and at inference the reason
    the same 74-second recording came back as six different transcripts in six
    requests. Under 25 seconds it looked stable only because a shorter
    recording gives the noise fewer frames in which to flip a decision."""
    (tmp_path / "config.yaml").write_text(
        "frontend_conf:\n  fs: 16000\n  n_mels: 80\n  lfr_m: 7\n", encoding="utf-8"
    )

    conf = SenseVoiceRecogniser(model_dir=tmp_path)._frontend_conf()

    assert conf["n_mels"] == 80
    assert conf["lfr_m"] == 7


def test_the_framing_the_model_was_trained_with_is_read_not_restated() -> None:
    """Only dither is ours to change. A checkpoint with different framing keeps
    it, which a hard-coded frontend config would silently overwrite."""
    recogniser = SenseVoiceRecogniser(model_dir=Path("does-not-exist"))

    fallback = recogniser._frontend_conf()

    assert fallback["fs"] == 16000
    assert "dither" not in fallback


@pytest.mark.asyncio
async def test_short_audio_is_given_the_declared_language_and_long_audio_is_not() -> None:
    """Half a second of a real 嗯 comes back as the Japanese うん。 -- loud, not
    silent, so the silence floor has nothing to say about it. The declared
    language reaches the model only there: over 320 corpus clips that the
    detector already called Chinese, passing it anyway still moved 20
    transcripts and made three worse, so above the threshold the call is
    unchanged.

    Driven through `transcribe` with audio on both sides of the threshold,
    because a test that reads the field back proves the field exists and
    nothing else."""
    from typing import Any

    seen: list[str] = []

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            seen.append(kwargs["language"])
            return [{"text": "<|zh|>嗯"}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused", language="zh")
    recogniser._model = Recorder()

    await recogniser.transcribe(_tone(0.5), 16_000)
    await recogniser.transcribe(_tone(3.0), 16_000)

    assert seen == ["zh", "auto"]


@pytest.mark.asyncio
async def test_declaring_auto_restores_the_behaviour_at_every_length() -> None:
    """The escape hatch for a deployment that is not spoken in one language."""
    from typing import Any

    seen: list[str] = []

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            seen.append(kwargs["language"])
            return [{"text": "<|zh|>嗯"}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused", language="auto")
    recogniser._model = Recorder()

    await recogniser.transcribe(_tone(0.5), 16_000)
    await recogniser.transcribe(_tone(3.0), 16_000)

    assert seen == ["auto", "auto"]


def test_a_language_the_model_cannot_read_is_refused_at_startup() -> None:
    """funasr takes an unknown language silently as "auto", so a typo would
    look configured and behave unconfigured."""
    from mindsurf_omni.service.config import ConfigurationError, Settings

    base = {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": "/w"}

    assert Settings.from_environment(base).asr_language == "zh"
    auto = Settings.from_environment({**base, "MINDSURF_ASR_LANGUAGE": "auto"})
    assert auto is not None and auto.asr_language == "auto"
    with pytest.raises(ConfigurationError, match="Chinese"):
        Settings.from_environment({**base, "MINDSURF_ASR_LANGUAGE": "Chinese"})


def test_a_transcript_of_only_foreign_script_is_not_delivered() -> None:
    """Asked to read a room with a fan in it, SenseVoice writes 그. and reports
    0.99 confidence in Korean. Gating on the reported language deletes ordinary
    Mandarin with it -- measured over 48 short spoken commands it calls 8% of
    them something other than Chinese, taking 咖啡 and 猫 down. This asks what
    the model wrote instead."""
    from mindsurf_omni.service.asr import writes_something

    for invented in ("그.", "うん。", "。", ".", "", "  ", "ら"):
        assert not writes_something(invented), invented
    for real in ("咖啡。", "嗯。", "demo", "B203", "60%", "Cafe。"):
        assert writes_something(real), real


@pytest.mark.asyncio
async def test_audio_shorter_than_one_frame_is_refused_not_crashed() -> None:
    """A single sample reached the frontend as `choose a window size 1` and left
    the caller with a bare 500."""
    from typing import Any

    seen: list[Any] = []

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            seen.append(kwargs)
            return [{"text": "<|zh|>。"}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    assert await recogniser.transcribe(_tone(0.01, amplitude=0.5), 16_000) == ("", None)
    assert seen == [], "the model was asked to read a fifth of a frame"


@pytest.mark.asyncio
async def test_the_models_own_nospeech_verdict_is_honoured() -> None:
    """It already said this is not speech; the service used to ship the `.`."""
    from typing import Any

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            return [{"text": "<|nospeech|><|Event_UNK|>."}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    assert await recogniser.transcribe(_tone(2.0), 16_000) == ("", None)


def test_steady_noise_is_caught_by_how_little_it_says() -> None:
    """Three seconds of mains hum comes back as 我. -- one Chinese character, so
    the script check passes it. Speaking rate is what separates them: noise sits
    at a third of a character per second against a person's 1.35, and the
    slowest real utterance measured is 0.67."""
    from mindsurf_omni.service.asr import said_enough

    assert not said_enough("我.", 3.0)
    assert not said_enough("I.", 3.0)
    assert said_enough("你好。", 1.2)
    assert said_enough("停。", 1.0)
    assert said_enough("你想看什么类型的电影，比如浪漫爱情、惊悚、恐怖。", 8.0)


def test_a_buffer_under_a_second_is_not_judged_by_rate() -> None:
    """A single word is over the line anyway, and the rate of a fraction of a
    second is not a rate."""
    from mindsurf_omni.service.asr import said_enough

    assert said_enough("好", 0.4)
    assert said_enough("", 0.4)


@pytest.mark.asyncio
async def test_a_recording_too_long_for_one_pass_is_refused_with_a_way_out() -> None:
    """Twenty minutes asked the allocator for 44 GiB and came back a bare 500,
    and the reserved block was still held when the next caller arrived."""
    from typing import Any

    from mindsurf_omni.service.engine import TooLongForModel

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            raise AssertionError("the model should not have been asked")

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    with pytest.raises(TooLongForModel, match="pieces"):
        await recogniser.transcribe(_tone(1200.0, amplitude=0.2), 16_000)


@pytest.mark.asyncio
async def test_a_long_silence_is_still_answered_not_refused() -> None:
    """Silence short-circuits before any GPU work, so the length refusal must
    sit below it: telling somebody to split a recording of nothing is advice
    they cannot use."""
    from typing import Any

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            raise AssertionError("the model should not have been asked")

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    assert await recogniser.transcribe(b"\x00\x00" * (16_000 * 1200), 16_000) == ("", None)


@pytest.mark.asyncio
async def test_the_sample_rate_reaches_the_recogniser() -> None:
    """It used to travel from the endpoint and stop one call short, so 48 kHz
    posted as 16 kHz produced a transcript that reads like Chinese and says
    something else -- 200, no exception, nothing in the log. Measured on four
    recordings, that mislabelling scores a character error rate of 1.0."""
    from typing import Any

    lengths: list[int] = []

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            lengths.append(len(kwargs["input"]))
            return [{"text": "<|zh|>你好，今天天气怎么样。"}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    await recogniser.transcribe(_tone(2.0, rate=48_000), 48_000)
    assert lengths == [pytest.approx(2.0 * 16_000, rel=0.01)]
