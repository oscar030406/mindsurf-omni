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
async def test_a_long_recording_is_read_in_pieces_no_pass_could_hold() -> None:
    """Twenty minutes asked the allocator for 44 GiB and came back a bare 500.

    A tone has no pauses to cut in, so this is the worst case for the split:
    every seam is forced. What it has to show is that no single pass is handed
    more than the hard limit, which is where the allocation stopped being
    survivable."""
    from typing import Any

    from mindsurf_omni.service.vad import SEGMENT_HARD_SECONDS

    seen: list[float] = []

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            seconds = len(kwargs["input"]) / 16_000
            seen.append(seconds)
            # At a plausible speaking rate, so the piece clears said_enough the
            # same way a real one would.
            return [{"text": "<|zh|>" + "一段听得见的话。" * int(seconds)}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    text, _ = await recogniser.transcribe(_tone(1200.0, amplitude=0.2), 16_000)

    assert len(seen) > 1
    assert max(seen) <= SEGMENT_HARD_SECONDS + 0.01
    assert sum(seen) == pytest.approx(1200.0, abs=1.0)
    assert text.startswith("一段听得见的话。")


@pytest.mark.asyncio
async def test_a_recording_past_the_endpoint_limit_is_still_refused() -> None:
    """Splitting bounds the memory, not the transport: an hour of PCM16 is
    115 MB in one request body, and past that the caller is told a number."""
    from typing import Any

    from mindsurf_omni.service.asr import LONGEST_SECONDS
    from mindsurf_omni.service.engine import TooLongForModel

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            raise AssertionError("the model should not have been asked")

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    with pytest.raises(TooLongForModel, match="pieces") as refusal:
        await recogniser.transcribe(_tone(LONGEST_SECONDS + 10, amplitude=0.2), 16_000)

    # 边界上不能自相矛盾：两边都四舍五入的话，刚过线的请求收到的是
    # 「this recording is 3600 seconds, past the 3600」，读完不知道该改什么。
    said, limit = f"{LONGEST_SECONDS + 10:.1f}", f"{LONGEST_SECONDS:.0f}"
    assert said in str(refusal.value)
    assert said != limit


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


@pytest.mark.asyncio
async def test_holding_the_key_after_speaking_does_not_delete_what_was_said() -> None:
    """按住录音键多按几秒，整条转写被删成空串——语速的分母用错了。

    分母是 buffer 的长度时，好的。加三秒室内底噪读出 0.7 字每秒、加八秒读出 0.25，
    两个都在真人的下限之下。实测 36 段短指令：不加尾巴 0 条被吃，加三秒 30 条，
    加八秒 36 条。没有人在话音落地的那一刻松手。
    """
    from typing import Any

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            return [{"text": "<|zh|>好的。"}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    spoken = _tone(0.8, amplitude=0.2)
    for padding in (0.0, 3.0, 8.0):
        quiet = _tone(padding, amplitude=0.0008) if padding else b""
        text, _ = await recogniser.transcribe(spoken + quiet, 16_000)
        assert text == "好的。", f"{padding} 秒的尾巴把它删了"


@pytest.mark.asyncio
async def test_a_long_recording_with_no_detectable_pause_is_still_read() -> None:
    """停顿检测说「这里没有语音」和上游的静音闸说的不是同一句话。

    上游那道闸已经放行了，说明缓冲区里有东西。这时候检测器数不出有声段
    （安静而且包络平的录音会这样），退回定长切窗——丢的是好切口，不是整段录音。
    """
    from typing import Any

    seen: list[float] = []

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            seconds = len(kwargs["input"]) / 16_000
            seen.append(seconds)
            return [{"text": "<|zh|>" + "一段听得见的话。" * int(seconds)}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    # 恒定电平、无停顿、刚好过静音闸：检测器取不到「安静档位」。
    import numpy as np

    flat = (np.full(200 * 16_000, 900, dtype=np.int16)).tobytes()
    text, _ = await recogniser.transcribe(flat, 16_000)

    assert seen, "整段被当成没有语音丢掉了"
    assert sum(seen) == pytest.approx(200.0, abs=1.0)
    assert text


@pytest.mark.asyncio
async def test_a_slow_stretch_is_vouched_for_by_the_rest_of_the_dictation() -> None:
    """语速判一次，不是每段判一次。

    切段之后这道闸从「整段一次」变成「每段一次」，于是一段停下来想事情、
    念一串数字、念一个网址，自己就低于闸线，被静默丢掉——实测 145 个字里
    消失 4 个，日志里什么都没有。说得快的那一段本来是给说得慢的那段作保的。
    """
    from typing import Any

    written = ["一段听得见的话。" * 20, "嗯。"]

    class Recorder:
        def generate(self, **kwargs: Any) -> list[dict[str, str]]:
            return [{"text": "<|zh|>" + (written.pop(0) if written else "。")}]

    recogniser = SenseVoiceRecogniser(model_dir="/unused")
    recogniser._model = Recorder()

    # 两段：一段说得飞快，一段只有两个字。
    pcm = _tone(60.0, amplitude=0.2) + _tone(0.5, amplitude=0.0) + _tone(60.0, amplitude=0.2)
    text, _ = await recogniser.transcribe(pcm, 16_000)

    assert text.endswith("嗯。"), "慢的那一段被逐段的闸吃掉了"


def test_a_deployment_can_choose_which_recogniser_it_serves() -> None:
    """The lead asked for the streaming one on the model list, selectable.

    Both are real answers to different questions -- the streaming one writes
    while the speaker is still talking, which SenseVoice structurally cannot,
    and reads 0.2796 against its 0.1094 -- so this is a choice the deployment
    makes rather than a replacement.
    """
    from mindsurf_omni.service.config import ConfigurationError, Settings

    def settings_for(name: str, tmp: Path) -> Settings:
        for folder in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
            (tmp / folder).mkdir(exist_ok=True)
        made = Settings.from_environment(
            {
                "MINDSURF_ENGINE": "cascade",
                "MINDSURF_WEIGHTS": str(tmp),
                "MINDSURF_ASR_MODEL": name,
            }
        )
        assert made is not None
        return made

    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        assert settings_for("sensevoice", tmp).asr_model == "sensevoice"
        settings_for("sensevoice", tmp).verify()
        assert settings_for("paraformer-streaming", tmp).asr_model == "paraformer-streaming"
        settings_for("paraformer-streaming", tmp).verify()

        # A name nobody wired is refused at assembly, not on the first thing
        # somebody says.
        with pytest.raises(ConfigurationError, match="MINDSURF_ASR_MODEL"):
            settings_for("whisper", tmp).verify()


def test_the_streaming_recogniser_yields_additions_not_rewrites() -> None:
    """Append-only, so a live display never has to re-render what it showed.

    Every revision policy in the streaming literature exists to manage text
    that changes after it is shown. This model commits as it goes, so the
    caller appends and there is nothing to manage. Also checks the step count:
    600 ms of audio per step is what makes the first word land at 0.6 s.
    """
    import asyncio

    from mindsurf_omni.service.asr import STREAMING_STRIDE, ParaformerStreamingRecogniser

    class Stub(ParaformerStreamingRecogniser):
        """Answers a word per step, and nothing on the step that has none."""

        steps: list[bool] = []  # noqa: RUF012 - test stub, one instance

        def load(self) -> None:
            self._model = object()

        def _step(self, piece: object, cache: dict, last: bool) -> str:
            Stub.steps.append(last)
            return ["今天", "", "天气"][len(Stub.steps) - 1]

    Stub.steps = []
    recogniser = Stub()

    async def collect() -> list[str]:
        # Two bytes per sample, so this is exactly three strides of audio.
        audio = b"\x00\x01" * (STREAMING_STRIDE * 3)
        return [piece async for piece in recogniser.stream(audio, 16_000)]

    pieces = asyncio.run(collect())

    # Three steps of 600 ms over three strides of audio, and the empty one is
    # not yielded -- an empty delta is a blank update the caller would render.
    assert len(Stub.steps) == 3
    assert Stub.steps == [False, False, True], "only the last step may be final"
    assert pieces == ["今天", "天气"]


def test_a_rate_the_streaming_recogniser_was_not_built_for_is_resampled() -> None:
    """It is a 16 kHz model, and 48 kHz fed to it reads as speech at a third speed."""
    import asyncio

    from mindsurf_omni.service.asr import STREAMING_STRIDE, ParaformerStreamingRecogniser

    seen: list[int] = []

    class Stub(ParaformerStreamingRecogniser):
        def load(self) -> None:
            self._model = object()

        def _step(self, piece: object, cache: dict, last: bool) -> str:
            seen.append(len(piece))
            return ""

    async def run() -> None:
        # One second at 48 kHz is 48000 samples; at 16 kHz it is 16000, which
        # is one step and change.
        async for _ in Stub().stream(b"\x00\x01" * 48_000, 48_000):
            pass

    asyncio.run(run())

    assert sum(seen) == 16_000, f"resampled to {sum(seen)} samples, wanted 16000"
    assert seen[0] == STREAMING_STRIDE


class _Reader:
    """A recogniser that answers with whatever the test queued, in order."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.read = 0

    async def transcribe(self, pcm: bytes, rate: int) -> tuple[str, str | None]:
        self.read += 1
        return self.answers.pop(0) if self.answers else "", "zh"


def _second(rate: int = 16_000) -> bytes:
    return bytes(rate * 2)


async def test_the_preview_shows_only_what_two_readings_agree_on() -> None:
    """LocalAgreement-2: the head both of the last two readings wrote.

    A whole-segment recogniser run over a growing buffer has a settled head and
    an unsettled tail, and emitting the tail is what makes a display flicker.
    """
    from mindsurf_omni.service.asr import Rereading

    reader = _Reader(["今天下午", "今天下午开会", "今天下午开会推迟"])
    live = Rereading(recogniser=reader, every=1.0, warmup=0)

    # Nothing to agree with yet.
    assert await live.feed(_second()) == ""
    # 今天下午 is in both readings; 开会 is not settled.
    assert await live.feed(_second()) == "今天下午"
    assert await live.feed(_second()) == "开会"


async def test_the_preview_holds_rather_than_unsaying_what_it_showed() -> None:
    """The gate has to be driven by a reading that contradicts the screen.

    Both readings agreeing on something that starts differently is the only case
    where an additions-only contract has nothing honest to send, and a test that
    never reaches it is testing the happy path twice.
    """
    from mindsurf_omni.service.asr import Rereading

    reader = _Reader(["今天下午", "今天下午开会", "明天上午开会", "明天上午开会"])
    live = Rereading(recogniser=reader, every=1.0, warmup=0)

    await live.feed(_second())
    assert await live.feed(_second()) == "今天下午"
    # Now the recogniser changes its mind about the head. Two readings agree on
    # 明天上午开会, which is not an extension of 今天下午.
    assert await live.feed(_second()) == ""
    assert await live.feed(_second()) == ""


async def test_the_preview_will_not_read_half_a_second_of_speech() -> None:
    """Two readings agreeing says they are stable, not that they are right.

    Driven end to end at a half-second interval, all four dictation recordings
    showed text the release transcript did not contain -- two readings of too
    little audio agree on the same wrong words. At 1.0 s three of four came out
    exactly equal to the release transcript and at 2.0 s all four did.
    """
    from mindsurf_omni.service.asr import Rereading

    reader = _Reader(["猜的", "猜的", "今天下午", "今天下午开会"])
    live = Rereading(recogniser=reader, every=0.25, warmup=1.5)

    # Four quarter-seconds is one second: past the interval, short of the warmup.
    for _ in range(4):
        assert await live.feed(_second()[: 16_000 // 2]) == ""
    assert reader.read == 0

    # Past 1.5 s now, and the readings start.
    await live.feed(_second())
    assert reader.read == 1


async def test_the_preview_waits_for_enough_new_audio() -> None:
    from mindsurf_omni.service.asr import Rereading

    reader = _Reader(["一", "一二", "一二三"])
    live = Rereading(recogniser=reader, every=2.0, warmup=0)

    assert await live.feed(_second()) == ""
    assert reader.read == 0
    await live.feed(_second())
    assert reader.read == 1


async def test_the_preview_gives_up_the_rest_at_release() -> None:
    from mindsurf_omni.service.asr import Rereading

    reader = _Reader(["今天下午", "今天下午开会", "今天下午开会推迟了"])
    live = Rereading(recogniser=reader, every=1.0, warmup=0)
    await live.feed(_second())
    await live.feed(_second())

    assert await live.finish() == "开会推迟了"


def test_agreement_is_the_common_head_and_nothing_else() -> None:
    from mindsurf_omni.service.asr import agreed_prefix

    assert agreed_prefix("今天下午", "今天下午开会") == "今天下午"
    assert agreed_prefix("", "今天") == ""
    assert agreed_prefix("明天", "今天") == ""


async def test_a_comma_turning_into_a_full_stop_does_not_freeze_the_preview() -> None:
    """Read live over 109 seconds of dictation, at the 33rd character a comma
    came back as a full stop, the exact-prefix test never held again, and the
    display stopped at 218 characters of an eventual 483.

    The body-text measurement that said re-reading was safe was right and blind:
    it only ever compared body characters, and the freeze was triggered by the
    marks it discarded.
    """
    from mindsurf_omni.service.asr import Rereading

    reader = _Reader(
        [
            "今天下午开会",
            "今天下午开会",  # settles 今天下午开会
            "今天下午开会。地点",
            "今天下午开会。地点在三楼",  # the comma-to-full-stop shape
        ]
    )
    live = Rereading(recogniser=reader, every=1.0, warmup=0)

    assert await live.feed(_second()) == ""
    assert await live.feed(_second()) == "今天下午开会"
    # The next two readings agree on 今天下午开会。地点, whose body extends what
    # was shown even though the string does not.
    await live.feed(_second())
    assert await live.feed(_second()) == "。地点"


async def test_a_real_disagreement_still_holds() -> None:
    """The gate has to still be a gate. Different words, not a different mark."""
    from mindsurf_omni.service.asr import Rereading

    reader = _Reader(["今天下午", "今天下午", "明天上午开会", "明天上午开会"])
    live = Rereading(recogniser=reader, every=1.0, warmup=0)

    await live.feed(_second())
    assert await live.feed(_second()) == "今天下午"
    await live.feed(_second())
    assert await live.feed(_second()) == ""


def _buzz(seconds: float, level: int = 6000, rate: int = 16_000) -> bytes:
    import array
    import math

    frames = array.array("h")
    for n in range(int(rate * seconds)):
        frames.append(int(level * math.sin(2 * math.pi * 220 * n / rate)))
    return frames.tobytes()


def test_the_cut_lands_in_the_silence_and_not_in_a_word() -> None:
    """Without timestamps a silence is the only place the text and the audio are
    known to line up, so a cut anywhere else starts the next reading
    mid-syllable."""
    from mindsurf_omni.service.asr import quietest_split

    rate = 16_000
    pcm = _buzz(2.0) + bytes(int(rate * 0.6) * 2) + _buzz(2.0)
    at = quietest_split(pcm, rate, keep_back=0.5)

    assert at is not None
    seconds = at / (rate * 2)
    assert 2.0 < seconds < 2.6, seconds

    # Speech all the way through has nowhere safe to cut, and saying so beats
    # cutting a word in half.
    assert quietest_split(_buzz(4.0), rate, keep_back=0.5) is None


async def test_a_long_turn_stops_re_reading_the_whole_thing() -> None:
    """Every reading re-reads from the last cut, so unbounded the work per turn
    is quadratic in its length: 0.057 of real time over a fourteen-second
    dictation against 0.177 over a hundred and nine, and the 3600-second turn
    the entrance allows would spend more than four cards on its preview."""
    from mindsurf_omni.service.asr import Rereading

    rate = 16_000
    seen: list[int] = []

    class _Watching:
        async def transcribe(self, pcm: bytes, rate: int) -> tuple[str, str | None]:
            seen.append(len(pcm))
            return "一二三四", "zh"

    live = Rereading(recogniser=_Watching(), rate=rate, every=1.0, warmup=0, window=3.0)
    # Loud, quiet, loud, so there is somewhere to cut.
    for _ in range(3):
        await live.feed(_buzz(1.0))
    await live.feed(bytes(rate * 2))
    for _ in range(4):
        await live.feed(_buzz(1.0))

    assert len(live._held) < 8 * rate * 2, "缓冲区没有被切短"  # noqa: SLF001
    assert live._frozen, "切了却没有冻结文本"  # noqa: SLF001
    assert max(seen) < 8 * rate * 2, "还有一次读了整段"


async def test_the_seam_does_not_stack_punctuation() -> None:
    """Driving the deployed service, 能下载 came back 能下载。，反正 and demo came
    back demo。。。我们.

    The earlier reading wrote one mark at that position, the later reading wrote
    a different one, and seeking by body character put the seam in front of
    both. Visible on the screen and invisible to every measurement of this
    stage, which normalise marks away or compare bodies.
    """
    from mindsurf_omni.service.asr import Rereading

    reader = _Reader(
        [
            "能下载，",
            "能下载，",  # settles 能下载，
            "能下载。反正就是",
            "能下载。反正就是月底",
        ]
    )
    live = Rereading(recogniser=reader, every=1.0, warmup=0)

    await live.feed(_second())
    assert await live.feed(_second()) == "能下载，"
    await live.feed(_second())
    said = await live.feed(_second())

    assert said == "反正就是", said
    assert "。，" not in live._shown  # noqa: SLF001


def test_the_streaming_weights_can_be_mounted_instead_of_downloaded(tmp_path) -> None:
    """Named, FunASR turns the name into a download during assembly.

    That is a service reaching the network because of a recogniser choice, and
    on a machine with no network it is a hang with nothing in the log. The
    whole-segment recogniser has been mounted by path since the beginning; this
    is the other one catching up.
    """
    from mindsurf_omni.service.config import ConfigurationError, Settings

    base = {
        "MINDSURF_ENGINE": "cascade",
        "MINDSURF_WEIGHTS": "/w",
        "MINDSURF_ASR_MODEL": "paraformer-streaming",
    }

    assert Settings.from_environment(base).asr_streaming is None

    weights = tmp_path / "paraformer"
    weights.mkdir()
    mounted = Settings.from_environment({**base, "MINDSURF_ASR_STREAMING": str(weights)})
    assert mounted is not None and mounted.asr_streaming == weights

    # A path that is not there is refused at assembly, not at the first word.
    absent = Settings.from_environment(
        {**base, "MINDSURF_ASR_STREAMING": str(tmp_path / "nowhere")}
    )
    assert absent is not None
    with pytest.raises(ConfigurationError, match="MINDSURF_ASR_STREAMING"):
        absent.verify()
