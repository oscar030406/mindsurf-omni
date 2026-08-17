"""Synthesis input cleaning, and the screen for instructions read aloud."""

from __future__ import annotations

import pytest

from mindsurf_omni.service.tts import (
    EDGE_PROSODY,
    EMOTION_INSTRUCTIONS,
    EdgeSynthesiser,
    Utterance,
    clean_for_speech,
    instruction_leaked,
    screen_batch,
)


def test_markdown_markers_are_removed() -> None:
    """They are shown, not spoken, and a synthesiser reads them as something."""
    assert clean_for_speech("**灵山大佛**通高 `88` 米") == "灵山大佛通高 88 米"


def test_em_dashes_become_pauses_rather_than_being_read() -> None:
    assert clean_for_speech("很好——真的很好") == "很好，真的很好"


def test_a_short_aside_is_kept_as_an_aside() -> None:
    """A speaker would say it, so it stays -- between pauses."""
    assert clean_for_speech("门票（旺季）二百一十元") == "门票，旺季，二百一十元"


def test_a_long_parenthetical_is_dropped() -> None:
    """A speaker would not read a paragraph in brackets, and it buries the sentence."""
    text = "灵山大佛（这里有一段很长很长很长很长很长很长的补充说明文字）值得一看"

    assert clean_for_speech(text) == "灵山大佛值得一看"


def test_line_breaks_become_sentence_ends() -> None:
    assert clean_for_speech("第一句\n第二句") == "第一句。第二句"


def test_repeated_punctuation_is_collapsed() -> None:
    """Doubled marks come from cleaning itself and read as a stumble."""
    assert clean_for_speech("好的。。，，然后呢") == "好的。然后呢"


def test_cleaning_leaves_ordinary_text_alone() -> None:
    assert clean_for_speech("今天天气真好") == "今天天气真好"


def test_cleaning_an_empty_string_is_empty_not_punctuation() -> None:
    """A stray comma would be synthesised as a pause with no content."""
    assert clean_for_speech("") == ""
    assert clean_for_speech("\n\n") == ""


def test_emotion_lives_beside_the_text_not_inside_it() -> None:
    utterance = Utterance(text="今天天气真好", emotion="happy")

    assert utterance.text == "今天天气真好"
    assert "开心" in utterance.instruction()


def test_an_unknown_emotion_falls_back_rather_than_failing() -> None:
    """A typo in a client should not take the turn down."""
    assert Utterance(text="x", emotion="ecstatic").instruction() == EMOTION_INSTRUCTIONS["neutral"]


def test_an_instruction_read_aloud_is_detected() -> None:
    """The real failure: the reply is there, with the instruction in front."""
    transcript = EMOTION_INSTRUCTIONS["happy"] + "今天天气真好"

    assert instruction_leaked("今天天气真好", transcript)


def test_a_clean_transcript_is_not_flagged() -> None:
    assert not instruction_leaked("今天天气真好", "今天天气真好")
    assert not instruction_leaked("今天天气真好", "")


def test_a_reply_that_merely_discusses_tone_is_not_flagged() -> None:
    """Containment would fire here; a prefix match does not."""
    assert not instruction_leaked("x", "这句话应该用开心热情的语气说才自然")


def test_screening_covers_the_batch_not_a_sample() -> None:
    """The leak is intermittent, so a spot check finds it in the batches it spared."""
    utterances = [Utterance(text=f"第{i}句") for i in range(4)]
    transcripts = [
        "第0句",
        EMOTION_INSTRUCTIONS["neutral"] + "第1句",
        "第2句",
        "",  # produced no speech at all
    ]

    suspect = screen_batch(utterances, transcripts)

    assert [utterance.text for utterance, _ in suspect] == ["第1句", "第3句"]


def test_misaligned_inputs_are_refused_rather_than_silently_zipped() -> None:
    """A shifted pair would blame the wrong sample and hide the real one."""
    with pytest.raises(ValueError, match="misaligned|against"):
        screen_batch([Utterance(text="a")], ["a", "b"])


# --------------------------------------------------------------------------
# The hosted synthesiser. The endpoint is faked; the decode is real, because
# an encoded blob that will not decode is one of the ways this actually fails.
# --------------------------------------------------------------------------


class _FakeEndpoint:
    """Records what was asked for and hands back real, decodable audio."""

    requests: list[tuple[str, str, dict[str, str]]] = []
    seconds: float = 0.5
    fail_first_n: int = 0

    def __init__(self, text: str, voice: str, **options: str) -> None:
        type(self).requests.append((text, voice, options))
        self._text = text

    async def stream(self):  # type: ignore[no-untyped-def]
        import io
        import math

        import numpy
        import soundfile

        cls = type(self)
        if cls.fail_first_n:
            cls.fail_first_n -= 1
            raise RuntimeError("NoAudioReceived")
        if not cls.seconds:
            return
        rate = 24_000
        count = int(rate * type(self).seconds)
        tone = numpy.array(
            [int(0.3 * 32767 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(count)],
            dtype=numpy.int16,
        )
        container = io.BytesIO()
        soundfile.write(container, tone, rate, format="WAV", subtype="PCM_16")
        yield {"type": "audio", "data": container.getvalue()}


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> type[_FakeEndpoint]:
    import sys
    import types

    _FakeEndpoint.requests = []
    _FakeEndpoint.seconds = 0.5
    _FakeEndpoint.fail_first_n = 0
    module = types.ModuleType("edge_tts")
    module.Communicate = _FakeEndpoint  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    return _FakeEndpoint


async def test_the_delivery_instruction_is_never_sent_as_text(
    endpoint: type[_FakeEndpoint],
) -> None:
    """This endpoint has no instruct mode, so a prefix would be read aloud every time.

    Not the intermittent leak screen_batch watches for -- a guaranteed one.
    """
    await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好", emotion="happy"))

    sent, _, _ = endpoint.requests[0]
    assert sent == "今天天气真好"
    assert EMOTION_INSTRUCTIONS["happy"] not in sent


async def test_emotion_is_carried_as_prosody_instead(endpoint: type[_FakeEndpoint]) -> None:
    await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好", emotion="happy"))

    _, _, options = endpoint.requests[0]
    assert options == EDGE_PROSODY["happy"]


async def test_an_unknown_emotion_falls_back_to_neutral_prosody(
    endpoint: type[_FakeEndpoint],
) -> None:
    await EdgeSynthesiser().synthesise(Utterance(text="x", emotion="ecstatic"))

    assert endpoint.requests[0][2] == EDGE_PROSODY["neutral"]


async def test_the_text_is_cleaned_before_it_is_spoken(endpoint: type[_FakeEndpoint]) -> None:
    """Otherwise every caller cleans it, and one forgets."""
    await EdgeSynthesiser().synthesise(Utterance(text="**灵山大佛**通高 `88` 米"))

    assert endpoint.requests[0][0] == "灵山大佛通高 88 米"


async def test_nothing_worth_saying_costs_no_request(endpoint: type[_FakeEndpoint]) -> None:
    """A request for silence spends a round trip and plays a click."""
    assert await EdgeSynthesiser().synthesise(Utterance(text="\n\n")) == b""
    assert endpoint.requests == []


async def test_returned_audio_is_pcm16_at_the_contract_rate(
    endpoint: type[_FakeEndpoint],
) -> None:
    """Anything else reaches the client as a chipmunk, reported as "sounds strange"."""
    from mindsurf_omni.contract import OUTPUT_SAMPLE_RATE

    pcm = await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好"))

    assert len(pcm) == int(OUTPUT_SAMPLE_RATE * 0.5) * 2


async def test_an_empty_response_raises_rather_than_returning_silence(
    endpoint: type[_FakeEndpoint],
) -> None:
    """Silence scores as the model having said nothing, blaming the model for a network fault."""
    endpoint.seconds = 0.0

    with pytest.raises(RuntimeError, match="no audio"):
        await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好"))


def test_a_synthesiser_is_never_eligible_to_judge_its_own_output() -> None:
    assert EdgeSynthesiser().eligible_as_judge is False


async def test_a_single_transient_failure_is_retried(endpoint: type[_FakeEndpoint]) -> None:
    """Measured at 2 turns in 160: the endpoint sometimes returns nothing.

    It sits behind a network, so one retry is the difference between a 1.25%
    turn failure rate and a negligible one.
    """
    endpoint.fail_first_n = 1

    pcm = await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好"))

    assert pcm  # the second attempt produced audio
    assert len(endpoint.requests) == 2


async def test_a_persistent_failure_is_raised_not_papered_over(
    endpoint: type[_FakeEndpoint],
) -> None:
    """Attempts are not a fix for something that is actually broken."""
    endpoint.fail_first_n = 5

    with pytest.raises(RuntimeError, match="twice"):
        await EdgeSynthesiser().synthesise(Utterance(text="今天天气真好"))

    assert len(endpoint.requests) == 2  # tried twice, then stopped


# --------------------------------------------------------------------------
# The local synthesiser. The weights are faked -- what is being checked is the
# contract around them: rate, emptiness, and that a card is touched once.
# --------------------------------------------------------------------------


def _fake_inner() -> object:
    """A real ``nn.Module`` with a real child called ``stop_head``.

    Not a plain object: assigning to a registered child is exactly what broke the
    first version of ``delay_stop``, and a fake without that rule let the unit
    tests pass while the first real synthesis died on "cannot assign ... as child
    module". A fake has to carry the constraint the real class carries, or it is
    testing a different object.
    """
    import torch

    class _FakeStopHead(torch.nn.Module):
        """Says stop every time, so a held budget shows up in the answers."""

        def forward(self, hidden: object) -> object:
            return torch.tensor([[0.0, 1.0]])

    class _FakeInner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stop_head = _FakeStopHead()

    return _FakeInner()


class _FakeVoxCPM:
    """Hands back a real float32 waveform at VoxCPM's own 16 kHz."""

    loads: int = 0
    calls: list[dict[str, object]] = []
    seconds: float = 0.5

    def __init__(self) -> None:
        type(self).loads += 1
        # The real class exposes its decoder here and delay_stop wraps the stop
        # head on it.
        self.tts_model = _fake_inner()

    @classmethod
    def from_pretrained(cls, model_id: str, **options: object) -> _FakeVoxCPM:
        return cls()

    def generate(self, **options: object):  # type: ignore[no-untyped-def]
        import numpy

        type(self).calls.append(options)
        count = int(16_000 * type(self).seconds)
        return numpy.zeros(count, dtype="float32")


@pytest.fixture
def voxcpm(monkeypatch: pytest.MonkeyPatch) -> type[_FakeVoxCPM]:
    import sys
    import types

    _FakeVoxCPM.loads = 0
    _FakeVoxCPM.calls = []
    _FakeVoxCPM.seconds = 0.5
    module = types.ModuleType("voxcpm")
    module.VoxCPM = _FakeVoxCPM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voxcpm", module)
    return _FakeVoxCPM


async def test_local_audio_arrives_as_pcm16_at_the_contract_rate(
    voxcpm: type[_FakeVoxCPM],
) -> None:
    """The model speaks at 16 kHz and the contract is 24; unresampled it is a chipmunk."""
    from mindsurf_omni.contract import OUTPUT_SAMPLE_RATE
    from mindsurf_omni.service.tts import VoxCPMSynthesiser

    pcm = await VoxCPMSynthesiser().synthesise(Utterance(text="今天天气真好"))

    assert len(pcm) == int(OUTPUT_SAMPLE_RATE * 0.5) * 2


async def test_the_weights_are_loaded_once_not_per_utterance(
    voxcpm: type[_FakeVoxCPM],
) -> None:
    """Half a billion parameters per turn would be slower than the endpoint it replaces."""
    from mindsurf_omni.service.tts import VoxCPMSynthesiser

    synthesiser = VoxCPMSynthesiser()
    await synthesiser.synthesise(Utterance(text="第一句"))
    await synthesiser.synthesise(Utterance(text="第二句"))

    assert voxcpm.loads == 1


async def test_the_delivery_instruction_is_not_prepended_locally_either(
    voxcpm: type[_FakeVoxCPM],
) -> None:
    """This model has no instruct mode, so a prefix would be read aloud every time."""
    from mindsurf_omni.service.tts import VoxCPMSynthesiser

    await VoxCPMSynthesiser().synthesise(Utterance(text="今天天气真好", emotion="happy"))

    assert voxcpm.calls[0]["text"] == "今天天气真好"


async def test_local_synthesis_of_nothing_costs_no_forward_pass(
    voxcpm: type[_FakeVoxCPM],
) -> None:
    from mindsurf_omni.service.tts import VoxCPMSynthesiser

    assert await VoxCPMSynthesiser().synthesise(Utterance(text="\n\n")) == b""
    assert voxcpm.calls == []


async def test_local_silence_raises_rather_than_being_scored_as_speech(
    voxcpm: type[_FakeVoxCPM],
) -> None:
    """An empty waveform would be charged to the model as having said nothing."""
    from mindsurf_omni.service.tts import VoxCPMSynthesiser

    voxcpm.seconds = 0.0

    with pytest.raises(RuntimeError, match="no audio"):
        await VoxCPMSynthesiser().synthesise(Utterance(text="今天天气真好"))


async def test_the_text_is_cleaned_before_the_local_model_sees_it(
    voxcpm: type[_FakeVoxCPM],
) -> None:
    from mindsurf_omni.service.tts import VoxCPMSynthesiser

    await VoxCPMSynthesiser().synthesise(Utterance(text="**灵山大佛**通高 `88` 米"))

    assert voxcpm.calls[0]["text"] == "灵山大佛通高 88 米"


def test_the_local_synthesiser_is_not_eligible_to_judge_either() -> None:
    from mindsurf_omni.service.tts import VoxCPMSynthesiser

    assert VoxCPMSynthesiser().eligible_as_judge is False


def test_only_the_tail_of_the_instruction_leaking_is_still_caught() -> None:
    """What the sister project actually observed, reproducibly.

    Their template put the instruction before a separator and the text after,
    and the synthesiser sometimes began reading just inside the instruction --
    so only "…的语气说吗？" survived into the audio. Matching the instruction's
    opening characters would have called that clean.
    """
    tail = EMOTION_INSTRUCTIONS["care"][-6:]

    assert instruction_leaked("今天天气真好", tail + "今天天气真好")


def test_a_reply_that_merely_discusses_tone_is_still_not_flagged() -> None:
    """Widening to any suffix must not widen to containment.

    Whatever fragment leaks is read before the reply; a reply about tone has
    those words in the middle. Flagging it would teach everyone to ignore this.
    """
    assert not instruction_leaked("x", "这句话应该用开心热情的语气说才自然")
    assert not instruction_leaked("x", "导游会用温柔关切的语气说这段话")


async def test_the_reference_clip_reaches_the_model(voxcpm: type[_FakeVoxCPM]) -> None:
    """Without it VoxCPM draws a speaker per call: one reply, several voices.

    The fields were always here; nothing set them, so every utterance the
    service produced was a fresh stranger.
    """
    from mindsurf_omni.service.tts import VoxCPMSynthesiser

    await VoxCPMSynthesiser(prompt_wav="reference.wav", prompt_text="参考音频说的话").synthesise(
        Utterance(text="今天天气真好")
    )

    assert voxcpm.calls[0]["prompt_wav_path"] == "reference.wav"
    assert voxcpm.calls[0]["prompt_text"] == "参考音频说的话"


def test_text_with_nothing_speakable_left_comes_back_empty() -> None:
    """ "。。。？！" survived the strip as "？！", which the hosted endpoint answers
    with no audio at all -- and this module turns that into a RuntimeError the
    speech endpoint had no choice but to serve as a 500."""
    assert clean_for_speech("。。。？！") == ""
    assert clean_for_speech("……") == ""
    assert clean_for_speech("？") == ""
    assert clean_for_speech("!!!") == ""


def test_punctuation_around_real_words_is_kept() -> None:
    """The check is "is there anything to say", not "strip punctuation"."""
    assert clean_for_speech("真的吗？太好了！") == "真的吗？太好了！"
    assert clean_for_speech("行，那就这样。") == "行，那就这样"


def test_the_stop_is_held_for_the_configured_number_of_patches() -> None:
    """VoxCPM ends a sentence a syllable short: it reads the stop flag off the
    state that produced the patch it has just appended, so the decision runs one
    patch behind the audio. Held twice, the final syllable goes from 100 ms to
    200 ms against edge's 150 -- measured with the trim sweep, and picked by a
    listener over both the current behaviour and holding four."""
    import torch

    from mindsurf_omni.service.tts import delay_stop

    model = _FakeVoxCPM()
    state = delay_stop(model, 2)
    head = model.tts_model.stop_head

    state.left = 2
    said = [int(head(torch.zeros(1)).argmax(dim=-1)[0]) for _ in range(4)]

    # The first two "stop"s become "keep going"; the third gets through.
    assert said == [0, 0, 1, 1]


def test_each_utterance_gets_its_own_budget() -> None:
    """Re-armed per utterance, or a long batch spends the whole allowance on its
    first sentence and every later one ends short again."""
    import torch

    from mindsurf_omni.service.tts import VoxCPMSynthesiser, delay_stop

    model = _FakeVoxCPM()
    synthesiser = VoxCPMSynthesiser(stop_delay=2)
    synthesiser._model = model
    synthesiser._stop_state = delay_stop(model, 2)
    head = model.tts_model.stop_head

    synthesiser._arm_stop_budget()
    first = [int(head(torch.zeros(1)).argmax(dim=-1)[0]) for _ in range(3)]
    synthesiser._arm_stop_budget()
    second = [int(head(torch.zeros(1)).argmax(dim=-1)[0]) for _ in range(3)]

    assert first == [0, 0, 1]
    assert second == [0, 0, 1]


def test_a_build_without_the_stop_head_refuses_rather_than_going_quiet() -> None:
    """A silently skipped patch here brings back a defect no number on this
    project can see. Upstream renaming it has to be an error, not a shrug."""
    from mindsurf_omni.service.tts import SynthesiserUnavailable, delay_stop

    class _Bare:
        pass

    with pytest.raises(SynthesiserUnavailable) as refused:
        delay_stop(_Bare(), 2)

    assert "stop_head" in str(refused.value)
