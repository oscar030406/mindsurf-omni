"""Fixed-text synthesis, which is how two synthesisers become comparable."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from scripts.speak_texts import build_synthesiser, load_texts, speak_all

from mindsurf_omni.contract import OUTPUT_SAMPLE_RATE


class _Fake:
    """Says a fixed length, or refuses on a named id's text."""

    def __init__(self, refuse: str = "") -> None:
        self.said: list[str] = []
        self.refuse = refuse

    async def synthesise(self, utterance) -> bytes:  # type: ignore[no-untyped-def]
        self.said.append(utterance.text)
        if self.refuse and self.refuse in utterance.text:
            raise RuntimeError("the endpoint returned no audio")
        return b"\x00\x01" * OUTPUT_SAMPLE_RATE  # one second of pcm16


def _texts(tmp_path: Path, count: int = 3) -> Path:
    path = tmp_path / "texts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"zh{i:03d}", "prompt": f"问题{i}", "text": f"第{i}句回答"})
            for i in range(count)
        ),
        encoding="utf-8",
    )
    return path


def test_the_fixed_text_is_the_reference(tmp_path: Path) -> None:
    """CER on this run is the synthesiser and the judge; the model sat it out."""
    fake = _Fake()

    samples = asyncio.run(
        speak_all(fake, load_texts(_texts(tmp_path)), tmp_path / "wav", "neutral")
    )

    assert [sample["reference_text"] for sample in samples] == [
        "第0句回答",
        "第1句回答",
        "第2句回答",
    ]
    assert fake.said == ["第0句回答", "第1句回答", "第2句回答"]


def test_a_refused_utterance_is_recorded_rather_than_ending_the_run(tmp_path: Path) -> None:
    """The expensive half is the whole set; losing it to one bad sample is the failure mode."""
    samples = asyncio.run(
        speak_all(_Fake(refuse="第1句"), load_texts(_texts(tmp_path)), tmp_path / "wav", "neutral")
    )

    assert len(samples) == 3
    assert samples[1]["audio_path"] is None
    assert "no audio" in samples[1]["error"]
    assert samples[2]["audio_path"] is not None


def test_the_audio_is_written_as_a_readable_wav(tmp_path: Path) -> None:
    """The judge opens files, not byte strings, and a headerless blob is refused by ffmpeg."""
    import soundfile

    samples = asyncio.run(
        speak_all(_Fake(), load_texts(_texts(tmp_path, 1)), tmp_path / "wav", "neutral")
    )

    audio, rate = soundfile.read(samples[0]["audio_path"], dtype="int16")

    assert rate == OUTPUT_SAMPLE_RATE
    assert len(audio) == OUTPUT_SAMPLE_RATE
    assert samples[0]["audio_seconds"] == pytest.approx(1.0)


def test_an_unknown_synthesiser_is_refused_before_anything_is_generated(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="names no synthesiser"):
        build_synthesiser("cosyvoice", "cpu")


def test_an_empty_text_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="no texts"):
        load_texts(path)


def test_half_a_clone_prompt_is_refused_before_anything_is_generated(tmp_path: Path) -> None:
    """Same rule as the service: a clip without its text clones nothing."""
    with pytest.raises(SystemExit, match="go together"):
        build_synthesiser("voxcpm", "cpu", prompt_wav=tmp_path / "reference.wav")
    with pytest.raises(SystemExit, match="go together"):
        build_synthesiser("voxcpm", "cpu", prompt_text="参考音频说的话")


def test_the_hosted_synthesiser_refuses_a_clone_prompt(tmp_path: Path) -> None:
    """It has one voice. Accepting the flag would silently produce the wrong arm."""
    with pytest.raises(SystemExit, match="only reaches voxcpm"):
        build_synthesiser("edge", "cpu", tmp_path / "reference.wav", "参考音频说的话")
