"""The dictation path's second stage, wired into the service."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mindsurf_omni.service.cascade import CascadeEngine
from mindsurf_omni.service.config import ConfigurationError, Settings, describe_components
from mindsurf_omni.service.polish import Polisher, build_prompt, reachable, subsequence_pointer


def _cascade(polisher: object = None) -> CascadeEngine:
    async def transcribe(pcm: bytes, rate: int) -> tuple[str, str | None]:
        return "那个今天天气怎么样", "zh"

    async def speak(text: str, settings: object) -> bytes:
        return b""

    return CascadeEngine(
        transcriber=transcribe,
        generator=lambda *_: None,  # type: ignore[arg-type]
        synthesiser=speak,
        components=[],
        token_spec=None,  # type: ignore[arg-type]
        polisher=polisher,
    )


def test_no_polisher_answers_none_rather_than_the_transcript() -> None:
    """A caller must tell "this service does not polish" from "nothing to polish"."""
    engine = _cascade()

    assert asyncio.run(engine.polish("那个今天天气怎么样")) is None


def test_a_wired_polisher_is_used() -> None:
    class _Fake:
        async def polish(self, transcript: str) -> str:
            return transcript.replace("那个", "")

    assert asyncio.run(_cascade(_Fake()).polish("那个今天天气怎么样")) == "今天天气怎么样"


def test_a_missing_polish_checkpoint_is_refused_at_startup(tmp_path: Path) -> None:
    """Otherwise the first dictation comes back unpolished and reads as a model doing nothing."""
    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH": str(tmp_path / "typo.pth"),
        }
    )
    assert settings is not None

    with pytest.raises(ConfigurationError, match="typo.pth"):
        settings.verify()

    (tmp_path / "typo.pth").write_bytes(b"")
    settings.verify()


def test_the_polisher_is_named_and_hashed_in_the_component_list(tmp_path: Path) -> None:
    """It rewrites what the user sees, so which weights did it belongs in the report."""
    checkpoint = tmp_path / "polish.pth"
    checkpoint.write_bytes(b"weights")
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_POLISH": str(checkpoint),
        }
    )
    assert settings is not None

    named = [c for c in describe_components(settings) if c.name == "polisher"]

    assert len(named) == 1
    assert named[0].sha256 is not None and len(named[0].sha256) == 64


def test_the_decode_can_only_walk_forward_through_the_transcript() -> None:
    """Same rule the measurement round settled on, now in the service."""
    source = [10, 11, 12, 13, 14]

    assert subsequence_pointer(source, [10, 13]) == 4
    assert reachable(source, 1, lookahead=2) == [11, 12]
    assert reachable(source, 1, lookahead=0) == [11, 12, 13, 14]


def test_empty_input_is_returned_untouched(tmp_path: Path) -> None:
    """No forward pass for nothing, and no chance to invent a sentence from silence."""
    polisher = Polisher(
        checkpoint=tmp_path / "unused.pth",
        tokenizer_dir=tmp_path,
        minimind_root=tmp_path,
    )

    assert asyncio.run(polisher.polish("   ")) == "   "


def test_the_instruction_is_the_one_the_model_was_trained_on() -> None:
    """Train and serve share these words; a flag would let them drift apart."""
    from scripts.train_polish import INSTRUCTION as trained

    assert build_prompt("你好") == f"{trained}\n\n你好"


def test_projection_keeps_the_deletions_and_drops_the_inventions() -> None:
    """The model's text intersected with the transcript, in order."""
    from mindsurf_omni.service.polish import project_onto

    transcript = "那个，今天天气怎么样？"

    # Filler removed and nothing added: projection changes nothing.
    assert project_onto(transcript, "今天天气怎么样？") == "今天天气怎么样？"
    # The model answered instead: only what it echoed survives.
    assert project_onto(transcript, "今天天气很好，适合出门") == "今天天气"
    # Untouched output projects back to itself.
    assert project_onto(transcript, transcript) == transcript


def test_the_first_word_is_protected_unless_it_is_a_filler() -> None:
    """Measured end to end: 你想看什么类型的电影 came back as 想看什么类型的电影."""
    from mindsurf_omni.service.polish import reachable

    source = [40, 41, 50, 51, 52]  # 40,41 are a filler; 50.. is the sentence

    free = reachable(source, 0, lookahead=3, protect_head=False)
    guarded = reachable(source, 0, lookahead=3, protect_head=True)
    with_door = reachable(source, 0, lookahead=3, fillers=((40, 41),), protect_head=True)

    assert 50 in free  # unguarded, the opening word can be skipped
    assert guarded == [40]  # guarded, only the first token
    assert 50 in with_door  # unless the skip is the filler itself


def test_a_projected_target_is_reachable_by_a_deletion_only_decoder() -> None:
    """What --project-targets buys: every training target is one the decoder can produce.

    45% of the pairs carry a clean text that is not a subsequence of the
    transcript -- the recogniser misheard a character, and no amount of
    deleting recovers it. Reaching for it is what makes the decoder skip.
    """
    from mindsurf_omni.service.polish import project_onto

    source, target = "那个今天天汽怎么样", "今天天气怎么样"  # 气 heard as 汽

    projected = project_onto(source, target)

    remaining = iter(source)
    assert all(character in remaining for character in projected)
    assert "气" not in projected  # unreachable, and no longer asked for


def test_a_long_dictation_is_polished_one_sentence_at_a_time() -> None:
    """164 seconds of speech is 718 characters and the model was trained on
    single sentences. Fed the whole buffer it returned 18 of them."""
    from mindsurf_omni.service.polish import split_sentences

    pieces = split_sentences("你好。今天天气怎么样？出门吧")
    assert pieces == ["你好。", "今天天气怎么样？", "出门吧"]
    # Joining the pieces back has to reproduce the transcript exactly, or the
    # split itself would lose text.
    text = "第一句。第二句！第三句？还没写完"
    assert "".join(split_sentences(text)) == text


def test_a_comma_does_not_end_a_sentence() -> None:
    """Cutting at every comma would take away the context the model needs to
    tell a filler 就是 from a copula one."""
    from mindsurf_omni.service.polish import split_sentences

    assert split_sentences("就是，我想说的是这个") == ["就是，我想说的是这个"]


def test_consumed_reads_how_far_the_decode_got() -> None:
    """Exact, not approximate: the output is always a subsequence of the input."""
    from mindsurf_omni.service.polish import consumed

    assert consumed("今天天气怎么样", "今天天气怎么样") == 1.0
    assert consumed("今天天气怎么样", "今天") < 0.5
    assert consumed("", "") == 1.0


class _Stub(Polisher):
    """A polisher whose model returns whatever the test tells it to.

    Overrides the batch entry point, not the single one: that is what the
    service path calls, and a stub on the road nobody drives tests nothing.
    """

    def _polish_batch(self, pieces: list[str]) -> list[str]:  # type: ignore[override]
        return [self.answers.get(piece, piece) for piece in pieces]  # type: ignore[attr-defined]


async def test_a_piece_the_model_truncated_is_thrown_away() -> None:
    """A dictation tool that silently drops what the user said is worse than
    one that leaves the filler in. Measured end to end: 74 s of speech came
    back at 0.39 of the transcript, 164 s at 0.03, HTTP 200 both times."""
    polisher = _Stub(checkpoint=Path("x"), tokenizer_dir=Path("y"), minimind_root=Path("z"))
    polisher.answers = {"嗯，第一句很长很长很长很长很长。": "嗯，第一"}  # type: ignore[attr-defined]

    out = await polisher.polish("嗯，第一句很长很长很长很长很长。")

    assert out == "嗯，第一句很长很长很长很长很长。"


async def test_a_piece_the_model_polished_properly_is_kept() -> None:
    polisher = _Stub(checkpoint=Path("x"), tokenizer_dir=Path("y"), minimind_root=Path("z"))
    polisher.answers = {"嗯，今天天气怎么样？": "今天天气怎么样？"}  # type: ignore[attr-defined]

    assert await polisher.polish("嗯，今天天气怎么样？") == "今天天气怎么样？"


def test_the_door_opens_for_the_spellings_the_recogniser_writes() -> None:
    """11.3% of the filler that reaches a transcript is not spelled the way it
    was said: SenseVoice writes 呃 as 饿 恶 鄂 扼 and 嗯 as 恩 摁 温."""
    from mindsurf_omni.service.polish import RECOGNISED_FILLERS

    assert "摁" in RECOGNISED_FILLERS
    # Left out on purpose: 温 appears 289 times in the corpus as an ordinary
    # word (水温), 饿 12 times (是不是饿了), 啊 11. A door onto those is a door
    # onto content.
    for ordinary in ("温", "饿", "啊"):
        assert ordinary not in RECOGNISED_FILLERS


def test_a_transcript_the_model_was_trained_on_is_not_split() -> None:
    """Splitting every sentence cost 0.039 of filler clearance over 986
    held-out transcripts, none of which was long enough to need it."""
    from mindsurf_omni.service.polish import TRAINED_LENGTH, group_sentences

    text = "嗯，今天天气怎么样？我想出门散步。"
    assert len(text) < TRAINED_LENGTH
    assert group_sentences(text) == [text]


def test_a_long_transcript_is_grouped_not_shredded() -> None:
    """Pieces up to the trained length, not one sentence per call."""
    from mindsurf_omni.service.polish import group_sentences

    text = "这是一句测试用的句子。" * 40
    pieces = group_sentences(text, longest=100)

    assert len(pieces) > 1
    assert all(len(piece) <= 100 for piece in pieces)
    assert "".join(pieces) == text


def test_a_single_sentence_past_the_line_is_left_whole() -> None:
    """A piece starting halfway through a sentence is worse input than a long one."""
    from mindsurf_omni.service.polish import group_sentences

    text = "没有标点的一长串字" * 30
    assert group_sentences(text, longest=50) == [text]


def test_a_piece_with_nothing_to_remove_never_reaches_the_model() -> None:
    """Skipping them removes 46.7% of the calls and the four numbers do not get
    worse -- the model was editing sentences with nothing to remove, and by
    construction those edits were over-deletion."""
    from mindsurf_omni.service.polish import worth_polishing

    assert not worth_polishing("会议纪要发群里了，麻烦大家今天下班前确认一下")
    assert worth_polishing("嗯，会议纪要发群里了")
    assert worth_polishing("我我想问一下报销流程")  # repetition, no filler word
    # 饿 is in the skip list but not in the decoder's door: calling the model on
    # 是不是饿了 costs one decode, opening the door on it costs the word.
    assert worth_polishing("鹦鹉一直叫是不是饿了")


async def test_the_skipped_pieces_come_back_untouched() -> None:
    polisher = _Stub(checkpoint=Path("x"), tokenizer_dir=Path("y"), minimind_root=Path("z"))
    polisher.answers = {}  # type: ignore[attr-defined]
    clean = "会议纪要发群里了。麻烦大家确认一下。"

    assert await polisher.polish(clean) == clean


def test_a_stranded_particle_is_dropped_with_the_words_it_belonged_to() -> None:
    """Deleting 我觉得 out of "我觉得吧这个真的挺好的" left "吧这个真的挺好的",
    which is not a sentence. Measured on the running service; it costs nothing
    on any of the four numbers, because 吧 is a character the source had."""
    from mindsurf_omni.service.polish import tidy

    assert tidy("吧这个真的挺好的") == "这个真的挺好的"
    assert tidy("说完了。吧我们走") == "说完了。我们走"


def test_a_particle_doing_its_job_is_left_alone() -> None:
    from mindsurf_omni.service.polish import tidy

    assert tidy("我们走吧") == "我们走吧"
    assert tidy("这个真的挺好的吧？") == "这个真的挺好的吧？"
    assert tidy("你说呢") == "你说呢"


def test_the_particles_that_can_open_a_sentence_are_not_touched() -> None:
    """ "啊，太好了" is ordinary, and deleting that 啊 would be deleting content."""
    from mindsurf_omni.service.polish import tidy

    assert tidy("啊，太好了！") == "啊，太好了！"
    assert tidy("呀，你来了。") == "呀，你来了。"


def test_stranded_punctuation_is_still_dropped() -> None:
    """The shape this started as, kept working now that the service runs it."""
    from mindsurf_omni.service.polish import tidy

    assert tidy("彩塑，？特别震撼") == "彩塑，特别震撼"
    assert tidy("？特别震撼") == "特别震撼"


def test_tidy_only_deletes_so_the_copy_constraint_holds() -> None:
    from mindsurf_omni.service.polish import consumed, tidy

    for text in ["吧这个真的挺好的", "彩塑，？特别震撼", "我们走吧", "啊，太好了！"]:
        assert consumed(text, tidy(text)) <= 1.0
        assert all(char in text for char in tidy(text))


def test_a_bridging_filler_takes_its_question_mark_with_it() -> None:
    """The mirror of the stranded-punctuation case: the filler was not deleted
    at all, and sits mid-sentence carrying its own mark. A listener meets that
    as a wrong sentence boundary -- 0.66 s of pause with the pitch held flat --
    and the four criteria price it at zero, because CER strips punctuation."""
    from mindsurf_omni.service.polish import tidy

    assert tidy("记了下来。对吧？训练好之后") == "记了下来。训练好之后"
    assert tidy("热水擦，怎么说呢？重油污就上小苏打") == "热水擦，重油污就上小苏打"
    assert tidy("开窗通风对吧？或者用除湿机") == "开窗通风或者用除湿机"


def test_a_mark_that_really_does_end_the_sentence_is_left_alone() -> None:
    """Nothing after it means there is no wrong boundary to remove -- and taking
    the filler there would strand the mark, which is the defect the other half
    of this function exists for."""
    from mindsurf_omni.service.polish import tidy

    assert tidy("这个方案怎么说呢？") == "这个方案怎么说呢？"
    assert tidy("开窗通风，对吧？") == "开窗通风，对吧？"


def test_only_the_bridging_three_get_this_rule() -> None:
    """Deleting every leftover vocabulary filler reads a better clearance and a
    worse retention, and does not remove a single wrong boundary. A question
    mark after ordinary words is the recogniser's, and nothing here can tell
    which sentence it belonged to."""
    from mindsurf_omni.service.polish import tidy

    assert tidy("我们要不要？再考虑一下别的方案") == "我们要不要？再考虑一下别的方案"
    assert tidy("那个方案还行？成本有点高") == "那个方案还行？成本有点高"


def test_dropping_a_bridging_filler_still_only_deletes() -> None:
    from mindsurf_omni.service.polish import consumed, tidy

    for text in ["记了下来。对吧？训练好之后", "热水擦，怎么说呢？重油污"]:
        assert consumed(text, tidy(text)) <= 1.0
        assert all(char in text for char in tidy(text))
