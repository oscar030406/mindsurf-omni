"""What counts as a sentence someone would dictate."""

from __future__ import annotations

from scripts.filter_dictation_register import is_dictation


def test_an_instruction_written_at_a_model_is_not_dictation() -> None:
    """The probe file is prompts, and its one empty output in the run was one."""
    assert not is_dictation("生成一个描述夏季沙滩度假的段落。", "asr_probe_scored")
    # The same sentence from a source that is not the probe file still has to
    # pass the other two rules on its own merits.
    assert is_dictation("生成一个描述夏季沙滩度假的段落。", "chat_refs_short_a_v1")


def test_a_numbered_list_is_typed_not_spoken() -> None:
    """118 rows of the 986 carry one, and a deleted 一、 is not the failure the lines are about."""
    assert not is_dictation("一、按照衣柜尺寸分类整理，2、按照重量分类。", "chat_refs_short_a_v1")
    assert not is_dictation("步骤如下。1. 先烧水。", "chat_refs_short_a_v1")


def test_a_year_inside_a_sentence_is_not_a_list_marker() -> None:
    """The rule keys on a following list punctuation mark, not on the digits."""
    assert is_dictation("中秋在农历8月15，正好处在秋季正中。", "talker_texts_zh_v1")
    assert is_dictation("每20分钟看远处20秒。", "talker_texts_zh_v1")


def test_a_line_break_means_a_prompt_plus_its_payload() -> None:
    """The second half of those rows is data, not speech."""
    assert not is_dictation("根据以下文本回答\n这首歌名为《起风了》。", "chat_refs_external_v1")
    assert not is_dictation("主人公：一只小猫\\n冲突：想吃鱼", "chat_refs_external_v1")


def test_an_ordinary_spoken_sentence_survives() -> None:
    assert is_dictation("白衬衫领子发黄怎么办？", "talker_texts_zh_v1")
