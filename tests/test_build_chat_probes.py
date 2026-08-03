"""Dedup and mix-matching, the two places a probe set goes wrong silently."""

from __future__ import annotations

from scripts.build_chat_probes import bigrams, parse_lines, targets, too_similar


def test_reordered_questions_count_as_duplicates() -> None:
    """Exact matching and token diffs both miss this; character bigrams do not."""
    assert too_similar("面怎么煮好吃", [bigrams("怎么煮面好吃")])


def test_different_questions_on_one_topic_survive() -> None:
    """Too aggressive a threshold empties the themes it is supposed to fill."""
    assert not too_similar("附近有什么餐厅", [bigrams("明天要下雨吗")])
    assert not too_similar("怎么挑西瓜", [bigrams("怎么挑螃蟹")])


def test_numbering_and_punctuation_come_off() -> None:
    reply = "1. 明天要下雨吗\n2、附近有什么好吃的\n- 「怎么挑西瓜」\n3) 今天限号吗？"
    assert parse_lines(reply) == ["明天要下雨吗", "附近有什么好吃的", "怎么挑西瓜", "今天限号吗"]


def test_lines_that_are_not_questions_are_dropped() -> None:
    """A model asked for 20 lines will pad with headings and apologies."""
    assert parse_lines("好的，以下是提问：\n\n明天要下雨吗\n") == ["明天要下雨吗"]


def test_the_target_mix_matches_the_set_being_extended() -> None:
    labelled = {f"p{i}": ("出行与交通" if i < 30 else "情绪与状态") for i in range(40)}
    wanted = targets(labelled, 400)
    assert sum(wanted.values()) == 400
    assert wanted["出行与交通"] == 300 and wanted["情绪与状态"] == 100


def test_unclassified_prompts_do_not_dilute_the_mix() -> None:
    """A theme the classifier could not name must not become a target theme."""
    labelled = {"a": "出行与交通", "b": "情绪与状态", "c": "未归类"}
    wanted = targets(labelled, 100)
    assert "未归类" not in wanted
    assert sum(wanted.values()) == 100
