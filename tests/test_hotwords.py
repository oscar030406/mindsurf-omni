"""热词：把说话人自己的专名放回去，同时不碰同音的普通词。

这一级的风险全在反方向。部署 和 不熟 同音，李工 和 理工 同音——表里放进任何
一个，另一个所在的每个普通句子都会被改成胡话。所以测试的重心不是"修好了几条"，
是"没碰的有多少"。
"""

from __future__ import annotations

import pytest

from mindsurf_omni.service.config import ConfigurationError
from mindsurf_omni.service.hotwords import build_table, correct


def test_a_term_that_sounds_like_a_common_word_is_refused_at_assembly() -> None:
    """部署 是 3333 的常用词，不熟 进表就意味着每一句 部署 都变成 不熟。

    在装配时拒绝，不是在请求时跳过：一张一半条目静默失效的表，操作员配了、
    服务报了就绪、词一次也没出现，而没有任何地方说得出是哪一半。
    """
    for word in ("不熟", "断连", "李工"):
        with pytest.raises(ConfigurationError, match="sounds exactly like"):
            build_table([word])


def test_a_term_with_no_common_homophone_is_admitted() -> None:
    table = build_table(["吕鑫", "通义千问"])

    assert table[("lv", "xin")] == "吕鑫"
    assert table[("tong", "yi", "qian", "wen")] == "通义千问"


def test_two_entries_that_sound_alike_are_refused() -> None:
    """同音的两条谁也说不清转写里那个音指的是哪一个。"""
    with pytest.raises(ConfigurationError, match="sound"):
        build_table(["吕鑫", "吕新"])


def test_a_latin_term_is_refused_rather_than_sitting_there_dead() -> None:
    with pytest.raises(ConfigurationError, match="two or more Chinese characters"):
        build_table(["Kubernetes"])


def test_the_recognisers_homophone_is_repaired() -> None:
    table = build_table(["吕鑫", "通义千问"])

    assert correct("请吕新过来一下。", table) == "请吕鑫过来一下。"
    assert correct("我们用通一千问做的。", table) == "我们用通义千问做的。"


def test_two_ordinary_words_that_together_rhyme_are_left_alone() -> None:
    """统一 和 前文 都是日常词，连起来的声音撞上 通义千问。

    第一版的闸看跨度本身和跨度的两端，这三样 统一前文 全过，
    先统一前文的术语 出来是 先通义千问的术语。干活的那个词整个在跨度**里面**，
    而里面没人看。
    """
    table = build_table(["通义千问"])

    assert correct("先统一前文的术语再往下写。", table) == "先统一前文的术语再往下写。"


def test_a_rare_dictionary_word_is_still_a_word() -> None:
    """履新 在 jieba 里是 8，低于入口闸的门槛，所以 吕鑫 收得进来。

    收进来之后就轮到句子这一级：分词器把 履新 读成一个词，那它就是说话人说的。
    """
    table = build_table(["吕鑫"])

    assert correct("新来的同事今天履新。", table) == "新来的同事今天履新。"
    assert correct("他刚刚履新上任。", table) == "他刚刚履新上任。"


def test_a_word_crossing_the_edge_of_the_span_protects_it() -> None:
    """不熟 在 不熟悉 里面，断联 在 断联系 里面——都是整词的一截。"""
    table = build_table(["吕鑫"])
    for sentence in ("这块我不熟悉。", "我们已经断联系很久了。", "他每天都锻炼身体。"):
        assert correct(sentence, table) == sentence


def test_an_empty_table_is_the_identity() -> None:
    assert correct("随便一句话。", build_table([])) == "随便一句话。"


def test_nothing_outside_the_transcript_or_the_table_can_appear() -> None:
    """这一级替换的是等字数的整词，所以输出里的字要么本来就在，要么在表里。"""
    table = build_table(["吕鑫", "通义千问"])
    source = "请吕新过来看看通一千问的输出。"
    out = correct(source, table)

    assert set(out) <= set(source) | set("吕鑫") | set("通义千问")
    assert len(out) == len(source)


def test_a_span_the_segmenter_reads_as_words_is_left_alone_even_when_it_is_wrong() -> None:
    """句首的 通一千问 被分词器切成 通/一千/问 三个词，所以这一级不碰它——
    即使那确实是识别器听错的。

    试过反过来：要求覆盖跨度的每一块都是**多字**词才算普通文本，好让句首那个也修回来。
    它把放行面从 2695 条真语料里 5.6% 的跨度开到 23.0%（净 +308%），
    因为中文单字词遍地都是——我觉得 切成 我/觉得，的时候 切成 的/时候，
    这些全部变成可改。一条召回换四倍的暴露面，不换。

    句中那个仍然会修，因为分词器在那里切出的 用通 跨过了跨度的左边界，
    对不齐。同一个错两个位置两种结果，难看，但难看的那一半是安全的那一半。
    """
    table = build_table(["通义千问"])

    assert correct("通一千问的接口改过了。", table) == "通一千问的接口改过了。"
    assert correct("我们用通一千问做的。", table) == "我们用通义千问做的。"


def test_the_recogniser_getting_the_neighbours_wrong_is_this_stages_ceiling() -> None:
    """识别器把 履新 听成 旅欣，这一级只看得见 旅欣——不是词，同音，就改了。

    留在这里是因为它是这套做法的天花板，不是可以修的缺陷：同音表读的是声音，
    而说话人说了一个和热词同音、又被识别器写坏的词时，没有信息能分开这两种。
    识别器听对的那一份（他刚刚履新上任）这一级不碰，那是闸能守住的部分。
    """
    table = build_table(["吕鑫"])

    assert correct("他刚刚履新上任。", table) == "他刚刚履新上任。"
    assert correct("新来的同事，今天旅欣。", table) == "新来的同事，今天吕鑫。"
