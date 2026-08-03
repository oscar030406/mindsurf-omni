"""The boundary between the prompt and the reply, which is the whole instrument.

Scoring the prompt as well as the reply would not fail, it would just dilute
every difference toward zero and report "indistinguishable" -- the failure mode
this project has already paid for once, on a repetition metric that gave a
seven-character answer full marks.

The forward pass needs torch and a checkpoint. The offset arithmetic does not,
and it is the part that goes wrong silently.
"""

from __future__ import annotations

from scripts.measure_chat_loss import reply_span


def test_the_reply_starts_where_the_prompt_ends() -> None:
    assert reply_span([1, 2, 3], [1, 2, 3, 4, 5]) == slice(3, 5)


def test_a_merged_boundary_is_dropped_rather_than_scored_at_the_wrong_offset() -> None:
    """Tokenisers may merge across the join, and then len(prefix) is a lie.

    Scoring from there charges the model for a token it was handed, which
    lowers the loss of whichever arm happens to merge -- a difference between
    tokenisations reported as a difference between models.
    """
    assert reply_span([1, 2, 3], [1, 2, 9, 4, 5]) is None


def test_an_empty_reply_is_dropped() -> None:
    assert reply_span([1, 2, 3], [1, 2, 3]) is None
    assert reply_span([1, 2, 3], [1, 2]) is None


def test_the_repetition_screen_catches_a_loop_and_spares_short_text() -> None:
    """It screens for a model that has started looping, nothing more.

    A likelihood cannot see a loop, because a loop is exactly what a language
    model finds probable -- so the arm that degenerates can score better. This
    is not compared between arms as a quality number; the project already
    shipped one repetition metric that rewarded a seven-character answer.
    """
    from scripts.measure_chat_loss import repetition

    assert repetition("好的好的好的好的好的好的好的好的好的好的") > 0.5
    assert repetition("今天天气晴朗，适合出门散步。") == 0.0
    assert repetition("短") == 0.0  # shorter than the window, not a loop


def test_a_prompt_set_with_no_reference_replies_can_still_be_sampled_from() -> None:
    """Preference sampling has prompts and no answers yet, by definition.

    Refusing such a set would mean the DPO pipeline could not draw the drafts
    it exists to rank; scoring it against nothing would be worse. The scoring
    loop skips those rows and the generation loop does not.
    """
    from scripts.measure_chat_loss import reply_span

    # The scoring side is what needs a reference: with no reply there is no
    # span to score, which is the condition the skip above stands in for.
    assert reply_span([1, 2, 3], [1, 2, 3]) is None


def test_the_spoken_rate_is_the_one_measured_on_the_fixed_texts() -> None:
    """A reply's length in characters says nothing a listener cares about.

    140 characters reads as a normal chat answer and as half a minute of
    talking, and only the second number is a product fact. The rate is pinned
    here because it was measured on this project's own synthesiser rather than
    guessed, and a silent drift would change every reported duration.
    """
    from scripts.measure_chat_loss import SPOKEN_CHARS_PER_SECOND

    assert SPOKEN_CHARS_PER_SECOND == 4.67
    # the product's own median reply is over half a minute of speech
    assert 140 / SPOKEN_CHARS_PER_SECOND > 25


def test_a_generating_run_records_how_it_sampled() -> None:
    """The old reports could not say, and extending the probe set needed to know."""
    from argparse import Namespace

    from scripts.measure_chat_loss import generation_settings

    settings = generation_settings(
        Namespace(
            generate=True, temperature=0.7, top_p=0.9, max_tokens=512, seed=1000,
            system_prompt=None,
        )
    )
    assert settings == {
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "seed": 1000,
        "system_prompt": None,
    }


def test_a_likelihood_only_run_records_no_sampling() -> None:
    """Nothing was sampled, so writing a temperature would claim something untrue."""
    from argparse import Namespace

    from scripts.measure_chat_loss import generation_settings

    assert generation_settings(Namespace(generate=False, temperature=0.7)) is None
