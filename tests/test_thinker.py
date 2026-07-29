"""Loading the Thinker, and the two ways it silently loads the wrong model.

torch is not in the test environment and the container does not carry it, so
what is exercised here is the weight selection and the refusals that do not
need it.

The load path itself was run against the real 359 MB checkpoint: 91 tensors,
all kept, the model built to exactly 89,864,448 parameters. The missing-tensor
refusal was checked the only way it can be -- by handing it a checkpoint with
every second tensor removed, which it declined, naming 45 of them. Both need
torch and a real file, so neither is repeated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mindsurf_omni.service.config import ConfigurationError, Settings
from mindsurf_omni.service.thinker import (
    THINKER_PARAMETERS,
    THINKER_SHAPE,
    ThinkerGenerator,
    thinker_weights,
)


def test_the_text_base_is_taken_whole() -> None:
    """Every tensor in a text-only checkpoint belongs to the text model."""
    state = {
        "model.embed_tokens.weight": 1,
        "model.layers.0.self_attn.q_proj.weight": 2,
        "lm_head.weight": 3,
    }

    assert thinker_weights(state) == state


def test_the_talker_and_the_projections_are_left_behind() -> None:
    """They belong to the native path, and strict=False would drop them silently.

    Silently is the problem: the same forgiveness hides a wrong config, so the
    caller can only check the parameter count if the input was filtered first.

    The prefixes are the ones sft_omni_768.pth actually carries, read off the
    file: model, lm_head, talker, audio_proj, vision_proj.
    """
    kept = thinker_weights(
        {
            "model.embed_tokens.weight": 1,
            "lm_head.weight": 2,
            "talker.text_scale": 3,
            "talker.layers.0.self_attn.q_proj.weight": 4,
            "audio_proj.0.weight": 5,
            "vision_proj.0.weight": 6,
        }
    )

    assert sorted(kept) == ["lm_head.weight", "model.embed_tokens.weight"]


def test_one_omni_checkpoint_yields_the_same_tensors_as_the_text_base() -> None:
    """What lets the same loader read either file, and the reason it is safe to.

    Counted on the real checkpoints: the base is 91 tensors, sft_omni_768.pth
    is 195 of which 90 model.* plus one lm_head.* are the same 91. If a future
    checkpoint changes that split, the parameter-count check in load() is what
    stops it being loaded as if nothing had changed.
    """
    omni = {f"model.layers.{index}.weight": index for index in range(90)}
    omni["lm_head.weight"] = 90
    omni.update({f"talker.layers.{index}.weight": index for index in range(92)})
    omni.update({f"audio_proj.{index}.weight": index for index in range(6)})
    omni.update({f"vision_proj.{index}.weight": index for index in range(6)})

    assert len(omni) == 195
    assert len(thinker_weights(omni)) == 91


def test_the_shape_is_stated_rather_than_left_to_upstream_defaults() -> None:
    """MiniMind defaults to GQA with four key-value heads and a narrower FFN.

    Those defaults build a model that loads our weights, runs, and is not the
    one that was trained -- the failure that reported 113.13M instead of
    152.06M and cost a night.
    """
    assert THINKER_SHAPE["num_key_value_heads"] == 8  # MHA, not MiniMind's 4
    assert THINKER_SHAPE["intermediate_size"] == 3584  # not the derived 1536
    assert THINKER_PARAMETERS == 89_864_448


def test_a_missing_checkout_says_which_path_and_why(tmp_path: Path) -> None:
    generator = ThinkerGenerator(
        checkpoint=tmp_path / "absent.pth",
        tokenizer_dir=tmp_path,
        minimind_root=tmp_path / "nowhere",
    )

    with pytest.raises(ConfigurationError, match="model definition is not at"):
        generator.load()


def test_a_missing_checkpoint_is_named(tmp_path: Path) -> None:
    (tmp_path / "root" / "model").mkdir(parents=True)
    (tmp_path / "root" / "model" / "model_minimind.py").write_text("", encoding="utf-8")
    generator = ThinkerGenerator(
        checkpoint=tmp_path / "absent.pth",
        tokenizer_dir=tmp_path,
        minimind_root=tmp_path / "root",
    )

    with pytest.raises(ConfigurationError, match="no Thinker checkpoint"):
        generator.load()


def test_a_checkpoint_without_a_checkout_is_refused_at_assembly(tmp_path: Path) -> None:
    """Half the configuration is worse than none: it looks configured."""
    from mindsurf_omni.service.factory import build

    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    settings = Settings.from_environment(
        {
            "MINDSURF_ENGINE": "cascade",
            "MINDSURF_WEIGHTS": str(tmp_path),
            "MINDSURF_THINKER": str(tmp_path / "sft.pth"),
        }
    )

    with pytest.raises(ConfigurationError, match="MINIMIND_O_ROOT"):
        build(settings)


def test_the_generator_stops_being_unwired_once_a_checkpoint_is_named(tmp_path: Path) -> None:
    from mindsurf_omni.service import factory

    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    common = {"MINDSURF_ENGINE": "cascade", "MINDSURF_WEIGHTS": str(tmp_path)}

    without = factory.build(Settings.from_environment(common))

    # torch is not in this environment; the text stage runs on a host that has
    # it, so the presence check is answered rather than the test skipped.
    real = factory._importable
    factory._importable = lambda module: module == "torch" or real(module)  # type: ignore[assignment]
    try:
        with_thinker = factory.build(
            Settings.from_environment(
                {
                    **common,
                    "MINDSURF_THINKER": str(tmp_path / "sft.pth"),
                    "MINIMIND_O_ROOT": str(tmp_path),
                }
            )
        )
    finally:
        factory._importable = real  # type: ignore[assignment]

    assert "generator" in without.unwired  # type: ignore[union-attr]
    assert "generator" not in with_thinker.unwired  # type: ignore[union-attr]


def test_a_named_checkpoint_without_torch_says_so(tmp_path: Path) -> None:
    """The image carries the runtime set only, so this is the container's answer."""
    from mindsurf_omni.service import factory

    for name in ("tokenizer", "SenseVoiceSmall", "mimi", "campplus"):
        (tmp_path / name).mkdir(exist_ok=True)
    real = factory._importable
    factory._importable = lambda module: module != "torch" and real(module)  # type: ignore[assignment]
    try:
        with pytest.raises(ConfigurationError, match="torch"):
            factory.build(
                Settings.from_environment(
                    {
                        "MINDSURF_ENGINE": "cascade",
                        "MINDSURF_WEIGHTS": str(tmp_path),
                        "MINDSURF_THINKER": str(tmp_path / "sft.pth"),
                        "MINIMIND_O_ROOT": str(tmp_path),
                    }
                )
            )
    finally:
        factory._importable = real  # type: ignore[assignment]


def test_a_consumer_that_stops_stops_the_generation() -> None:
    """A caller that reads part of a reply must not leave the model generating.

    The streamer's queue is unbounded, so nothing pushes back on `generate`
    when the reader goes away: it runs to max_new_tokens on the GPU and puts
    the rest of the reply somewhere nobody will look. It raises nothing, and
    surfaces only as later turns being slower -- which on the native path cost
    this project three days and two wrong explanations of its own latency.

    Needs torch and transformers, which the test environment does not carry;
    it runs where the service runs.
    """
    import asyncio

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import torch

    from mindsurf_omni.service.engine import GenerationSettings

    steps = 0

    class _Tokeniser:
        def __call__(self, text: str, return_tensors: str = "pt") -> Any:
            return type("_Encoded", (), {"input_ids": torch.zeros((1, 1), dtype=torch.long)})()

        def decode(self, value: Any, **kwargs: Any) -> str:
            # The streamer decodes the whole cache each step and emits the
            # diff, so a constant string would look like no new text at all.
            return "字" * len(value)

    class _Model:
        def generate(self, **kwargs: Any) -> None:
            nonlocal steps
            streamer = kwargs["streamer"]
            # Tolerant on purpose: without the fix there is no criteria to
            # read, and this should then run to 200 and fail the assertion
            # rather than raise in the thread and hang the reader.
            criteria = kwargs.get("stopping_criteria") or []
            ids = torch.zeros((1, 1), dtype=torch.long)
            for _ in range(200):
                if any(one(ids, None) for one in criteria):
                    break
                steps += 1
                streamer.put(ids)
            streamer.end()

    class _Generator(ThinkerGenerator):
        def load(self) -> None:
            pass

        def prompt(self, messages: list[dict[str, str]]) -> str:
            return "问"

    generator = _Generator(
        checkpoint=Path("unused"), tokenizer_dir=Path("unused"), minimind_root=Path("unused")
    )
    generator._model = _Model()
    generator._tokenizer = _Tokeniser()

    async def read_two() -> int:
        stream = generator.generate([{"role": "user", "content": "问"}], GenerationSettings())
        seen = 0
        async for _ in stream:
            seen += 1
            if seen == 2:
                break
        await stream.aclose()
        return seen

    assert asyncio.run(read_two()) == 2
    assert steps < 200, "generation ran to completion after the consumer stopped"


def test_shape_and_parameter_count_cannot_drift_apart() -> None:
    """The guard works only because the count belongs to the shape.

    Two independent settings would let a caller widen the expected count until
    the wrong config passed, which is the exact failure this check exists to
    catch -- a model that loads, runs, answers, and is not the one trained.
    """
    from mindsurf_omni.service.thinker import VARIANTS

    assert VARIANTS["mindsurf"] == (THINKER_SHAPE, THINKER_PARAMETERS)
    for name, (shape, count) in VARIANTS.items():
        assert isinstance(shape, dict), name
        assert count > 0, name
    # Upstream's defaults are a different model, not a relabelling of ours.
    assert VARIANTS["upstream-default"][1] != THINKER_PARAMETERS


def test_an_unknown_variant_is_refused_by_name() -> None:
    generator = ThinkerGenerator(
        checkpoint=Path("unused"),
        tokenizer_dir=Path("unused"),
        minimind_root=Path("unused"),
        variant="whatever-upstream-calls-it",
    )
    # Asserting on the message. Every path out of load() raises ConfigurationError,
    # so without it this would pass on the missing checkout instead.
    with pytest.raises(ConfigurationError, match="unknown Thinker variant"):
        generator.load()
