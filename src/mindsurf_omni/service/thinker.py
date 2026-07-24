"""The cascade's text stage: MiniMind's Thinker, answering in text.

The cascade is ASR, then this, then speech. Only the text transformer is
needed here -- the audio encoder is loaded separately by the recogniser, and
the Talker is the native path's business -- so this loads the Thinker alone
rather than the whole omni model. That matters on a box that is also training:
the full model pulls SenseVoice in a second time for nothing.

Two things about the weights are not guessable from the file name.

``thinker`` is an alias for ``model`` inside MiniMindOmni, installed with
``object.__setattr__`` so it is not a registered submodule. The Thinker's
tensors are therefore saved under ``model.*`` and ``lm_head.*``, in both the
text-only base and the omni checkpoints -- which is what lets one loader read
either.

And the config cannot come from MiniMind's defaults. Ours is MHA with a wider
feed-forward; the defaults are GQA with four key-value heads and a narrower
one, and ``load_state_dict`` is forgiving enough to build the wrong model and
run it. That failure already cost this project a night of training, so the
parameter count is checked rather than trusted.

torch is imported inside the methods. It is not in the runtime set and the
container does not carry it: a module-level import would make the service
unimportable in the image it ships in.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mindsurf_omni.service.config import ConfigurationError
from mindsurf_omni.service.engine import GenerationSettings

# Our base, which MiniMind's defaults do not describe. Stated here because the
# alternative is a model that loads, runs, and is not the one that was trained.
THINKER_SHAPE = {
    "hidden_size": 768,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,  # MHA; MiniMind defaults to 4
    "intermediate_size": 3584,  # MiniMind derives 1536 from hidden_size
    "vocab_size": 6400,
    "tie_word_embeddings": True,
}
THINKER_PARAMETERS = 89_864_448
IM_END = 2


def thinker_weights(state: dict[str, Any]) -> dict[str, Any]:
    """The text transformer's tensors, from a base or an omni checkpoint.

    An omni checkpoint also carries ``talker.*`` and the projections. Passing
    those to the text model would be silently dropped by ``strict=False``,
    which is the same forgiveness that hides a wrong config, so they are
    removed here and the count is checked by the caller.
    """
    return {
        key: value
        for key, value in state.items()
        if key.startswith(("model.", "lm_head.")) and not key.startswith("model.talker")
    }


@dataclass(slots=True)
class ThinkerGenerator:
    """Loads the Thinker on first use and streams text deltas from it."""

    checkpoint: Path
    tokenizer_dir: Path
    minimind_root: Path
    device: str = "cpu"
    _model: Any = field(default=None)
    _tokenizer: Any = field(default=None)

    def load(self) -> None:
        if self._model is not None:
            return

        # Before the imports, deliberately. These answers need no torch, and
        # the environment most likely to be misconfigured is the container,
        # which has no torch -- "No module named 'torch'" would be a true
        # statement about the wrong problem.
        definition = self.minimind_root / "model" / "model_minimind.py"
        if not definition.is_file():
            raise ConfigurationError(
                f"MiniMind's model definition is not at {definition}; set MINIMIND_O_ROOT "
                "to a checkout, because the Thinker is built from their class rather than "
                "a copy of it"
            )
        if not self.checkpoint.is_file():
            raise ConfigurationError(f"no Thinker checkpoint at {self.checkpoint}")

        import sys

        import torch
        from transformers import AutoTokenizer

        sys.path.insert(0, str(definition.parent))
        import importlib.util

        spec = importlib.util.spec_from_file_location("model_minimind", definition)
        if spec is None or spec.loader is None:  # pragma: no cover - unreadable file
            raise ConfigurationError(f"cannot import {definition}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.MiniMindForCausalLM(module.MiniMindConfig(**THINKER_SHAPE))
        state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
        weights = thinker_weights(state)
        model.load_state_dict(weights, strict=False)

        # The check that would have caught the silent 113.13M run: a wrong
        # config still loads and still generates, just not from our base.
        count = sum(parameter.numel() for parameter in model.parameters())
        if count != THINKER_PARAMETERS:
            raise ConfigurationError(
                f"the Thinker built to {count:,} parameters, but our base is "
                f"{THINKER_PARAMETERS:,} -- the config does not match the checkpoint"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
        self._model = model.to(self.device).eval()

    def prompt(self, messages: list[dict[str, str]]) -> str:
        return str(
            self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    async def generate(
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        """Yield text deltas as they are produced.

        Generation is synchronous and blocking, so it runs on a thread and the
        deltas come back through the streamer. Yielding only at the end would
        cost the whole reply before the first sound -- the single largest term
        in the latency budget, and the reason the cascade cuts at the first
        clause at all.
        """
        import threading

        import torch
        from transformers import TextIteratorStreamer

        self.load()
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        input_ids = self._tokenizer(self.prompt(messages), return_tensors="pt").input_ids.to(
            self.device
        )

        def run() -> None:
            with torch.inference_mode():
                self._model.generate(
                    input_ids=input_ids,
                    max_new_tokens=settings.max_tokens,
                    temperature=max(settings.temperature, 1e-5),
                    top_p=settings.top_p,
                    do_sample=settings.temperature > 0,
                    eos_token_id=IM_END,
                    streamer=streamer,
                )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        iterator = iter(streamer)
        sentinel = object()
        while True:
            # next() on the streamer blocks until the model produces a token;
            # doing that on the event loop would stall every other request on
            # this worker for the length of a reply.
            delta = await asyncio.to_thread(next, iterator, sentinel)
            if delta is sentinel:
                break
            if delta:
                yield str(delta)
