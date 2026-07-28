"""The native path: audio tokens straight through the model.

No transcript in the middle. Speech goes in as Mimi codes, the Thinker reasons
over them alongside text, and the Talker emits codes back -- so tone, pacing
and interruption survive a round trip that a transcript would flatten.

Two things here were only learnable by running it. The Talker's codes come
out on a diagonal -- codebook i starts i steps late -- and MiniMind-O's
``stream_generate`` already undoes that, handing back whole frames of eight;
taking its per-layer buffers instead would decode to noise. And a chunk
decoded on its own starts with an audible edge, because the codec carries
temporal context, so each chunk is decoded with a lead-in that is then thrown
away.

The chunking here is the whole latency argument. The Talker emits Mimi codes at
12.5 Hz, so 25 codes are two seconds of speech; decoding every 4 frames means
audio starts flowing about 320 ms into a reply instead of after it. That is the
same lever the cascade pulls by cutting at the first clause -- applied at the
token level, where it costs less.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mindsurf_omni.contract import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    ComponentInfo,
    TokenSpec,
)
from mindsurf_omni.service.audio import peak_normalise, resample, trim_silence
from mindsurf_omni.service.config import ConfigurationError
from mindsurf_omni.service.engine import (
    EngineDescription,
    GenerationSettings,
    SpeechChunk,
    SpeechEngine,
)

# Mimi's frame rate. Everything about the streaming budget follows from it.
MIMI_FRAME_RATE_HZ = 12.5
DEFAULT_CHUNK_FRAMES = 4  # about 320 ms
# Frames of context handed to the decoder ahead of each chunk and then dropped.
# Upstream's streaming demo uses two, "to ease chunk-boundary breakage".
DEFAULT_OVERLAP_FRAMES = 2
# The codebook holds 2048 entries; anything at or above the pad token is a
# marker rather than audio, and upstream zeroes it before decoding.
FIRST_MARKER_CODE = 2049
# Stands in the text for one frame of encoded audio; the model writes the
# encoder's output over these embeddings.
AUDIO_PLACEHOLDER = "<|audio_pad|>"


class OmniModel(Protocol):
    """What this path needs from MiniMind-O's model.

    Declared as a Protocol so the engine can be built and tested without the
    checkout: the real class satisfies it structurally, and a stub in a test
    satisfies it too.
    """

    def stream_generate(self, input_ids: Any, audio_inputs: Any | None, **kwargs: Any) -> Any: ...


@dataclass(slots=True)
class NativeConfig:
    chunk_frames: int = DEFAULT_CHUNK_FRAMES
    overlap_frames: int = DEFAULT_OVERLAP_FRAMES
    max_new_tokens: int = 512
    # Trailing dead air between chunks concatenates into an audible stutter.
    trim_chunks: bool = True

    def __post_init__(self) -> None:
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be at least 1")

    @property
    def chunk_ms(self) -> float:
        return self.chunk_frames / MIMI_FRAME_RATE_HZ * 1000


class NativeEngine(SpeechEngine):
    """Speech in, speech out, with no transcript in between."""

    def __init__(
        self,
        model: OmniModel,
        codec: Any,
        tokenizer: Any,
        token_spec: TokenSpec,
        components: list[ComponentInfo],
        config: NativeConfig | None = None,
        device: str = "cpu",
        recogniser: Any = None,
    ) -> None:
        self._model = model
        self._codec = codec
        self._tokenizer = tokenizer
        self._spec = token_spec
        self._components = components
        self._config = config or NativeConfig()
        self._device = device
        # Only for /v1/audio/transcriptions. A reply never goes through it --
        # the model reads the audio itself -- so it stays optional.
        self._recogniser = recogniser

    def describe(self) -> EngineDescription:
        return EngineDescription(path="native", components=self._components)

    def token_spec(self) -> TokenSpec:
        return self._spec

    async def transcribe(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        """Available, but not on the path a reply takes.

        The native path never needs a transcript to answer. This exists so the
        endpoint behaves the same whichever engine is configured, and so a
        caller can read back what was heard.
        """
        return await self._recognise(pcm, sample_rate)

    async def _recognise(self, pcm: bytes, sample_rate: int) -> tuple[str, str | None]:
        if self._recogniser is None:
            raise ConfigurationError(
                "this native instance has no recogniser wired; a reply does not need one, "
                "so /v1/audio/transcriptions is the only thing affected"
            )
        return await self._recogniser.transcribe(pcm, sample_rate)  # type: ignore[no-any-return]

    async def complete(  # type: ignore[override]
        self, messages: list[dict[str, str]], settings: GenerationSettings
    ) -> AsyncIterator[str]:
        """Text only, from a model that is also producing audio as it goes.

        The audio is discarded here rather than not produced: the Talker runs
        off the same forward pass, so asking for text alone does not make it
        cheaper. What it does make is comparable to the cascade's text stage.
        """
        async for delta, _ in self._turn(messages, settings, audio=None):
            if delta:
                yield delta

    def speak(self, text: str, settings: GenerationSettings) -> AsyncIterator[SpeechChunk]:
        """Not something this path can do, and the reason is not a missing wire.

        The native model speaks its own words: the training pairs an assistant
        turn with the codes for *that* turn, so there is no task in it for
        "read this text aloud". Handing it someone else's sentence asks for
        something it was never shown.

        The endpoint stays in the contract because the cascade answers it. On
        this path the honest answer is to say so -- reaching for the cascade's
        synthesiser instead would report `path: native` over audio a hosted
        service produced.
        """
        raise ConfigurationError(
            "the native path has no text-to-speech step: the model speaks the words it "
            "generates, and was never trained to read text it did not write. Use "
            "/v1/realtime, or MINDSURF_ENGINE=cascade for arbitrary text"
        )

    async def respond(  # type: ignore[override]
        self, pcm: bytes, sample_rate: int, settings: GenerationSettings
    ) -> AsyncIterator[SpeechChunk]:
        """Speech in, speech out, with no transcript in between.

        This is the path's whole argument. The reply's text and its audio come
        out of one forward pass, so the audio starts flowing while the sentence
        is still being decided rather than after it is finished.
        """
        frames: list[list[int]] = []
        emitted = 0
        overlap = self._config.overlap_frames
        size = self._config.chunk_frames

        async for delta, frame in self._turn(None, settings, audio=(pcm, sample_rate)):
            if frame is not None:
                frames.append(frame)
            if len(frames) - emitted >= size:
                lead = min(overlap, emitted)
                audio = self._decode(frames[emitted - lead : emitted + size], lead)
                emitted += size
                yield SpeechChunk(pcm=audio, text=delta or None)
            elif delta:
                # Text runs ahead of audio by the codebook delay, so a turn
                # emits words before it emits sound. Holding them back would
                # spend the model's head start.
                yield SpeechChunk(pcm=b"", text=delta)

        if len(frames) > emitted:
            lead = min(overlap, emitted)
            yield SpeechChunk(pcm=self._decode(frames[emitted - lead :], lead), is_final=True)
        else:
            yield SpeechChunk(pcm=b"", is_final=True)

    async def _turn(
        self,
        messages: list[dict[str, str]] | None,
        settings: GenerationSettings,
        audio: tuple[bytes, int] | None,
    ) -> AsyncIterator[tuple[str, list[int] | None]]:
        """One generation, as (text delta, audio frame) pairs.

        Generation is synchronous and holds the GIL between steps, so it runs
        on a thread and the pairs come back through a queue. Driving it on the
        event loop would stall every other request for the length of a reply.
        """
        import queue
        import threading

        outbox: queue.Queue[Any] = queue.Queue()
        done = object()

        def produce() -> None:
            try:
                for item in self._stream(messages, settings, audio):
                    outbox.put(item)
            except Exception as error:  # noqa: BLE001 - carried to the consumer
                outbox.put(error)
            finally:
                outbox.put(done)

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        while True:
            item = await asyncio.to_thread(outbox.get)
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _stream(
        self,
        messages: list[dict[str, str]] | None,
        settings: GenerationSettings,
        audio: tuple[bytes, int] | None,
    ) -> Iterator[tuple[str, list[int] | None]]:
        """Drive MiniMind-O's generator, turning ids into text deltas."""
        import torch

        arguments: dict[str, Any] = {}
        if audio is not None:
            features, lengths = self._encode_audio(*audio)
            arguments["audio_inputs"], arguments["audio_lens"] = features, lengths
            prompt = self._prompt(messages or [], audio_frames=int(lengths[0]))
        else:
            prompt = self._prompt(messages or [])

        input_ids = torch.tensor(self._tokenizer(prompt).data["input_ids"], dtype=torch.long)[
            None, ...
        ].to(self._device)

        said = ""
        with torch.no_grad():
            for ids, frame in self._model.stream_generate(
                input_ids,
                eos_token_id=self._spec.special_tokens.get("im_end", 2),
                max_new_tokens=settings.max_tokens,
                temperature=max(settings.temperature, 1e-5),
                top_p=settings.top_p,
                rp=1.0,
                use_cache=True,
                return_audio_codes=True,
                **arguments,
            ):
                delta = ""
                if ids is not None:
                    whole = self._tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
                    # A partial multi-byte character decodes to the replacement
                    # character; emitting it would put a black diamond in the
                    # transcript and in anything scoring it.
                    if whole and not whole.endswith("�"):
                        delta, said = whole[len(said) :], whole
                if delta or frame is not None:
                    yield delta, frame

    def _prompt(self, messages: list[dict[str, str]], audio_frames: int = 0) -> str:
        """The chat template, with one placeholder per encoder frame.

        The audio does not arrive as a side input the model consults: its
        features are written over the embeddings of a run of ``<|audio_pad|>``
        tokens in the user turn, one per frame the encoder produced. Leaving
        them out is not a partial prompt -- the model simply never sees the
        audio, and answers the empty turn it did get. That failure is silent
        and reads as a model that cannot understand speech.
        """
        turns = list(messages)
        if audio_frames > 0:
            placeholder = AUDIO_PLACEHOLDER * audio_frames
            spoken_turn = {"role": "user", "content": placeholder}
            last_user = max(
                (index for index, turn in enumerate(turns) if turn["role"] == "user"),
                default=None,
            )
            if last_user is None:
                turns.append(spoken_turn)
            else:
                turns[last_user] = spoken_turn
        return str(
            self._tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
        )

    def _encode_audio(self, pcm: bytes, sample_rate: int) -> tuple[Any, Any]:
        """Waveform to the encoder features the model splices into the prompt."""
        import numpy
        import torch

        if sample_rate != INPUT_SAMPLE_RATE:
            pcm = resample(pcm, sample_rate, INPUT_SAMPLE_RATE)
        waveform = numpy.frombuffer(pcm, dtype=numpy.int16).astype(numpy.float32) / 32768.0

        # The same two values the training pipeline derives: the fbank frames
        # and the number of them the encoder will actually produce. Passing the
        # padded length instead would have the model attend to silence it was
        # never trained to see there.
        inputs = self._model.audio_processor(  # type: ignore[attr-defined]
            waveform, sampling_rate=INPUT_SAMPLE_RATE, return_tensors="pt"
        )
        valid = int(inputs.attention_mask.sum().item())
        features = inputs.input_features.squeeze(0).unsqueeze(0).to(self._device)
        return features, torch.tensor([valid], device=self._device)

    def _decode(self, frames: list[list[int]], lead: int) -> bytes:
        """Mimi codes to PCM, minus the lead-in that was only there for context."""
        import torch

        usable = [frame for frame in frames if frame and len(frame) == 8]
        if not usable:
            return b""
        codes = torch.tensor(usable, dtype=torch.long, device=self._device).T.unsqueeze(0)
        codes = torch.where(codes >= FIRST_MARKER_CODE, torch.zeros_like(codes), codes)
        with torch.no_grad():
            waveform = self._codec.decode(codes).audio_values.squeeze().float().cpu().numpy()
        if lead:
            waveform = waveform[round(lead * len(waveform) / len(usable)) :]
        pcm = (waveform * 32767).astype("int16").tobytes()
        return prepare_output(
            pcm, self._codec.config.sampling_rate, OUTPUT_SAMPLE_RATE, self._config.trim_chunks
        )


def chunk_codes(
    codes: list[list[int]], chunk_frames: int, overlap: int = 0
) -> list[tuple[list[list[int]], int]]:
    """Group Mimi frames into decodable chunks, each with its lead-in.

    Each frame is one value per codebook, so a chunk is a list of frames rather
    than a flat list. Splitting on the wrong axis would interleave codebooks
    and decode to noise -- which is why this is a named function with a test
    rather than a slice expression inside the generation loop.

    Returned with the number of lead-in frames, because the decoder has
    temporal context and a chunk decoded from nothing starts with an audible
    edge. Upstream's streaming demo decodes ``chunk + overlap`` frames and
    throws the lead-in away; doing it here rather than at the call site keeps
    the two numbers together, since discarding the wrong amount silently
    shortens every chunk.

    The final chunk is kept even when short: dropping it would clip the end of
    every reply.
    """
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be at least 1")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    chunks = []
    for start in range(0, len(codes), chunk_frames):
        lead = min(overlap, start)
        chunks.append((codes[start - lead : start + chunk_frames], lead))
    return chunks


def prepare_output(pcm: bytes, sample_rate: int, target_rate: int, trim: bool) -> bytes:
    """Normalise one decoded chunk for playback.

    Trimming happens before resampling: the trim threshold is in sample
    amplitude, which resampling preserves, but the margin is in milliseconds,
    which only matches the rate it was measured at.
    """
    if trim:
        pcm = trim_silence(pcm, rate=sample_rate)
    pcm = peak_normalise(pcm)
    return resample(pcm, sample_rate, target_rate)


# Thinker plus Talker plus the projections, as the training reports it.
# Our shape on both halves, and the graft: our Thinker beside upstream's
# narrower Talker. Which one a checkpoint needs is read off the checkpoint --
# a shape passed by configuration is a shape someone can get wrong, and the
# way it goes wrong is a Talker left at random initialisation that still emits
# codes.
OMNI_PARAMETERS = {"mindsurf": 152_059_650, "graft": 139_083_522}
TALKER_WIDTH = {"mindsurf": 3584, "graft": 2432}


def detect_shape(state: dict[str, Any]) -> str:
    """Which shape this checkpoint's Talker was trained at.

    Read rather than configured. The graft carries upstream's Talker beside
    our Thinker, so its Talker feed-forward is 2432 wide where ours is 3584,
    and building the wrong one does not produce a helpful error at serving
    time -- it produces twenty size mismatches, or worse, silence.
    """
    for key, tensor in state.items():
        if key.startswith("talker.") and key.endswith("mlp.gate_proj.weight"):
            width = int(tensor.shape[0])
            for name, expected in TALKER_WIDTH.items():
                if width == expected:
                    return name
            raise ConfigurationError(
                f"the Talker's feed-forward is {width} wide, which is neither ours "
                f"({TALKER_WIDTH['mindsurf']}) nor upstream's ({TALKER_WIDTH['graft']})"
            )
    # No Talker tensors at all: a text-only checkpoint, which the native path
    # cannot serve. Say so here rather than at the first missing key.
    raise ConfigurationError("this checkpoint has no Talker; the native path needs one")


def load_omni(
    checkpoint: Path,
    minimind_root: Path,
    tokenizer_dir: Path,
    audio_encoder: Path,
    codec_dir: Path,
    device: str = "cpu",
) -> tuple[Any, Any, Any]:
    """Build MiniMind-O at our base's shape and load one checkpoint into it.

    The same two traps as the Thinker loader, and for the same reason: the
    upstream defaults build a narrower model that still accepts our weights,
    and ``strict=False`` leaves anything absent at its random initialisation.
    Both are checked rather than trusted.

    The checkout is imported as a package because ``model_omni`` does
    ``from .model_minimind import *``; loading it by file path fails on the
    relative import.
    """
    for name, path in (("checkout", minimind_root / "model"), ("checkpoint", checkpoint)):
        if not path.exists():
            raise ConfigurationError(f"the native path's {name} is not at {path}")

    import sys

    import torch
    from transformers import AutoTokenizer, MimiModel

    if str(minimind_root) not in sys.path:
        sys.path.insert(0, str(minimind_root))
    from model import model_minimind, model_omni  # type: ignore[import-not-found]

    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    shape = detect_shape(state)

    original = model_minimind.MiniMindConfig.__init__
    # On a graft the override goes to the Thinker alone. OmniConfig is the
    # Thinker's config; the Talker builds a plain MiniMindConfig and has to
    # keep the library defaults upstream trained it with. Widening it here
    # would make its twenty tensors mismatch, which is how this failed before
    # the detection existed.
    thinker_only = shape == "graft"

    def patched(self: Any, *args: Any, **kwargs: Any) -> None:
        if not (thinker_only and type(self) is model_minimind.MiniMindConfig):
            kwargs.setdefault("intermediate_size", 3584)
            kwargs.setdefault("num_key_value_heads", 8)
        original(self, *args, **kwargs)

    model_minimind.MiniMindConfig.__init__ = patched
    try:
        config = model_omni.OmniConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
        # The vision tower was cut from this project; the loader warns and
        # returns None for a path that does not exist, which is what we want.
        model = model_omni.MiniMindOmni(
            config,
            audio_encoder_path=str(audio_encoder),
            vision_model_path=str(minimind_root / "model" / "vision-not-used"),
        )
    finally:
        model_minimind.MiniMindConfig.__init__ = original

    count = sum(parameter.numel() for parameter in model.parameters())
    if count != OMNI_PARAMETERS[shape]:
        raise ConfigurationError(
            f"the omni model built to {count:,} parameters, but a {shape!r} checkpoint "
            f"should give {OMNI_PARAMETERS[shape]:,} -- the config does not match it"
        )

    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if key != "lm_head.weight"]
    if missing:
        raise ConfigurationError(
            f"{len(missing)} tensors are not in {checkpoint.name} and stayed at random "
            f"initialisation: {missing[:4]}{' …' if len(missing) > 4 else ''} -- a Talker "
            "half like that still emits codes, and they decode to noise"
        )

    if model.audio_encoder is not None:
        model.audio_encoder.to(device)
    codec = MimiModel.from_pretrained(str(codec_dir)).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return model.eval().to(device), tokenizer, codec
