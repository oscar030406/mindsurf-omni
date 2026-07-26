"""Talker-isolated evaluation: fixed text, teacher-forced, paired across checkpoints.

The free-running native evaluation conflates two questions: what the model
chose to say, and how well it spoke it. The text changes every run -- sampling
at temperature 0.7 -- so per-sample CER spreads with the text, the noise floor
lands around ±0.03, and CER has never been eligible to certify the 0.05 effect
it declares an interest in. The benchmark survey's answer is the standard TTS
protocol: hold the text fixed and teacher-force it, so the only thing that
varies between two checkpoints is the speaking.

Teacher forcing is not a trick here, it is the training objective. T2A trains
the Talker to emit audio codes for the assistant text it is conditioned on;
forcing that text at evaluation reproduces the training-time conditioning
exactly. Every checkpoint speaks the same 160 sentences, so comparisons are
paired sample-by-sample -- the text variance that dominated the free-run noise
floor is simply gone from the axis being measured.

The generation loop mirrors ``stream_generate`` step for step -- the one-step
audio lag, the per-codebook diagonal, the temperature-0.2 top-50 audio
sampling, the repetition penalty over the last three codes, the enter-then-pad
text tail. It has to: the codes come out on that diagonal, and a loop that
disagrees with it decodes to noise. The only divergence is where the text
token comes from -- a queue of the target's tokens instead of a sample.

What this does not measure: whether the model would have *chosen* good text
(the free-run eval keeps that), and naturalness (measure_naturalness.py reads
the same manifest). Both remain separate numbers on purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.contract import OUTPUT_SAMPLE_RATE  # noqa: E402
from mindsurf_omni.service.audio import to_wav  # noqa: E402

# Our base's shape. The upstream release uses the trainer defaults; both are
# stated fully because a shape left to defaults has already cost one training
# run and one wrong conclusion each.
SHAPES = {
    "mindsurf": {"intermediate_size": 3584, "num_key_value_heads": 8},
    "upstream-default": {},
    # A grafted checkpoint: our Thinker beside upstream's Talker. The two halves
    # were built with different shapes, so the override cannot be global -- it
    # goes into OmniConfig, which the Thinker reads, while the Talker builds a
    # fresh MiniMindConfig and keeps the library defaults upstream trained with.
    "graft": {"intermediate_size": 3584, "num_key_value_heads": 8},
}
# Which shapes patch MiniMindConfig globally (both halves) rather than passing
# the override to OmniConfig (Thinker only).
GLOBAL_SHAPE_PATCH = {"mindsurf", "upstream-default"}
ENTER_TOKEN = 201  # what stream_generate emits on the step after the text ends
PAD_TOKEN = 0
EOS_TOKEN = 2


def forced_text_plan(target_ids: list[int]) -> list[int]:
    """The text tokens the loop will feed, in order, ending with EOS.

    Everything after this plan is the enter-then-pad tail stream_generate
    produces once the text is finished; that part depends on how long the
    audio runs and stays in the loop.
    """
    return [*target_ids, EOS_TOKEN]


def load_texts(path: Path) -> list[dict[str, str]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no texts in {path}")
    for row in rows:
        if not row.get("text", "").strip():
            raise SystemExit(f"row {row.get('id')} has no text; an empty target scores as silence")
    return rows


def build_model(
    checkpoint: Path, minimind_root: Path, audio_encoder: Path, shape: str, device: str
) -> Any:
    import torch

    if str(minimind_root) not in sys.path:
        sys.path.insert(0, str(minimind_root))
    from model import model_minimind, model_omni  # type: ignore[import-not-found]

    overrides = SHAPES[shape]
    config_kwargs: dict[str, Any] = {}
    if overrides and shape in GLOBAL_SHAPE_PATCH:
        original = model_minimind.MiniMindConfig.__init__

        def patched(self: Any, *args: Any, **kwargs: Any) -> None:
            for key, value in overrides.items():
                kwargs.setdefault(key, value)
            original(self, *args, **kwargs)

        model_minimind.MiniMindConfig.__init__ = patched  # type: ignore[method-assign]
    elif overrides:
        config_kwargs = dict(overrides)

    model = model_omni.MiniMindOmni(
        model_omni.OmniConfig(hidden_size=768, num_hidden_layers=8, use_moe=False, **config_kwargs),
        audio_encoder_path=str(audio_encoder),
        vision_model_path=str(checkpoint.parent / "nonexistent-vision"),
    )
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    report = model.load_state_dict(state, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise SystemExit(
            f"{checkpoint.name} does not fit the {shape!r} shape: "
            f"{len(report.missing_keys)} missing, {len(report.unexpected_keys)} unexpected -- "
            "a half-loaded model speaks, plausibly, from random weights"
        )
    return model.eval().to(device)


@dataclass(frozen=True)
class AudioSampling:
    """How audio codes are drawn. Upstream's values are the default.

    ``temperature`` accepts a list of eight to set one per codebook. The
    diagnostic found the codebooks are not alike -- codebook 0 picks the argmax
    86.8% of the time and codebook 7 only 70.8% -- so a single number is an
    assumption, not a given.
    """

    temperature: float | list[float] = 0.2
    top_k: int = 50
    penalty: float = 1.05
    penalty_window: int = 3

    def for_codebook(self, layer: int) -> float:
        if isinstance(self.temperature, list):
            return self.temperature[layer]
        return self.temperature


DEFAULT_SAMPLING = AudioSampling()


def speak_forced(
    model: Any,
    tokenizer: Any,
    prompt: str,
    text: str,
    device: str,
    seed: int,
    max_steps: int = 640,
    sampling: AudioSampling = DEFAULT_SAMPLING,
    observer: Any = None,
) -> tuple[list[list[int]], int]:
    """Teacher-force the text and collect whole Mimi frames.

    Follows stream_generate's schedule exactly; see the module docstring for
    why divergence is not an option. That is also why this is the only copy:
    the schedule is delicate enough that three drifting transcriptions of it
    would eventually disagree, and the one that decodes to noise would not
    announce itself. Sweeps and diagnostics call in here rather than restating
    it -- ``sampling`` varies the knobs, ``observer`` watches each draw.
    """
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)  # audio sampling stays stochastic; the text is not

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor(
        tokenizer(rendered).data["input_ids"], dtype=torch.long, device=device
    )[None, ...]
    plan = forced_text_plan(tokenizer(text, add_special_tokens=False).data["input_ids"])

    start_pos, past, text_finished, first_finished = input_ids.shape[1], None, False, True
    audio_codes: list[list[int]] = [[] for _ in range(8)]
    audio_stop: list[int | None] = [None] * 8
    audio_buffer = torch.full(
        (1, 8, start_pos), model.audio_pad_token, dtype=torch.long, device=device
    )
    frames: list[list[int]] = []

    with torch.no_grad():
        while input_ids.shape[1] < start_pos + max_steps:
            if past is None:
                out = model.forward(
                    torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1),
                    past_key_values=past,
                    use_cache=True,
                )
            else:
                out = model.forward(
                    torch.cat((audio_buffer[:, :, -1:], input_ids[:, -1:].unsqueeze(1)), dim=1),
                    past_key_values=past,
                    use_cache=True,
                )
            past = out.past_key_values

            step = input_ids.shape[1] - start_pos
            if text_finished:
                text_token = ENTER_TOKEN if first_finished else PAD_TOKEN
                first_finished = False
            else:
                text_token = plan[step] if step < len(plan) else EOS_TOKEN

            audio_step = step - 1
            for layer, logits in enumerate(out.audio_logits):
                if audio_step < layer:
                    audio_codes[layer].append(model.audio_pad_token)
                    continue
                scores = logits[0, -1, :].clone() / sampling.for_codebook(layer)
                if sampling.penalty != 1.0:
                    for previous in audio_codes[layer][-sampling.penalty_window :]:
                        value = scores[previous]
                        scores[previous] = torch.where(
                            value > 0, value / sampling.penalty, value * sampling.penalty
                        )
                top_values, top_indices = scores.topk(min(sampling.top_k, scores.shape[-1]))
                probabilities = functional.softmax(top_values, dim=-1)
                code = int(top_indices[torch.multinomial(probabilities, 1)])
                if observer is not None:
                    # What the sampler chose against what greedy would have,
                    # from the same logits -- the only place both are in hand.
                    observer(layer, float(probabilities[0]), code == int(top_indices[0]))
                audio_codes[layer].append(code)
                if audio_stop[layer] is None and code >= 2048:
                    audio_stop[layer] = len(audio_codes[layer]) - 1

            if text_finished and all(stop is not None for stop in audio_stop):
                break

            input_ids = torch.cat((input_ids, torch.tensor([[text_token]], device=device)), dim=1)
            audio_buffer = torch.cat(
                (
                    audio_buffer,
                    torch.full((1, 8, 1), model.audio_pad_token, dtype=torch.long, device=device),
                ),
                dim=2,
            )
            for layer in range(min(audio_step + 1, 8)):
                audio_buffer[0, layer, -1] = audio_codes[layer][-1]

            if audio_step >= 7:
                frame = [audio_codes[layer][step - 7 + layer] for layer in range(8)]
                active = sum(
                    1
                    for layer in range(8)
                    if audio_stop[layer] is None or step - 7 + layer < audio_stop[layer]
                )
                if active >= 8:
                    frames.append(frame)

            if not text_finished and text_token == EOS_TOKEN:
                text_finished = True

    return frames, input_ids.shape[1] - start_pos


def parse_temperature(value: str) -> float | list[float]:
    """One number for all eight codebooks, or eight numbers.

    Eight is not a convenience: the diagnostic found codebook 0 agrees with
    greedy 86.8% of the time and codebook 7 only 70.8%, so "one temperature for
    the stack" is an assumption nobody has tested. Anything other than one or
    eight values is refused rather than broadcast -- a four-value list would
    silently mean something this loop does not implement.
    """
    parts = [piece.strip() for piece in value.split(",") if piece.strip()]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) != 8:
        raise SystemExit(
            f"--audio-temperature takes 1 or 8 values, got {len(parts)}; "
            "there are eight codebooks and no rule for spreading four across them"
        )
    return [float(piece) for piece in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--shape", choices=sorted(SHAPES), default="mindsurf")
    parser.add_argument("--texts", type=Path, default=Path("configs/talker_texts_zh_v1.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", required=True, type=Path)
    parser.add_argument("--audio-encoder", required=True, type=Path)
    parser.add_argument("--codec", required=True, type=Path, help="Mimi model directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--limit", type=int, help="first N texts only, for a smoke run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--audio-temperature",
        default="0.2",
        help="one value, or eight comma-separated -- one per codebook. The "
        "inherited 0.2 is what upstream's stream_generate hardcodes, so leaving "
        "it alone measures the released decoding path",
    )
    parser.add_argument(
        "--audio-penalty",
        type=float,
        default=DEFAULT_SAMPLING.penalty,
        help="repetition penalty over the last --audio-penalty-window codes",
    )
    parser.add_argument("--audio-penalty-window", type=int, default=DEFAULT_SAMPLING.penalty_window)
    args = parser.parse_args()

    sampling = AudioSampling(
        temperature=parse_temperature(args.audio_temperature),
        top_k=DEFAULT_SAMPLING.top_k,
        penalty=args.audio_penalty,
        penalty_window=args.audio_penalty_window,
    )

    import torch
    from transformers import AutoTokenizer, MimiModel

    texts = load_texts(args.texts)[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    model = build_model(
        args.checkpoint, args.minimind_root, args.audio_encoder, args.shape, args.device
    )
    codec = MimiModel.from_pretrained(str(args.codec)).eval().to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    args.output.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    generation_seconds = 0.0
    audio_seconds = 0.0

    for index, row in enumerate(texts):
        started = time.perf_counter()
        frames, steps = speak_forced(
            model,
            tokenizer,
            row["prompt"],
            row["text"],
            args.device,
            seed=args.seed + index,
            sampling=sampling,
        )
        elapsed = time.perf_counter() - started

        path = args.output / f"{row['id']}.wav"
        seconds = 0.0
        if frames:
            codes = torch.tensor(frames, dtype=torch.long, device=args.device).T.unsqueeze(0)
            codes = torch.where(codes >= 2049, torch.zeros_like(codes), codes)
            with torch.no_grad():
                waveform = codec.decode(codes).audio_values.squeeze().float().cpu().numpy()
            seconds = len(waveform) / OUTPUT_SAMPLE_RATE
            pcm = (waveform * 32767).astype("int16").tobytes()
            path.write_bytes(to_wav(pcm, OUTPUT_SAMPLE_RATE))

        generation_seconds += elapsed
        audio_seconds += seconds
        samples.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                # The forced text: on this protocol the reference really is
                # what the Talker was told to say.
                "reference_text": row["text"],
                "audio_path": str(path) if frames else None,
                "elapsed_ms": elapsed * 1000,
                "audio_seconds": seconds,
                "steps": steps,
                **({} if frames else {"error": "no complete frames"}),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(texts)}", flush=True)

    failed = [sample["id"] for sample in samples if "error" in sample]
    manifest = {
        "generated_by": {
            "path": "talker-isolated",
            "model": args.checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "shape": args.shape,
            "components": [{"name": f"talker teacher-forced, shape {args.shape}"}],
            "text_source": "fixed",
            "texts_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
            # What actually ran, not what the default was when this line was
            # written. A manifest that reports the default while the run used
            # something else is how a sweep becomes eight identical-looking
            # reports.
            "sampling": {
                "audio_temperature": sampling.temperature,
                "audio_top_k": sampling.top_k,
                "audio_penalty": sampling.penalty,
                "audio_penalty_window": sampling.penalty_window,
                "seed": args.seed,
            },
        },
        # The survey's reproduction checklist: efficiency alongside quality.
        "efficiency": {
            "rtf": (generation_seconds / audio_seconds) if audio_seconds else None,
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else None
            ),
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "probe_count": len(texts),
        "generated": len(samples) - len(failed),
        "failed": failed,
        "samples": samples,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rtf = manifest["efficiency"]["rtf"]
    vram = manifest["efficiency"]["peak_vram_bytes"] or 0
    print(f"\ngenerated {manifest['generated']}/{len(texts)}  failed {failed[:5]}")
    print(f"RTF {rtf:.2f}  peak VRAM {vram >> 20} MiB")
    print(f"清单 {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
