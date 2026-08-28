"""Speak a fixed set of texts with one synthesiser, for comparing synthesisers.

The cascade's CER answers "did the synthesiser say the reply the model wrote".
Two synthesisers can only be compared on that if the text is held fixed --
otherwise the arms differ by what was said as well as by who said it, and the
subtraction is meaningless. So this reads the same 160 sentences the Talker
benchmark uses, hands them to one synthesiser, and writes the manifest the
transcribe-then-score half of the harness already understands.

The model is not involved. Nothing here measures the Thinker or the Talker:
the reference text is the file's text, so whatever CER comes back is the floor
that the synthesiser and the judge contribute between them. That floor is worth
measuring precisely because every cascade number is read against it -- change
the synthesiser and 0.0325 is no longer the reference frame it was.

Latency is recorded per utterance and is *reported only*. The instrument for
that is a quiet card, and this laptop has read 291 / 960 / 2488 ms out of the
same code.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mindsurf_omni.contract import OUTPUT_SAMPLE_RATE  # noqa: E402
from mindsurf_omni.data.synthesis import (  # noqa: E402
    EdgeSynthesiser,
    Utterance,
    VoxCPMSynthesiser,
    stream_utterance,
)
from mindsurf_omni.service.audio import to_wav  # noqa: E402


def build_synthesiser(
    name: str,
    device: str,
    prompt_wav: Path | None = None,
    prompt_text: str | None = None,
) -> Any:
    if name == "edge":
        if prompt_wav or prompt_text:
            raise SystemExit("edge has one fixed voice; a clone prompt only reaches voxcpm")
        return EdgeSynthesiser()
    if name == "voxcpm":
        # Both or neither, same as the service: VoxCPM ignores half a reference
        # and then draws a speaker per call, which is the arm this flag exists
        # to separate from.
        if bool(prompt_wav) != bool(prompt_text):
            raise SystemExit(
                "--prompt-wav and --prompt-text go together; a clip without its text clones nothing"
            )
        return VoxCPMSynthesiser(
            device=device,
            prompt_wav=str(prompt_wav) if prompt_wav else None,
            prompt_text=prompt_text,
        )
    raise SystemExit(f"{name!r} names no synthesiser; 'edge' and 'voxcpm' are wired")


def load_texts(path: Path) -> list[dict[str, str]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise SystemExit(f"no texts in {path}")
    return rows


async def speak_all(
    synthesiser: Any,
    texts: list[dict[str, str]],
    output: Path,
    emotion: str,
    stream: bool = False,
) -> list[dict[str, Any]]:
    """One at a time, because that is how the cascade calls it.

    Batching would change the number this run exists to produce: an external
    report has a 4090 going from p99 450 ms at concurrency 1 to 1465 ms at 4,
    and the cascade issues one clause at a time anyway.
    """
    output.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []

    # Weights before the clock starts. A local synthesiser loads on its first
    # call, and charging seven seconds of that to the first utterance puts a
    # number in the manifest that no later utterance can reproduce.
    warm_up = getattr(synthesiser, "load", None)
    if warm_up is not None:
        warm_up()

    for index, row in enumerate(texts):
        path = output / f"{row['id']}.wav"
        started = time.perf_counter()
        failure = None
        pcm = b""
        first_chunk_ms = None
        try:
            utterance = Utterance(text=row["text"], emotion=emotion)
            if stream:
                # The number this mode exists for is the first piece, not the
                # last: a clause that takes two seconds to finish is audible
                # long before it is done, and that gap is the whole reason the
                # streaming path was written.
                async for piece in stream_utterance(synthesiser, utterance):
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - started) * 1000
                    pcm += piece
            else:
                pcm = await synthesiser.synthesise(utterance)
        except Exception as error:  # noqa: BLE001 - one bad sample is data, not the end
            failure = f"{type(error).__name__}: {error}"
        elapsed = (time.perf_counter() - started) * 1000

        if pcm:
            path.write_bytes(to_wav(pcm, OUTPUT_SAMPLE_RATE))
        elif failure is None:
            failure = "the synthesiser returned no audio"

        samples.append(
            {
                "id": row["id"],
                "prompt": row.get("prompt", ""),
                # The fixed text really is the reference here: it is what the
                # synthesiser was told to say, word for word.
                "reference_text": row["text"],
                "audio_path": str(path) if pcm else None,
                "elapsed_ms": elapsed,
                "audio_seconds": len(pcm) / 2 / OUTPUT_SAMPLE_RATE,
                **({} if first_chunk_ms is None else {"first_chunk_ms": first_chunk_ms}),
                **({} if failure is None else {"error": failure}),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(texts)}", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthesiser", required=True, choices=("edge", "voxcpm"))
    parser.add_argument("--texts", type=Path, default=Path("configs/talker_texts_zh_v1.jsonl"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="first N texts only, for a smoke run")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="take the audio piece by piece and record when the first arrived. "
        "The whole-clause time is what a caller waits for today; this is what "
        "it would wait for once the cascade yields pieces",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--emotion",
        default="neutral",
        help="only 'edge' carries it, as prosody; the local one has no instruct "
        "mode and the manifest records that the request had nowhere to go",
    )
    parser.add_argument(
        "--prompt-wav",
        type=Path,
        help="a clip for voxcpm to clone. Without one it draws a speaker per "
        "call, so the voice changes between utterances -- which is what this "
        "flag exists to measure against",
    )
    parser.add_argument("--prompt-text", help="what the clip says, word for word")
    args = parser.parse_args()

    texts = load_texts(args.texts)[: args.limit]
    synthesiser = build_synthesiser(
        args.synthesiser, args.device, args.prompt_wav, args.prompt_text
    )
    print(f"合成 {len(texts)} 条，合成器 {args.synthesiser}")
    print("模型没有参与：参考文本就是固定文本，所以这一轮量的是合成器加判官的底噪")

    samples = asyncio.run(speak_all(synthesiser, texts, args.output, args.emotion, args.stream))
    spoken = [sample for sample in samples if sample.get("audio_path")]
    failed = [sample["id"] for sample in samples if not sample.get("audio_path")]

    manifest: dict[str, Any] = {
        "generated_by": {
            "path": "synthesiser-only",
            "model": f"tts-{args.synthesiser}",
            "components": [{"name": f"tts-{args.synthesiser}"}],
            # The same word generate_speech_samples.py uses for a run the model
            # sat out, so a reader who knows one knows the other.
            "text_source": "fixed",
            "texts_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
            "emotion": args.emotion,
            "emotion_carried": args.synthesiser == "edge",
            # Which voice spoke, by the file it was cloned from. Two voxcpm runs
            # are otherwise indistinguishable in the manifest and one of them
            # has a different speaker in every utterance.
            "voice_prompt": (
                {
                    "path": str(args.prompt_wav),
                    "sha256": hashlib.sha256(args.prompt_wav.read_bytes()).hexdigest(),
                    "text": args.prompt_text,
                }
                if args.prompt_wav
                else None
            ),
        },
        "efficiency": {
            # Present only in streaming mode, and it is the number that decides
            # whether the cascade fits its budget: the whole-clause figure is
            # what a caller waits for today.
            "first_chunk_ms_median": (
                statistics.median(
                    sample["first_chunk_ms"] for sample in spoken if "first_chunk_ms" in sample
                )
                if any("first_chunk_ms" in sample for sample in spoken)
                else None
            ),
            "first_chunk_ms_p95": (
                sorted(sample["first_chunk_ms"] for sample in spoken if "first_chunk_ms" in sample)[
                    int(0.95 * sum("first_chunk_ms" in sample for sample in spoken)) - 1
                ]
                if sum("first_chunk_ms" in sample for sample in spoken) >= 20
                else None
            ),
            "synthesis_ms_median": (
                statistics.median(sample["elapsed_ms"] for sample in spoken) if spoken else None
            ),
            "synthesis_ms_p95": (
                sorted(sample["elapsed_ms"] for sample in spoken)[int(0.95 * len(spoken)) - 1]
                if len(spoken) >= 20
                else None
            ),
            "rtf": (
                sum(sample["elapsed_ms"] for sample in spoken)
                / 1000
                / sum(sample["audio_seconds"] for sample in spoken)
                if spoken and sum(sample["audio_seconds"] for sample in spoken)
                else None
            ),
        },
        "probe_count": len(texts),
        "generated": len(spoken),
        "failed": failed,
        "samples": samples,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    efficiency = manifest["efficiency"]
    print(f"\n成功 {len(spoken)}/{len(texts)}  失败 {failed[:5]}")
    if efficiency["synthesis_ms_median"]:
        print(
            f"合成中位 {efficiency['synthesis_ms_median']:.0f} ms  "
            f"RTF {efficiency['rtf']:.2f}  —— 仅报告，延迟要独占卡"
        )
    print(f"清单 {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
