"""What VoxCPM's speed costs, and what buying it back costs.

The local synthesiser runs at RTF 0.95-1.00 against edge's 0.08: reading thirty
seconds of dictation aloud spends thirty seconds synthesising it, and streaming
only means keeping up with no margin. Of the three faults on this model, speed
is the one that decides whether it can be used at all, so it goes first.

``VoxCPM.generate`` has two knobs that trade speed for quality and one that
trades it for warm-up, and the service currently takes the default of all
three::

    inference_timesteps=10   diffusion steps; cost is close to linear in this
    cfg_value=2.0            guidance; above 1.0 it is two forward passes a step
    optimize=False           torch.compile, set when the model is built

Measuring them together rather than one at a time would not say which one paid,
so this sweeps them and reports each arm's RTF next to what the arm still
sounds like. The quality reading here is read-back: synthesise, hand the audio
to SenseVoice, compare characters. It answers "is it still the same words",
which is what falls over first when the steps come down. It does **not** answer
"does it still sound like the same person" -- point ``measure_voice_consistency``
at the same directories for that, which is why each arm is written out with the
manifest that script expects.

Timing needs the card to itself. Sharing it with a running service has already
produced a reading in the wrong direction once on this project, so this refuses
to start if the service is answering on the port it is told about::

    python scripts/measure_voxcpm_cost.py \\
        --prompt-wav refs/reference_zh_5s_loud.wav \\
        --prompt-text "您好，请问您有什么需要帮忙的问题吗？" \\
        --timesteps 10 6 4 --cfg 2.0 1.0 \\
        --out <scratch>/voxcpm_cost --report artifacts/voxcpm-cost-2026-08-16.json

``--optimize`` is a whole-run flag, not an arm: torch.compile happens when the
weights are built, so comparing it means comparing two processes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.evaluation.metrics import character_error_rate  # noqa: E402

# The polished form of the four dictated notes, which is what this product
# actually asks the synthesiser to read. Digits, letters and units left in on
# purpose -- they are the part the read-aloud half has to deal with, and the
# part that goes wrong first when the steps come down.
NOTES = [
    "我想说一下明天的安排啊，会议改到下午3点了，因为上午张老师有课，地点还是在B203。",
    "报销的事情，我今天问了一下，说需要发票原件，还要填一个表表在OA系统上能下载，月底之前交上去就行。",
    "方案还行，但是有一个问题就是成本有点高，我们要不要？再考虑一下别的方案。比如说第二个方案也不错。",
    "客户那边催了两次了，说是希望下周能看到demo。我们这边进度大概完成了60%吧，应该来得及。",
]


def refuse_if_the_card_is_busy(base: str) -> None:
    """A timing run that shares the card reads the wrong direction, not a wrong number.

    Measured once already: with the service up, the arm with an extra model in
    it came back faster than the arm without. Both numbers were queueing.
    """
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=2):
            raise SystemExit(
                f"something is answering {base}/health -- stop it before timing, or pass "
                "--base '' if you are certain the card is idle"
            )
    except (urllib.error.URLError, OSError):
        return


def synthesise(
    model: Any, text: str, prompt: tuple[str, str], timesteps: int, cfg: float
) -> tuple[Any, float]:
    started = time.perf_counter()
    waveform = model.generate(
        text=text,
        prompt_wav_path=prompt[0],
        prompt_text=prompt[1],
        normalize=True,
        inference_timesteps=timesteps,
        cfg_value=cfg,
    )
    return waveform, (time.perf_counter() - started) * 1000


def write_arm(directory: Path, rows: list[dict[str, Any]], waveforms: list[Any], rate: int) -> None:
    """The layout ``measure_voice_consistency`` reads, so the timbre number
    comes from the instrument that already exists rather than a second copy."""
    import soundfile

    directory.mkdir(parents=True, exist_ok=True)
    samples = []
    for row, waveform in zip(rows, waveforms, strict=True):
        name = f"{row['id']}.wav"
        soundfile.write(directory / name, waveform, rate)
        # reference_text and audio_seconds are what measure_voice_consistency
        # reads; writing anything else here means that script loads the clips,
        # finds no text, and divides by zero characters.
        samples.append(
            {
                "id": row["id"],
                "reference_text": row["text"],
                "audio_seconds": row["audio_seconds"],
                "audio_path": name,
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps({"samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="openbmb/VoxCPM-0.5B")
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--asr", type=Path, required=True, help="SenseVoiceSmall 目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[10, 6, 4])
    parser.add_argument("--cfg", type=float, nargs="+", default=[2.0])
    parser.add_argument("--optimize", action="store_true", help="torch.compile，整轮的开关")
    parser.add_argument("--repeat", type=int, default=1, help="每条文本跑几遍，取中位")
    parser.add_argument(
        "--texts",
        type=Path,
        help="jsonl，读 polished 字段当要念的文本。不给就用上面那四条笔记——"
        "四条够看 RTF，不够看音色（六对），音色要几十条",
    )
    parser.add_argument("--limit", type=int, default=40, help="配合 --texts")
    parser.add_argument("--base", default="http://127.0.0.1:8099")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()

    if arguments.base:
        refuse_if_the_card_is_busy(arguments.base)

    prompt = (arguments.prompt_wav, arguments.prompt_text)
    texts = NOTES
    if arguments.texts:
        texts = [
            row["polished"]
            for row in (
                json.loads(line)
                for line in arguments.texts.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if row.get("polished", "").strip()
        ][: arguments.limit]
        print(f"{len(texts)} 条文本，来自 {arguments.texts.name}", flush=True)

    import asyncio

    from voxcpm import VoxCPM

    from mindsurf_omni.service.asr import SenseVoiceRecogniser

    print(f"loading {arguments.model_id} (optimize={arguments.optimize})", flush=True)
    model = VoxCPM.from_pretrained(
        arguments.model_id,
        load_denoiser=False,
        optimize=arguments.optimize,
        device=arguments.device,
    )
    recogniser = SenseVoiceRecogniser(model_dir=arguments.asr, device=arguments.device)
    recogniser.load()

    # The first call pays for CUDA context, kernel autotuning and -- with
    # optimize -- the compile. Timing it would charge the whole run for a cost
    # a service pays once at startup.
    print("warm-up", flush=True)
    synthesise(model, texts[0], prompt, max(arguments.timesteps), max(arguments.cfg))

    rate = 16_000
    report: dict[str, Any] = {"optimize": arguments.optimize, "arms": {}}
    for timesteps in arguments.timesteps:
        for cfg in arguments.cfg:
            name = f"t{timesteps}_cfg{cfg:g}" + ("_compiled" if arguments.optimize else "")
            rows, waveforms = [], []
            for index, text in enumerate(texts):
                spent, waveform = [], None
                for _ in range(arguments.repeat):
                    waveform, milliseconds = synthesise(model, text, prompt, timesteps, cfg)
                    spent.append(milliseconds)
                seconds = len(waveform) / rate
                pcm = (waveform.clip(-1.0, 1.0) * 32767).astype("int16").tobytes()
                heard, _ = asyncio.run(recogniser.transcribe(pcm, rate))
                rows.append(
                    {
                        "id": f"note{index + 1}",
                        "text": text,
                        "milliseconds": round(statistics.median(spent), 1),
                        "audio_seconds": round(seconds, 2),
                        "rtf": round(statistics.median(spent) / 1000 / seconds, 3)
                        if seconds
                        else None,
                        "readback": heard,
                        "cer": round(character_error_rate(text, heard, fold_numbers=True), 4),
                    }
                )
                waveforms.append(waveform)
            write_arm(arguments.out / name, rows, waveforms, rate)
            report["arms"][name] = {
                "inference_timesteps": timesteps,
                "cfg_value": cfg,
                "rtf_median": round(statistics.median(r["rtf"] for r in rows), 3),
                "rtf_max": round(max(r["rtf"] for r in rows), 3),
                "cer_median": round(statistics.median(r["cer"] for r in rows), 4),
                "cer_max": round(max(r["cer"] for r in rows), 4),
                "notes": rows,
            }
            block = report["arms"][name]
            print(
                f"{name:<20} RTF 中位 {block['rtf_median']:.3f}（最大 {block['rtf_max']:.3f}）"
                f"  回读 CER 中位 {block['cer_median']:.4f}（最大 {block['cer_max']:.4f}）",
                flush=True,
            )

    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {arguments.report}")


if __name__ == "__main__":
    main()
