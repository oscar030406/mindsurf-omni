"""Produce the audio the evaluation harness scores.

Kept apart from scoring on purpose. Generation needs a GPU and the model;
scoring needs an independent recogniser and none of the above. Splitting them
means the expensive half runs once and the cheap half can be re-run whenever a
metric changes, against exactly the same audio.

The manifest records which path produced the audio and which model. A report
that cannot say whether it measured the native or the cascade path has measured
nothing, and by the time someone asks, the service has usually been
reconfigured.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_probes(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no probes in {path}")
    return rows


async def generate(
    base: str,
    probes: list[dict[str, str]],
    output: Path,
    timeout: float,
    text_source: str,
    sampling: dict[str, float],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        models = (await client.get("/v1/models")).json().get("data", [])
        if not models:
            raise SystemExit("the service reports no model; set MINDSURF_ENGINE before generating")
        served = models[0]

        for index, probe in enumerate(probes):
            started = time.perf_counter()
            if text_source == "probe":
                # Nothing is asked of the model: the probe text is spoken as
                # written, so the CER that comes back is the floor the
                # synthesiser and the judge contribute between them. Recorded
                # in the manifest, because a floor read as a model score would
                # be the most flattering wrong number this project could print.
                text = probe["prompt"]
            else:
                # Sent explicitly rather than left to the server default, so
                # what the manifest records is what was used rather than what
                # this script assumed the server would do.
                reply = await client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": probe["prompt"]}],
                        **sampling,
                    },
                )
                if reply.status_code != 200:
                    samples.append(
                        {
                            "id": probe["id"],
                            "prompt": probe["prompt"],
                            "error": f"chat returned {reply.status_code}",
                        }
                    )
                    continue
                text = reply.json()["choices"][0]["message"]["content"]

            path = output / f"{probe['id']}.wav"
            spoken = False
            try:
                audio = await client.post("/v1/audio/speech", json={"input": text})
                if audio.status_code == 200:
                    path.write_bytes(audio.content)
                    spoken = True
                else:
                    failure = f"speech returned {audio.status_code}"
            except Exception as error:  # noqa: BLE001 - one bad sample is data
                # Synthesis is a streaming response, so a fault after the
                # headers arrives as a truncated body rather than a status.
                # Letting it propagate ends the run at whichever sample hit it
                # and discards every sample already generated -- the expensive
                # half, and the one this script exists to run only once.
                failure = f"{type(error).__name__}: {error}"

            samples.append(
                {
                    "id": probe["id"],
                    "prompt": probe["prompt"],
                    # What the model said is the reference for CER: we are
                    # measuring whether the synthesiser said it, not whether
                    # the model answered well.
                    "reference_text": text,
                    "audio_path": str(path) if spoken else None,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    **({} if spoken else {"error": failure}),
                }
            )
            if (index + 1) % 10 == 0:
                print(f"  {index + 1}/{len(probes)}", flush=True)

    failed = [sample for sample in samples if "error" in sample or not sample.get("audio_path")]
    return {
        "generated_by": {
            "path": served.get("path"),
            "model": served.get("id"),
            "components": served.get("components"),
            "licence": served.get("licence"),
            # "probe" means the model never spoke: this run measures the
            # instrument, not the model. Downstream reports carry it forward.
            "text_source": text_source,
            # Two runs at different temperatures are two different measurements
            # and would otherwise be indistinguishable in the artifacts.
            "sampling": None if text_source == "probe" else sampling,
        },
        "probe_count": len(probes),
        "generated": len(samples) - len(failed),
        # Named rather than silently dropped: a shrunken sample size changes
        # every noise floor computed from it.
        "failed": [sample["id"] for sample in failed],
        "samples": samples,
    }


async def generate_realtime(
    base: str,
    probes: list[dict[str, str]],
    output: Path,
    timeout: float,
    stimulus: Path,
) -> dict[str, Any]:
    """Speech in, speech out, over the WebSocket -- the native path's own shape.

    The cascade is measured by asking for a reply and then asking for it to be
    spoken. The native path has no second step: one turn produces the words and
    the sound together, and it cannot read text it did not write. So it is
    driven the way it is meant to be driven, with audio, and the stimulus is a
    fixed set of spoken probes rather than something synthesised per run --
    the input has to be identical across runs for the outputs to be comparable.
    """
    import base64

    import soundfile
    import websockets

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from mindsurf_omni.contract import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE
    from mindsurf_omni.service.audio import resample, to_wav

    output.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []

    async with httpx.AsyncClient(base_url=base, timeout=timeout) as client:
        models = (await client.get("/v1/models")).json().get("data", [])
        if not models:
            raise SystemExit("the service reports no model; set MINDSURF_ENGINE before generating")
        served = models[0]

    parsed = urlparse(base)
    url = f"{'wss' if parsed.scheme == 'https' else 'ws'}://{parsed.netloc}/v1/realtime"

    for index, probe in enumerate(probes):
        spoken_probe = stimulus / f"{probe['id']}.wav"
        if not spoken_probe.is_file():
            samples.append(
                {
                    "id": probe["id"],
                    "prompt": probe["prompt"],
                    "error": f"no stimulus audio at {spoken_probe}",
                }
            )
            continue

        heard, rate = soundfile.read(spoken_probe, dtype="int16")
        pcm = resample(heard.tobytes(), rate, INPUT_SAMPLE_RATE)
        path = output / f"{probe['id']}.wav"
        started = time.perf_counter()
        text, audio, failure = "", bytearray(), None

        try:
            async with websockets.connect(url, max_size=None, open_timeout=timeout) as socket:
                await socket.recv()  # session.created
                await socket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm).decode("ascii"),
                        }
                    )
                )
                await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
                while True:
                    event = json.loads(await asyncio.wait_for(socket.recv(), timeout=timeout))
                    kind = event.get("type")
                    if kind == "response.text.delta":
                        text += event.get("delta", "")
                    elif kind == "response.audio.delta":
                        audio += base64.b64decode(event.get("audio", ""))
                    elif kind == "error":
                        failure = str(event.get("error", {}).get("message"))[:200]
                        break
                    elif kind == "response.done":
                        break
        except Exception as error:  # noqa: BLE001 - one bad turn is data
            failure = f"{type(error).__name__}: {error}"

        if audio and failure is None:
            path.write_bytes(to_wav(bytes(audio), OUTPUT_SAMPLE_RATE))
        elif failure is None:
            failure = "the turn produced no audio"

        samples.append(
            {
                "id": probe["id"],
                "prompt": probe["prompt"],
                # The model's own words, spoken by the model itself -- so on
                # this path CER really is about the model, not a synthesiser.
                "reference_text": text,
                "audio_path": str(path) if (audio and failure is None) else None,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                **({} if failure is None else {"error": failure}),
            }
        )
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(probes)}", flush=True)

    failed = [sample for sample in samples if "error" in sample or not sample.get("audio_path")]
    return {
        "generated_by": {
            "path": served.get("path"),
            "model": served.get("id"),
            "components": served.get("components"),
            "licence": served.get("licence"),
            "text_source": "model",
            "sampling": None,  # the realtime session carries its own defaults
            "stimulus": str(stimulus),
        },
        "probe_count": len(probes),
        "generated": len(samples) - len(failed),
        "failed": [sample["id"] for sample in failed],
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--probes", type=Path, default=Path("configs/speech_probes_zh_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/speech_samples"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="512 tokens of Chinese is around 1100 characters, or 75 seconds "
        "spoken as one turn. The judge stops early on audio that long when it "
        "is repetitive, and the missing text is charged to the synthesiser as "
        "deletions -- so a run that means to measure CER should cap this",
    )
    parser.add_argument(
        "--via",
        choices=("http", "realtime"),
        default="http",
        help="'realtime' drives /v1/realtime with spoken probes, which is the only "
        "way to measure the native path: it produces words and sound in one turn "
        "and cannot read text it did not write",
    )
    parser.add_argument(
        "--stimulus",
        type=Path,
        help="directory of <probe id>.wav to speak at the model, for --via realtime; "
        "a fixed set, so two runs differ by the model and not by the input",
    )
    parser.add_argument(
        "--text-source",
        choices=("model", "probe"),
        default="model",
        help="'probe' speaks the probe text as written instead of asking the "
        "model, which measures what the synthesiser and the judge cost between "
        "them -- the floor any model score has to be read against",
    )
    args = parser.parse_args()

    probes = load_probes(args.probes)
    print(f"生成 {len(probes)} 条，来自 {args.base}")
    if args.text_source == "probe":
        print("文本来自探针，模型没有参与——这一轮量的是仪器底噪，不是模型")

    if args.via == "realtime":
        if args.stimulus is None:
            raise SystemExit("--via realtime needs --stimulus pointing at spoken probes")
        _write(
            asyncio.run(
                generate_realtime(args.base, probes, args.output, args.timeout, args.stimulus)
            ),
            args.output,
        )
        return

    report = asyncio.run(
        generate(
            args.base,
            probes,
            args.output,
            args.timeout,
            args.text_source,
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
        )
    )

    _write(report, args.output)


def _write(report: dict[str, Any], output: Path) -> None:
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n路径 {report['generated_by']['path']}")
    print(f"成功 {report['generated']}/{report['probe_count']}")
    if report["failed"]:
        print(f"失败 {len(report['failed'])} 条: {report['failed'][:5]}")
    print(f"清单 {manifest}")


if __name__ == "__main__":
    main()
