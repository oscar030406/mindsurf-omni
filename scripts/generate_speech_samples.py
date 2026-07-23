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
    base: str, probes: list[dict[str, str]], output: Path, timeout: float
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
            reply = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": probe["prompt"]}]},
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

            audio = await client.post("/v1/audio/speech", json={"input": text})
            path = output / f"{probe['id']}.wav"
            if audio.status_code == 200:
                path.write_bytes(audio.content)

            samples.append(
                {
                    "id": probe["id"],
                    "prompt": probe["prompt"],
                    # What the model said is the reference for CER: we are
                    # measuring whether the synthesiser said it, not whether
                    # the model answered well.
                    "reference_text": text,
                    "audio_path": str(path) if audio.status_code == 200 else None,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
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
        },
        "probe_count": len(probes),
        "generated": len(samples) - len(failed),
        # Named rather than silently dropped: a shrunken sample size changes
        # every noise floor computed from it.
        "failed": [sample["id"] for sample in failed],
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--probes", type=Path, default=Path("configs/speech_probes_zh_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/speech_samples"))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    probes = load_probes(args.probes)
    print(f"生成 {len(probes)} 条，来自 {args.base}")

    report = asyncio.run(generate(args.base, probes, args.output, args.timeout))

    manifest = args.output / "manifest.json"
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
