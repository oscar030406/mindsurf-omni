"""Transcribe generated audio with an independent recogniser.

The step between generating audio and scoring it, kept separate because it is
the one that must not use our own encoder. Scoring a model with one of its own
components is circular: shared failure modes cancel, and the number flatters
exactly where it should warn.

So this refuses a recogniser whose lineage matches the model under test, and
records which one it used in the output. A CER without that provenance cannot
be checked by anyone reading it later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Recognisers considered independent of what we build. Whisper is trained by a
# different group on different data; SenseVoice is ours and is not on the list.
INDEPENDENT = {"whisper", "whisper-large", "paraformer"}


class JudgeError(RuntimeError):
    """Raised rather than falling back to a recogniser that cannot judge."""


def verify_independent(lineage: str, model_under_test: str) -> None:
    if lineage not in INDEPENDENT:
        raise JudgeError(
            f"{lineage!r} is not on the independent list {sorted(INDEPENDENT)}; "
            "scoring a model with a component of that model is circular"
        )
    if lineage in model_under_test:
        raise JudgeError(
            f"the judge {lineage!r} appears in the model under test "
            f"{model_under_test!r}; their failure modes would cancel"
        )


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("samples"):
        raise SystemExit(f"{path} lists no samples")
    return manifest  # type: ignore[no-any-return]


def transcribe_all(manifest: dict[str, Any], transcribe: Any, lineage: str) -> list[dict[str, Any]]:
    """Transcribe each sample, recording failures rather than skipping them."""
    model_under_test = " ".join(
        str(component.get("name", ""))
        for component in manifest.get("generated_by", {}).get("components") or []
    )
    verify_independent(lineage, model_under_test)

    rows = []
    for sample in manifest["samples"]:
        audio_path = sample.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            # Recorded as empty rather than omitted: a sample that produced no
            # audio is a failure to measure, not a sample that never existed.
            rows.append({**sample, "transcript": "", "transcribed": False})
            continue
        try:
            transcript = transcribe(Path(audio_path))
        except Exception as error:  # noqa: BLE001 - one bad file must not end the run
            rows.append({**sample, "transcript": "", "error": str(error), "transcribed": False})
            continue
        rows.append({**sample, "transcript": transcript, "transcribed": True})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="small", help="Whisper size")
    parser.add_argument("--language", default="zh")
    args = parser.parse_args()

    import whisper  # imported here so the module is usable without it installed

    manifest = load_manifest(args.manifest)
    model = whisper.load_model(args.model)

    def transcribe(path: Path) -> str:
        result = model.transcribe(str(path), language=args.language, fp16=False)
        return str(result.get("text", "")).strip()

    rows = transcribe_all(manifest, transcribe, lineage="whisper")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    transcribed = sum(1 for row in rows if row.get("transcribed"))
    print(f"判官 whisper-{args.model}（独立于被测模型）")
    print(f"转写 {transcribed}/{len(rows)}")
    if transcribed < len(rows):
        print(f"未能转写 {len(rows) - transcribed} 条——它们以空转写计入，不被丢弃")
    print(f"输出 {args.output}")


if __name__ == "__main__":
    main()
