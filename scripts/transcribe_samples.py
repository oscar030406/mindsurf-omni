"""Transcribe generated audio with an independent recogniser.

The step between generating audio and scoring it, kept separate because it is
the one that must not use our own encoder. Scoring a model with one of its own
components is circular: shared failure modes cancel, and the number flatters
exactly where it should warn.

So this refuses a recogniser whose lineage matches the model under test, and
records which one it used in the output. A CER without that provenance cannot
be checked by anyone reading it later.

Two judges are wired, and which one ran is not a detail. On the same 160 clips
whisper-small read 0.2981 and paraformer-zh read 0.2172 -- a 0.08 gap, larger
than the effect most comparisons here are trying to resolve. So the judge's
name goes into every row rather than only into this script's stdout, and
``evaluate_speech.py`` refuses to compare two runs that were not judged by the
same one.
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


def transcribe_all(
    manifest: dict[str, Any], transcribe: Any, lineage: str, judge: str = ""
) -> list[dict[str, Any]]:
    """Transcribe each sample, recording failures rather than skipping them."""
    model_under_test = " ".join(
        str(component.get("name", ""))
        for component in manifest.get("generated_by", {}).get("components") or []
    )
    verify_independent(lineage, model_under_test)
    # Carried per row rather than in a header, because the rows are what gets
    # merged, re-scored and compared -- a header would be dropped by the first
    # tool that rewrites the file, and measure_naturalness rewrites it.
    stamp = {"judge": judge or lineage}

    rows = []
    for sample in manifest["samples"]:
        audio_path = sample.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            # Recorded as empty rather than omitted: a sample that produced no
            # audio is a failure to measure, not a sample that never existed.
            rows.append({**sample, **stamp, "transcript": "", "transcribed": False})
            continue
        try:
            transcript = transcribe(Path(audio_path))
        except Exception as error:  # noqa: BLE001 - one bad file must not end the run
            rows.append(
                {**sample, **stamp, "transcript": "", "error": str(error), "transcribed": False}
            )
            continue
        rows.append({**sample, **stamp, "transcript": transcript, "transcribed": True})
    return rows


def load_whisper(size: str, language: str) -> tuple[Any, str, str]:
    """OpenAI's recogniser: a different group, different data, different failures."""
    import whisper  # imported here so the module is usable without it installed

    model = whisper.load_model(size)

    def transcribe(path: Path) -> str:
        result = model.transcribe(str(path), language=language, fp16=False)
        return str(result.get("text", "")).strip()

    return transcribe, "whisper", f"whisper-{size}"


def load_paraformer() -> tuple[Any, str, str]:
    """The judge every gating comparison in this project was measured with.

    Mandarin-native and non-autoregressive, so it does not hallucinate a
    fluent sentence over unclear audio the way an autoregressive decoder can --
    which matters here, because the failure being measured is speech that is
    hard to make out. It is also the reason the two judges disagree by 0.08.

    Independent of what we build despite arriving through funasr: FunASR is the
    toolkit, paraformer-zh is not SenseVoice and shares no weights with the
    encoder in our own model.
    """
    from funasr import AutoModel

    # disable_update: the toolkit otherwise reaches out to check for a newer
    # version on construction, which turns an offline run into a stall and,
    # worse, could change the instrument between two runs being compared.
    model = AutoModel(model="paraformer-zh", disable_update=True)

    def transcribe(path: Path) -> str:
        result = model.generate(input=str(path))
        return str(result[0].get("text", "")).strip() if result else ""

    return transcribe, "paraformer", "paraformer-zh"


def refuse_if_the_judge_never_ran(rows: list[dict[str, Any]]) -> None:
    """Zero transcripts is the recogniser failing, not the model going silent.

    Counting a failure as silence is right for a handful of clips and wrong for
    all of them: a judge that never loaded produces a perfect CER of 1.0 and a
    file that looks exactly like a measurement. That happened -- whisper could
    not find ffmpeg on the server, wrote 158 empty transcripts, and said so only
    by setting transcribed:false on every row.
    """
    if not rows or any(row.get("transcribed") for row in rows):
        return
    errors = sorted({str(row.get("error")) for row in rows if row.get("error")})
    raise SystemExit(
        f"nothing transcribed at all ({len(rows)} clips): the judge failed to run rather "
        f"than the model failing to speak, and this output would score as total silence. "
        f"First error: {errors[:1]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="small", help="Whisper size")
    parser.add_argument("--language", default="zh")
    parser.add_argument(
        "--judge",
        choices=("whisper", "paraformer"),
        default="whisper",
        help="paraformer-zh is what every gating comparison here was measured "
        "with; the two disagree by 0.08 CER on the same audio, so an arm judged "
        "by one may not be compared against an arm judged by the other",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    if args.judge == "paraformer":
        transcribe, lineage, judge = load_paraformer()
    else:
        transcribe, lineage, judge = load_whisper(args.model, args.language)

    rows = transcribe_all(manifest, transcribe, lineage=lineage, judge=judge)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    transcribed = sum(1 for row in rows if row.get("transcribed"))
    print(f"判官 {judge}（独立于被测模型）")
    print(f"转写 {transcribed}/{len(rows)}")
    if transcribed < len(rows):
        print(f"未能转写 {len(rows) - transcribed} 条——它们以空转写计入，不被丢弃")
    print(f"输出 {args.output}")

    refuse_if_the_judge_never_ran(rows)


if __name__ == "__main__":
    main()
