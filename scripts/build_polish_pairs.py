"""Training pairs for the polisher: what the recogniser heard, what was meant.

The floor measurement decided the shape of this file. On 160 sentences the
polish job was 762 insertions against 96 substitutions -- **89% of it is
deleting spoken filler**, the recogniser's own error rate is 0.0063, and the
transcript comes back punctuated and broken in the same places as the original
(recall 0.939). So the pairs here are built to teach deletion, and the
correction half rides along rather than being manufactured.

Each pair goes through the production loop rather than being written by hand:
clean text -> filler injected -> edge-tts reads it -> SenseVoice hears it. The
input side is therefore a real transcript, with the recogniser's punctuation
habits and its real confusions (它/他, 地/的) in it, not an imitation of one.

Two things are deliberate and easy to get wrong:

* **Some pairs carry no filler at all.** A model trained only on dirty input
  learns that its job is always to remove something, and then it removes
  content from a clean sentence. The clean fraction is what makes "change
  nothing" a valid answer.
* **A pair whose audio came back garbled is dropped, and the count is
  reported.** The floor run put the median at 0.008 CER, so a pair at 0.3 is a
  synthesis failure rather than a hard example, and training on it teaches the
  polisher to invent.

    python scripts/build_polish_pairs.py --texts artifacts/polish_train/pool.jsonl \
        --asr-dir D:/environment/models/mindsurf-local/SenseVoiceSmall \
        --output artifacts/polish_train/pairs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mindsurf_omni.contract import OUTPUT_SAMPLE_RATE  # noqa: E402
from mindsurf_omni.data.synthesis import EdgeSynthesiser, Utterance  # noqa: E402
from mindsurf_omni.evaluation.metrics import character_error_rate  # noqa: E402
from scripts.inject_fillers import inject  # noqa: E402

# Above this against the sentence that was read aloud, the audio is broken
# rather than the example being hard. The floor run's median was 0.008.
GARBLED = 0.30

PUNCTUATION = "。，、？！；：,.?!;:"


def align_final_punctuation(target: str, heard: str) -> str:
    """Give the target the sentence-final mark the transcript already has.

    Nearly half the pool is user-side prompts written without a final mark
    ("白衬衫领子发黄怎么办"), and SenseVoice correctly writes one ("...怎么办？").
    Left alone, 1464 of 3169 pairs teach the polisher to delete the question
    mark, and a dictation product that eats the punctuation the recogniser got
    right is worse than one that does nothing.

    This does *not* move the main criterion: ``normalise_for_cer`` strips
    punctuation from both sides, so CER never saw it. The reason is the product
    output, not the number -- said explicitly because the first draft of this
    comment claimed the opposite.

    Only the last character, and only when the target has none. Anything more
    would be editing the target, which is the one side of the pair that has to
    stay the sentence a person meant.
    """
    if target and heard and target[-1] not in PUNCTUATION and heard[-1] in PUNCTUATION:
        return target + heard[-1]
    return target


def load_texts(paths: list[Path], limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            # Deduplicated on the text itself: the pool is assembled from
            # several files that overlap, and a duplicated target is a pair the
            # model sees twice for no extra signal.
            if not text or text in seen:
                continue
            seen.add(text)
            rows.append({"id": row.get("id") or f"t{len(rows):06d}", "text": text})
    return rows[:limit] if limit else rows


async def speak_many(
    synthesiser: EdgeSynthesiser, texts: list[str], concurrency: int
) -> list[bytes]:
    """Audio for a batch, several requests at a time.

    Concurrency is fine here and is not in ``speak_texts.py``: that one measures
    what a caller waits for, and this one is a data pipeline where only the
    total matters.
    """
    guard = asyncio.Semaphore(concurrency)

    async def one(text: str) -> bytes:
        async with guard:
            try:
                return await synthesiser.synthesise(Utterance(text=text))
            except Exception:  # noqa: BLE001 - a failed clip is a dropped pair
                return b""

    return list(await asyncio.gather(*(one(text) for text in texts)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asr-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--clean-fraction",
        type=float,
        default=0.15,
        help="share of pairs with nothing injected, so 'change nothing' stays a valid answer",
    )
    parser.add_argument("--rate", type=float, default=0.35, help="filler chance per clause")
    parser.add_argument("--cap", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.05,
        help="share marked val. Split by a hash of the target text, so the same "
        "sentence never lands on both sides of a rerun",
    )
    args = parser.parse_args()

    from mindsurf_omni.service.asr import SenseVoiceRecogniser

    texts = load_texts(args.texts, args.limit)
    if not texts:
        raise SystemExit("文本池是空的")
    print(f"文本池 {len(texts)} 条（已按正文去重）", flush=True)

    recogniser = SenseVoiceRecogniser(model_dir=args.asr_dir, device=args.device)
    synthesiser = EdgeSynthesiser()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = dropped_silent = dropped_garbled = clean_kept = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(texts), args.batch):
            batch = texts[start : start + args.batch]
            spoken_texts, records = [], []
            for row in batch:
                rng = random.Random(f"{args.seed}:{row['id']}")
                # The clean share is drawn per row from the row's own stream, so
                # it does not move when the batch size changes.
                if rng.random() < args.clean_fraction:
                    spoken, injections = row["text"], []
                else:
                    spoken, injections = inject(row["text"], rng, args.rate, args.cap)
                spoken_texts.append(spoken)
                records.append((row, spoken, injections))

            audio = asyncio.run(speak_many(synthesiser, spoken_texts, args.concurrency))
            for (row, spoken, injections), pcm in zip(records, audio, strict=True):
                if not pcm:
                    dropped_silent += 1
                    continue
                heard, language = asyncio.run(recogniser.transcribe(pcm, OUTPUT_SAMPLE_RATE))
                if not heard.strip():
                    dropped_silent += 1
                    continue
                if character_error_rate(spoken, heard, fold_numbers=True) > GARBLED:
                    dropped_garbled += 1
                    continue
                digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                split = "val" if int(digest[:8], 16) / 0xFFFFFFFF < args.holdout else "train"
                handle.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            # What the polisher reads at run time.
                            "source": heard,
                            # What it should write: the sentence before anyone
                            # spoke it, with the sentence-final mark aligned so
                            # the pair is not also about punctuation.
                            "target": align_final_punctuation(row["text"], heard),
                            "spoken": spoken,
                            "injections": injections,
                            "language": language,
                            "split": split,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                clean_kept += not injections
            print(
                f"  {min(start + args.batch, len(texts))}/{len(texts)}  写出 {written}",
                flush=True,
            )

    print(
        f"{written} 对写到 {args.output}\n"
        f"  没出声/空转写丢弃 {dropped_silent}，转写乱掉（CER > {GARBLED}）丢弃 {dropped_garbled}\n"
        f"  其中不含注入的 {clean_kept} 对（{clean_kept / max(1, written):.1%}）"
    )


if __name__ == "__main__":
    main()
