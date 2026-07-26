"""How many training samples lose the end of their speech to max_seq_len.

Out-of-range target codes are not padded, flagged, or errored on. The dataset
writes neither the input nor the label past ``max_length``
(``omni_dataset.py`` line 315), so a sequence cap set too low means the ends
of long utterances were never supervised and nothing in the log says so.

The answer this produced is a negative one, which is why the script is kept:
our A2A stages lose target codes on 16.9% and 7.5% of samples where
upstream's 1024 loses none -- but upstream's own T2A line takes the argparse
default of 512 and loses 32.5%, twice our worst, and that is the arm that
reaches CER 0.0763. Truncation is a real deviation and not the cause.

Runs on the training host, CPU only, on a sample of row groups rather than
the whole file: the parquets are 5-6 GB and the point is the magnitude of a
percentage, not its third digit.

    python scripts/measure_truncation.py --data ../dataset/sft_a2a.parquet \
        --minimind-root ~/omni/minimind-o --caps 640 768 1024
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Any

# Every position past the cap is a code that vanishes, so the requirement is
# the assistant's start plus the codebook diagonal plus the frame count.
CODEBOOK_DIAGONAL = 8


def audio_frames(answer_audios: Any) -> int:
    """Frames the Talker must emit: eight interleaved codes each, plus stop."""
    tokens = (
        answer_audios[-1]
        if isinstance(answer_audios, list) and answer_audios and isinstance(answer_audios[0], list)
        else answer_audios
    )
    return (len(tokens) // 8) + 1 if tokens else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="parquet to sample")
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[512, 640, 768, 1024])
    parser.add_argument("--row-groups", type=int, default=3)
    parser.add_argument("--per-group", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--with-input-audio",
        action="store_true",
        help="run the real SenseVoice frontend to count the <|audio_pad|> "
        "positions the input speech occupies. Required for A2A, where they "
        "are most of the prompt; pointless for T2A, which has no input speech",
    )
    args = parser.parse_args()

    root = args.minimind_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(str(root / "model"))
    processor = None
    if args.with_input_audio:
        import contextlib
        import logging

        logging.getLogger().setLevel(logging.ERROR)
        with contextlib.redirect_stdout(io.StringIO()):
            from funasr import AutoModel  # type: ignore[import-not-found]

            loaded = AutoModel(
                model=str(root / "model" / "SenseVoiceSmall"),
                trust_remote_code=True,
                disable_update=True,
                device="cpu",
            )
        from model.model_omni import SenseVoiceAudioProcessor  # type: ignore[import-not-found]

        processor = SenseVoiceAudioProcessor(loaded.kwargs["frontend"].eval())

    handle = pq.ParquetFile(args.data)
    print(f"{args.data}: {handle.num_row_groups} row groups, {handle.metadata.num_rows} rows")
    rng = random.Random(args.seed)
    required = []
    for group in rng.sample(
        range(handle.num_row_groups), min(args.row_groups, handle.num_row_groups)
    ):
        table = handle.read_row_group(group)
        for index in rng.sample(range(table.num_rows), min(args.per_group, table.num_rows)):
            conversations = table["conversations"][index].as_py()
            if isinstance(conversations, str):
                conversations = json.loads(conversations)
            text = "".join(str(turn.get("content", "")) for turn in conversations)
            length = len(tokeniser(text).input_ids)
            if processor is not None:
                length += input_audio_positions(table, index, processor)
            required.append(
                length + CODEBOOK_DIAGONAL + audio_frames(table["answer_audios"][index].as_py())
            )

    values = np.array(required)
    p50, p90, p99 = np.percentile(values, [50, 90, 99])
    print(f"n = {len(values)}")
    print(f"需要的长度  p50 {p50:.0f}  p90 {p90:.0f}  p99 {p99:.0f}  max {values.max():.0f}")
    for cap in args.caps:
        print(f"max_seq_len {cap:4d}: {100.0 * (values > cap).mean():5.1f}% 的样本丢掉目标码")


def input_audio_positions(table: Any, index: int, processor: Any) -> int:
    """One ``<|audio_pad|>`` per SenseVoice frame, so this is the real cost."""
    import numpy as np
    import soundfile as sf

    audios = table["question_audios"][index].as_py()
    if not audios:
        return 0
    raw = audios[-1] if isinstance(audios, list) else audios
    if not raw:
        return 0
    wav, rate = sf.read(io.BytesIO(raw))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if rate != 16000:
        import librosa

        wav = librosa.resample(wav.astype(float), orig_sr=rate, target_sr=16000)
    features = processor(
        wav.astype(np.float32), sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    )
    return int(features.attention_mask.sum().item())


if __name__ == "__main__":
    main()
