"""The tagging arm's inference half.

The trainer imports from here, so the columns the head was trained on and the
columns it is served are one piece of code. Training-only concerns (labelling,
the optimiser, the unfrozen backbone) stay in the script.
"""

from __future__ import annotations

from typing import Any

# How many following tokens the head is allowed to peek at, as input
# embeddings. Two is enough for 就是 / 你知道吧 to be resolvable and keeps the
# head small; it is a knob the ablation can move.
LOOKAHEAD = 2

# Repetition columns are read on characters. Named so a checkpoint can say so
# and inference can refuse a head that was trained on the other unit.
REPETITION_UNIT = "character"


def token_spans(tokeniser: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and the character span each one covers.

    Walked by decoding rather than read from an offset mapping: this tokenizer is
    not a fast one, and a wrong span here mislabels the training data silently.
    """
    ids = list(tokeniser(text).input_ids)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index in range(len(ids)):
        piece = tokeniser.decode(ids[: index + 1], skip_special_tokens=True)
        end = len(piece)
        spans.append((cursor, max(cursor, end)))
        cursor = max(cursor, end)
    return ids, spans


def repetition_features(
    text: str, spans: list[tuple[int, int]], torch: Any, device: str, longest: int
) -> Any:
    """Whether this token overlaps an exact adjacent repetition, read on characters.

    Handed over as a feature because the head cannot compute it: it reads one
    position at a time, so 时间时间 looks like two ordinary words.

    Characters, not token ids. A BPE vocabulary does not cut where a repetition
    starts -- 地点还是还是在B203 tokenises to 地点/还是/还/是在, so the second 还是
    straddles a boundary and the columns stay dark on a repetition that is plain in
    the text.

    Two columns per length: "the next k characters repeat me" and "I repeat the
    previous k", because which copy an alignment marks as deleted is not fixed.
    Overlap is enough to mark a token, and length 1 is included. This is an input,
    where a false positive costs one column the head can learn to discount, not a
    label, where it would cost content.
    """
    out = torch.zeros((len(spans), 2 * longest), device=device)
    for size in range(1, longest + 1):
        opens: set[int] = set()
        closes: set[int] = set()
        for start in range(len(text) - 2 * size + 1):
            if text[start : start + size] == text[start + size : start + 2 * size]:
                opens.update(range(start, start + size))
                closes.update(range(start + size, start + 2 * size))
        for index, (begin, end) in enumerate(spans):
            covered = range(begin, min(end, len(text)))
            if any(position in opens for position in covered):
                out[index, 2 * (size - 1)] = 1.0
            if any(position in closes for position in covered):
                out[index, 2 * (size - 1) + 1] = 1.0
    return out


def feature_width(embedding_size: int, lookahead: int, repetition: int = 0) -> int:
    """How wide one token's row is, so the head is built to match it.

    One statement for both the frozen and unfrozen paths. Computed separately, the
    two widths drift and the mismatch only surfaces at inference.
    """
    return embedding_size * (1 + lookahead) + 2 * repetition


def assemble(
    hidden: Any,
    embeddings: Any,
    text: str,
    spans: list[tuple[int, int]],
    torch: Any,
    device: str,
    lookahead: int,
    repetition: int,
) -> Any:
    """The hidden state, the lookahead embeddings, and the repetition columns.

    Shared by the frozen and unfrozen paths for the reason ``feature_width`` is: a
    second copy drifts, and a head trained without the feature still writes the
    feature's name into its checkpoint.
    """
    pieces = [hidden]
    for step in range(1, lookahead + 1):
        shifted = torch.zeros_like(embeddings)
        if embeddings.shape[0] > step:
            shifted[:-step] = embeddings[step:]
        pieces.append(shifted)
    if repetition:
        pieces.append(repetition_features(text, spans, torch, device, repetition))
    return torch.cat(pieces, dim=-1)


def features(
    model: Any,
    ids: list[int],
    torch: Any,
    device: str,
    lookahead: int,
    repetition: int = 0,
    text: str = "",
    spans: list[tuple[int, int]] | None = None,
) -> Any:
    """One hidden state per token, plus the next tokens' input embeddings.

    ``repetition`` appends the hand-crafted columns, read off ``text`` rather than
    off the ids -- see ``repetition_features``. Zero keeps the old width, so a head
    trained before this existed still loads and still means the same thing.
    """
    if repetition and spans is None:
        raise ValueError(
            "repetition columns are read on characters, so `text` and `spans` are "
            "needed; pass the pair `token_spans` returned"
        )
    tensor = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=tensor, output_hidden_states=True)
        # MiniMind returns the last layer only, as (batch, seq, hidden) --
        # not HF's tuple-per-layer. Handled by shape rather than by trusting
        # either convention: indexing the wrong axis here yields a tensor that
        # still trains and means nothing.
        hidden = out.hidden_states
        hidden = hidden[0] if hidden.dim() == 3 else hidden
        hidden = hidden.float()
        embeddings = model.get_input_embeddings()(tensor)[0].float()
    return assemble(hidden, embeddings, text, spans or [], torch, device, lookahead, repetition)
