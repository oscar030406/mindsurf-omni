"""The tagging arm's inference half.

Split out of ``scripts/train_polish_tagger.py`` on 2026-08-16, when the service
started running the tagger: a script may import from the service, and the
service may not import from a script. The trainer imports these from here, so
the columns the head was trained on and the columns it is served are one piece
of code -- the drift that hid a broken ``--repetition`` for a whole round
started as exactly this kind of second copy.

Training-only concerns (labelling, the optimiser, the unfrozen backbone) stay
in the script.
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

    Walked by decoding rather than read from an offset mapping: this tokenizer
    is not a fast one, and a wrong span here silently mislabels the training
    data -- which is the failure that cannot be seen in a loss curve.
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

    Handed over as a feature because the head cannot compute it. It reads one
    position at a time, so 时间时间 looks like two ordinary words -- measured,
    the tagger clears 0.437 of the injected repetition against the generative
    arm's 0.603, and that gap is the whole reason the filler-clearance line is
    still failing. Everything else it needs is in the hidden state; this is not.

    **Characters, not token ids.** The first version compared token ids, and a
    BPE vocabulary does not cut where a repetition starts: 地点还是还是在B203
    tokenises to 地点 / 还是 / 还 / 是在, so the second 还是 is split across a
    boundary with 在 and the columns stayed dark on a repetition that is plain
    in the text. Measured on the four dictated notes -- 应该应该 and 这边这边
    tokenise cleanly and were removed, 还是还是 did not and survived every arm.
    Reading the characters and projecting onto the spans lights all of them.

    The literature does not compute this on subwords either: disfluency is
    annotated on words, and subword tokens inherit their parent word's label
    (Kumar et al., ACL 2026, doi 10.18653/v1/2026.acl-long.2137).

    Two columns per length: "the next k characters repeat me" and "I repeat the
    previous k". Both, because which copy the alignment marks as deleted is not
    fixed for two identical spans, and the head should be free to learn either.

    A token overlapping a marked span is marked. It is an input, not a label --
    the conservative "only if wholly covered" rule belongs to `label_tokens`,
    where being wrong costs content; being wrong here costs the head one column
    it can learn to discount.

    Length 1 is included, unlike the merge rule's exemption: there the cost of a
    false positive is keeping a deletion, here it is one input column.
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

    Stated once. The unfrozen path used to compute this itself and left the
    repetition columns out of the arithmetic, which is half of why
    ``--repetition`` never worked.
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

    Shared by the frozen and unfrozen paths rather than written twice. It was
    written twice, and the two copies drifted: the frozen one grew the
    repetition columns and the unfrozen one did not, so ``--repetition 3``
    trained a head that had never seen the feature, wrote ``repetition: 3``
    into the checkpoint anyway, and then failed at inference with a shape
    mismatch -- 2310 columns arriving at a 2304-wide head. Nothing caught it
    because nothing had ever run the flag end to end.
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

    ``repetition`` appends the hand-crafted repetition columns, which are read
    off ``text`` rather than off the ids -- see ``repetition_features``. Zero
    keeps the old width, so a head trained before this existed still loads and
    still means the same thing.
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
