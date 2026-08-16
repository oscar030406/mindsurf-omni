"""Judge each token instead of rewriting the sentence.

Six rounds of generative polishing landed on one curve: every configuration
trades filler removal against content retention, and no point on it satisfies
both lines. The curve is the problem, not the point -- so this changes the
shape of the task rather than the size of the model.

The prior art says the same thing, and says it for our exact situation.
Disfluency removal is classically token-level tagging rather than generation
(the Switchboard line of work). LaserTagger -- "Encode, Tag, Realize"
(EMNLP 2019) -- casts editing as KEEP/DELETE/ADD tags and reports three
properties that all apply here: it beats seq2seq when training examples number
in the thousands rather than the millions, it is far less prone to
hallucination, and it is orders of magnitude faster at inference.

Two deliberate differences from that work:

* **Only KEEP and DELETE.** The floor measurement put the job at 89% deletion
  and 0.6% substitution, and substitution was given up on purpose
  (DECISIONS §16). ADD would reintroduce exactly the freedom that produced the
  invention this whole line is trying to remove.
* **A causal backbone with a bought-in lookahead.** LaserTagger encodes with
  BERT, which sees both sides; the Thinker is causal, and whether 就是 is
  filler or content often depends on what follows. So the head reads the
  hidden state at the token *plus the input embeddings of the next two* --
  right context without touching the backbone.

The backbone is frozen. What trains is one linear layer, and the features are
computed once and reused, so a run is minutes rather than an hour.

    python scripts/train_polish_tagger.py --checkpoint out/sft_merge_768.pth \
        --pairs artifacts/polish_train/pairs_v2.jsonl \
        --minimind-root ~/omni/minimind-o --output out/polish_tagger.pt
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from scripts.train_dpo import load_thinker  # noqa: E402

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


def label_tokens(source: str, target: str, spans: list[tuple[int, int]]) -> list[int]:
    """1 where the token is filler the polisher should drop.

    **Only pure deletions count.** Aligning the transcript against the clean
    original produces two kinds of mismatch: characters the transcript has and
    the original does not (the filler -- learnable, and what we want removed),
    and characters the recogniser simply got wrong (a substitution -- 12.6% of
    the mismatched characters, measured on the training set). Labelling the
    second kind as DELETE teaches two wrong things at once: that a
    misrecognised character should vanish, which loses content, and that a
    tagger can tell which character was misheard, which it cannot -- nothing in
    the input marks it.

    The first version of this file labelled by survival rather than by opcode
    and made exactly that mistake; the tagger's deletion precision sat at 0.398
    with the backbone frozen and at 0.398 with three blocks unfrozen, which is
    what an unlearnable share of the labels looks like.

    Conservative on partial coverage: a token only partly inside a deleted span
    is KEPT, because deleting it would take content with it.
    """
    doomed: set[int] = set()
    matcher = difflib.SequenceMatcher(None, source, target, autojunk=False)
    for tag, start, end, _, _ in matcher.get_opcodes():
        if tag == "delete":
            doomed.update(range(start, end))
    labels = []
    for start, end in spans:
        covered = list(range(start, min(end, len(source))))
        labels.append(1 if covered and all(index in doomed for index in covered) else 0)
    return labels


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


def main_unfrozen(
    args: Any, model: Any, tokeniser: Any, rows: list[dict[str, Any]], tuned: list[Any]
) -> None:
    """The same objective with the top blocks training, so the comparison with
    the generative arm is about the objective rather than about capacity."""
    import random

    import torch

    prepared = []
    for row in rows:
        source = row["source"]
        ids, spans = token_spans(tokeniser, source)
        if not ids or len(ids) > args.max_tokens:
            continue
        prepared.append(
            (row.get("split"), ids, source, spans, label_tokens(source, row["target"], spans))
        )
    train = [item for item in prepared if item[0] != "val"]
    heldout = [item for item in prepared if item[0] == "val"]
    positives = sum(sum(item[-1]) for item in train)
    total = sum(len(item[-1]) for item in train)
    weight = torch.tensor([1.0, (total - positives) / max(1, positives)], device=args.device)
    print(f"训练 {len(train)} 句（删 {positives}/{total} 个 token）", flush=True)

    hidden_size = feature_width(
        model.model.embed_tokens.weight.shape[1], args.lookahead, args.repetition
    )
    head = torch.nn.Linear(hidden_size, 2).to(args.device)
    optimiser = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": args.learning_rate}, {"params": tuned, "lr": 1e-5}]
    )
    generator = random.Random(args.seed)

    def logits_for(ids: list[int], text: str, spans: list[tuple[int, int]]) -> Any:
        tensor = torch.tensor([ids], device=args.device)
        out = model(input_ids=tensor, output_hidden_states=True)
        states = out.hidden_states
        states = states[0] if states.dim() == 3 else states
        embeddings = model.get_input_embeddings()(tensor)[0]
        # Through the shared assembler rather than rebuilt here: this is the
        # copy that drifted, and gradients still flow because `assemble` only
        # concatenates -- the no_grad lives in `features`, not in it.
        return head(
            assemble(
                states.float(),
                embeddings.float(),
                text,
                spans,
                torch,
                args.device,
                args.lookahead,
                args.repetition,
            )
        )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        order = list(range(len(train)))
        generator.shuffle(order)
        running = 0.0
        for step, index in enumerate(order, start=1):
            _, ids, source, spans, labels = train[index]
            loss = torch.nn.functional.cross_entropy(
                logits_for(ids, source, spans),
                torch.tensor(labels, device=args.device),
                weight=weight,
            )
            (loss / 8).backward()
            running += float(loss.detach())
            if step % 8 == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_([*head.parameters(), *tuned], 1.0)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            if step % 400 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(order)} loss {running / step:.4f}", flush=True
                )
        model.eval()
        head.eval()
        true_positive = predicted_positive = actual_positive = 0
        with torch.no_grad():
            for _, ids, source, spans, labels in heldout:
                probability = torch.softmax(logits_for(ids, source, spans), dim=-1)[:, 1]
                predicted = (probability > 0.5).long().tolist()
                for guess, truth in zip(predicted, labels, strict=True):
                    true_positive += guess == 1 and truth == 1
                    predicted_positive += guess == 1
                    actual_positive += truth == 1
        precision = true_positive / max(1, predicted_positive)
        recall = true_positive / max(1, actual_positive)
        history.append(
            {"epoch": epoch, "loss": running / len(order), "precision": precision, "recall": recall}
        )
        print(
            f"epoch {epoch} loss {running / len(order):.4f} "
            f"删除精确 {precision:.3f} 召回 {recall:.3f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in head.state_dict().items()},
            "lookahead": args.lookahead,
            "repetition": args.repetition,
            # Which unit the repetition columns were read on. Heads written
            # before 2026-08-16 compared token ids and carry no such field;
            # loading one now would feed it columns that mean something else at
            # the same width, and nothing would raise. Inference refuses that.
            "repetition_unit": REPETITION_UNIT,
            "hidden": hidden_size,
            "checkpoint": args.checkpoint.name,
            "unfreeze": args.unfreeze,
        },
        str(args.output),
    )
    print(f"写入 {args.output}")
    if args.backbone_output:
        from mindsurf_omni.service.thinker import thinker_weights

        parent = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
        tuned_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        merged = dict(parent)
        for key in thinker_weights(parent):
            if key in tuned_state:
                merged[key] = tuned_state[key]
        args.backbone_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, str(args.backbone_output))
        print(f"主干写入 {args.backbone_output}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "pairs": len(rows),
                    "unfreeze": args.unfreeze,
                    # Which arm this is. The frozen path's report has carried
                    # lookahead since it was written; this one did not, so the
                    # first run with repetition columns produced a report that
                    # could not be told apart from a run without them.
                    "lookahead": args.lookahead,
                    "repetition": args.repetition,
                    "checkpoint": args.checkpoint.name,
                    "pairs_file": args.pairs.name,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"报告 {args.report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer"))
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--variant", default="mindsurf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lookahead", type=int, default=LOOKAHEAD)
    parser.add_argument(
        "--repetition",
        type=int,
        default=0,
        help="append 2N hand-crafted columns marking exact adjacent repetitions "
        "of length 1..N. The head reads one position at a time and cannot "
        "compute this; it clears 0.437 of the injected repetition against the "
        "generative arm's 0.603, and that gap is what the filler line fails on",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--unfreeze",
        type=int,
        default=0,
        help="train the top N transformer blocks along with the head. 0 keeps "
        "the backbone frozen, which is a linear probe -- and a linear probe "
        "against a fully fine-tuned generator is not a comparison of "
        "objectives, it is a comparison of capacity",
    )
    parser.add_argument(
        "--backbone-output",
        type=Path,
        help="where to write the tuned backbone when --unfreeze is used; "
        "inference needs both halves",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    rows = [
        json.loads(line)
        for line in args.pairs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    model, tokeniser = load_thinker(args.checkpoint, args, trainable=args.unfreeze > 0)
    model.eval()
    tuned_parameters: list[Any] = []
    if args.unfreeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for block in model.model.layers[-args.unfreeze :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
                tuned_parameters.append(parameter)
        print(
            f"解冻最后 {args.unfreeze} 层，共 {sum(p.numel() for p in tuned_parameters):,} 个参数",
            flush=True,
        )

    if args.unfreeze:
        main_unfrozen(args, model, tokeniser, rows, tuned_parameters)
        return

    train_x, train_y, val_x, val_y = [], [], [], []
    deleted = kept = 0
    for index, row in enumerate(rows, start=1):
        source = row["source"]
        ids, spans = token_spans(tokeniser, source)
        if not ids or len(ids) > args.max_tokens:
            continue
        labels = label_tokens(source, row["target"], spans)
        matrix = features(
            model, ids, torch, args.device, args.lookahead, args.repetition, source, spans
        ).cpu()
        answers = torch.tensor(labels)
        deleted += int(answers.sum())
        kept += len(labels) - int(answers.sum())
        if row.get("split") == "val":
            val_x.append(matrix)
            val_y.append(answers)
        else:
            train_x.append(matrix)
            train_y.append(answers)
        if index % 200 == 0:
            print(f"  特征 {index}/{len(rows)}", flush=True)

    features_train = torch.cat(train_x).to(args.device)
    labels_train = torch.cat(train_y).to(args.device)
    features_val = torch.cat(val_x).to(args.device)
    labels_val = torch.cat(val_y).to(args.device)
    print(
        f"训练 {features_train.shape[0]} 个 token（删 {int(labels_train.sum())}），"
        f"留出 {features_val.shape[0]} 个；全集删/留 = {deleted}/{kept}",
        flush=True,
    )

    head = torch.nn.Linear(features_train.shape[1], 2).to(args.device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    # The classes are far apart in size -- deletion is a small share of tokens
    # -- and an unweighted loss would sit at "keep everything", which is the
    # arm we already have.
    weight = torch.tensor(
        [1.0, float(labels_train.numel() - labels_train.sum()) / max(1, int(labels_train.sum()))],
        device=args.device,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(head(features_train), labels_train, weight=weight)
        loss.backward()
        optimiser.step()
        head.eval()
        with torch.no_grad():
            probability = torch.softmax(head(features_val), dim=-1)[:, 1]
            predicted = (probability > 0.5).long()
            true_positive = int(((predicted == 1) & (labels_val == 1)).sum())
            precision = true_positive / max(1, int((predicted == 1).sum()))
            recall = true_positive / max(1, int((labels_val == 1).sum()))
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "precision": precision,
                "recall": recall,
            }
        )
        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch} loss {float(loss.detach()):.4f} "
                f"删除精确 {precision:.3f} 召回 {recall:.3f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.cpu() for key, value in head.state_dict().items()},
            "lookahead": args.lookahead,
            "repetition": args.repetition,
            # Which unit the repetition columns were read on. Heads written
            # before 2026-08-16 compared token ids and carry no such field;
            # loading one now would feed it columns that mean something else at
            # the same width, and nothing would raise. Inference refuses that.
            "repetition_unit": REPETITION_UNIT,
            "hidden": features_train.shape[1],
            "checkpoint": args.checkpoint.name,
        },
        str(args.output),
    )
    print(f"写入 {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "pairs": len(rows),
                    "tokens_train": int(features_train.shape[0]),
                    "tokens_val": int(features_val.shape[0]),
                    "delete_share": deleted / max(1, deleted + kept),
                    "lookahead": args.lookahead,
                    "repetition": args.repetition,
                    "history": history,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
