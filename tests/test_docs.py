"""Everything the shipped documents name must exist, and match the code.

A README citing a script that was renamed, or quoting a default the contract no
longer carries, is worse than no README: it is read under pressure and sends the
reader somewhere wrong. The public tree carries three documents, so these check
those three and the example client they point at.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
# What a stranger reads: the front page and the two cards that travel with the
# weights and the listening packs.
DOCS = [
    ROOT / "README.md",
    ROOT / "configs" / "release" / "MODEL_CARD.md",
    ROOT / "configs" / "release" / "LISTENING_DATASET_CARD.md",
]
# Plus the one the backend and the frontend read to wire this up. It quotes
# figures inside tables where the qualifier sits in a neighbouring column, so it
# gets the link and script checks but not the bare-number one. It was three
# files until 2026-08-06 -- capabilities and the runbook were folded in, because
# one audience reading three documents kept them out of sync with each other.
SHIPPED = DOCS + [
    ROOT / "docs" / "INTEGRATION.md",
    ROOT / "docs" / "TRAINING.md",
]


@pytest.mark.parametrize("doc", SHIPPED, ids=lambda path: path.name)
def test_every_script_a_document_cites_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")

    missing = [
        script
        for script in re.findall(r"(scripts/[a-z_]+\.py)", text)
        if not (ROOT / script).is_file()
    ]

    assert not missing, f"{doc.name} cites scripts that do not exist: {missing}"


@pytest.mark.parametrize("doc", SHIPPED, ids=lambda path: path.name)
def test_every_repository_path_a_document_links_exists(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    base = doc.parent

    missing = []
    for target in re.findall(
        r"\]\((\.\./[A-Za-z][\w./-]*|[a-z][a-z_/]*(?:\.(?:md|py|json|toml|yml))?)\)", text
    ):
        if not (base / target).resolve().exists():
            missing.append(target)

    assert not missing, f"{doc.name} links to missing files: {missing}"


def test_no_document_quotes_a_number_without_saying_what_it_is_worth() -> None:
    """A bare CER reads as a verdict; ours is reported-only and must say so."""
    for doc in DOCS:
        text = doc.read_text(encoding="utf-8")
        for line in text.splitlines():
            if re.search(r"CER\s*[:：为=]?\s*\d+(\.\d+)?\s*%?", line):
                assert any(
                    marker in line for marker in ("±", "极限", "阈值", "分辨", "假设", "例", "到")
                ), f"{doc.name} quotes a CER without qualifying it: {line.strip()}"


def test_the_a2a_script_stages_the_projector_before_unfreezing() -> None:
    """One-step training lets a badly-aligned projector drag the language weights.

    The loss curve looks healthy while it happens, so the ordering is the only
    thing preventing it.
    """
    script = (ROOT / "scripts" / "run_a2a.sh").read_text(encoding="utf-8")

    # Match the invocations, not any occurrence of the names: the log file is
    # called a2a_full.log and appears near the top.
    projector = script.index('run "a2a_proj"')
    unfrozen = script.index('run "a2a_full"')

    assert projector < unfrozen
    assert "--mode audio_proj" in script[projector:unfrozen]
    assert "--mode" not in script[unfrozen:]  # the second stage trains everything
    # The second stage uses a much smaller rate: the Thinker arrives trained,
    # and a large step here unlearns it.
    assert "--learning_rate 5e-4" in script[projector:unfrozen]
    assert "--learning_rate 2e-5" in script[unfrozen:]


def test_the_a2a_script_does_not_roll_a_failed_stage_into_the_next() -> None:
    """The second stage would train from whatever the first left behind."""
    script = (ROOT / "scripts" / "run_a2a.sh").read_text(encoding="utf-8")

    assert 'exit "$status"' in script


def test_the_example_client_only_uses_endpoints_that_exist() -> None:
    """A copyable example that calls a missing route is worse than no example."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")
    app = (ROOT / "src" / "mindsurf_omni" / "service" / "app.py").read_text(encoding="utf-8")

    called = set(re.findall(r"/v1/[a-z/-]+", example))
    implemented = set(re.findall(r'@app\.(?:get|post|websocket)\("(/v1/[^"]+)"', app))

    assert called <= implemented, f"the example calls: {sorted(called - implemented)}"


def test_the_example_does_not_hardcode_the_sample_rate() -> None:
    """Assuming it is how a client ends up playing 24 kHz audio at 16 kHz."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")

    assert "x-sample-rate" in example
    assert "input_sample_rate" in example  # read from session.created too


def test_the_example_checks_the_licence_before_using_the_output() -> None:
    """It is the first thing a backend author should see."""
    example = (ROOT / "examples" / "minimal_client.py").read_text(encoding="utf-8")

    assert "commercial_use_permitted" in example


def test_the_a2a_guard_matches_a_process_not_a_substring() -> None:
    """A bare `pgrep -f` has produced a false positive here twice.

    Once it matched an ssh command carrying the path as an argument, once a
    dry run still winding down. Both times the guard refused to start a run
    that nothing was actually blocking.
    """
    script = (ROOT / "scripts" / "run_a2a.sh").read_text(encoding="utf-8")

    assert "python.*train_omni" in script  # anchored on the interpreter
    assert "--data_path" in script  # and on an argument only a real run has


def test_the_trainer_can_keep_upstream_talker_shape() -> None:
    """A grafted checkpoint dies silently without this: the loader skips
    mismatched tensors, so the graft evaporates and the run looks healthy."""
    source = (ROOT / "scripts" / "train_omni.py").read_text(encoding="utf-8")

    assert "MINDSURF_TALKER_SHAPE" in source
    # Discrimination has to be by class -- OmniConfig is the Thinker's config,
    # the Talker builds a plain MiniMindConfig.
    assert "type(self) is MiniMindConfig" in source


def test_the_codec_roundtrip_manifest_points_at_the_rebuilt_audio() -> None:
    """A manifest left pointing at the input flatters, and does it silently.

    Downstream reads audio_path. If the round trip rewrites the clips but not
    the manifest, transcription and UTMOS score the originals while the report
    claims to describe the codec ceiling -- and the ceiling comes out equal to
    the input, which is exactly the answer that ends the investigation.
    """
    from scripts.codec_roundtrip import repoint_manifest

    source, target = Path("artifacts/tts_edge"), Path("artifacts/tts_edge_mimi")
    manifest = repoint_manifest(
        {"samples": [{"id": "zh000", "audio_path": str(source / "zh000.wav")}]},
        source,
        target,
        12.5,
    )

    assert manifest["samples"][0]["audio_path"] == str(target / "zh000.wav")
    stamp = manifest["generated_by"]["codec_roundtrip"]
    assert stamp["codec"] == "mimi" and stamp["codebooks"] == 8
    # The stamp has to say a model did not make these, or a reader takes the
    # ceiling for a result.
    assert "not a model reading" in stamp["note"]


def test_the_manifest_rewrite_survives_a_windows_path() -> None:
    r"""This one already fired: 160 clips, every transcript empty.

    Manifests are written on Windows and consumed on Linux. Path(...).name is
    not portable between them -- on POSIX a backslash is an ordinary character,
    so the whole "artifacts\tts_edge\zh000.wav" comes back as the filename,
    the rewritten path names nothing, and the report reads as a model that
    never spoke. Which is exactly what it read as.
    """
    from scripts.codec_roundtrip import basename, repoint_manifest

    assert basename(r"artifacts\tts_edge\zh000.wav") == "zh000.wav"
    assert basename("artifacts/tts_edge/zh000.wav") == "zh000.wav"
    assert basename("zh000.wav") == "zh000.wav"

    target = Path("/srv/omni/codec_out")
    manifest = repoint_manifest(
        {"samples": [{"audio_path": r"artifacts\tts_edge\zh000.wav"}]},
        Path("in"),
        target,
        None,
    )
    assert manifest["samples"][0]["audio_path"] == str(target / "zh000.wav")
