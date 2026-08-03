"""The scrubber has to be portable and narrow, and it has already been neither.

Portable: manifests are written on Windows and archived from Linux, and
Path(...).name reads a whole Windows path as one filename on POSIX. That
difference cost this project a run of 160 clips whose transcripts all came back
empty, so the filename split is tested against both separators.

Narrow: a reference text that happens to mention a directory is prose, not a
path field, and rewriting it would edit evidence. Only the known path fields
move.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.scrub_artifact_paths import filename, is_personal, scrub, scrub_jsonl


def test_a_filename_survives_either_separator() -> None:
    assert filename("/home/oscar/omni/out/zh000.wav") == "zh000.wav"
    assert filename(r"artifacts\tts_edge\zh000.wav") == "zh000.wav"
    assert filename("zh000.wav") == "zh000.wav"


def test_only_personal_roots_count() -> None:
    assert is_personal("/home/oscar/omni/zh000.wav")
    assert is_personal("/Users/someone/clips/zh000.wav")
    assert is_personal(r"D:\UserData\Desktop\run\zh000.wav")
    # A relative archive path is already what we want, and a service path on a
    # shared host names nobody.
    assert not is_personal(r"artifacts\tts_edge\zh000.wav")
    assert not is_personal("/srv/omni/out/zh000.wav")


def test_prose_that_mentions_a_directory_is_left_alone() -> None:
    """Rewriting a transcript would edit the evidence the row exists to hold."""
    row = {
        "audio_path": "/home/oscar/omni/out/zh000.wav",
        "reference_text": "把文件放在 /home/oscar 下面",
        "transcript": "把 文件 放 在 home oscar 下面",
    }

    assert scrub(row) == 1
    assert row["audio_path"] == "zh000.wav"
    assert row["reference_text"] == "把文件放在 /home/oscar 下面"


def test_a_jsonl_archive_is_rewritten_line_for_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"id": "zh000", "audio_path": "/home/oscar/o/zh000.wav"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"id": "zh001", "audio_path": "zh001.wav"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    assert scrub_jsonl(path, apply=True) == 1

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["audio_path"] for row in rows] == ["zh000.wav", "zh001.wav"]
    assert [row["id"] for row in rows] == ["zh000", "zh001"]


def test_check_mode_reports_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    original = json.dumps({"audio_path": "/home/oscar/o/zh000.wav"}, ensure_ascii=False) + "\n"
    path.write_text(original, encoding="utf-8")

    assert scrub_jsonl(path, apply=False) == 1
    assert path.read_text(encoding="utf-8") == original


def test_naming_one_file_scans_it_rather_than_reporting_nothing(tmp_path: Path) -> None:
    """rglob on a file yields nothing, so this call used to pass silently."""
    import subprocess
    import sys

    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        '{"audio_path": "/home/someone/run/zh000.wav"}\n', encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "scripts/scrub_artifact_paths.py", "--check", str(evidence)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout
    assert "1 处路径" in result.stdout
