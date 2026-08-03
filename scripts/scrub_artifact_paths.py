"""Reduce archived audio paths to filenames before the evidence is committed.

Every row a generation run writes carries the absolute path the clip had on the
machine that made it, and that machine is somebody's home directory. The rule
against shipping absolute paths is not decoration: those strings name a user
and a directory layout, and they are worthless to a reader anyway, because the
clips themselves never enter the repository -- they are gitignored audio.

So the archive keeps the filename, which is what identifies a clip inside its
run, and records once where the clips live. Nothing downstream reads these
paths: scoring reads transcripts and reference text, and a fresh run writes its
own manifest. What re-reading an archive by path would have found is a
directory that only ever existed on one host.

    python scripts/scrub_artifact_paths.py --check      # report, change nothing
    python scripts/scrub_artifact_paths.py artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Matches what the release gate calls a personal path, narrowed to the leading
# component: everything after it is a directory layout we also do not want.
PERSONAL_PREFIXES = ("/home/", "/Users/")
WINDOWS_MARKERS = (":\\Users", ":/Users", ":\\UserData", ":/UserData", ":\\home", ":/home")
# Not an allowlist of fields: the prosody report keys its arms by whatever
# label the run was given, so the field names are unbounded. The rule is
# inverted instead -- rewrite any value that is a personal path, unless it sits
# in a field that holds prose, and unless it contains whitespace. A transcript
# that mentions a directory is evidence and must not be edited; a path never
# has a space in it here.
PROSE_FIELDS = frozenset(
    {
        "reference_text",
        "transcript",
        "text",
        "reply",
        "prompt",
        "why",
        "note",
        "chosen",
        "rejected",
        "left",
        "right",
        "instruction",
    }
)
NOTE_FIELD = "audio_location"
NOTE = (
    "clips stay on the machine that generated them, under that run's output "
    "directory; the archive keeps filenames because the audio itself is not "
    "committed"
)


def is_personal(value: str) -> bool:
    return value.startswith(PERSONAL_PREFIXES) or any(m in value for m in WINDOWS_MARKERS)


def filename(value: str) -> str:
    """Last component under either separator.

    Path(...).name is not portable between the two: on POSIX a backslash is an
    ordinary character, so a Windows path comes back whole. This project has
    already lost a run to that exact difference.
    """
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def scrub(node: Any) -> int:
    """Rewrite personal path fields in place, returning how many changed."""
    changed = 0
    if isinstance(node, dict):
        for key, value in node.items():
            looks_like_path = isinstance(value, str) and not any(ch.isspace() for ch in value)
            if key not in PROSE_FIELDS and looks_like_path and is_personal(value):
                node[key] = filename(value)
                changed += 1
            else:
                changed += scrub(value)
    elif isinstance(node, list):
        for item in node:
            changed += scrub(item)
    return changed


def scrub_json(path: Path, apply: bool) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = scrub(payload)
    if changed and apply:
        if isinstance(payload, dict):
            payload.setdefault(NOTE_FIELD, NOTE)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return changed


def scrub_jsonl(path: Path, apply: bool) -> int:
    rows, changed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        changed += scrub(row)
        rows.append(row)
    if changed and apply:
        body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        path.write_text(body + "\n", encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("artifacts")])
    parser.add_argument("--check", action="store_true", help="report without writing")
    args = parser.parse_args()

    total, touched = 0, 0
    for root in args.roots:
        # rglob on a file yields nothing, so naming one evidence file used to
        # print "0 paths" and exit 0 -- the same output as a clean scan. The
        # documented habit is to check a file the moment it comes off the
        # training host, and that is exactly the call that silently passed.
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if path.suffix not in (".json", ".jsonl") or not path.is_file():
                continue
            handler = scrub_jsonl if path.suffix == ".jsonl" else scrub_json
            changed = handler(path, apply=not args.check)
            if changed:
                touched += 1
                total += changed
                print(f"  {path}: {changed}")

    verb = "会改" if args.check else "已改"
    print(f"{verb} {touched} 个文件、{total} 处路径")
    raise SystemExit(1 if (args.check and total) else 0)


if __name__ == "__main__":
    main()
