r"""The training recipes, checked for the way a shell script lies quietly.

`bash -n` parses. It does not notice a literal backslash-n where a line
continuation was meant, and that is exactly what a careless edit produced here:
two flags collapsed onto one line with `\n` sitting between them as text. The
script parsed, would have run, and would have silently dropped the flag after
it -- in this case `--num_workers`, on a script that launches multi-hour GPU
runs.

Cheap structural checks, because the alternative place to find this is in the
log of a run that has already spent the afternoon.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RECIPES = sorted((Path(__file__).resolve().parent.parent / "scripts").glob("run_*.sh"))
# `--flag "value"` followed by a run of spaces and another flag: the fingerprint
# of a continuation that stopped continuing.
COLLAPSED = re.compile(r'--\S+ "[^"]*"\s{3,}--')


def test_there_are_recipes_to_check() -> None:
    """A glob matching nothing would make everything below vacuously pass."""
    assert RECIPES


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda path: path.name)
def test_no_literal_backslash_n(recipe: Path) -> None:
    for number, line in enumerate(recipe.read_text(encoding="utf-8").splitlines(), 1):
        assert "\n" not in line, f"{recipe.name}:{number} literal backslash-n: {line.strip()}"


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda path: path.name)
def test_no_collapsed_continuation(recipe: Path) -> None:
    for number, line in enumerate(recipe.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        assert not COLLAPSED.search(line), f"{recipe.name}:{number} collapsed: {line.strip()}"
