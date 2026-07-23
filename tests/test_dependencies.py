"""Every external import must be declared somewhere.

CI caught this the hard way: pyarrow was imported by the data loader and named
in no dependency set, so it worked on every machine that happened to have it
and failed on the clean one. The audit is cheap enough to run as a test, so a
future gap fails here rather than on a fresh checkout.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Provided by MiniMind-O's checkout at run time, not by a package. Imported
# inside functions so that importing our modules never requires them.
VENDORED = {"model", "dataset", "trainer"}

# Import name to distribution name, where they differ.
ALIASES = {"whisper": "openai-whisper", "yaml": "pyyaml"}


def declared_packages() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    names = set()
    for requirement in project.get("dependencies", []) + [
        item for group in project.get("optional-dependencies", {}).values() for item in group
    ]:
        name = requirement.split(">=")[0].split("==")[0].split("[")[0].strip()
        names.add(name.lower())
    return names


def imported_packages() -> dict[str, str]:
    """External top-level imports, mapped to one file that uses each."""
    found: dict[str, str] = {}
    for directory in ("src", "tests", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    modules = [node.module.split(".")[0]]
                for module in modules:
                    if (
                        module in sys.stdlib_module_names
                        or module in {"mindsurf_omni", "scripts", "tests"}
                        or module.startswith("_")
                    ):
                        continue
                    found.setdefault(module, str(path.relative_to(ROOT)))
    return found


def test_every_import_is_declared_or_explicitly_vendored() -> None:
    declared = declared_packages()
    undeclared = {
        module: source
        for module, source in imported_packages().items()
        if module not in VENDORED and ALIASES.get(module, module).lower() not in declared
    }

    assert not undeclared, "imported but declared nowhere: " + ", ".join(
        f"{module} (in {source})" for module, source in sorted(undeclared.items())
    )


def test_the_runtime_set_stays_small() -> None:
    """It is what goes into the container, and every entry is something to patch.

    torch alone is about 2.5 GB; a server that only answers HTTP should not
    carry it.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = {
        requirement.split(">=")[0].split("[")[0].strip().lower()
        for requirement in data["project"]["dependencies"]
    }

    assert "torch" not in runtime
    assert "pyarrow" not in runtime  # training data, not request handling
    assert len(runtime) <= 8


def test_vendored_modules_are_imported_lazily() -> None:
    """MiniMind-O's modules exist only inside its checkout.

    A module-level import of them would make our package unimportable
    anywhere else -- including in this test run.
    """
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not (set(names) & VENDORED), (
                f"{path.name} imports a vendored module at top level"
            )
