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
# Our own scripts/, imported by path from a test rather than installed.
# Our own scripts/, imported by path from a test rather than installed;
# and vllm, which only ever runs inside the vLLM container image.
VENDORED = {"model", "dataset", "trainer", "build_graded_targets", "inject_fillers", "vllm"}

# Import name to distribution name, where they differ by more than punctuation.
ALIASES = {"whisper": "openai-whisper", "yaml": "pyyaml", "sklearn": "scikit-learn"}


def canonical(name: str) -> str:
    """PEP 503 names: underscore and hyphen are the same distribution.

    Without this the audit reports ``edge_tts`` as undeclared while
    ``edge-tts`` sits in pyproject -- a false alarm that invites the next
    person to silence the whole check.
    """
    return name.lower().replace("_", "-")


def declared_packages() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    names = set()
    for requirement in project.get("dependencies", []) + [
        item for group in project.get("optional-dependencies", {}).values() for item in group
    ]:
        name = requirement.split(">=")[0].split("==")[0].split("[")[0].strip()
        names.add(canonical(name))
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
        if module not in VENDORED and canonical(ALIASES.get(module, module)) not in declared
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


def test_the_container_installs_declared_dependencies_not_a_copied_list() -> None:
    """A hand-written list in the Dockerfile drifts from pyproject silently.

    The drift only shows up as an ImportError inside a running container,
    which is the most expensive place to find it.
    """
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    install_lines = [
        line for line in dockerfile.splitlines() if "pip install" in line and "#" not in line
    ]

    assert install_lines, "the Dockerfile installs nothing"
    for line in install_lines:
        # Installing the project itself picks up whatever pyproject declares.
        # An extra counts: `.[dictation]` still names a set pyproject owns, and
        # the drift this guards against is a package spelled out here.
        unquoted = line.replace('"', "").replace("'", "")
        assert "pip install --no-cache-dir ." in unquoted or "-r " in line, (
            f"packages named by hand in the Dockerfile: {line.strip()}"
        )


def test_the_container_does_not_bake_in_the_weights() -> None:
    """They are 359 MB, carry a non-commercial licence, and version separately."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "VOLUME" in dockerfile
    for pattern in ("COPY release", "COPY out", "COPY weights", "ADD release"):
        assert pattern not in dockerfile


def test_the_container_runs_as_a_non_root_user() -> None:
    """It reads weights and answers HTTP; it writes nothing else."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER omni" in dockerfile
    assert dockerfile.index("USER omni") < dockerfile.index("CMD ")
