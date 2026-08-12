"""Checks on the example notebook.

Split the same way as the rest of the suite. The offline tests read the
notebook as JSON and need nothing beyond the standard library, so CI runs them
on every push: they catch the failure mode a notebook actually has, which is
drifting away from the API it documents without anyone noticing. Executing it
needs the live services, so that test is marked ``network`` and deselected by
default -- run it with ``pytest -m network``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

import old_imagery

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "getting-started.ipynb"

# Cell sources may contain IPython magics (`%matplotlib inline`) and shell
# escapes, which are not Python and would fail to parse.
_MAGIC = re.compile(r"^\s*[%!]")
_USAGE = re.compile(r"\bold_imagery\.(\w+)")


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _python(cell: dict) -> str:
    """The cell's source with magic lines blanked out, keeping line numbers."""
    source = "".join(cell["source"])
    return "\n".join("" if _MAGIC.match(line) else line for line in source.splitlines())


def test_notebook_exists_and_is_a_notebook() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert _code_cells(notebook), "the tour has no code left in it"


def test_every_code_cell_parses() -> None:
    for index, cell in enumerate(_code_cells(_notebook())):
        try:
            ast.parse(_python(cell))
        except SyntaxError as exc:  # pragma: no cover - only on a broken notebook
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_notebook_only_uses_the_public_api() -> None:
    """The notebook is documentation, so it must not reach past ``__all__``."""
    used = {
        name
        for cell in _code_cells(_notebook())
        for name in _USAGE.findall("".join(cell["source"]))
    }
    assert used, "the notebook never calls old_imagery"
    unexported = used - set(old_imagery.__all__)
    assert not unexported, (
        f"the notebook uses names that are not exported: {sorted(unexported)}. "
        "Either export them deliberately or stop documenting them."
    )


def test_notebook_covers_every_public_function() -> None:
    """A tour that quietly stops covering a function is a stale tour."""
    source = "".join("".join(cell["source"]) for cell in _code_cells(_notebook()))
    for name in ("availability", "download", "esri_wayback_releases", "esri_mosaic_as_of"):
        assert f"old_imagery.{name}(" in source, f"{name} is no longer demonstrated"


def test_notebook_is_committed_without_outputs() -> None:
    """Executed outputs would embed fetched Google/Esri imagery in this repo.

    The library retrieves imagery but does not license it, so the committed
    notebook carries none: run it locally to see the pictures. Stripping also
    keeps diffs readable and the repository small.
    """
    for index, cell in enumerate(_code_cells(_notebook())):
        assert not cell.get("outputs"), (
            f"code cell {index} has committed outputs. Clear all outputs before "
            "committing (Kernel > Restart & Clear Output, or `nbstripout`)."
        )
        assert cell.get("execution_count") is None, (
            f"code cell {index} has an execution count; clear outputs before committing"
        )


@pytest.mark.network
def test_notebook_runs_end_to_end() -> None:
    """Execute every cell against the live services.

    Slow on a cold cache -- the Esri sections alone issue a few hundred
    metadata queries -- so this is deselected by default like the rest of the
    network suite.
    """
    nbformat = pytest.importorskip("nbformat", reason="pip install -e '.[examples]'")
    nbclient = pytest.importorskip("nbclient", reason="pip install -e '.[examples]'")
    pytest.importorskip("ipykernel", reason="pip install -e '.[examples]'")
    pytest.importorskip("matplotlib", reason="pip install -e '.[examples]'")

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        # Relative links in the notebook resolve from its own directory.
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()

    # Executed in memory only: the file on disk keeps its stripped outputs.
    executed = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in executed)
