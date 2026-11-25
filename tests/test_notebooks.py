"""
Execute the demo notebook end to end.

``%pip`` lines are removed so the run uses the checked-out package, the
kernel runs in a temporary directory with a headless matplotlib backend,
and Plotly renders to JSON instead of opening a browser. Marked ``docs``
together with the docs snippet tests.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ["Phased_Array_Demo.ipynb"]


@pytest.mark.docs
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_executes(name, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")

    nb = nbformat.read(ROOT / name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = "\n".join(
                line for line in cell.source.splitlines()
                if not line.lstrip().startswith(("%pip", "!pip"))
            )

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PLOTLY_RENDERER", "json")
    client = nbclient.NotebookClient(
        nb, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
