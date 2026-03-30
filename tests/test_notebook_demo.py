"""Notebook smoke test — gated behind @pytest.mark.notebook."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "bgf_demo.ipynb"


@pytest.mark.notebook
def test_notebook_executes_without_error(tmp_path):
    """Execute bgf_demo.ipynb end-to-end and assert no cell raises an error."""
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        pytest.skip("nbformat / nbconvert not installed")

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(PROJECT_ROOT)}})
    # If no exception was raised, all cells executed successfully.


def test_notebook_file_exists():
    """Non-gated: verify the notebook file is present and valid JSON."""
    import json

    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)
    assert nb.get("nbformat") == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 5
