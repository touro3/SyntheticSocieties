"""Tests for scripts/export_figures_hires.py.

Phase 20 — LaTeX paper figure export.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from export_figures_hires import (
    _FIGURE_MAP,
    export_figures,
)

# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def tiny_png(tmp_path: Path) -> Path:
    """Create a minimal 10×10 RGBA PNG using matplotlib."""
    import matplotlib.pyplot as plt

    png = tmp_path / "test_figure.png"
    fig, ax = plt.subplots(figsize=(0.1, 0.1), dpi=10)
    ax.set_facecolor("blue")
    ax.axis("off")
    fig.savefig(str(png), dpi=10)
    plt.close(fig)
    return png


@pytest.fixture()
def source_dir_with_one_figure(tmp_path: Path, tiny_png: Path) -> Path:
    """Source dir containing one known figure."""
    src = tmp_path / "analysis" / "figures"
    src.mkdir(parents=True)
    # Copy as the first entry in _FIGURE_MAP
    first_src_filename = next(iter(_FIGURE_MAP.values()))
    (src / first_src_filename).write_bytes(tiny_png.read_bytes())
    return src


# ── _FIGURE_MAP ──────────────────────────────────────────────────────────


class TestFigureMap:
    def test_map_is_non_empty(self):
        assert len(_FIGURE_MAP) > 0

    def test_all_values_are_png(self):
        for stem, filename in _FIGURE_MAP.items():
            assert filename.endswith(".png"), f"{stem} -> {filename} is not a PNG"

    def test_all_keys_are_strings(self):
        for k in _FIGURE_MAP:
            assert isinstance(k, str)

    def test_key_figures_present(self):
        """Key paper figures must be in the map."""
        required = {
            "phase_c_macro_comparison",
            "grafo_A_ablated",
            "grafo_B_grounded",
            "bad_apple_resilience",
        }
        assert required.issubset(set(_FIGURE_MAP.keys()))


# ── export_figures ────────────────────────────────────────────────────────


class TestExportFigures:
    def test_creates_dest_dir(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "paper" / "figures"
        result = export_figures(src, dest)
        assert dest.exists()

    def test_missing_source_returns_false(self, tmp_path: Path):
        src = tmp_path / "empty_src"
        src.mkdir()
        dest = tmp_path / "dest"
        result = export_figures(src, dest, pdf=True, png=True)
        # All entries should be False (no source PNGs)
        assert all(not v for v in result.values())

    def test_returns_dict_with_all_keys(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        dest = tmp_path / "dest"
        result = export_figures(src, dest)
        assert set(result.keys()) == set(_FIGURE_MAP.keys())

    def test_existing_figure_exports_png(self, tmp_path: Path, source_dir_with_one_figure: Path):
        dest = tmp_path / "dest"
        first_stem = next(iter(_FIGURE_MAP.keys()))
        result = export_figures(source_dir_with_one_figure, dest, dpi=10, pdf=False, png=True)
        assert result[first_stem] is True
        out_png = dest / f"{first_stem}.png"
        assert out_png.exists()
        assert out_png.stat().st_size > 0

    def test_existing_figure_exports_pdf(self, tmp_path: Path, source_dir_with_one_figure: Path):
        dest = tmp_path / "dest"
        first_stem = next(iter(_FIGURE_MAP.keys()))
        result = export_figures(source_dir_with_one_figure, dest, dpi=10, pdf=True, png=False)
        assert result[first_stem] is True
        out_pdf = dest / f"{first_stem}.pdf"
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 0

    def test_no_pdf_no_png_still_returns_true_for_existing(self, tmp_path: Path, source_dir_with_one_figure: Path):
        """With pdf=False and png=False, success is vacuously True for existing files."""
        dest = tmp_path / "dest"
        first_stem = next(iter(_FIGURE_MAP.keys()))
        result = export_figures(source_dir_with_one_figure, dest, dpi=10, pdf=False, png=False)
        # File exists, no export ops triggered → success = True
        assert result[first_stem] is True
