"""Export all key paper figures at publication quality.

Phase 20 — LaTeX paper reproducibility.

Regenerates figures at 300 DPI (PNG) and PDF (vector) into paper/figures/.
Run after any simulation update to refresh the publication-quality exports.

Usage:
    python scripts/export_figures_hires.py
    python scripts/export_figures_hires.py --figures-dir paper/figures
    python scripts/export_figures_hires.py --dpi 600
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # non-interactive backend


# ── Key figures referenced in main.tex ─────────────────────────────────
# Maps paper/figures/<dest>.pdf → analysis/figures/<src>.png
_FIGURE_MAP: dict[str, str] = {
    "phase_c_macro_comparison": "phase_c_macro_comparison.png",
    "grafo_A_ablated": "grafo_A_ablated.png",
    "grafo_B_grounded": "grafo_B_grounded.png",
    "bad_apple_resilience": "bad_apple_resilience.png",
    "ladder_ablation": "ladder_ablation.png",
    "lorenz_curves_all": "lorenz_curves_all.png",
    "llm_grounding_comparison": "llm_grounding_comparison.png",
    "calibration_gap": "calibration_gap.png",
    "ess_population_heatmap": "ess_population_heatmap.png",
    "empirical_simulation_results": "empirical_simulation_results.png",
    "results_dashboard": "results_dashboard.png",
    "wealth_stress_trajectories": "wealth_stress_trajectories.png",
    "perturbation_robustness": "perturbation_robustness.png",
    "diversity_collapse": "diversity_collapse.png",
    "ablation_effect": "ablation_effect.png",
}


def _png_to_hires_pdf(
    src_png: Path,
    dest_pdf: Path,
    dpi: int,
) -> bool:
    """Convert a PNG to a publication-quality PDF via matplotlib.

    Reads the PNG as a raster image and embeds it in a tight-layout PDF.
    This preserves the visual appearance while producing a PDF container
    that LaTeX can include as a vector-compatible format.

    Args:
        src_png: Source PNG file.
        dest_pdf: Destination PDF file (will be created/overwritten).
        dpi: Output DPI for rasterisation (300 minimum for publication).

    Returns:
        True on success, False if the source file does not exist.
    """
    if not src_png.exists():
        return False

    img = plt.imread(str(src_png))
    h, w = img.shape[:2]

    fig_w = w / dpi
    fig_h = h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(dest_pdf), format="pdf", dpi=dpi, bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)
    return True


def _copy_hires_png(
    src_png: Path,
    dest_png: Path,
    dpi: int,
) -> bool:
    """Re-save a PNG at the requested DPI via matplotlib.

    Args:
        src_png: Source PNG file.
        dest_png: Destination PNG file.
        dpi: Output DPI.

    Returns:
        True on success, False if the source file does not exist.
    """
    if not src_png.exists():
        return False

    img = plt.imread(str(src_png))
    h, w = img.shape[:2]

    fig_w = w / dpi
    fig_h = h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(dest_png), format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)
    return True


def export_figures(
    source_dir: Path,
    dest_dir: Path,
    dpi: int = 300,
    pdf: bool = True,
    png: bool = True,
) -> dict[str, bool]:
    """Export all key paper figures to the destination directory.

    Args:
        source_dir: Directory containing source PNG figures (analysis/figures/).
        dest_dir: Output directory for publication-quality figures (paper/figures/).
        dpi: Dots per inch for output (300 = minimum publication quality).
        pdf: Whether to export PDF versions.
        png: Whether to export high-DPI PNG versions.

    Returns:
        Dict mapping figure name → success status.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    for stem, src_filename in _FIGURE_MAP.items():
        src_path = source_dir / src_filename

        if not src_path.exists():
            results[stem] = False
            continue

        success = True

        if pdf:
            dest_pdf = dest_dir / f"{stem}.pdf"
            ok = _png_to_hires_pdf(src_path, dest_pdf, dpi=dpi)
            success = success and ok

        if png:
            dest_png = dest_dir / f"{stem}.png"
            ok = _copy_hires_png(src_path, dest_png, dpi=dpi)
            success = success and ok

        results[stem] = success

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export publication-quality figures for the BGF LaTeX paper."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("analysis/figures"),
        help="Directory containing source PNG figures.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for publication-quality figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI (default: 300, minimum for publication).",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF export (export PNG only).",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG export (export PDF only).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    results = export_figures(
        source_dir=args.source_dir,
        dest_dir=args.figures_dir,
        dpi=args.dpi,
        pdf=not args.no_pdf,
        png=not args.no_png,
    )

    ok = sum(v for v in results.values())
    total = len(results)
    missing = [k for k, v in results.items() if not v]

    print(f"Exported {ok}/{total} figures to {args.figures_dir} at {args.dpi} DPI.")
    if missing:
        print(f"Missing source files ({len(missing)}): {', '.join(missing)}")
        print("Run the simulation pipeline first to generate these figures.")
