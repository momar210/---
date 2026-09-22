#!/usr/bin/env python3
"""
Generate the final metric-comparison figure.

This script produces a single PNG that compares four similarity metrics
on synthetic EDS spectra at different intensities:

    1. Cosine similarity       (scale invariant)
    2. Pearson correlation     (scale invariant after centering)
    3. Euclidean distance      (scale sensitive)
    4. Manhattan distance      (scale sensitive)

The figure argues that cosine similarity is the preferred metric for
EDS spectrum matching because it measures the ANGLE between vectors
rather than their absolute magnitude.

Usage:
    python scripts/generate_metric_comparison.py
    python scripts/generate_metric_comparison.py --output diagrams
    python scripts/generate_metric_comparison.py -o out --dpi 200 -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so the script works on headless servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger("metric_comparison")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_OUTPUT_DIR = Path("diagrams")
DEFAULT_FILENAME = "final_metric_comparison.png"
DEFAULT_DPI = 150
DEFAULT_VECTOR_SIZE = 200

# Tolerance used to treat vector norms as zero.
_EPS = 1e-12

COLORS = {
    "header": "#2C3E50",
    "muted": "#7F8C8D",
    "cosine": "#27AE60",
    "cosine_edge": "#1E8449",
    "pearson": "#F39C12",
    "pearson_edge": "#D68910",
    "distance": "#E74C3C",
    "distance_edge": "#C0392B",
    "spec_1": "#3498DB",
    "spec_2": "#27AE60",
    "spec_3": "#E74C3C",
    "spec_4": "#9B59B6",
    "row_cosine": "#E8F8F5",
    "row_pearson": "#FEF9E7",
    "row_distance": "#FDEDEC",
}


# ============================================================================
# METRIC IMPLEMENTATIONS
# ============================================================================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the cosine similarity between two vectors, in [-1, 1]."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < _EPS or norm_b < _EPS:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the Euclidean (L2) distance between two vectors."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b))


def pearson_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the Pearson correlation coefficient, in [-1, 1]."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Guard against constant (zero-variance) vectors.
    if float(np.std(a)) < _EPS or float(np.std(b)) < _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the Manhattan (L1) distance between two vectors."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    return float(np.sum(np.abs(a - b)))


# ============================================================================
# SYNTHETIC SPECTRUM GENERATION
# ============================================================================

def make_base_spectrum(size: int = DEFAULT_VECTOR_SIZE) -> np.ndarray:
    """
    Build a synthetic EDS-like spectrum with four Gaussian peaks.

    Returns:
        A 1-D array of length `size`, normalized to a maximum of 1.0.
    """
    x = np.arange(size)
    peaks: Sequence[Tuple[int, float]] = [
        (30, 0.8),
        (80, 0.5),
        (120, 1.0),
        (160, 0.3),
    ]

    spectrum = np.zeros(size, dtype=np.float64)
    for position, height in peaks:
        spectrum += height * np.exp(-((x - position) ** 2) / 50.0)

    max_val = float(np.max(spectrum))
    if max_val < _EPS:
        raise RuntimeError("Synthetic spectrum is degenerate (all zeros).")
    return spectrum / max_val


# ============================================================================
# FIGURE CONSTRUCTION
# ============================================================================

def _add_title(fig) -> None:
    """Add the top title block to the figure."""
    ax = fig.add_axes([0.05, 0.92, 0.9, 0.06])
    ax.axis("off")

    ax.text(
        0.5, 0.7,
        "Metric Comparison: Why Cosine Similarity Is the Preferred Option",
        fontsize=18, fontweight="bold",
        ha="center", va="center", color=COLORS["header"],
    )
    ax.text(
        0.5, 0.1,
        "Analysis of scale invariance and mineral identification "
        "performance for EDS spectra",
        fontsize=11, ha="center", va="center",
        color=COLORS["muted"], style="italic",
    )


def _add_spectra_panel(fig, spectra: List[np.ndarray], scales: Sequence[float]) -> None:
    """Plot the same mineral at different intensities."""
    ax = fig.add_axes([0.06, 0.62, 0.42, 0.26])

    palette = [
        COLORS["spec_1"],
        COLORS["spec_2"],
        COLORS["spec_3"],
        COLORS["spec_4"],
    ]

    for i, (spectrum, scale) in enumerate(zip(spectra, scales)):
        ax.plot(
            spectrum,
            color=palette[i % len(palette)],
            linewidth=2.5,
            label=f"Scale {scale}\u00d7",
            alpha=0.85,
        )

    ax.set_title(
        "Same Mineral at Different Intensities\n"
        "(Simulating Different EDS Acquisition Times)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Vector dimension (energy)", fontsize=10)
    ax.set_ylabel("Intensity", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, len(spectra[0]))


def _add_metric_table(
    fig,
    ref: np.ndarray,
    spectra: List[np.ndarray],
    scales: Sequence[float],
) -> None:
    """Draw the metric value table with color-coded highlights."""
    ax = fig.add_axes([0.55, 0.62, 0.42, 0.26])
    ax.axis("off")

    rows = []
    for scale, spectrum in zip(scales, spectra):
        rows.append([
            f"{scale}\u00d7",
            f"{cosine_similarity(ref, spectrum):.3f}",
            f"{euclidean_distance(ref, spectrum):.2f}",
            f"{pearson_correlation(ref, spectrum):.3f}",
            f"{manhattan_distance(ref, spectrum):.1f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Scale", "Cosine", "Euclidean", "Pearson", "Manhattan"],
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.18, 0.22, 0.18, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Header row.
    for col in range(5):
        table[(0, col)].set_facecolor(COLORS["header"])
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    # Highlight the cosine column (scale-invariant -> green).
    for row in range(1, len(rows) + 1):
        table[(row, 1)].set_facecolor(COLORS["row_cosine"])

    # Highlight distance metrics (scale-sensitive -> red).
    for row in range(1, len(rows) + 1):
        for col in (2, 4):
            table[(row, col)].set_facecolor(COLORS["row_distance"])

    ax.set_title(
        "Metric Behavior under Changes in Scale",
        fontsize=11, fontweight="bold", pad=10,
    )


def _add_bar_chart(
    fig,
    rect: List[float],
    title: str,
    scales: Sequence[float],
    values: Sequence[float],
    facecolor: str,
    edgecolor: str,
    ylabel: str,
    *,
    ylim: Optional[Tuple[float, float]] = None,
    value_format: str = "{:.2f}",
    annotate_color: Optional[str] = None,
    axhline: Optional[float] = None,
) -> None:
    """Draw a single annotated bar chart panel."""
    ax = fig.add_axes(rect)

    bars = ax.bar(
        scales, values,
        color=facecolor, edgecolor=edgecolor, linewidth=2,
    )

    if axhline is not None:
        ax.axhline(
            y=axhline, color=facecolor, linestyle="--",
            alpha=0.5, linewidth=1.5,
        )

    ax.set_title(title, fontsize=10, fontweight="bold", color=facecolor)
    ax.set_xlabel("Scale factor")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)

    label_color = annotate_color or facecolor
    for bar, value in zip(bars, values):
        ax.annotate(
            value_format.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            color=label_color,
        )


def _add_summary_block(fig, ref: np.ndarray, spectra: List[np.ndarray], scales: Sequence[float]) -> None:
    """Draw the summary table and conclusion banner."""
    ax = fig.add_axes([0.05, 0.02, 0.9, 0.25])
    ax.axis("off")

    cos_vals = [cosine_similarity(ref, s) for s in spectra]
    euc_vals = [euclidean_distance(ref, s) for s in spectra]
    man_vals = [manhattan_distance(ref, s) for s in spectra]
    pear_vals = [pearson_correlation(ref, s) for s in spectra]

    summary_rows = [
        [
            "Cosine Similarity",
            f"CONSTANT ({_span(cos_vals, '.2f')})",
            "scale invariant",
            "RECOMMENDED",
        ],
        [
            "Pearson Correlation",
            f"CONSTANT ({_span(pear_vals, '.2f')})",
            "requires centering",
            "ALTERNATIVE",
        ],
        [
            "Euclidean Distance",
            f"VARIABLE ({_span(euc_vals, '.1f')})",
            "magnitude sensitive",
            "NOT RECOMMENDED",
        ],
        [
            "Manhattan Distance",
            f"VARIABLE ({_span(man_vals, '.0f')})",
            "magnitude sensitive",
            "NOT RECOMMENDED",
        ],
    ]

    row_colors = [
        COLORS["row_cosine"],
        COLORS["row_pearson"],
        COLORS["row_distance"],
        COLORS["row_distance"],
    ]

    table = ax.table(
        cellText=summary_rows,
        colLabels=["Metric", "Response to Scale", "Characteristic", "Recommendation"],
        cellLoc="center",
        loc="upper center",
        colWidths=[0.26, 0.26, 0.24, 0.24],
        bbox=[0.02, 0.35, 0.96, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for col in range(4):
        table[(0, col)].set_facecolor(COLORS["header"])
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    for row, color in enumerate(row_colors, start=1):
        for col in range(4):
            table[(row, col)].set_facecolor(color)

    # Recommendation column colors.
    table[(1, 3)].set_text_props(color=COLORS["cosine"], fontweight="bold")
    table[(2, 3)].set_text_props(color=COLORS["pearson"], fontweight="bold")
    table[(3, 3)].set_text_props(color=COLORS["distance"], fontweight="bold")
    table[(4, 3)].set_text_props(color=COLORS["distance"], fontweight="bold")

    # Footnote.
    ax.text(
        0.5, 0.22,
        "Values shown are the observed range across scale factors "
        f"{scales[0]}\u00d7 to {scales[-1]}\u00d7.",
        fontsize=9, ha="center", va="center",
        color=COLORS["muted"], style="italic",
    )

    # Conclusion banner.
    conclusion = (
        "CONCLUSION: Cosine Similarity is the preferred option because it "
        "measures the ANGLE between vectors rather than absolute distance.\n"
        "This makes it well suited to EDS spectra, where SHAPE is more "
        "important than MAGNITUDE."
    )
    rect = FancyBboxPatch(
        (0.05, 0.0), 0.9, 0.14,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["cosine"],
        alpha=0.15,
        edgecolor=COLORS["cosine"],
        linewidth=2,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)

    ax.text(
        0.5, 0.07, conclusion,
        fontsize=10, ha="center", va="center",
        color=COLORS["cosine_edge"], fontweight="bold",
        transform=ax.transAxes,
    )


def _span(values: Sequence[float], fmt: str) -> str:
    """Return a compact 'min – max' string for a sequence of numbers."""
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < 1e-9:
        return fmt.format(vmax)
    return f"{fmt.format(vmin)} \u2013 {fmt.format(vmax)}"


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_comprehensive_comparison(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    filename: str = DEFAULT_FILENAME,
    dpi: int = DEFAULT_DPI,
    vector_size: int = DEFAULT_VECTOR_SIZE,
) -> Path:
    """
    Generate the comprehensive metric-comparison figure.

    Args:
        output_dir: Directory where the PNG is written.
        filename: Output filename (default `final_metric_comparison.png`).
        dpi: Output resolution.
        vector_size: Length of the synthetic spectra.

    Returns:
        Path to the saved PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # --- Build synthetic spectra -------------------------------------------
    base = make_base_spectrum(size=vector_size)
    scales = [0.5, 1.0, 1.5, 2.0]
    spectra = [base * s for s in scales]
    ref = spectra[1]  # scale 1.0 is the reference

    # --- Assemble the figure -----------------------------------------------
    fig = plt.figure(figsize=(16, 12))

    _add_title(fig)
    _add_spectra_panel(fig, spectra, scales)
    _add_metric_table(fig, ref, spectra, scales)

    # Row 2: bar charts.
    cos_vals = [cosine_similarity(ref, s) for s in spectra]
    pear_vals = [pearson_correlation(ref, s) for s in spectra]
    euc_vals = [euclidean_distance(ref, s) for s in spectra]
    man_vals = [manhattan_distance(ref, s) for s in spectra]

    _add_bar_chart(
        fig, [0.06, 0.32, 0.2, 0.22],
        "Cosine Similarity\n(SCALE INVARIANT)",
        scales, cos_vals,
        facecolor=COLORS["cosine"], edgecolor=COLORS["cosine_edge"],
        ylabel="Similarity", ylim=(0, 1.15),
        value_format="{:.2f}", axhline=1.0,
    )
    _add_bar_chart(
        fig, [0.30, 0.32, 0.2, 0.22],
        "Pearson Correlation\n(SCALE INVARIANT)",
        scales, pear_vals,
        facecolor=COLORS["pearson"], edgecolor=COLORS["pearson_edge"],
        ylabel="Correlation", ylim=(0, 1.15),
        value_format="{:.2f}",
    )
    _add_bar_chart(
        fig, [0.54, 0.32, 0.2, 0.22],
        "Euclidean Distance\n(SCALE SENSITIVE)",
        scales, euc_vals,
        facecolor=COLORS["distance"], edgecolor=COLORS["distance_edge"],
        ylabel="Distance", value_format="{:.1f}",
    )
    _add_bar_chart(
        fig, [0.78, 0.32, 0.2, 0.22],
        "Manhattan Distance\n(SCALE SENSITIVE)",
        scales, man_vals,
        facecolor=COLORS["distance"], edgecolor=COLORS["distance_edge"],
        ylabel="Distance", value_format="{:.0f}",
    )

    # Row 3: summary block.
    _add_summary_block(fig, ref, spectra, scales)

    # --- Save --------------------------------------------------------------
    fig.savefig(
        output_path,
        bbox_inches="tight",
        facecolor="white",
        dpi=dpi,
    )
    plt.close(fig)

    logger.info("Generated: %s", output_path)
    return output_path


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final metric-comparison figure demonstrating "
            "why cosine similarity is preferred for EDS spectra."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the figure is saved.",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="Output filename.",
    )
    parser.add_argument(
        "--dpi",
        type=int, default=DEFAULT_DPI,
        help="Output resolution.",
    )
    parser.add_argument(
        "--vector-size",
        type=int, default=DEFAULT_VECTOR_SIZE,
        help="Number of dimensions in the synthetic spectra.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    """Configure the root logger for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on failure."""
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        path = generate_comprehensive_comparison(
            output_dir=Path(args.output),
            filename=args.filename,
            dpi=args.dpi,
            vector_size=args.vector_size,
        )
        print(f"Generated: {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to generate metric comparison figure.")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())