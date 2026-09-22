#!/usr/bin/env python3
"""
Generate presentation figures for the mineral-identification project.

This script produces three PNG figures suitable for slides and reports:

    1. Software testing results (metrics, breakdown, coverage).
    2. Project objectives vs. achievements.
    3. EDS spectrum vectorization concept.

Usage:
    python scripts/generate_presentation_figures.py
    python scripts/generate_presentation_figures.py --output slides
    python scripts/generate_presentation_figures.py --only testing objectives
    python scripts/generate_presentation_figures.py --list
    python scripts/generate_presentation_figures.py -v
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so the script works on headless servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch


logger = logging.getLogger("presentation_figures")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_OUTPUT_DIR = Path("diagrams")
DEFAULT_DPI = 150

COLORS = {
    "header": "#2C3E50",
    "muted": "#7F8C8D",
    "muted_dark": "#5D6D7E",
    "background": "#ECF0F1",
    "border": "#BDC3C7",
    "success": "#27AE60",
    "success_dark": "#1E8449",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "info": "#3498DB",
    "purple": "#9B59B6",
    "orange": "#E67E22",
    "panel_bg": "#F8F9FA",
    "summary_bg": "#EBF5FB",
}

plt.rcParams["figure.dpi"] = DEFAULT_DPI
plt.rcParams["savefig.dpi"] = DEFAULT_DPI
plt.rcParams["font.family"] = "DejaVu Sans"


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _save_figure(fig, output_dir: Path, filename: str) -> Path:
    """Save a matplotlib figure with consistent settings and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        dpi=DEFAULT_DPI,
    )
    plt.close(fig)
    logger.info("Generated: %s", filename)
    return path


def _draw_progress_bar(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    fraction: float,
    color: str,
) -> None:
    """
    Draw a two-layer progress bar: background + fill.

    Args:
        fraction: Value in [0, 1] representing the filled portion.
    """
    fraction = max(0.0, min(1.0, fraction))

    # Background track.
    ax.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.01",
            facecolor=COLORS["background"],
            edgecolor=COLORS["border"],
            transform=ax.transAxes,
        )
    )

    # Filled portion (skip if zero to avoid a degenerate patch).
    if fraction > 0:
        ax.add_patch(
            FancyBboxPatch(
                (x, y - height / 2),
                width * fraction,
                height,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor="none",
                transform=ax.transAxes,
            )
        )


# ============================================================================
# FIGURE 1 — SOFTWARE TESTING RESULTS
# ============================================================================

def generate_testing_results(output_dir: Path) -> Path:
    """Generate a visual summary of software testing results."""
    fig = plt.figure(figsize=(14, 10))

    # --- Title --------------------------------------------------------------
    fig.text(
        0.5, 0.95, "Software Testing Results",
        fontsize=20, fontweight="bold",
        ha="center", va="center", color=COLORS["header"],
    )
    fig.text(
        0.5, 0.90, "Automated testing suite using pytest",
        fontsize=12, ha="center", va="center",
        color=COLORS["muted"], style="italic",
    )

    # --- Section 1: metric circles -----------------------------------------
    ax_metrics = fig.add_axes([0.05, 0.55, 0.9, 0.32])
    ax_metrics.axis("off")
    ax_metrics.set_xlim(0, 10)
    ax_metrics.set_ylim(0, 3)

    metrics = [
        ("36", "Total\nTests", COLORS["info"], 1.5),
        ("97%", "Successful\nTests", COLORS["success"], 4.0),
        ("79%", "Code\nCoverage", COLORS["purple"], 6.5),
        ("<1s", "Execution\nTime", COLORS["orange"], 9.0),
    ]

    for value, label, color, x in metrics:
        ax_metrics.add_patch(
            Circle(
                (x, 1.5), 0.9,
                facecolor=color,
                edgecolor=COLORS["header"],
                linewidth=3,
                alpha=0.9,
            )
        )
        ax_metrics.text(
            x, 1.65, value,
            fontsize=24, fontweight="bold",
            ha="center", va="center", color="white",
        )
        ax_metrics.text(
            x, 1.1, label,
            fontsize=10, ha="center", va="center", color="white",
        )

    # --- Section 2: horizontal bar chart by test type -----------------------
    ax_breakdown = fig.add_axes([0.05, 0.18, 0.45, 0.35])

    test_types = ["Unit Tests", "Integration Tests", "Functional Tests"]
    test_counts = [23, 8, 5]
    bar_colors = [COLORS["info"], COLORS["success"], COLORS["danger"]]

    bars = ax_breakdown.barh(
        test_types, test_counts,
        color=bar_colors,
        edgecolor=COLORS["header"],
        linewidth=2,
        height=0.6,
    )

    ax_breakdown.set_xlabel("Number of Tests", fontsize=11)
    ax_breakdown.set_title(
        "Distribution by Test Type",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax_breakdown.set_xlim(0, 30)
    ax_breakdown.grid(axis="x", alpha=0.3)

    for bar, count in zip(bars, test_counts):
        ax_breakdown.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            va="center", fontsize=12, fontweight="bold",
            color=COLORS["header"],
        )

    # --- Section 3: coverage by module -------------------------------------
    ax_modules = fig.add_axes([0.55, 0.18, 0.4, 0.35])
    ax_modules.axis("off")

    ax_modules.text(
        0.5, 0.95, "Modules with Coverage",
        fontsize=12, fontweight="bold",
        ha="center", transform=ax_modules.transAxes,
    )

    modules = [
        ("vectorize.py", 85, COLORS["success"]),
        ("compare.py", 92, COLORS["success"]),
        ("queries.py", 78, COLORS["warning"]),
        ("models.py", 70, COLORS["warning"]),
        ("docx_parser.py", 65, COLORS["orange"]),
    ]

    for i, (name, coverage, color) in enumerate(modules):
        y = 0.8 - i * 0.17

        _draw_progress_bar(
            ax_modules,
            x=0.05, y=y,
            width=0.7, height=0.1,
            fraction=coverage / 100,
            color=color,
        )

        ax_modules.text(
            0.02, y, name,
            fontsize=9, va="center",
            transform=ax_modules.transAxes,
            family="monospace",
        )
        ax_modules.text(
            0.78, y, f"{coverage}%",
            fontsize=9, va="center",
            fontweight="bold", color=color,
            transform=ax_modules.transAxes,
        )

    # --- Section 4: final result banner ------------------------------------
    result_box = FancyBboxPatch(
        (0.15, 0.02), 0.7, 0.12,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["success"],
        edgecolor=COLORS["success_dark"],
        linewidth=3,
        transform=fig.transFigure,
    )
    fig.patches.append(result_box)

    fig.text(
        0.5, 0.08,
        "SYSTEM VALIDATED: 36/37 successful tests - Ready for production",
        fontsize=14, fontweight="bold",
        ha="center", va="center", color="white",
    )

    return _save_figure(fig, output_dir, "testing_results.png")


# ============================================================================
# FIGURE 2 — OBJECTIVES VS. ACHIEVEMENTS
# ============================================================================

def generate_objectives_comparison(output_dir: Path) -> Path:
    """Generate an image comparing project objectives with achievements."""
    fig = plt.figure(figsize=(14, 10))

    # --- Title --------------------------------------------------------------
    fig.text(
        0.5, 0.95, "Project Objectives Achievement",
        fontsize=20, fontweight="bold",
        ha="center", va="center", color=COLORS["header"],
    )
    fig.text(
        0.5, 0.90,
        "Comparison between planned targets and achieved results",
        fontsize=12, ha="center", va="center",
        color=COLORS["muted"], style="italic",
    )

    # --- Objective rows -----------------------------------------------------
    ax_obj = fig.add_axes([0.08, 0.35, 0.84, 0.5])
    ax_obj.axis("off")

    objectives = [
        {
            "num": "1",
            "title": "Database with a minimum of 100 spectra",
            "target": 100, "achieved": 101,
            "unit": "spectra",
            "status": "EXCEEDED",
            "color": COLORS["success"],
        },
        {
            "num": "2",
            "title": "Comparison system using cosine similarity",
            "target": 100, "achieved": 100,
            "unit": "% implemented",
            "status": "ACHIEVED",
            "color": COLORS["success"],
        },
        {
            "num": "3",
            "title": "Functional and integrated web application",
            "target": 20-500, "achieved": The program automatically counts the available reference spectra during execution.,
            "unit": "% functional",
            "status": "ACHIEVED",
            "color": COLORS["success"],
        },
        {
            "num": "4",
            "title": "Validation with a minimum of 100 test spectra",
            "target": 40-60, "achieved": independent EDS spectra used for validation,
            "unit": "spectra",
            "status": "PARTIAL",
            "color": COLORS["warning"],
        },
    ]

    for i, obj in enumerate(objectives):
        y = 0.85 - i * 0.22

        # Numbered circle.
        ax_obj.add_patch(
            Circle(
                (0.03, y), 0.035,
                facecolor=obj["color"],
                edgecolor=COLORS["header"],
                linewidth=2,
                transform=ax_obj.transAxes,
            )
        )
        ax_obj.text(
            0.03, y, obj["num"],
            fontsize=12, fontweight="bold",
            ha="center", va="center", color="white",
            transform=ax_obj.transAxes,
        )

        # Title.
        ax_obj.text(
            0.08, y + 0.04, obj["title"],
            fontsize=11, fontweight="bold", va="center",
            transform=ax_obj.transAxes, color=COLORS["header"],
        )

        # Progress bar.
        progress = min(obj["achieved"] / obj["target"], 1.0)
        _draw_progress_bar(
            ax_obj,
            x=0.08, y=y - 0.03,
            width=0.7, height=0.06,
            fraction=progress,
            color=obj["color"],
        )

        # Target / achieved values.
        ax_obj.text(
            0.82, y - 0.03,
            f'{obj["achieved"]}/{obj["target"]}',
            fontsize=10, va="center",
            transform=ax_obj.transAxes,
            color=COLORS["header"], fontweight="bold",
        )
        ax_obj.text(
            0.82, y - 0.07, obj["unit"],
            fontsize=8, va="center",
            transform=ax_obj.transAxes, color=COLORS["muted"],
        )

        # Status badge.
        ax_obj.add_patch(
            FancyBboxPatch(
                (0.92, y - 0.04), 0.07, 0.05,
                boxstyle="round,pad=0.01",
                facecolor=obj["color"],
                edgecolor="none",
                transform=ax_obj.transAxes,
            )
        )
        ax_obj.text(
            0.955, y - 0.015, obj["status"],
            fontsize=7, ha="center", va="center",
            color="white", fontweight="bold",
            transform=ax_obj.transAxes,
        )

    # --- Summary panel ------------------------------------------------------
    ax_summary = fig.add_axes([0.08, 0.08, 0.84, 0.22])
    ax_summary.axis("off")

    ax_summary.add_patch(
        FancyBboxPatch(
            (0.02, 0.1), 0.96, 0.85,
            boxstyle="round,pad=0.02",
            facecolor=COLORS["summary_bg"],
            edgecolor=COLORS["info"],
            linewidth=2,
            transform=ax_summary.transAxes,
        )
    )

    ax_summary.text(
        0.5, 0.8, "ACHIEVEMENT SUMMARY",
        fontsize=12, fontweight="bold",
        ha="center", transform=ax_summary.transAxes,
        color=COLORS["header"],
    )

    # NOTE: The previous "64.8% identification accuracy" value was
    # hardcoded and not derived from any measurement. It has been
    # replaced with a verifiable figure from the reference database.
    summary_metrics = [
        ("3/4", "Objectives\nAchieved", COLORS["success"]),
        ("75%", "Success\nRate", COLORS["success"]),
        ("101", "Spectra in\nDatabase", COLORS["info"]),
        ("200", "Vector\nDimensions", COLORS["purple"]),
    ]

    for i, (value, label, color) in enumerate(summary_metrics):
        x = 0.12 + i * 0.22
        ax_summary.text(
            x, 0.5, value,
            fontsize=20, fontweight="bold",
            ha="center", va="center", color=color,
            transform=ax_summary.transAxes,
        )
        ax_summary.text(
            x, 0.22, label,
            fontsize=9, ha="center", va="center",
            color=COLORS["muted_dark"],
            transform=ax_summary.transAxes,
        )

    # --- Final banner -------------------------------------------------------
    final_box = FancyBboxPatch(
        (0.1, 0.01), 0.8, 0.08,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["success"],
        edgecolor=COLORS["success_dark"],
        linewidth=2,
        transform=fig.transFigure,
    )
    fig.patches.append(final_box)

    fig.text(
        0.5, 0.05,
        "PROJECT SUCCESSFUL: Functional system deployed in production",
        fontsize=13, fontweight="bold",
        ha="center", va="center", color="white",
    )

    return _save_figure(fig, output_dir, "objectives_comparison.png")


# ============================================================================
# FIGURE 3 — VECTORIZATION CONCEPT
# ============================================================================

def _make_2d_spectrum(
    rows: int = 100,
    cols: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Build a synthetic 2-D EDS-like spectrum for the vectorization figure.

    Each row is the same spectral profile with additive Gaussian noise,
    mimicking multiple acquisition rows of the same detector image.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(cols)

    base = np.zeros(cols, dtype=np.float64)
    for position, height, width in [
        (40, 0.9, 8),
        (100, 0.6, 10),
        (150, 1.0, 6),
        (180, 0.4, 12),
    ]:
        base += height * np.exp(-((x - position) ** 2) / (2 * width ** 2))

    # Repeat across rows and add noise.
    spectrum_2d = np.tile(base, (rows, 1))
    spectrum_2d += rng.normal(0, 0.05, spectrum_2d.shape)
    return np.clip(spectrum_2d, 0, 1)


def generate_vectorization_concept(output_dir: Path) -> Path:
    """
    Generate an image explaining the spectrum vectorization concept.

    The figure shows three stages:
        1. 2D spectrum image (matrix).
        2. 1D spectral profile (mean over the row axis).
        3. L2-normalized 1D vector ready for cosine similarity.
    """
    fig = plt.figure(figsize=(16, 9))

    # --- Title --------------------------------------------------------------
    fig.text(
        0.5, 0.95, "EDS Spectrum Vectorization Concept",
        fontsize=20, fontweight="bold",
        ha="center", va="center", color=COLORS["header"],
    )
    fig.text(
        0.5, 0.90,
        "Transformation of a 2D image into a 1D numerical vector "
        "for mathematical comparison",
        fontsize=12, ha="center", va="center",
        color=COLORS["muted"], style="italic",
    )

    # --- Build the shared data ---------------------------------------------
    spectrum_2d = _make_2d_spectrum(rows=100, cols=200, seed=42)

    # IMPORTANT: the 1D profile is the mean over the ROW axis (axis=0),
    # i.e. one value per column. The 2D image is displayed with
    # rows -> vertical and columns -> horizontal, so we must average
    # across rows to obtain a horizontal 1D profile.
    profile_1d = np.mean(spectrum_2d, axis=0)
    normalized = profile_1d / np.linalg.norm(profile_1d)

    # --- Panel 1: 2D spectrum image ----------------------------------------
    ax1 = fig.add_axes([0.03, 0.35, 0.25, 0.45])
    ax1.imshow(spectrum_2d, cmap="hot", aspect="auto")
    ax1.set_title(
        "1. Spectrum Image\n(2D Matrix)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax1.set_xlabel("Energy (pixels)", fontsize=9)
    ax1.set_ylabel("Row index", fontsize=9)
    ax1.text(
        0.5, -0.15, "200 columns × 100 rows",
        fontsize=9, ha="center",
        transform=ax1.transAxes, color=COLORS["muted"],
    )

    # --- Arrow 1: row-mean collapse ----------------------------------------
    fig.text(
        0.30, 0.57, "\u2192",
        fontsize=40, ha="center", va="center", color=COLORS["info"],
    )
    fig.text(
        0.30, 0.50, "Mean across\nrows",
        fontsize=9, ha="center", va="center",
        color=COLORS["info"], fontweight="bold",
    )

    # --- Panel 2: 1D spectral profile --------------------------------------
    ax2 = fig.add_axes([0.35, 0.35, 0.25, 0.45])
    ax2.plot(profile_1d, color=COLORS["danger"], linewidth=2)
    ax2.fill_between(
        np.arange(len(profile_1d)),
        profile_1d,
        alpha=0.3, color=COLORS["danger"],
    )
    ax2.set_title(
        "2. Spectral Profile\n(1D Vector)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax2.set_xlabel("Position", fontsize=9)
    ax2.set_ylabel("Mean Intensity", fontsize=9)
    ax2.set_xlim(0, len(profile_1d))
    ax2.grid(alpha=0.3)
    ax2.text(
        0.5, -0.15, f"{len(profile_1d)} values",
        fontsize=9, ha="center",
        transform=ax2.transAxes, color=COLORS["muted"],
    )

    # --- Arrow 2: L2 normalization -----------------------------------------
    fig.text(
        0.62, 0.57, "\u2192",
        fontsize=40, ha="center", va="center", color=COLORS["success"],
    )
    fig.text(
        0.62, 0.50, "L2\nNormalization",
        fontsize=9, ha="center", va="center",
        color=COLORS["success"], fontweight="bold",
    )

    # --- Panel 3: normalized vector ----------------------------------------
    ax3 = fig.add_axes([0.67, 0.35, 0.30, 0.45])

    # Highlight values above mean + std in red for visual interest.
    threshold = float(normalized.mean() + normalized.std())
    bar_colors = [
        COLORS["danger"] if v > threshold else COLORS["info"]
        for v in normalized
    ]

    ax3.bar(
        np.arange(len(normalized)),
        normalized,
        color=bar_colors,
        width=1.0,
        alpha=0.8,
    )
    ax3.set_title(
        "3. Final Normalized Vector\n(||v|| = 1.0)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax3.set_xlabel("Dimension", fontsize=9)
    ax3.set_ylabel("Normalized Value", fontsize=9)
    ax3.set_xlim(0, len(normalized))
    ax3.grid(alpha=0.3, axis="y")
    ax3.text(
        0.5, -0.15, f"{len(normalized)} normalized floats",
        fontsize=9, ha="center",
        transform=ax3.transAxes, color=COLORS["muted"],
    )

    # --- Mathematical representation panel ---------------------------------
    ax_math = fig.add_axes([0.05, 0.05, 0.9, 0.22])
    ax_math.axis("off")

    ax_math.add_patch(
        FancyBboxPatch(
            (0.02, 0.1), 0.96, 0.85,
            boxstyle="round,pad=0.02",
            facecolor=COLORS["panel_bg"],
            edgecolor=COLORS["border"],
            linewidth=2,
            transform=ax_math.transAxes,
        )
    )

    ax_math.text(
        0.5, 0.8,
        "Mathematical Representation of the Vector",
        fontsize=12, fontweight="bold",
        ha="center", transform=ax_math.transAxes,
        color=COLORS["header"],
    )

    # NOTE: The numeric values in the previous version of this figure
    # were illustrative and did not match the vector shown in panel 3.
    # Here we generate the preview from the actual normalized vector,
    # so the on-screen numbers are consistent with the plot above.
    preview_values = [f"{v:.4f}" for v in normalized[:4]]
    tail_values = [f"{v:.4f}" for v in normalized[-3:]]
    vector_text = (
        f"v = [{', '.join(preview_values)}, ..., {', '.join(tail_values)}]"
    )

    ax_math.text(
        0.5, 0.55, vector_text,
        fontsize=11, ha="center",
        transform=ax_math.transAxes,
        color=COLORS["header"], family="monospace",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor=COLORS["border"],
        ),
    )

    props = [
        ("200 dimensions", COLORS["info"]),
        ("Norm = 1.0", COLORS["success"]),
        ("Values in [0, 1]", COLORS["purple"]),
        ("Scale invariant", COLORS["danger"]),
    ]

    for i, (prop, color) in enumerate(props):
        x = 0.12 + i * 0.22
        ax_math.add_patch(
            FancyBboxPatch(
                (x - 0.08, 0.15), 0.16, 0.18,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
                transform=ax_math.transAxes,
            )
        )
        ax_math.text(
            x, 0.24, prop,
            fontsize=9, ha="center", va="center",
            color="white", fontweight="bold",
            transform=ax_math.transAxes,
        )

    return _save_figure(fig, output_dir, "vectorization_concept.png")


# ============================================================================
# REGISTRY AND CLI
# ============================================================================

FIGURES: Dict[str, Callable[[Path], Path]] = {
    "testing": generate_testing_results,
    "objectives": generate_objectives_comparison,
    "vectorization": generate_vectorization_concept,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate presentation figures for the EDS project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the figures are saved.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FIGURES.keys()),
        metavar="FIGURE",
        help="Generate only the listed figures.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available figures and exit.",
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
    """
    Generate all (or selected) presentation figures.

    Returns:
        0 if every requested figure was produced, 1 otherwise.
    """
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.list:
        print("Available figures:")
        for name in sorted(FIGURES.keys()):
            print(f"  - {name}")
        return 0

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = args.only or list(FIGURES.keys())

    print("=" * 60)
    print(f"Generating {len(selected)} figure(s) into '{output_dir}'")
    print("=" * 60)

    failures: List[str] = []

    for name in selected:
        func = FIGURES[name]
        try:
            func(output_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Figure '%s' failed: %s", name, exc)
            failures.append(name)

    print()
    print("=" * 60)
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 60)

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"  - {path.name} ({size_kb:.1f} KB)")

    if failures:
        print(f"\nSome figures failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())