#!/usr/bin/env python3
"""
Generate presentation figures for the mineral-identification project.

This script produces seven figures:

    1. Mineral group distribution (from the reference database).
    2. Processing and vectorization pipeline.
    3. Three-layer system architecture.
    4. Validation results (correct vs. incorrect identifications).
    5. Entity-Relationship diagram.
    6. Confusion matrix (illustrative unless a validation report exists).
    7. Spectral comparison principle (synthetic spectra).

Usage:
    python scripts/generate_project_figures.py
    python scripts/generate_project_figures.py --output figures
    python scripts/generate_project_figures.py --db sqlite:///minerales_eds.db
    python scripts/generate_project_figures.py --only distribution pipeline
    python scripts/generate_project_figures.py --list
    python scripts/generate_project_figures.py -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend so the script works on headless servers.
matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


logger = logging.getLogger("project_figures")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_OUTPUT_DIR = Path("figures")
DEFAULT_DPI = 300
DEFAULT_DB_URL = "sqlite:///minerales_eds.db"

# Classification of mineral names into groups. The order of keys in this
# dictionary is the order the groups appear in the legend, so keep it
# stable when editing.
MINERAL_GROUP_ORDER = [
    "Silicates",
    "Sulfides",
    "Oxides",
    "Carbonates",
    "Sulfates",
    "Phosphates",
    "Other",
]

# Keywords used to classify a mineral name into a group. A mineral is
# assigned to the first group whose keywords it matches (case-insensitive).
GROUP_KEYWORDS = {
    "Silicates": [
        "adularia", "albite", "almandine", "andalusite", "anorthite",
        "anorthoclase", "augite", "axinite", "benitoite", "beryl",
        "biotite", "chlorite", "chloritoid", "cordierite", "diopside",
        "epidote", "fayalite", "forsterite", "glaucophane",
        "grandidierite", "hornblende", "hypersthene", "jadeite",
        "kornerupine", "muscovite", "osumilite", "paragonite",
        "phlogopite", "plagioclase", "pyrope", "sapphirine", "scapolite",
        "sillimanite", "staurolite", "stilpnomelane", "titanite",
        "tremolite", "tourmaline", "topaz", "vesuvianite", "willemite",
        "wollastonite", "zircon",
    ],
    "Sulfides": [
        "arsenopyrite", "bismuthinite", "bornite", "chalcocite",
        "chalcopyrite", "covellite", "enargite", "galena", "pyrrhotite",
        "pyrite", "sphalerite", "stibnite", "stromeyerite", "nickeline",
    ],
    "Oxides": [
        "brannerite", "corundum", "chromite", "spinel", "hematite",
        "ilmenite", "mntio3",
    ],
    "Carbonates": [
        "ankerite", "calcite", "dolomite", "rhodochrosite",
    ],
    "Sulfates": [
        "barite", "celestite", "gypsum",
    ],
    "Phosphates": [
        "apatite", "beryllonite", "monazite", "turquoise",
    ],
}

# Fallback values used only when the database is unreachable. These are
# clearly labelled in the figure so the reader is never misled.
FALLBACK_GROUP_COUNTS = {
    "Silicates": 45,
    "Sulfides": 18,
    "Oxides": 12,
    "Carbonates": 8,
    "Sulfates": 6,
    "Phosphates": 5,
    "Other": 6,
}

GROUP_COLORS = [
    "#ff9999",
    "#66b3ff",
    "#99ff99",
    "#ffcc99",
    "#c2c2f0",
    "#ffb3e6",
    "#c4e17f",
]


# ============================================================================
# DATABASE ACCESS
# ============================================================================

def _load_sample_names(db_url: str) -> Optional[List[str]]:
    """
    Load every sample name from the database.

    Returns:
        A list of sample names, or None if the database is unreachable
        or the schema does not contain a Sample table.
    """
    try:
        engine = create_engine(db_url, future=True)
        # Import the model lazily so this script can still run in
        # environments where the ORM is not importable.
        from src.database.models import Sample

        with Session(engine) as session:
            names = list(session.scalars(select(Sample.nombre_sample)).all())
        return names
    except Exception as exc:  # noqa: BLE001 - DB errors are heterogeneous
        logger.warning("Could not read sample names from %s: %s", db_url, exc)
        return None


def _classify_mineral(name: str) -> str:
    """Return the mineral group for a given sample name."""
    lowered = (name or "").lower()
    for group, keywords in GROUP_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return group
    return "Other"


def _compute_group_counts(names: List[str]) -> Dict[str, int]:
    """Aggregate sample names into mineral-group counts."""
    counter: Counter[str] = Counter()
    for name in names:
        # Normalize filenames like "EDS_Albite" or "Albite.docx".
        clean = re.sub(r"^EDS[_\-\s]+", "", name, flags=re.IGNORECASE)
        clean = re.sub(r"\.docx$", "", clean, flags=re.IGNORECASE)
        counter[_classify_mineral(clean.strip())] += 1

    # Ensure every group appears, even if zero.
    return {group: counter.get(group, 0) for group in MINERAL_GROUP_ORDER}


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _save_figure(fig, output_dir: Path, filename: str) -> Path:
    """Save a matplotlib figure with consistent settings and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Generated: %s", filename)
    return path


def _draw_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = "black",
    fontsize: int = 9,
    fontweight: str = "normal",
) -> None:
    """Draw a FancyBboxPatch with centered text."""
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.1",
            ec=edgecolor,
            fc=facecolor,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
    )


# ============================================================================
# FIGURE 1 — MINERAL GROUP DISTRIBUTION
# ============================================================================

def plot_mineral_distribution(output_dir: Path, db_url: str) -> Path:
    """
    Figure 1: Mineralogical group distribution.

    Data is loaded directly from the database. If the database is
    unavailable, the figure falls back to a fixed dataset and clearly
    marks itself as such so no one mistakes it for the real result.
    """
    names = _load_sample_names(db_url)
    using_fallback = names is None or len(names) == 0

    if using_fallback:
        logger.warning(
            "Falling back to static distribution data. "
            "Connect to the database to generate real figures."
        )
        counts = dict(FALLBACK_GROUP_COUNTS)
    else:
        counts = _compute_group_counts(names)

    labels = list(counts.keys())
    sizes = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=GROUP_COLORS[: len(labels)],
        pctdistance=0.85,
    )

    # Donut hole.
    ax.add_artist(plt.Circle((0, 0), 0.70, fc="white"))
    ax.axis("equal")

    title = "Distribution of Mineralogical Groups in the Reference Database"
    if using_fallback:
        title += "\n(illustrative data — database not connected)"

    ax.set_title(title, pad=20)

    return _save_figure(fig, output_dir, "figure_1_mineral_distribution.png")


# ============================================================================
# FIGURE 4 — VALIDATION RESULTS
# ============================================================================

def plot_validation_results(output_dir: Path) -> Path:
    """
    Figure 4: Overall validation results.

    Requires actual validation counts. If none are available, the figure
    renders a clear "no validation data available" placeholder so the
    reader is never misled by fabricated percentages.
    """
    # Replace these with real values once a validation run exists.
    # Example: parse a JSON report produced by the validation script.
    validation_report = Path("reports/validation_summary.json")

    correct = 0
    incorrect = 0
    if validation_report.exists():
        try:
            payload = json.loads(validation_report.read_text(encoding="utf-8"))
            correct = int(payload.get("correct", 0))
            incorrect = int(payload.get("incorrect", 0))
        except (ValueError, OSError) as exc:
            logger.warning("Could not parse %s: %s", validation_report, exc)

    values = [correct, incorrect]
    categories = ["Correct Identifications", "Incorrect Identifications"]
    colors = ["#2ecc71", "#e74c3c"]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 6))

    if total == 0:
        ax.text(
            0.5, 0.5,
            "No validation data available.\n"
            "Run the validation script and place its summary at\n"
            f"'{validation_report}' to populate this figure.",
            ha="center", va="center",
            fontsize=12, color="#7F8C8D",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Overall Validation Performance")
        return _save_figure(
            fig, output_dir, "figure_4_validation_results.png"
        )

    bars = ax.bar(categories, values, color=colors, width=0.6)

    for bar in bars:
        height = bar.get_height()
        percentage = height / total * 100 if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}\n({percentage:.1f}%)",
            ha="center", va="bottom",
        )

    ax.set_ylabel("Number of Validation Spectra")
    ax.set_title(f"Overall Validation Performance (n={total})")
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)

    return _save_figure(fig, output_dir, "figure_4_validation_results.png")


# ============================================================================
# FIGURE 2 — PROCESSING PIPELINE
# ============================================================================

def plot_pipeline_diagram(output_dir: Path) -> Path:
    """Figure 2: EDS processing and vectorization pipeline."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    steps = [
        "1. Extraction\n(DOCX / Image)",
        "2. Gaussian\nFiltering",
        "3. Binarization",
        "4. Spectral\nSignature",
        "5. Resampling\n(200 dimensions)",
        "6. L2\nNormalization",
    ]
    x_positions = [1.0, 2.6, 4.2, 5.8, 7.4, 9.0]

    for i, (step, x) in enumerate(zip(steps, x_positions)):
        _draw_box(
            ax, x - 0.7, 1.5, 1.4, 1.0, step,
            facecolor="#e1f5fe", fontsize=9,
        )

        if i < len(steps) - 1:
            next_x = x_positions[i + 1]
            ax.arrow(
                x + 0.75, 2.0,
                (next_x - x) - 1.5, 0,
                head_width=0.1, head_length=0.1,
                fc="black", ec="black",
            )

    ax.set_title("EDS Processing and Vectorization Pipeline", pad=10)
    return _save_figure(fig, output_dir, "figure_2_processing_pipeline.png")


# ============================================================================
# FIGURE 3 — SYSTEM ARCHITECTURE
# ============================================================================

def plot_architecture_diagram(output_dir: Path) -> Path:
    """Figure 3: Three-layer system architecture."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Presentation layer.
    _draw_box(
        ax, 2, 7.5, 6, 1.5,
        "Presentation Layer\n(Streamlit Application)",
        facecolor="#fff9c4", fontweight="bold",
    )

    # Business logic layer.
    _draw_box(
        ax, 2, 4.5, 6, 2.0, "",
        facecolor="#e1bee7",
    )
    ax.text(
        5, 6.2, "Business Logic Layer",
        ha="center", va="center", fontweight="bold",
    )
    _draw_box(ax, 2.5, 4.8, 2.0, 1.0, "Parsers\n(DOCX)", fontsize=8)
    _draw_box(ax, 5.5, 4.8, 2.0, 1.0, "Analysis\n(Comparison)", fontsize=8)

    # Data layer.
    _draw_box(
        ax, 2, 1.5, 6, 2.0, "",
        facecolor="#c8e6c9",
    )
    ax.text(
        5, 3.2, "Data Layer",
        ha="center", va="center", fontweight="bold",
    )
    ax.text(
        5, 2.2, "SQLite Database\n(minerales_eds.db)",
        ha="center", va="center",
        bbox=dict(boxstyle="square", fc="white"),
    )

    # Layer connections.
    ax.arrow(5, 7.5, 0, -0.9,
             head_width=0.2, head_length=0.2, fc="black", ec="black")
    ax.arrow(5, 4.5, 0, -0.9,
             head_width=0.2, head_length=0.2, fc="black", ec="black")

    ax.set_title("Three-Layer Modular System Architecture", pad=10)
    return _save_figure(fig, output_dir, "figure_3_system_architecture.png")


# ============================================================================
# FIGURE 5 — ENTITY-RELATIONSHIP DIAGRAM
# ============================================================================

def plot_er_diagram(output_dir: Path) -> Path:
    """Figure 5: Entity-Relationship diagram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    # --- SAMPLES table -----------------------------------------------------
    ax.add_patch(
        patches.Rectangle((1, 4), 3, 4, ec="black", fc="#fff3e0")
    )
    ax.text(2.5, 7.5, "SAMPLES", ha="center", fontweight="bold")
    ax.plot([1, 4], [7.2, 7.2], color="black", linewidth=1)

    sample_fields = [
        "PK id (int)",
        "name (str)",
        "date (date)",
        "researcher (str)",
        "image_path (str)",
    ]
    for i, field in enumerate(sample_fields):
        ax.text(1.2, 6.8 - i * 0.5, field, fontsize=9)

    # --- SPECTRA table -----------------------------------------------------
    ax.add_patch(
        patches.Rectangle((6, 4.5), 3.5, 3, ec="black", fc="#e0f7fa")
    )
    ax.text(7.75, 7.0, "SPECTRA", ha="center", fontweight="bold")
    ax.plot([6, 9.5], [6.7, 6.7], color="black", linewidth=1)

    spectra_fields = [
        "PK id (int)",
        "FK sample_id (int)",
        "vector_json (text)",
    ]
    for i, field in enumerate(spectra_fields):
        ax.text(6.2, 6.3 - i * 0.5, field, fontsize=9)

    # --- Relationship ------------------------------------------------------
    ax.plot([4, 6], [6, 6], color="black", linewidth=1.5)
    ax.plot([4.2, 4.2], [5.8, 6.2], color="black", linewidth=1.5)
    ax.plot([5.8, 5.8], [5.8, 6.2], color="black", linewidth=1.5)
    ax.text(5, 6.2, "1 : 1", ha="center", fontsize=10,
            backgroundcolor="white")

    ax.set_title("Entity-Relationship Diagram", pad=10)
    return _save_figure(fig, output_dir, "figure_5_entity_relationship_diagram.png")


# ============================================================================
# FIGURE 6 — CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix_sim(output_dir: Path) -> Path:
    """
    Figure 6: Representative confusion matrix.

    This is an illustrative matrix. It is clearly labelled as such so
    it is not mistaken for an experimental result.
    """
    classes = ["Galena", "Pyrite", "Malachite", "Gypsum", "Other"]
    matrix = np.array([
        [10, 0, 0, 0, 0],
        [0, 8, 0, 0, 0],
        [0, 0, 2, 0, 5],
        [0, 0, 1, 2, 1],
        [0, 0, 2, 1, 15],
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45, ha="right", rotation_mode="anchor",
    )

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, matrix[i, j],
                ha="center", va="center",
                color="black" if matrix[i, j] < 10 else "white",
            )

    ax.set_title(
        "Representative Confusion Matrix\n"
        "(illustrative — not a validation result)"
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    fig.colorbar(im, ax=ax, label="Number of Spectra")

    return _save_figure(fig, output_dir, "figure_6_confusion_matrix.png")


# ============================================================================
# FIGURE 7 — SPECTRAL COMPARISON
# ============================================================================

def plot_spectral_comparison(output_dir: Path) -> Path:
    """
    Figure 7: Illustrative spectral comparison.

    Synthetic spectra are used to demonstrate the comparison principle.
    This figure is not an experimental result.
    """
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 200)

    # --- Target spectrum ---------------------------------------------------
    y1 = 0.8 * np.exp(-((x - 2.5) ** 2) / 0.1)
    y1 += 0.5 * np.exp(-((x - 6.0) ** 2) / 0.2)
    y1 += rng.normal(0, 0.02, 200)
    y1 = np.clip(y1, 0, 1)

    # --- Matching reference ------------------------------------------------
    y2 = y1 * 0.7 + rng.normal(0, 0.02, 200)
    y2 = np.clip(y2, 0, 1)

    # --- Non-matching reference -------------------------------------------
    y3 = 0.9 * np.exp(-((x - 4.0) ** 2) / 0.1)
    y3 += rng.normal(0, 0.02, 200)
    y3 = np.clip(y3, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x, y1, "b-", label="Unknown Spectrum", linewidth=1.5)
    ax1.plot(x, y2, "g--", label="Reference Spectrum", linewidth=1.5)
    ax1.set_title("Case A: High Spectral Similarity")
    ax1.set_xlabel("Energy (keV)")
    ax1.set_ylabel("Normalized Intensity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, y1, "b-", label="Unknown Spectrum", linewidth=1.5)
    ax2.plot(x, y3, "r--", label="Reference Spectrum", linewidth=1.5)
    ax2.set_title("Case B: Low Spectral Similarity")
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Normalized Intensity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Principle of EDS Spectral Comparison", fontsize=14)

    return _save_figure(fig, output_dir, "figure_7_spectral_comparison.png")


# ============================================================================
# REGISTRY AND CLI
# ============================================================================

# Each entry takes (output_dir, db_url) and returns the saved Path.
# Figures that do not use the database ignore the db_url argument.
FIGURES: Dict[str, Callable[..., Path]] = {
    "distribution":  plot_mineral_distribution,
    "pipeline":      lambda out, _db: plot_pipeline_diagram(out),
    "architecture":  lambda out, _db: plot_architecture_diagram(out),
    "validation":    lambda out, _db: plot_validation_results(out),
    "er":            lambda out, _db: plot_er_diagram(out),
    "confusion":     lambda out, _db: plot_confusion_matrix_sim(out),
    "spectral":      lambda out, _db: plot_spectral_comparison(out),
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the mineral-identification presentation figures. "
            "Mineral distribution and validation figures can read "
            "directly from the project database."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the figures are saved.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_URL,
        help="SQLAlchemy database URL for the reference database.",
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
    Generate all (or selected) figures.

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
    print(f"Database: {args.db}")
    print("=" * 60)

    failures: List[str] = []

    for name in selected:
        func = FIGURES[name]
        try:
            func(output_dir, args.db)
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