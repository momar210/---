#!/usr/bin/env python3
"""
Generate documentation diagrams for the mineral-identification system.

This script renders five figures to an output directory:

    1. Entity-Relationship (ER) diagram of the database schema.
    2. Processing pipeline: DOCX/JPG -> 200-dimensional vector.
    3. Cosine similarity formula.
    4. Geometric interpretation of cosine similarity.
    5. Grayscale EDS spectrum processing (from a sample .docx).

Usage:
    python scripts/generate_diagrams.py
    python scripts/generate_diagrams.py --output diagrams
    python scripts/generate_diagrams.py --sample-docx path/to/spectrum.docx
    python scripts/generate_diagrams.py --only er pipeline formula
    python scripts/generate_diagrams.py --list

Every diagram is generated independently. If one fails, the others still
run, and the script exits with code 1 if any diagram could not be produced.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib

# Use a non-interactive backend so the script works on headless servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    from docx import Document
except ImportError:  # pragma: no cover
    Document = None  # type: ignore[assignment]


logger = logging.getLogger("generate_diagrams")


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

DEFAULT_OUTPUT_DIR = Path("diagrams")
DEFAULT_SAMPLE_DIR = Path("reference_data")

# Shared palette used across diagrams.
COLORS = {
    "header": "#2C3E50",
    "primary_key": "#E74C3C",
    "foreign_key": "#3498DB",
    "field": "#ECF0F1",
    "border": "#34495E",
    "muted": "#7F8C8D",
    "muted_light": "#95A5A6",
    "input": "#3498DB",
    "process": "#2ECC71",
    "transform": "#9B59B6",
    "output": "#E74C3C",
    "success": "#27AE60",
}

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
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
        edgecolor="none",
    )
    plt.close(fig)
    logger.info("Generated: %s", filename)
    return path


def _draw_field_row(
    ax,
    x: float,
    y: float,
    width: float,
    name: str,
    dtype: str,
    constraint: str,
    facecolor: str,
    *,
    is_key: bool = False,
) -> None:
    """
    Draw one field row inside a table in the ER diagram.

    `is_key` means PK or FK, so text is drawn in white and bold.
    """
    rect = FancyBboxPatch(
        (x, y - 0.25),
        width,
        0.5,
        boxstyle="round,pad=0.02",
        facecolor=facecolor,
        edgecolor=COLORS["border"],
        linewidth=1,
    )
    ax.add_patch(rect)

    if is_key:
        text_color = "white"
        name_weight = "bold"
    else:
        text_color = "black"
        name_weight = "normal"

    ax.text(
        x + 0.2, y, name,
        fontsize=10, fontweight=name_weight, va="center", color=text_color,
    )
    ax.text(
        x + 2.0, y, dtype,
        fontsize=9, va="center",
        color=text_color if is_key else COLORS["muted"],
    )
    ax.text(
        x + width - 0.3, y, constraint,
        fontsize=8, va="center", ha="right",
        color=text_color if is_key else COLORS["muted_light"],
    )


# ============================================================================
# DIAGRAM 1 — ENTITY-RELATIONSHIP
# ============================================================================

def generate_er_diagram(output_dir: Path) -> Path:
    """Generate the Entity-Relationship diagram of the database."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # --- SAMPLES table header ----------------------------------------------
    header1 = FancyBboxPatch(
        (0.5, 4.5), 4, 0.7,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["header"],
        edgecolor=COLORS["border"],
        linewidth=2,
    )
    ax.add_patch(header1)
    ax.text(
        2.5, 4.85, "SAMPLES",
        fontsize=14, fontweight="bold", color="white",
        ha="center", va="center",
    )

    sample_fields = [
        ("id", "INTEGER", "PK", COLORS["primary_key"], True),
        ("sample_name", "STRING", "NOT NULL", COLORS["field"], False),
        ("date", "DATETIME", "DEFAULT NOW()", COLORS["field"], False),
        ("researcher", "STRING", "NULLABLE", COLORS["field"], False),
        ("image_path", "STRING", "NULLABLE", COLORS["field"], False),
    ]
    for i, (name, dtype, constraint, color, is_key) in enumerate(sample_fields):
        _draw_field_row(
            ax,
            x=0.5,
            y=4.3 - i * 0.6,
            width=4,
            name=name,
            dtype=dtype,
            constraint=constraint,
            facecolor=color,
            is_key=is_key,
        )

    # --- VECTORIZED_SPECTRA table header -----------------------------------
    header2 = FancyBboxPatch(
        (7, 4.5), 4.5, 0.7,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["header"],
        edgecolor=COLORS["border"],
        linewidth=2,
    )
    ax.add_patch(header2)
    ax.text(
        9.25, 4.85, "VECTORIZED_SPECTRA",
        fontsize=12, fontweight="bold", color="white",
        ha="center", va="center",
    )

    spectra_fields = [
        ("id", "INTEGER", "PK", COLORS["primary_key"], True),
        ("sample_id", "INTEGER", "FK", COLORS["foreign_key"], True),
        ("vector_json", "TEXT", "NOT NULL", COLORS["field"], False),
    ]
    for i, (name, dtype, constraint, color, is_key) in enumerate(spectra_fields):
        _draw_field_row(
            ax,
            x=7,
            y=4.3 - i * 0.6,
            width=4.5,
            name=name,
            dtype=dtype,
            constraint=constraint,
            facecolor=color,
            is_key=is_key,
        )

    ax.text(
        9.25, 2.4, "(JSON array of 200 floats)",
        fontsize=8, ha="center", va="center",
        color=COLORS["muted"], style="italic",
    )

    # --- Relationship line --------------------------------------------------
    ax.plot([4.5, 7], [4.05, 4.05], color=COLORS["border"], linewidth=2)

    # "1" on the samples side
    ax.text(4.7, 4.25, "1", fontsize=12, fontweight="bold", va="center")
    ax.plot([4.5, 4.5], [3.9, 4.2], color=COLORS["border"], linewidth=2)

    # "1" on the spectra side
    ax.text(6.7, 4.25, "1", fontsize=12, fontweight="bold", va="center")
    ax.plot([7, 7], [3.9, 4.2], color=COLORS["border"], linewidth=2)

    # Relationship label
    ax.text(
        5.75, 4.4, "contains",
        fontsize=10, ha="center", va="center",
        style="italic", color=COLORS["muted"],
    )

    # --- Legend -------------------------------------------------------------
    legend_y = 1.2
    legend_items = [
        (1.0, COLORS["primary_key"], "PK = Primary Key"),
        (4.0, COLORS["foreign_key"], "FK = Foreign Key"),
        (7.5, COLORS["field"], "Regular field"),
    ]
    for x, color, label in legend_items:
        ax.add_patch(
            FancyBboxPatch(
                (x, legend_y), 0.4, 0.3,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor=COLORS["border"],
            )
        )
        ax.text(x + 0.6, legend_y + 0.15, label, fontsize=9, va="center")

    # --- Title --------------------------------------------------------------
    ax.text(
        6, 6.5,
        "Entity-Relationship Diagram: EDS Spectrum Database",
        fontsize=14, fontweight="bold",
        ha="center", va="center",
    )

    return _save_figure(fig, output_dir, "er_diagram.png")


# ============================================================================
# DIAGRAM 2 — PROCESSING PIPELINE
# ============================================================================

def generate_pipeline_diagram(output_dir: Path) -> Path:
    """Generate the spectrum-processing pipeline diagram."""
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    steps = [
        ("DOCX File\nor JPG Image", "input", "System input"),
        ("1. EXTRACTION", "process", "Extracts the EDS spectrum image"),
        ("2. READING", "process", "Reads image as float (0–1)"),
        ("3. FILTERING", "transform", "5 × 5 Gaussian filter"),
        ("4. GRAYSCALE", "transform", "Converts to a single channel"),
        ("5. BINARIZATION", "transform", "Binary mask (threshold 0.99)"),
        ("6. CROPPING", "transform", "Isolates the region of interest"),
        ("7. SPECTRAL SIGNATURE", "transform", "Spectral profile (column mean)"),
        ("8. RESIZE", "transform", "Resizes to 200 dimensions"),
        ("9. NORMALIZATION", "transform", "L2 normalization"),
        ("FINAL VECTOR\n[0.1, 0.2, ...]", "output", "Array of 200 floats"),
    ]

    box_height = 0.9
    box_width = 4.5
    start_y = 13
    spacing = 1.15
    center_x = 5

    for i, (name, step_type, desc) in enumerate(steps):
        y = start_y - i * spacing
        color = COLORS[step_type]

        rect = FancyBboxPatch(
            (center_x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor=COLORS["header"],
            linewidth=2,
            alpha=0.9,
        )
        ax.add_patch(rect)

        ax.text(
            center_x, y, name,
            fontsize=11, fontweight="bold",
            ha="center", va="center", color="white",
        )
        ax.text(
            center_x + box_width / 2 + 0.3, y, desc,
            fontsize=9, ha="left", va="center",
            color=COLORS["muted"], style="italic",
        )

        if i < len(steps) - 1:
            arrow_y = y - box_height / 2 - 0.05
            ax.annotate(
                "",
                xy=(center_x, arrow_y - 0.15),
                xytext=(center_x, arrow_y),
                arrowprops=dict(
                    arrowstyle="->",
                    color=COLORS["header"],
                    lw=2,
                ),
            )

    # --- Legend -------------------------------------------------------------
    legend_items = [
        ("Input", COLORS["input"]),
        ("Processing", COLORS["process"]),
        ("Transformation", COLORS["transform"]),
        ("Output", COLORS["output"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        x = 0.5 + i * 2.3
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.3), 0.4, 0.3,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor=COLORS["header"],
            )
        )
        ax.text(x + 0.5, 0.45, label, fontsize=8, va="center")

    return _save_figure(fig, output_dir, "processing_pipeline.png")


# ============================================================================
# DIAGRAM 3 — COSINE SIMILARITY FORMULA
# ============================================================================

def generate_cosine_formula(output_dir: Path) -> Path:
    """Generate an image showing the cosine similarity formula."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    formula = (
        r"$\mathrm{Similarity}(\mathbf{A}, \mathbf{B}) = "
        r"\frac{\mathbf{A} \cdot \mathbf{B}}"
        r"{\|\mathbf{A}\| \times \|\mathbf{B}\|} = "
        r"\frac{\sum_{i=1}^{n} A_i \times B_i}"
        r"{\sqrt{\sum_{i=1}^{n} A_i^2} "
        r"\times \sqrt{\sum_{i=1}^{n} B_i^2}}$"
    )
    ax.text(
        0.5, 0.65, formula,
        fontsize=18, ha="center", va="center",
        transform=ax.transAxes,
    )

    explanations = [
        r"$\mathbf{A}, \mathbf{B}$: Spectral feature vectors (200 dimensions)",
        r"$\mathbf{A} \cdot \mathbf{B}$: Dot product of the vectors",
        r"$\|\mathbf{A}\|, \|\mathbf{B}\|$: L2 norms (magnitudes) of the vectors",
        r"$n = 200$: Dimension of the vector space",
    ]
    for i, text in enumerate(explanations):
        ax.text(
            0.5, 0.35 - i * 0.1, text,
            fontsize=11, ha="center", va="center",
            transform=ax.transAxes, color=COLORS["header"],
        )

    ax.text(
        0.5, 0.9, "Cosine Similarity Formula",
        fontsize=16, fontweight="bold",
        ha="center", va="center", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.02,
        "Range: [0, 1], where 1 = identical vectors and 0 = orthogonal vectors",
        fontsize=10, ha="center", va="center",
        transform=ax.transAxes,
        color=COLORS["output"], fontweight="bold",
    )

    return _save_figure(fig, output_dir, "cosine_similarity_formula.png")


# ============================================================================
# DIAGRAM 4 — VECTOR INTERPRETATION
# ============================================================================

def generate_vector_interpretation(output_dir: Path) -> Path:
    """Generate a geometric interpretation of identical/orthogonal vectors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- LEFT: Identical vectors (cos 0° = 1) ------------------------------
    ax1 = axes[0]
    ax1.set_xlim(-0.5, 2)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="k", linewidth=0.5)
    ax1.axvline(x=0, color="k", linewidth=0.5)

    ax1.annotate(
        "", xy=(1.5, 1.2), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=COLORS["output"], lw=3),
    )
    ax1.text(1.6, 1.3, r"$\mathbf{A}$", fontsize=14,
             color=COLORS["output"], fontweight="bold")

    ax1.annotate(
        "", xy=(1.45, 1.15), xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="->", color=COLORS["input"],
            lw=3, linestyle="--",
        ),
    )
    ax1.text(1.55, 1.0, r"$\mathbf{B}$", fontsize=14,
             color=COLORS["input"], fontweight="bold")

    theta = np.linspace(0, np.arctan(1.2 / 1.5), 30)
    ax1.plot(0.4 * np.cos(theta), 0.4 * np.sin(theta),
             color=COLORS["success"], lw=2)
    ax1.text(0.5, 0.15, r"$\theta = 0°$",
             fontsize=12, color=COLORS["success"])

    ax1.set_title(r"Identical Vectors: $\cos(0°) = 1$",
                  fontsize=14, fontweight="bold", pad=10)
    ax1.text(0.75, -0.35, "Similarity = 100%",
             fontsize=12, ha="center",
             color=COLORS["success"], fontweight="bold")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")

    # --- RIGHT: Orthogonal vectors (cos 90° = 0) ---------------------------
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 2)
    ax2.set_ylim(-0.5, 2)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linewidth=0.5)
    ax2.axvline(x=0, color="k", linewidth=0.5)

    ax2.annotate(
        "", xy=(1.5, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=COLORS["output"], lw=3),
    )
    ax2.text(1.6, 0.1, r"$\mathbf{A}$", fontsize=14,
             color=COLORS["output"], fontweight="bold")

    ax2.annotate(
        "", xy=(0, 1.5), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=COLORS["input"], lw=3),
    )
    ax2.text(0.1, 1.6, r"$\mathbf{B}$", fontsize=14,
             color=COLORS["input"], fontweight="bold")

    ax2.plot([0.2, 0.2, 0], [0, 0.2, 0.2],
             color=COLORS["success"], lw=2)
    ax2.text(0.35, 0.35, r"$\theta = 90°$",
             fontsize=12, color=COLORS["success"])

    ax2.set_title(r"Orthogonal Vectors: $\cos(90°) = 0$",
                  fontsize=14, fontweight="bold", pad=10)
    ax2.text(0.75, -0.35, "Similarity = 0%",
             fontsize=12, ha="center",
             color=COLORS["output"], fontweight="bold")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")

    fig.suptitle(
        "Geometric Interpretation of Cosine Similarity",
        fontsize=16, fontweight="bold", y=1.02,
    )

    return _save_figure(fig, output_dir, "vector_interpretation.png")


# ============================================================================
# DIAGRAM 5 — GRAYSCALE EDS SPECTRUM
# ============================================================================

def _read_first_image_from_docx(docx_path: Path) -> Optional[bytes]:
    """Return the raw bytes of the first image embedded in a .docx file."""
    if Document is None:
        raise RuntimeError("python-docx is not installed")
    doc = Document(str(docx_path))
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            return rel.target_part.blob
    return None


def _find_sample_docx(sample_dir: Path) -> Optional[Path]:
    """Return the first .docx file in a directory, or None."""
    if not sample_dir.is_dir():
        return None
    files = sorted(sample_dir.glob("*.docx"))
    return files[0] if files else None


def generate_grayscale_spectrum(
    output_dir: Path,
    sample_docx: Optional[Path] = None,
) -> Optional[Path]:
    """
    Extract and display an EDS spectrum image in grayscale.

    Returns:
        Path to the main figure, or None if no sample .docx was found.
    """
    if cv2 is None:
        logger.error("opencv-python is not installed; skipping diagram 5.")
        return None

    docx_path = sample_docx or _find_sample_docx(DEFAULT_SAMPLE_DIR)
    if docx_path is None or not docx_path.exists():
        logger.warning(
            "No sample .docx found. Provide --sample-docx or place a file "
            "under '%s'. Skipping grayscale spectrum diagram.",
            DEFAULT_SAMPLE_DIR,
        )
        return None

    mineral_name = docx_path.stem
    logger.info("Processing sample: %s", docx_path)

    # --- Extract the image --------------------------------------------------
    img_data = _read_first_image_from_docx(docx_path)
    if img_data is None:
        logger.error("No embedded image found in %s", docx_path)
        return None

    nparr = np.frombuffer(img_data, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_color is None:
        logger.error("Failed to decode the embedded image in %s", docx_path)
        return None

    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_filtered = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # --- Three-panel comparison --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Image (Color)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    axes[1].set_title("2. Grayscale", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(img_filtered, cmap="gray")
    axes[2].set_title("3. 5 × 5 Gaussian Filter", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(
        f"EDS Spectrum Processing: {mineral_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    _save_figure(fig, output_dir, "grayscale_spectrum_processing.png")

    # --- Grayscale-only version --------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.imshow(img_gray, cmap="gray")
    ax2.set_title(
        f"EDS Spectrum in Grayscale\n{mineral_name}",
        fontsize=14, fontweight="bold",
    )
    ax2.axis("off")
    path = _save_figure(fig2, output_dir, "grayscale_spectrum_only.png")

    return path


# ============================================================================
# REGISTRY AND CLI
# ============================================================================

DIAGRAMS: Dict[str, Callable[[Path], Optional[Path]]] = {
    "er": generate_er_diagram,
    "pipeline": generate_pipeline_diagram,
    "formula": generate_cosine_formula,
    "vectors": generate_vector_interpretation,
    "grayscale": generate_grayscale_spectrum,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate documentation diagrams for the EDS pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated figures are saved.",
    )
    parser.add_argument(
        "--sample-docx",
        default=None,
        help="Path to a sample .docx for the grayscale spectrum diagram.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(DIAGRAMS.keys()),
        metavar="DIAGRAM",
        help="Generate only the listed diagrams.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available diagrams and exit.",
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
    Generate all (or selected) diagrams.

    Returns:
        0 if every requested diagram was produced, 1 otherwise.
    """
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.list:
        print("Available diagrams:")
        for name in sorted(DIAGRAMS.keys()):
            print(f"  - {name}")
        return 0

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_docx = Path(args.sample_docx) if args.sample_docx else None

    selected = args.only or list(DIAGRAMS.keys())

    print("=" * 60)
    print(f"Generating {len(selected)} diagram(s) into '{output_dir}'")
    print("=" * 60)

    failures: List[str] = []

    for name in selected:
        func = DIAGRAMS[name]
        try:
            if name == "grayscale":
                result = func(output_dir, sample_docx)  # type: ignore[call-arg]
            else:
                result = func(output_dir)
            if result is None and name == "grayscale":
                # Grayscale diagram is optional: missing sample is not fatal.
                logger.info("Grayscale diagram skipped (no sample .docx).")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Diagram '%s' failed: %s", name, exc)
            failures.append(name)

    print()
    print("=" * 60)
    print(f"Output directory: {output_dir.resolve()}")
    print("=" * 60)

    # --- File listing -------------------------------------------------------
    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"  - {path.name} ({size_kb:.1f} KB)")

    if failures:
        print(f"\nSome diagrams failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())