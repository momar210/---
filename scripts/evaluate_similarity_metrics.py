#!/usr/bin/env python3
"""
Comparison of similarity/distance metrics for mineral identification.

Compares four metrics on a validation set of DOCX spectra:
    - Cosine Similarity
    - Euclidean Distance
    - Pearson Correlation
    - Manhattan Distance

For each metric, the script:
    1. Loads the reference database of vectorized spectra.
    2. Loads each validation DOCX, extracts its vector.
    3. Ranks every reference by the metric.
    4. Takes the top-1 as the prediction.
    5. Checks whether the prediction matches the expected mineral.

Outputs:
    - Console report with accuracy per metric.
    - diagrams/metric_accuracy_comparison.png
    - diagrams/metric_ranking_table.png
    - diagrams/scale_invariance_comparison.png
    - diagrams/metric_comparison_results.json

Usage:
    python scripts/compare_metrics.py
    python scripts/compare_metrics.py --tests tests/test_data -o diagrams
    python scripts/compare_metrics.py --min-accuracy 0.7 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

# Use a non-interactive backend so the script works on headless servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.analysis.vectorize import vectorize_spectrum  # noqa: F401 (used by callers)
from src.parsers.docx_parser import extract_and_vectorize_spectrum
from src.database.connection import SessionLocal
from src.database.models import Sample, VectorizedSpectrum


logger = logging.getLogger("compare_metrics")


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_OUTPUT_DIR = Path("diagrams")
DEFAULT_TEST_DIR = Path("tests/test_data")
DEFAULT_DPI = 150

COLORS = {
    "header": "#2C3E50",
    "muted": "#7F8C8D",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "info": "#3498DB",
    "warning": "#F39C12",
    "neutral": "#95A5A6",
}

# Tolerance for treating vector norms as zero.
_EPS = 1e-12


# ============================================================================
# METRIC IMPLEMENTATIONS
# ============================================================================

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity: angle between vectors. Range [0, 1] for non-negative data."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < _EPS or norm_b < _EPS:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Euclidean distance: straight-line distance. Lower = more similar."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b))


def pearson_correlation(v1: np.ndarray, v2: np.ndarray) -> float:
    """Pearson correlation: linear correlation. Range [-1, 1]."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return 0.0
    if float(np.std(a)) < _EPS or float(np.std(b)) < _EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def manhattan_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Manhattan distance: sum of absolute differences. Lower = more similar."""
    a = np.asarray(v1, dtype=np.float64).ravel()
    b = np.asarray(v2, dtype=np.float64).ravel()
    return float(np.sum(np.abs(a - b)))


# ============================================================================
# MINERAL NAME NORMALIZATION AND EQUIVALENCE
# ============================================================================

EQUIVALENCES: Dict[str, List[str]] = {
    "calcite":    ["calcite", "calcita"],
    "pyrite":     ["pyrite", "pirita"],
    "biotite":    ["biotite", "biotita"],
    "epidote":    ["epidote", "epidota"],
    "galena":     ["galena"],
    "magnetite":  ["magnetite", "magnetita"],
    "hematite":   ["hematite", "hematita"],
    "celestine":  ["celestite", "celestina", "celestine"],
    "celestite":  ["celestina", "celestine"],
    "amethyst":   ["quartz", "cuarzo", "amethyst", "amatista"],
    "goethite":   ["goethite", "goethita", "hematite", "hematita"],
    "quartz":     ["quartz", "cuarzo", "amethyst"],
}


def normalize_for_comparison(name: str) -> str:
    """Normalize a mineral name for comparison."""
    name = (name or "").lower().strip()
    name = re.sub(r"[-_]\d+$", "", name)
    name = re.sub(r"[-_]sample\d*$", "", name)
    name = name.replace("iron_oxide", "iron oxide")
    return name.strip()


def are_minerals_equivalent(expected: str, predicted: str) -> bool:
    """Check whether two mineral names are considered equivalent."""
    exp = normalize_for_comparison(expected)
    pred = normalize_for_comparison(predicted)

    if exp == pred:
        return True
    if exp in pred or pred in exp:
        return True
    if exp in EQUIVALENCES and pred in EQUIVALENCES[exp]:
        return True
    if pred in EQUIVALENCES and exp in EQUIVALENCES[pred]:
        return True
    return False


def extract_mineral_name(filename: str) -> str:
    """Extract the mineral name from a file name."""
    name = Path(filename).stem
    prefixes = ["EDS_", "EDS ", "Eds_", "Eds ", "Element ", "Eds-"]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = re.sub(r"[_-]\d+.*$", "", name)
    name = re.sub(r"_data$", "", name)
    return name.strip().lower()


# ============================================================================
# DATABASE LOADING
# ============================================================================

@dataclass
class ReferenceRecord:
    """A single reference spectrum loaded from the database."""

    id: int
    name: str
    vector: np.ndarray
    image_path: Optional[str] = None


def load_reference_records(session: Session) -> List[ReferenceRecord]:
    """
    Load every (sample, vector) pair from the database.

    Skips rows whose vector cannot be parsed as a numeric array.
    """
    stmt = (
        select(
            Sample.id,
            Sample.nombre_sample,
            Sample.ruta_imagen,
            VectorizedSpectrum.vector,
        )
        .join(VectorizedSpectrum, VectorizedSpectrum.sample_id == Sample.id)
    )

    records: List[ReferenceRecord] = []
    for sid, name, image_path, raw in session.execute(stmt):
        vec = _parse_vector(raw)
        if vec is None:
            logger.warning("Skipping sample %s: unreadable vector", sid)
            continue
        records.append(
            ReferenceRecord(id=sid, name=name, vector=vec, image_path=image_path)
        )
    return records


def _parse_vector(raw) -> Optional[np.ndarray]:
    """Parse a stored vector (JSON string, list, or array) into float64."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    try:
        vec = np.asarray(raw, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        return None
    return vec


# ============================================================================
# VALIDATION LOOP
# ============================================================================

@dataclass
class MetricState:
    """Running state for a single metric during validation."""

    name: str
    func: callable
    higher_is_better: bool
    correct: int = 0
    total: int = 0
    confidences: List[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0


def _normalize_confidence(metric_name: str, raw_score: float) -> float:
    """
    Map a raw metric score to [0, 1] for cross-metric comparison.

    - Cosine similarity: already in [0, 1] for non-negative data.
    - Pearson correlation: in [-1, 1], rescaled to [0, 1].
    - Distances: mapped via 1 / (1 + d) so that 0 distance -> 1.0.
    """
    if metric_name == "Cosine Similarity":
        return float(np.clip(raw_score, 0.0, 1.0))
    if metric_name == "Pearson Correlation":
        return float((raw_score + 1.0) / 2.0)
    return float(1.0 / (1.0 + max(raw_score, 0.0)))


def _evaluate_one_file(
    test_file: Path,
    records: List[ReferenceRecord],
    metrics: Dict[str, MetricState],
    strict: bool,
) -> bool:
    """
    Evaluate a single validation file against all metrics.

    Returns True if the file was successfully processed (even if the
    prediction was wrong), False if it was skipped.
    """
    expected = extract_mineral_name(test_file.name)

    try:
        vector = extract_and_vectorize_spectrum(str(test_file))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Extraction failed for %s: %s", test_file.name, exc)
        return False

    if vector is None:
        logger.warning("No vector produced for %s", test_file.name)
        return False

    vector = np.asarray(vector, dtype=np.float64).ravel()

    for state in metrics.values():
        scored = [(rec, state.func(vector, rec.vector)) for rec in records]
        if not scored:
            continue
        scored.sort(
            key=lambda pair: pair[1],
            reverse=state.higher_is_better,
        )
        top_record, top_score = scored[0]

        is_correct = are_minerals_equivalent(expected, top_record.name)

        state.total += 1
        if is_correct:
            state.correct += 1
        state.confidences.append(_normalize_confidence(state.name, top_score))

        if strict and not is_correct:
            logger.info(
                "[%s] %s -> predicted %s (score=%.4f)",
                state.name, expected, top_record.name, top_score,
            )

    return True


def run_metric_comparison(
    test_dir: Path,
    *,
    strict: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Run the metric comparison and return the aggregated results.

    Args:
        test_dir: Directory containing validation .docx files.
        strict: If True, log every incorrect prediction.

    Returns:
        A dict mapping metric name to a dict with keys:
        accuracy, correct, total, avg_confidence.
    """
    print("=" * 70)
    print("SIMILARITY / DISTANCE METRIC COMPARISON")
    print("=" * 70)

    session = SessionLocal()
    try:
        records = load_reference_records(session)
    finally:
        session.close()

    print(f"\nReference spectra in database: {len(records)}")
    if not records:
        logger.error("No reference spectra found. Populate the database first.")
        return {}

    test_files = sorted(test_dir.glob("*.docx"))
    print(f"Validation files in {test_dir}: {len(test_files)}")
    if not test_files:
        logger.error("No validation .docx files found.")
        return {}

    metrics: Dict[str, MetricState] = {
        "Cosine Similarity": MetricState(
            name="Cosine Similarity",
            func=cosine_similarity,
            higher_is_better=True,
        ),
        "Euclidean Distance": MetricState(
            name="Euclidean Distance",
            func=euclidean_distance,
            higher_is_better=False,
        ),
        "Pearson Correlation": MetricState(
            name="Pearson Correlation",
            func=pearson_correlation,
            higher_is_better=True,
        ),
        "Manhattan Distance": MetricState(
            name="Manhattan Distance",
            func=manhattan_distance,
            higher_is_better=False,
        ),
    }

    print("\nRunning validation with each metric...\n")

    processed = 0
    for test_file in test_files:
        if _evaluate_one_file(test_file, records, metrics, strict):
            processed += 1

    print(f"\nProcessed {processed}/{len(test_files)} validation files.")

    results: Dict[str, Dict[str, float]] = {}
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name, state in metrics.items():
        if state.total == 0:
            continue
        results[name] = {
            "accuracy": state.accuracy,
            "correct": state.correct,
            "total": state.total,
            "avg_confidence": state.avg_confidence,
        }
        print(f"\n{name}:")
        print(f"  Accuracy:           {state.accuracy:.2%} "
              f"({state.correct}/{state.total})")
        print(f"  Average confidence: {state.avg_confidence:.2%}")

    return results


# ============================================================================
# CHART 1 — ACCURACY BAR CHART
# ============================================================================

def _chart_accuracy_bar(
    results: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> Path:
    """Bar chart of accuracy per metric."""
    metrics = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in metrics]

    best = max(accuracies)
    worst = min(accuracies)
    colors = [
        COLORS["success"] if a == best
        else COLORS["danger"] if a == worst
        else COLORS["neutral"]
        for a in accuracies
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(metrics, accuracies, color=colors,
                  edgecolor=COLORS["header"], linewidth=2)

    for bar, accuracy in zip(bars, accuracies):
        ax.annotate(
            f"{accuracy:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=COLORS["header"],
        )

    ax.axhline(y=best, color=COLORS["success"],
               linestyle="--", alpha=0.5, linewidth=2)
    ax.axhline(y=50, color=COLORS["danger"],
               linestyle=":", alpha=0.5, linewidth=1.5,
               label="Baseline (50%)")

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_title(
        "Accuracy Comparison of Similarity / Distance Metrics",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor=COLORS["success"], edgecolor=COLORS["header"],
              label="Highest accuracy"),
        Patch(facecolor=COLORS["danger"], edgecolor=COLORS["header"],
              label="Lowest accuracy"),
        Patch(facecolor=COLORS["neutral"], edgecolor=COLORS["header"],
              label="Other metrics"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    path = output_dir / "metric_accuracy_comparison.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=DEFAULT_DPI)
    plt.close(fig)
    logger.info("Generated: %s", path)
    return path


# ============================================================================
# CHART 2 — RANKING TABLE
# ============================================================================

def _chart_ranking_table(
    results: Dict[str, Dict[str, float]],
    output_dir: Path,
    *,
    n_validation: int,
) -> Path:
    """
    Visual ranking table for the four metrics.

    The subtitle reports the actual number of validation files processed,
    not a hardcoded count.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    sorted_metrics = sorted(
        results.items(),
        key=lambda kv: kv[1]["accuracy"],
        reverse=True,
    )

    ax.text(0.5, 0.95, "Metric Accuracy Comparison",
            fontsize=18, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes)

    ax.text(0.5, 0.89,
            f"Validation using {n_validation} independent test spectra",
            fontsize=11, ha="center", va="top",
            transform=ax.transAxes,
            color=COLORS["muted"], style="italic")

    rank_colors = [COLORS["success"], COLORS["info"],
                   COLORS["warning"], COLORS["danger"]]
    rank_labels = ["Highest", "2nd", "3rd", "4th"]

    for i, (metric_name, data) in enumerate(sorted_metrics):
        y = 0.75 - i * 0.18

        ax.add_patch(FancyBboxPatch(
            (0.05, y - 0.06), 0.9, 0.14,
            boxstyle="round,pad=0.02",
            facecolor=rank_colors[i], alpha=0.15,
            edgecolor=rank_colors[i], linewidth=2,
            transform=ax.transAxes,
        ))

        ax.add_patch(FancyBboxPatch(
            (0.07, y - 0.02), 0.08, 0.06,
            boxstyle="round,pad=0.01",
            facecolor=rank_colors[i],
            edgecolor=COLORS["header"], linewidth=1,
            transform=ax.transAxes,
        ))

        ax.text(0.11, y + 0.01, f"{i + 1}",
                fontsize=14, fontweight="bold",
                ha="center", va="center",
                transform=ax.transAxes, color="white")

        ax.text(0.18, y + 0.02, metric_name,
                fontsize=13, fontweight="bold",
                ha="left", va="center",
                transform=ax.transAxes, color=COLORS["header"])

        ax.text(0.55, y + 0.02, f"{data['accuracy']:.1%}",
                fontsize=18, fontweight="bold",
                ha="center", va="center",
                transform=ax.transAxes, color=rank_colors[i])

        ax.text(0.75, y + 0.02,
                f"{data['correct']}/{data['total']} correct",
                fontsize=10, ha="left", va="center",
                transform=ax.transAxes, color=COLORS["muted"])

        ax.text(0.92, y + 0.02, rank_labels[i],
                fontsize=10, fontweight="bold",
                ha="right", va="center",
                transform=ax.transAxes, color=rank_colors[i])

    explanation = (
        "Cosine similarity is expected to be less sensitive to scale because:\n"
        "  - It is SCALE-INVARIANT: different acquisition times may produce\n"
        "    different intensities while preserving the spectral shape.\n"
        "  - It measures the ANGLE between vectors rather than absolute distance.\n"
        "  - EDS spectra may vary in magnitude while retaining characteristic shape."
    )
    ax.text(0.5, 0.08, explanation,
            fontsize=9, ha="center", va="center",
            transform=ax.transAxes, color=COLORS["header"],
            bbox=dict(boxstyle="round", facecolor="#FEF9E7",
                      edgecolor=COLORS["warning"], alpha=0.8),
            family="monospace")

    path = output_dir / "metric_ranking_table.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=DEFAULT_DPI)
    plt.close(fig)
    logger.info("Generated: %s", path)
    return path


# ============================================================================
# CHART 3 — SCALE INVARIANCE DEMONSTRATION
# ============================================================================

def _chart_scale_invariance(output_dir: Path) -> Path:
    """Four-panel figure demonstrating scale invariance of cosine similarity."""
    # --- Synthetic spectrum ------------------------------------------------
    base = np.zeros(200)
    for position, height in [(30, 0.8), (80, 0.5), (120, 1.0), (160, 0.3)]:
        base += height * np.exp(-((np.arange(200) - position) ** 2) / 50)
    base = base / np.max(base)

    scales = [0.5, 1.0, 1.5, 2.0]
    scaled = [base * s for s in scales]
    ref = scaled[1]  # scale 1.0 as reference

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: spectra at different scales ------------------------------
    ax1 = axes[0, 0]
    palette = [COLORS["info"], COLORS["success"],
               COLORS["danger"], "#9B59B6"]
    for i, (spectrum, scale) in enumerate(zip(scaled, scales)):
        ax1.plot(spectrum, color=palette[i], linewidth=2,
                 label=f"Scale {scale}\u00d7", alpha=0.8)
    ax1.set_title("Spectra at Different Scales\n"
                  "(Same mineral, different acquisition time)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Vector Dimension")
    ax1.set_ylabel("Intensity")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # --- Panel 2: cosine similarity ----------------------------------------
    ax2 = axes[0, 1]
    cos_vals = [cosine_similarity(ref, s) for s in scaled]
    bars = ax2.bar(scales, cos_vals, color=COLORS["success"],
                   edgecolor=COLORS["header"], linewidth=2)
    ax2.axhline(y=1.0, color=COLORS["success"], linestyle="--", alpha=0.5)
    ax2.set_title("Cosine Similarity\n(Scale invariant)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Scale Factor")
    ax2.set_ylabel("Similarity")
    ax2.set_ylim(0, 1.1)
    for bar, value in zip(bars, cos_vals):
        ax2.annotate(f"{value:.3f}",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold",
                     color=COLORS["success"])

    # --- Panel 3: Euclidean distance ---------------------------------------
    ax3 = axes[1, 0]
    euc_vals = [euclidean_distance(ref, s) for s in scaled]
    bars = ax3.bar(scales, euc_vals, color=COLORS["danger"],
                   edgecolor=COLORS["header"], linewidth=2)
    ax3.set_title("Euclidean Distance\n(Sensitive to scale)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("Scale Factor")
    ax3.set_ylabel("Distance")
    for bar, value in zip(bars, euc_vals):
        ax3.annotate(f"{value:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold",
                     color=COLORS["danger"])

    # --- Panel 4: Manhattan distance ---------------------------------------
    ax4 = axes[1, 1]
    man_vals = [manhattan_distance(ref, s) for s in scaled]
    bars = ax4.bar(scales, man_vals, color=COLORS["danger"],
                   edgecolor=COLORS["header"], linewidth=2)
    ax4.set_title("Manhattan Distance\n(Sensitive to scale)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("Scale Factor")
    ax4.set_ylabel("Distance")
    for bar, value in zip(bars, man_vals):
        ax4.annotate(f"{value:.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="bold",
                     color=COLORS["danger"])

    fig.suptitle(
        "Demonstration of the Scale Invariance of Cosine Similarity",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    path = output_dir / "scale_invariance_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=DEFAULT_DPI)
    plt.close(fig)
    logger.info("Generated: %s", path)
    return path


def generate_comparison_charts(
    results: Dict[str, Dict[str, float]],
    output_dir: Path,
    *,
    n_validation: int,
) -> List[Path]:
    """Generate all comparison charts and return their paths."""
    return [
        _chart_accuracy_bar(results, output_dir),
        _chart_ranking_table(results, output_dir, n_validation=n_validation),
        _chart_scale_invariance(output_dir),
    ]


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare cosine, Euclidean, Pearson, and Manhattan metrics "
            "on validation DOCX spectra."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tests", "-t",
        default=str(DEFAULT_TEST_DIR),
        help="Directory containing validation .docx files.",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where figures and JSON are saved.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Log every incorrect prediction.",
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

    test_dir = Path(args.tests)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.is_dir():
        logger.error("Test directory not found: %s", test_dir)
        return 1

    print("\n" + "=" * 70)
    print("STARTING METRIC COMPARISON")
    print("=" * 70)

    results = run_metric_comparison(test_dir, strict=args.strict)
    if not results:
        logger.error("No results produced. Check the database and test files.")
        return 1

    print("\n" + "=" * 70)
    print("GENERATING COMPARATIVE CHARTS")
    print("=" * 70)

    # Number of validation files that actually contributed to the results.
    n_validation = max((int(v["total"]) for v in results.values()), default=0)

    generate_comparison_charts(
        results, output_dir, n_validation=n_validation
    )

    # --- Save JSON ---------------------------------------------------------
    serializable = {
        key: {
            "accuracy": value["accuracy"],
            "accuracy_percent": f"{value['accuracy']:.1%}",
            "correct": int(value["correct"]),
            "total": int(value["total"]),
            "avg_confidence": value["avg_confidence"],
        }
        for key, value in results.items()
    }
    json_path = output_dir / "metric_comparison_results.json"
    json_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {json_path}")

    # --- Final summary -----------------------------------------------------
    best = max(results.items(), key=lambda kv: kv[1]["accuracy"])
    worst = min(results.items(), key=lambda kv: kv[1]["accuracy"])

    print("\n" + "=" * 70)
    print("METRIC COMPARISON COMPLETED")
    print("=" * 70)
    print(f"\n Highest accuracy: {best[0]} ({best[1]['accuracy']:.1%})")
    print(f" Lowest accuracy:  {worst[0]} ({worst[1]['accuracy']:.1%})")
    print(
        f"\n Accuracy difference: "
        f"{(best[1]['accuracy'] - worst[1]['accuracy']) * 100:.1f} "
        f"percentage points"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())