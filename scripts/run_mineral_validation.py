
"""
Validation script for the mineral identification system.

This script:
1. Loads all reference spectra from the SQLite database.
2. Loads the independent validation spectra from tests/test_data.
3. Extracts and vectorizes each validation EDS spectrum.
4. Compares each validation spectrum against all reference spectra using:
   - Cosine similarity
   - Pearson correlation
   - Euclidean distance
   - Manhattan distance
5. Selects the best reference spectrum for each metric.
6. Determines whether the predicted mineral is correct.
7. Calculates an independent accuracy for each metric.
8. Calculates per-mineral statistics.
9. Generates confusion matrices.
10. Saves the complete results to JSON.

The reference database and validation dataset must be kept separate.
The database should contain the 101 reference spectra, while
testing data should contain the 54 independent validation spectra.
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------
# Make project root available
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from src.parsers.docx_parser import extract_and_vectorize_spectrum
from src.database.connection import SessionLocal
from src.database.queries import get_all_samples
from src.database.models import VectorizedSpectrum


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_FOLDER = PROJECT_ROOT / "test" / "test_data"

OUTPUT_JSON = PROJECT_ROOT / "validation_results_four_metrics.json"

METRIC_NAMES = [
    "cosine",
    "pearson",
    "euclidean",
    "manhattan"
]


# =====================================================================
# Mineral-name handling
# =====================================================================

def extract_mineral_name(filename: str) -> str:
    """
    Extract the mineral name from a validation filename.

    Examples
    --------
    EDS_Galena_01.docx
        -> galena

    EDS_Pyrite_02.docx
        -> pyrite
    """

    name = Path(filename).stem

    prefixes = [
        "EDS_",
        "EDS ",
        "Eds_",
        "Eds ",
        "Element ",
        "Eds-"
    ]

    for prefix in prefixes:

        if name.startswith(prefix):
            name = name[len(prefix):]

    # Remove trailing sample numbers and suffixes
    name = re.sub(
        r"[_-]\d+.*$",
        "",
        name
    )

    name = re.sub(
        r"_data$",
        "",
        name
    )

    return name.strip().lower()


def normalize_mineral_name(name: str) -> str:
    """
    Normalize mineral names before comparison.

    Equivalent names are mapped to a common representation.
    """

    name = name.lower().strip()

    equivalents = {

        # Canonical English names
        "magnetite": "magnetite",
        "calcite": "calcite",
        "malachite": "malachite",
        "pyrite": "pyrite",
        "biotite": "biotite",
        "epidote": "epidote",
        "goethite": "goethite",
        "celestite": "celestite",
        "galena": "galena",
        "gypsum": "gypsum",
    }

    return equivalents.get(
        name,
        name
    )


def minerals_match(
    expected: str,
    predicted: str
) -> bool:
    """
    Determine whether the expected and predicted mineral names
    represent the same mineral.
    """

    expected_norm = normalize_mineral_name(
        expected
    )

    predicted_norm = normalize_mineral_name(
        predicted
    )

    return (
        expected_norm == predicted_norm
        or expected_norm in predicted_norm
        or predicted_norm in expected_norm
    )


# =====================================================================
# Spectral metrics
# =====================================================================

def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Calculate cosine similarity.

    Higher values indicate greater similarity.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=float
    )

    vector_b = np.asarray(
        vector_b,
        dtype=float
    )

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )


def pearson_correlation(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Calculate Pearson correlation coefficient.
    Higher values indicate greater similarity.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=float
    )

    vector_b = np.asarray(
        vector_b,
        dtype=float
    )

    std_a = np.std(vector_a)
    std_b = np.std(vector_b)

    if std_a == 0 or std_b == 0:
        return 0.0

    correlation = np.corrcoef(
        vector_a,
        vector_b
    )[0, 1]

    if np.isnan(correlation):
        return 0.0

    return float(correlation)


def euclidean_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Calculate Euclidean distance.

    Lower values indicate greater similarity.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=float
    )

    vector_b = np.asarray(
        vector_b,
        dtype=float
    )

    return float(
        np.linalg.norm(
            vector_a - vector_b
        )
    )


def manhattan_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray
) -> float:
    """
    Calculate Manhattan distance.

    Lower values indicate greater similarity.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=float
    )

    vector_b = np.asarray(
        vector_b,
        dtype=float
    )

    return float(
        np.sum(
            np.abs(
                vector_a - vector_b
            )
        )
    )


# =====================================================================
# Calculate all four metrics
# =====================================================================

def calculate_metrics(
    test_vector: np.ndarray,
    reference_vector: np.ndarray
) -> dict:
    """
    Calculate all four comparison metrics for one pair of spectra.
    """

    return {

        "cosine": cosine_similarity(
            test_vector,
            reference_vector
        ),

        "pearson": pearson_correlation(
            test_vector,
            reference_vector
        ),

        "euclidean": euclidean_distance(
            test_vector,
            reference_vector
        ),

        "manhattan": manhattan_distance(
            test_vector,
            reference_vector
        )
    }


# =====================================================================
# Select best reference spectrum
# =====================================================================

def select_best_match(
    comparisons: list,
    metric: str
) -> dict:
    """
    Select the best reference spectrum for a specified metric.

    Cosine and Pearson:
        Higher value = better match.

    Euclidean and Manhattan:
        Lower value = better match.
    """

    if metric in (
        "cosine",
        "pearson"
    ):

        return max(
            comparisons,
            key=lambda x: x[metric]
        )

    if metric in (
        "euclidean",
        "manhattan"
    ):

        return min(
            comparisons,
            key=lambda x: x[metric]
        )

    raise ValueError(
        f"Unknown metric: {metric}"
    )


# =====================================================================
# Load reference database
# =====================================================================

def load_reference_database():
    """
    Load all reference spectra from minerales_eds.db.

    Returns
    -------
    list
        Reference spectra containing:
        id
        mineral name
        vector
        source
    """

    session = SessionLocal()

    try:

        samples = get_all_samples(
            session
        )

        reference_spectra = []

        for sample in samples:

            spectrum = (
                session.query(
                    VectorizedSpectrum
                )
                .filter_by(
                    muestra_id=sample.id
                )
                .first()
            )

            if spectrum is None:
                continue

            vector = np.asarray(
                spectrum.vector,
                dtype=float
            )

            reference_spectra.append({

                "id": sample.id,

                "nombre": sample.nombre_muestra,

                "vector": vector,

                "fuente": sample.investigador,

                "ruta": sample.ruta_imagen
            })

        return reference_spectra

    finally:

        session.close()


# =====================================================================
# Validate one spectrum
# =====================================================================

def validate_spectrum(
    test_file: Path,
    reference_spectra: list
) -> dict:
    """
    Validate one independent spectrum against all reference spectra.
    """

    expected_mineral = extract_mineral_name(
        test_file.name
    )

    # --------------------------------------------------------------
    # Extract and vectorize validation spectrum
    # --------------------------------------------------------------

    test_vector = extract_and_vectorize_spectrum(
        str(test_file)
    )

    if test_vector is None:
        raise ValueError(
            "Unable to extract a valid spectrum."
        )

    test_vector = np.asarray(
        test_vector,
        dtype=float
    )

    if test_vector.size != 200:

        raise ValueError(
            "Expected a 200-dimensional vector, "
            f"but received {test_vector.size} dimensions."
        )

    # --------------------------------------------------------------
    # Compare against every reference spectrum
    # --------------------------------------------------------------

    comparisons = []

    for reference in reference_spectra:

        metrics = calculate_metrics(
            test_vector,
            reference["vector"]
        )

        comparisons.append({

            "id": reference["id"],

            "nombre": reference["nombre"],

            "fuente": reference["fuente"],

            **metrics
        })

    # --------------------------------------------------------------
    # Find best result for each metric
    # --------------------------------------------------------------

    metric_results = {}

    for metric in METRIC_NAMES:

        best = select_best_match(
            comparisons,
            metric
        )

        predicted_mineral = best["nombre"]

        correct = minerals_match(
            expected_mineral,
            predicted_mineral
        )

        metric_results[metric] = {

            "predicted": predicted_mineral,

            "correct": correct,

            "score": best[metric],

            "reference_id": best["id"],

            "reference_source": best["fuente"]
        }

    return {

        "file": test_file.name,

        "expected": expected_mineral,

        "metrics": metric_results
    }


# =====================================================================
# Calculate summary statistics
# =====================================================================

def calculate_metric_summary(
    results: list,
    metric: str
) -> dict:
    """
    Calculate accuracy and prediction statistics for one metric.
    """

    valid_results = [
        result
        for result in results
        if metric in result["metrics"]
    ]

    correct = sum(
        result["metrics"][metric]["correct"]
        for result in valid_results
    )

    incorrect = (
        len(valid_results)
        - correct
    )

    accuracy = (
        correct / len(valid_results)
        if valid_results
        else 0.0
    )

    scores = [
        result["metrics"][metric]["score"]
        for result in valid_results
    ]

    return {

        "metric": metric,

        "evaluated": len(valid_results),

        "correct": int(correct),

        "incorrect": int(incorrect),

        "accuracy": float(accuracy),

        "mean_score": (
            float(np.mean(scores))
            if scores
            else 0.0
        ),

        "minimum_score": (
            float(np.min(scores))
            if scores
            else 0.0
        ),

        "maximum_score": (
            float(np.max(scores))
            if scores
            else 0.0
        )
    }


# =====================================================================
# Per-mineral statistics
# =====================================================================

def calculate_mineral_statistics(
    results: list,
    metric: str
) -> dict:
    """
    Calculate TP, FP, and FN statistics for each mineral.
    """

    statistics = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    )

    for result in results:

        expected = normalize_mineral_name(
            result["expected"]
        )

        predicted = normalize_mineral_name(
            result["metrics"][metric]["predicted"]
        )

        if result["metrics"][metric]["correct"]:

            statistics[expected]["tp"] += 1

        else:

            statistics[expected]["fn"] += 1

            statistics[predicted]["fp"] += 1

    return dict(statistics)


# =====================================================================
# Main validation
# =====================================================================

def run_validation():
    """
    Execute the complete four-metric validation.
    """

    print("=" * 80)
    print(
        "MINERAL IDENTIFICATION SYSTEM "
        "FOUR-METRIC VALIDATION"
    )
    print("=" * 80)

    # --------------------------------------------------------------
    # Load reference database
    # --------------------------------------------------------------

    print(
        "\nLoading reference spectra "
        "from the SQLite database..."
    )

    reference_spectra = (
        load_reference_database()
    )

    print(
        f"Reference spectra loaded: "
        f"{len(reference_spectra)}"
    )

    if len(reference_spectra) != 101:

        print(
            "\nWARNING:"
            "\nThe expected reference database "
            "contains 101 spectra."
            f"\nCurrent database contains "
            f"{len(reference_spectra)} spectra."
        )

    if not reference_spectra:

        raise RuntimeError(
            "No reference spectra were found "
            "in the database."
        )

    # --------------------------------------------------------------
    # Locate validation data
    # --------------------------------------------------------------

    if not TEST_FOLDER.exists():

        raise FileNotFoundError(
            "Validation directory not found:\n"
            f"{TEST_FOLDER}"
        )

    test_files = sorted(
        TEST_FOLDER.glob("*.docx")
    )

    print(
        f"\nValidation DOCX files found: "
        f"{len(test_files)}"
    )

    if len(test_files) != 54:

        print(
            "\nWARNING:"
            "\nThe expected independent validation "
            "dataset contains 54 spectra."
            f"\nCurrent validation directory contains "
            f"{len(test_files)} DOCX files."
        )

    # --------------------------------------------------------------
    # Validate each spectrum
    # --------------------------------------------------------------

    results = []

    failed_files = []

    print(
        "\n" + "=" * 80
    )

    print(
        "RUNNING INDEPENDENT VALIDATION"
    )

    print(
        "=" * 80
    )

    for index, test_file in enumerate(
        test_files,
        start=1
    ):

        print(
            f"\n[{index}/{len(test_files)}] "
            f"{test_file.name}"
        )

        try:

            result = validate_spectrum(
                test_file,
                reference_spectra
            )

            results.append(result)

            expected = result["expected"]

            print(
                f"Expected mineral: {expected}"
            )

            for metric in METRIC_NAMES:

                metric_result = (
                    result["metrics"][metric]
                )

                status = (
                    "CORRECT"
                    if metric_result["correct"]
                    else "INCORRECT"
                )

                predicted = (
                    metric_result["predicted"]
                )

                score = metric_result["score"]

                print(
                    f"  {metric.capitalize():<10} "
                    f"-> {predicted:<20} "
                    f"score={score:.6f} "
                    f"[{status}]"
                )

        except Exception as error:

            failed_files.append({

                "file": test_file.name,

                "error": str(error)
            })

            print(
                f"  ERROR: {error}"
            )

    # --------------------------------------------------------------
    # Calculate final metrics
    # --------------------------------------------------------------

    summaries = {}

    for metric in METRIC_NAMES:

        summaries[metric] = (
            calculate_metric_summary(
                results,
                metric
            )
        )

    # --------------------------------------------------------------
    # Print final accuracy table
    # --------------------------------------------------------------

    print(
        "\n\n" + "=" * 80
    )

    print(
        "FINAL FOUR-METRIC VALIDATION RESULTS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"{'Metric':<15}"
        f"{'Evaluated':>12}"
        f"{'Correct':>12}"
        f"{'Incorrect':>12}"
        f"{'Accuracy':>12}"
    )

    print("-" * 63)

    for metric in METRIC_NAMES:

        summary = summaries[metric]

        print(
            f"{metric.capitalize():<15}"
            f"{summary['evaluated']:>12}"
            f"{summary['correct']:>12}"
            f"{summary['incorrect']:>12}"
            f"{summary['accuracy']:>11.2%}"
        )

    print("-" * 63)

    # --------------------------------------------------------------
    # Per-mineral statistics
    # --------------------------------------------------------------

    mineral_statistics = {}

    print(
        "\n" + "=" * 80
    )

    print(
        "PER-MINERAL VALIDATION STATISTICS"
    )

    print(
        "=" * 80
    )

    for metric in METRIC_NAMES:

        statistics = (
            calculate_mineral_statistics(
                results,
                metric
            )
        )

        mineral_statistics[metric] = (
            statistics
        )

        print(
            f"\n{metric.upper()}"
        )

        print(
            f"{'Mineral':<20}"
            f"{'TP':>8}"
            f"{'FP':>8}"
            f"{'FN':>8}"
        )

        print("-" * 44)

        for mineral in sorted(
            statistics
        ):

            values = statistics[mineral]

            print(
                f"{mineral:<20}"
                f"{values['tp']:>8}"
                f"{values['fp']:>8}"
                f"{values['fn']:>8}"
            )

    # --------------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------------

    output = {

        "dataset": {

            "reference_spectra": len(
                reference_spectra
            ),

            "validation_files": len(
                test_files
            ),

            "successfully_evaluated": len(
                results
            ),

            "failed": len(
                failed_files
            )
        },

        "metrics": summaries,

        "mineral_statistics":
            mineral_statistics,

        "failed_files":
            failed_files,

        "results":
            results
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nComplete validation results saved to:"
    )

    print(
        OUTPUT_JSON
    )

    # --------------------------------------------------------------
    # Final concise summary
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "ACCURACY SUMMARY"
    )

    print(
        "=" * 80
    )

    for metric in METRIC_NAMES:

        accuracy = summaries[
            metric
        ]["accuracy"]

        print(
            f"{metric.capitalize():<12}: "
            f"{accuracy:.2%}"
        )

    print(
        "=" * 80
    )

    return output


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":

    run_validation()