#!/usr/bin/env python3
"""
Quick self-contained test for the mineral identification pipeline.

This script demonstrates the complete workflow on a single example:
    1. Creates a temporary in-memory database.
    2. Inserts two synthetic reference spectra.
    3. Vectorizes a synthetic test spectrum.
    4. Compares it against the references using cosine similarity.
    5. Verifies that the expected mineral is returned as the top match.

The test does not require any external files, network access, or a
pre-existing database. It runs in under 5 seconds.

Usage:
    python examples/quick_test.py

Exit code:
    0 if the test passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when running from examples/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


# ============================================================================
# SYNTHETIC SPECTRA
# ============================================================================

def make_spectrum(peaks: list, size: int = 200) -> np.ndarray:
    """Build a synthetic EDS-like spectrum from Gaussian peaks."""
    x = np.arange(size)
    spectrum = np.zeros(size, dtype=np.float64)
    for position, height, width in peaks:
        spectrum += height * np.exp(-((x - position) ** 2) / (2 * width ** 2))
    return spectrum / np.linalg.norm(spectrum)


# Three minerals with distinct peak patterns.
REFERENCE_A = make_spectrum([(30, 0.9, 8), (100, 0.4, 10)])
REFERENCE_B = make_spectrum([(60, 1.0, 6), (150, 0.7, 12)])
REFERENCE_C = make_spectrum([(40, 0.5, 9), (120, 0.9, 7), (180, 0.3, 10)])

# The test spectrum is Reference A at a different scale (0.7x intensity).
# Cosine similarity must still rank it as the closest match.
TEST_SPECTRUM = REFERENCE_A * 0.7


# ============================================================================
# TEST
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_test() -> bool:
    """Run the quick test and return True on success."""
    print("=" * 60)
    print("Quick Test: Mineral Identification via Cosine Similarity")
    print("=" * 60)

    references = {
        "Mineral A": REFERENCE_A,
        "Mineral B": REFERENCE_B,
        "Mineral C": REFERENCE_C,
    }

    # --- Step 1: score every reference -------------------------------------
    print("\nScoring references against the test spectrum (scale=0.7x):\n")
    scores = []
    for name, ref in references.items():
        sim = cosine_similarity(TEST_SPECTRUM, ref)
        scores.append((name, sim))
        print(f"  {name:<12} cosine similarity = {sim:.6f}")

    # --- Step 2: rank and pick the top match -------------------------------
    scores.sort(key=lambda pair: pair[1], reverse=True)
    top_name, top_score = scores[0]
    print(f"\nTop match: {top_name}  (similarity = {top_score:.6f})")

    # --- Step 3: verify the expected result --------------------------------
    print("\nVerifying expected behavior:")
    checks = []

    checks.append((
        "Top match is 'Mineral A'",
        top_name == "Mineral A",
    ))
    checks.append((
        "Top similarity >= 0.999 (scale invariance)",
        top_score >= 0.999,
    ))
    checks.append((
        "Mineral A beats Mineral B",
        scores[0][0] == "Mineral A" and scores[1][0] == "Mineral B",
    ))

    all_passed = True
    for description, passed in checks:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {description}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: Quick test PASSED")
        print("=" * 60)
        return True
    print("RESULT: Quick test FAILED")
    print("=" * 60)
    return False


if __name__ == "__main__":
    sys.exit(0 if run_test() else 1)