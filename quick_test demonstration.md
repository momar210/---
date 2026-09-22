# Examples

This directory contains a self-contained quick test that demonstrates
the core idea of the mineral identification system: **cosine similarity
is invariant to scale**.

## What the test does

1. Builds three synthetic EDS-like spectra with distinct peak patterns.
2. Creates a test spectrum by scaling one of them (0.7x intensity).
3. Scores each reference against the test spectrum using cosine similarity.
4. Verifies that the correct mineral is returned as the top match,
   despite the intensity difference.

## How to run

From the project root:

```bash
python examples/quick_test.py
============================================================
Quick Test: Mineral Identification via Cosine Similarity
============================================================

Scoring references against the test spectrum (scale=0.7x):

  Mineral A    cosine similarity = 1.000000
  Mineral B    cosine similarity = 0.612344
  Mineral C    cosine similarity = 0.487291

Top match: Mineral A  (similarity = 1.000000)

Verifying expected behavior:
  [PASS] Top match is 'Mineral A'
  [PASS] Top similarity >= 0.999 (scale invariance)
  [PASS] Mineral A beats Mineral B

============================================================
RESULT: Quick test PASSED
============================================================