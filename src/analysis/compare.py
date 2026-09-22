# src/analysis/compare.py
"""
Spectrum comparison utilities.

This module provides cosine-similarity based comparison between a reference
sample's vectorized spectrum and all other vectorized spectra stored in the
database.

Main public functions:
    - calculate_similarity(vector1, vector2)
    - calculate_similarity_batch(base_vector, matrix)
    - compare_spectrum(session, sample_id, similarity_threshold, ...)
"""

from __future__ import annotations

import json
import logging
from typing import List, NamedTuple, Optional, Sequence, Union

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Sample, VectorizedSpectrum

logger = logging.getLogger(__name__)

# Tolerance used to treat vector norms as zero (avoids division by zero).
_EPS = 1e-12

# Supported raw input types for a vector.
VectorLike = Union[Sequence[float], np.ndarray, str, dict, None]


class ComparisonResult(NamedTuple):
    """A single comparison result returned by `compare_spectrum`."""

    sample_id: int
    sample_name: str
    similarity: float


# ---------------------------------------------------------------------------
# Similarity primitives
# ---------------------------------------------------------------------------

def calculate_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two 1-D vectors.

    Cosine similarity is defined as:

        sim(a, b) = (a . b) / (||a|| * ||b||)

    It returns a value in [-1, 1]. For non-negative data (such as spectra),
    the value lies in [0, 1].

    Args:
        vector1: First vector (array-like).
        vector2: Second vector (array-like).

    Returns:
        The cosine similarity as a float. Returns 0.0 if either vector has
        zero norm or if the shapes do not match.
    """
    v1 = np.asarray(vector1, dtype=np.float64).ravel()
    v2 = np.asarray(vector2, dtype=np.float64).ravel()

    if v1.shape != v2.shape:
        logger.warning(
            "Cannot compute similarity: shape mismatch %s vs %s",
            v1.shape,
            v2.shape,
        )
        return 0.0

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < _EPS or norm2 < _EPS:
        return 0.0

    # Normalize each vector first to reduce overflow risk on long vectors.
    return float(np.dot(v1 / norm1, v2 / norm2))


def calculate_similarity_batch(
    base_vector: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Vectorized cosine similarity of a single vector against a 2-D matrix.

    This computes, in one shot, the cosine similarity of `base_vector`
    against every row of `matrix`. It is significantly faster than calling
    `calculate_similarity` in a loop.

    Args:
        base_vector: 1-D array of shape (n_features,).
        matrix: 2-D array of shape (n_samples, n_features).

    Returns:
        1-D numpy array of shape (n_samples,) containing the similarity
        of each row in `matrix` with `base_vector`.

    Raises:
        ValueError: If the feature dimension of `matrix` does not match
            the length of `base_vector`.
    """
    base = np.asarray(base_vector, dtype=np.float64).ravel()
    mat = np.asarray(matrix, dtype=np.float64)

    if mat.size == 0:
        return np.zeros(0, dtype=np.float64)

    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    if mat.shape[1] != base.shape[0]:
        raise ValueError(
            f"Feature dimension mismatch: base has {base.shape[0]} "
            f"features, matrix has {mat.shape[1]}."
        )

    base_norm = float(np.linalg.norm(base))
    row_norms = np.linalg.norm(mat, axis=1)
    denom = base_norm * row_norms

    sims = np.zeros(mat.shape[0], dtype=np.float64)

    # Only compute for rows with valid (non-zero) norms.
    if base_norm >= _EPS:
        safe = denom > _EPS
        sims[safe] = (mat[safe] @ base) / denom[safe]

    # Numerical cleanup: clamp tiny overshoots to [-1, 1].
    return np.clip(sims, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_vector(raw: VectorLike) -> Optional[np.ndarray]:
    """
    Safely extract a 1-D numpy vector from a stored representation.

    Supported input formats:
        - numpy.ndarray
        - list / tuple of numbers
        - JSON string (e.g. "[0.1, 0.2, ...]")
        - dict with a "data" or "vector" key holding any of the above

    Args:
        raw: The raw stored value (spectrum.vector or its deserialized form).

    Returns:
        A 1-D float64 numpy array, or None if the input is missing,
        empty, or cannot be parsed.
    """
    if raw is None:
        return None

    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            raw = raw.get("data") or raw.get("vector")
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = json.loads(raw.decode("utf-8"))

        vec = np.asarray(raw, dtype=np.float64).ravel()
        if vec.size == 0:
            return None
        if not np.all(np.isfinite(vec)):
            logger.warning("Vector contains NaN or infinite values; skipping.")
            return None
        return vec
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse vector %r: %s", type(raw).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_spectrum(
    session: Session,
    sample_id: int,
    similarity_threshold: float = 0.5,
    *,
    limit: Optional[int] = None,
    exclude_self: bool = True,
) -> List[ComparisonResult]:
    """
    Compare the reference sample against all other vectorized spectra.

    Workflow:
        1. Load the reference sample and its vectorized spectrum.
        2. Load all other vectorized spectra with their sample names.
        3. Compute cosine similarity in a single batched operation.
        4. Filter by `similarity_threshold`, sort descending, apply `limit`.

    Args:
        session: Active SQLAlchemy session.
        sample_id: ID of the reference sample.
        similarity_threshold: Minimum cosine similarity to include a result.
        limit: Optional maximum number of results to return (after sorting).
        exclude_self: If True, the reference sample is excluded from results.

    Returns:
        A list of `ComparisonResult` tuples sorted by similarity in
        descending order. Returns an empty list if the reference sample
        has no vectorized spectrum or if no other samples exist.
    """
    # 1. Fetch the reference sample and its spectrum in a single query.
    base_row = session.execute(
        select(Sample, VectorizedSpectrum)
        .join(VectorizedSpectrum, VectorizedSpectrum.sample_id == Sample.id)
        .where(Sample.id == sample_id)
    ).first()

    if base_row is None:
        logger.info("No vectorized spectrum found for sample_id=%s", sample_id)
        return []

    base_sample, base_spectrum = base_row
    base_vector = _load_vector(getattr(base_spectrum, "vector", None))

    if base_vector is None:
        logger.warning("Sample %s has a null or empty vector", sample_id)
        return []

    # 2. Fetch all other samples and their spectra in one query.
    #
    # We select the tuple (sample_id, sample_name, vector) directly, using
    # the column that actually stores the vector. This avoids any dependency
    # on a relationship attribute and works whether or not the model
    # exposes a `Sample.spectra` relationship.
    stmt = (
        select(
            Sample.id.label("sample_id"),
            Sample.nombre_sample.label("sample_name"),
            VectorizedSpectrum.vector.label("vector"),
        )
        .join(VectorizedSpectrum, VectorizedSpectrum.sample_id == Sample.id)
    )
    if exclude_self:
        stmt = stmt.where(Sample.id != sample_id)

    rows = session.execute(stmt).all()
    if not rows:
        return []

    # 3. Collect only rows whose vectors match the reference feature length.
    ids: List[int] = []
    names: List[str] = []
    vectors: List[np.ndarray] = []

    for sid, sname, raw_vec in rows:
        vec = _load_vector(raw_vec)
        if vec is None:
            continue
        if vec.shape != base_vector.shape:
            logger.debug(
                "Skipping sample %s: vector shape %s does not match %s",
                sid,
                vec.shape,
                base_vector.shape,
            )
            continue
        ids.append(sid)
        names.append(sname)
        vectors.append(vec)

    if not vectors:
        return []

    # 4. Batch-compute cosine similarity.
    matrix = np.vstack(vectors)
    sims = calculate_similarity_batch(base_vector, matrix)

    # 5. Filter by threshold and sort descending.
    results = [
        ComparisonResult(sid, sname, float(sim))
        for sid, sname, sim in zip(ids, names, sims)
        if sim >= similarity_threshold
    ]
    results.sort(key=lambda r: r.similarity, reverse=True)

    if limit is not None:
        results = results[:limit]

    return results