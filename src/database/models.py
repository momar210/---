# src/database/models.py
"""
SQLAlchemy ORM models for the mineral-identification database.

Schema
------
Two tables:

    samples                 samples
    --------------------     --------------------------------
    id                       primary key
    sample_name           mineral / sample name
    fecha                    creation timestamp
    investigador             researcher name (nullable)
    ruta_imagen              path to the source file (nullable)

    vectorized_spectra   vectorized spectra
    --------------------     --------------------------------
    id                       primary key
    sample_id               foreign key to samples.id
    vector_json              JSON array of 200 floats

Relationship: one sample -> one spectrum (1 : 1).

Public classes:
    - Base
    - sample
    - VectorizedSpectrum
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Union

import numpy as np
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

from .connection import engine


logger = logging.getLogger(__name__)


# ============================================================================
# BASE
# ============================================================================

Base = declarative_base()


# ============================================================================
# SAMPLE
# ============================================================================

class sample(Base):
    """
    A mineral sample.

    Columns:
        id              Primary key.
        sample_name  Human-readable name (unique, case-sensitive).
        fecha           Creation timestamp, set by the database server.
        investigador    Optional researcher name.
        ruta_imagen     Optional path to the source DOCX or image.

    Relationship:
        vector_spectrum  The single vectorized spectrum for this sample,
                         or None. Cascade-deleted with the sample.
    """

    __tablename__ = "samples"

    # A sample name must be unique so that re-running the builder does
    # not create duplicates. This is what enables the skip-existing
    # behavior in `scripts/build_reference_database.py`.
    __table_args__ = (
        UniqueConstraint("sample_name", name="uq_sample_name"),
    )

    id = Column(Integer, primary_key=True, index=True)
    sample_name = Column(String, nullable=False, index=True)
    fecha = Column(DateTime(timezone=True), server_default=func.now())
    investigador = Column(String, nullable=True)
    ruta_imagen = Column(String, nullable=True)

    vector_spectrum = relationship(
        "VectorizedSpectrum",
        uselist=False,
        back_populates="sample",
        cascade="all, delete-orphan",
        single_parent=True,
    )

    def __repr__(self) -> str:
        return (
            f"<sample(id={self.id}, sample_name={self.sample_name!r})>"
        )


# ============================================================================
# VECTORIZED SPECTRUM
# ============================================================================

class VectorizedSpectrum(Base):
    """
    The L2-normalized feature vector associated with a sample.

    The vector is stored as a JSON string so the schema is portable
    across SQLite, PostgreSQL, and MySQL without needing a native
    array type. The `vector` property provides transparent
    conversion to and from a plain Python list or numpy array.

    Columns:
        id              Primary key.
        sample_id      Foreign key to samples.id.
        vector_json     JSON-encoded list of floats.

    Relationship:
        sample         The owning sample instance.
    """

    __tablename__ = "vectorized_spectra"

    id = Column(Integer, primary_key=True, index=True)
    sample_id = Column(
        Integer,
        ForeignKey("samples.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    vector_json = Column(Text, nullable=False)

    sample = relationship("sample", back_populates="vector_spectrum")

    # ------------------------------------------------------------------
    # Vector property
    # ------------------------------------------------------------------

    @property
    def vector(self) -> Optional[List[float]]:
        """
        Return the stored vector as a Python list of floats.

        Returns None if the underlying JSON is empty, invalid, or does
        not decode to a numeric list. Failures are logged rather than
        raised so a single bad row does not break a whole query.
        """
        if not self.vector_json:
            return None
        try:
            decoded = json.loads(self.vector_json)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "Invalid vector_json for VectorizedSpectrum id=%s: %s",
                self.id, exc,
            )
            return None

        if not isinstance(decoded, list):
            logger.warning(
                "vector_json for id=%s is not a list (got %s).",
                self.id, type(decoded).__name__,
            )
            return None
        return decoded

    @vector.setter
    def vector(self, value: Union[List[float], np.ndarray, None]) -> None:
        """
        Store a list or numpy array as a JSON string.

        Accepts:
            - list / tuple of numbers
            - numpy.ndarray (any shape; flattened with .ravel())
            - None (stores an empty JSON array)

        Raises:
            TypeError: If `value` is not one of the accepted types.
            ValueError: If the array contains NaN or infinite values.
        """
        if value is None:
            self.vector_json = json.dumps([])
            return

        if isinstance(value, np.ndarray):
            arr = value.astype(np.float64).ravel()
            if not np.all(np.isfinite(arr)):
                raise ValueError("vector contains NaN or infinite values")
            payload = arr.tolist()
        elif isinstance(value, (list, tuple)):
            payload = [float(x) for x in value]
            if any(x != x or x in (float("inf"), float("-inf")) for x in payload):
                raise ValueError("vector contains NaN or infinite values")
        else:
            raise TypeError(
                f"vector must be a list, tuple, numpy.ndarray, or None; "
                f"got {type(value).__name__}"
            )

        self.vector_json = json.dumps(payload)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def vector_array(self) -> Optional[np.ndarray]:
        """Return the vector as a float64 numpy array, or None."""
        vec = self.vector
        if vec is None:
            return None
        try:
            return np.asarray(vec, dtype=np.float64)
        except (TypeError, ValueError):
            return None

    @property
    def dimension(self) -> int:
        """Return the number of elements in the vector, or 0 if unreadable."""
        vec = self.vector
        return len(vec) if vec else 0

    def __repr__(self) -> str:
        return (
            f"<VectorizedSpectrum(id={self.id}, "
            f"sample_id={self.sample_id}, dim={self.dimension})>"
        )