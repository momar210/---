# src/database/queries.py
"""
Database query helpers for samples and vectorized spectra.
This module wraps common CRUD operations on the `Sample` and
`VectorizedSpectrum` ORM models. Each function takes an active
SQLAlchemy `Session` as its first argument so callers control
transaction scope.
Public functions:
    - create_tables()
    - insert_sample(session, name, researcher, image_path)
    - insert_spectrum(session, sample_id, vector)
    - get_all_samples(session)
    - get_sample_by_id(session, sample_id)
    - get_spectrum_by_sample_id(session, sample_id)
    - count_samples(session)
    - get_all_samples_with_vectors(session)
    - update_sample(session, sample_id, name, researcher)
    - delete_sample(session, sample_id)
    - delete_all_samples(session)
    - bulk_insert_samples(session, rows)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, List, Optional, Sequence, Union

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .connection import SessionLocal, engine
from .models import Base, Sample, VectorizedSpectrum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

def create_tables() -> None:
    """Create all database tables defined by the ORM metadata."""
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """Drop all database tables. Useful for tests and local resets."""
    Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# Sample operations
# ---------------------------------------------------------------------------

def insert_sample(
    session: Session,
    nombre_sample: str,
    investigador: Optional[str] = None,
    ruta_imagen: Optional[str] = None,
) -> Sample:
    """
    Insert a new sample and return the persisted ORM object.

    Args:
        session: Active SQLAlchemy session.
        nombre_sample: Name of the sample (required).
        investigador: Optional researcher name.
        ruta_imagen: Optional path to the sample image.

    Returns:
        The newly created and refreshed `Sample` instance.

    Raises:
        ValueError: If `nombre_sample` is empty.
        SQLAlchemyError: If the commit fails; the session is rolled back.
    """
    if not nombre_sample or not nombre_sample.strip():
        raise ValueError("nombre_sample must be a non-empty string")

    new_sample = Sample(
        nombre_sample=nombre_sample.strip(),
        investigador=investigador,
        ruta_imagen=ruta_imagen,
    )

    try:
        session.add(new_sample)
        session.commit()
        session.refresh(new_sample)
        return new_sample
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to insert sample %r", nombre_sample)
        raise


def get_all_samples(session: Session) -> List[Sample]:
    """Return every sample in the database, ordered by id."""
    return list(session.scalars(select(Sample).order_by(Sample.id)).all())


def get_sample_by_id(session: Session, sample_id: int) -> Optional[Sample]:
    """Return a sample by its primary key, or None if not found."""
    return session.get(Sample, sample_id)


def count_samples(session: Session) -> int:
    """Return the total number of samples in the database."""
    return int(session.scalar(select(func.count()).select_from(Sample)) or 0)


def get_all_samples_with_vectors(session: Session) -> List[Sample]:
    """
    Return all samples that have at least one associated vectorized spectrum.

    Uses an INNER JOIN so samples without spectra are excluded.
    """
    stmt = (
        select(Sample)
        .join(VectorizedSpectrum, VectorizedSpectrum.sample_id == Sample.id)
        .order_by(Sample.id)
    )
    return list(session.scalars(stmt).unique().all())


def update_sample(
    session: Session,
    sample_id: int,
    nombre_sample: Optional[str] = None,
    investigador: Optional[str] = None,
    ruta_imagen: Optional[str] = None,
) -> Optional[Sample]:
    """
    Update an existing sample's fields. Only non-None values are applied.

    Args:
        session: Active SQLAlchemy session.
        sample_id: Primary key of the sample to update.
        nombre_sample: New name (optional).
        investigador: New researcher name (optional).
        ruta_imagen: New image path (optional).

    Returns:
        The updated `Sample` instance, or None if no sample was found.
    """
    sample = session.get(Sample, sample_id)
    if sample is None:
        logger.info("No sample found with id=%s", sample_id)
        return None

    if nombre_sample is not None:
        sample.nombre_sample = nombre_sample
    if investigador is not None:
        sample.investigador = investigador
    if ruta_imagen is not None:
        sample.ruta_imagen = ruta_imagen

    try:
        session.commit()
        session.refresh(sample)
        return sample
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to update sample id=%s", sample_id)
        raise


def delete_sample(session: Session, sample_id: int) -> bool:
    """
    Delete a sample and all of its associated vectorized spectra.

    Args:
        session: Active SQLAlchemy session.
        sample_id: Primary key of the sample to delete.

    Returns:
        True if a sample was deleted, False otherwise.
    """
    sample = session.get(Sample, sample_id)
    if sample is None:
        return False

    try:
        # Delete associated spectra first to respect foreign key constraints.
        session.query(VectorizedSpectrum).filter_by(sample_id=sample_id).delete(
            synchronize_session=False
        )
        session.delete(sample)
        session.commit()
        return True
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to delete sample id=%s", sample_id)
        raise


def delete_all_samples(session: Session) -> int:
    """
    Delete all spectra and samples from the database.

    Returns:
        The number of samples that were deleted.
    """
    try:
        session.query(VectorizedSpectrum).delete(synchronize_session=False)
        deleted = session.query(Sample).delete(synchronize_session=False)
        session.commit()
        return int(deleted or 0)
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to delete all samples")
        raise


# ---------------------------------------------------------------------------
# Spectrum operations
# ---------------------------------------------------------------------------

def _serialize_vector(vector: Union[Sequence[float], np.ndarray, str]) -> str:
    """
    Convert a vector into a JSON string suitable for storage.

    Accepts a list, tuple, numpy array, or an already-serialized JSON string.
    """
    if isinstance(vector, str):
        # Validate that it is proper JSON before storing.
        json.loads(vector)
        return vector
    if isinstance(vector, np.ndarray):
        return json.dumps(vector.astype(float).ravel().tolist())
    if isinstance(vector, (list, tuple)):
        return json.dumps([float(x) for x in vector])
    raise TypeError(f"Unsupported vector type: {type(vector)!r}")


def insert_spectrum(
    session: Session,
    sample_id: int,
    vector: Union[Sequence[float], np.ndarray, str],
) -> VectorizedSpectrum:
    """
    Create a vectorized spectrum associated with a sample.

    Args:
        session: Active SQLAlchemy session.
        sample_id: Foreign key to the owning sample.
        vector: A list, tuple, numpy array, or JSON string of floats.

    Returns:
        The newly created and refreshed `VectorizedSpectrum` instance.

    Raises:
        ValueError: If the referenced sample does not exist.
        TypeError: If the vector type is unsupported.
        SQLAlchemyError: If the commit fails.
    """
    if session.get(Sample, sample_id) is None:
        raise ValueError(f"Cannot insert spectrum: sample_id={sample_id} does not exist")

    serialized = _serialize_vector(vector)

    spectrum = VectorizedSpectrum(sample_id=sample_id)
    spectrum.vector = serialized

    try:
        session.add(spectrum)
        session.commit()
        session.refresh(spectrum)
        return spectrum
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to insert spectrum for sample_id=%s", sample_id)
        raise


def get_spectrum_by_sample_id(
    session: Session, sample_id: int
) -> Optional[VectorizedSpectrum]:
    """Return the first vectorized spectrum for a sample, or None."""
    return (
        session.query(VectorizedSpectrum)
        .filter_by(sample_id=sample_id)
        .first()
    )


def get_spectrum_by_id(
    session: Session, spectrum_id: int
) -> Optional[VectorizedSpectrum]:
    """Return a vectorized spectrum by its primary key, or None."""
    return session.get(VectorizedSpectrum, spectrum_id)


def delete_spectrum(session: Session, sample_id: int) -> bool:
    """
    Delete the vectorized spectrum associated with a sample.

    Returns:
        True if a spectrum was deleted, False otherwise.
    """
    spectrum = (
        session.query(VectorizedSpectrum)
        .filter_by(sample_id=sample_id)
        .first()
    )
    if spectrum is None:
        return False

    try:
        session.delete(spectrum)
        session.commit()
        return True
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Failed to delete spectrum for sample_id=%s", sample_id)
        raise


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------

def bulk_insert_samples(
    session: Session,
    rows: Iterable[dict],
) -> List[Sample]:
    """
    Insert many samples in a single transaction.

    Args:
        session: Active SQLAlchemy session.
        rows: Iterable of dictionaries with keys matching `Sample` fields.

    Returns:
        The list of inserted `Sample` objects (with IDs populated).
    """
    objects = [Sample(**row) for row in rows]
    try:
        session.add_all(objects)
        session.commit()
        for obj in objects:
            session.refresh(obj)
        return objects
    except SQLAlchemyError:
        session.rollback()
        logger.exception("Bulk insert of samples failed")
        raise


def get_session() -> Session:
    """
    Convenience helper to open a new session from the configured factory.

    Remember to close it, e.g.:

        with get_session() as session:
            ...
    """
    return SessionLocal()