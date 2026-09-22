# main.py
"""
Command-line entry point for the mineral EDS spectrum pipeline.

Workflow:
    1. Parse command-line arguments.
    2. Ensure database tables exist.
    3. Extract and vectorize the spectrum from a .docx file.
    4. Persist the sample and its vectorized spectrum.
    5. Compare the new sample against previously stored spectra.

"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from src.analysis.compare import compare_spectrum
from src.database.connection import SessionLocal
from src.database.queries import (
    count_samples,
    create_tables,
    insert_sample,
    insert_spectrum,
)
from src.parsers.docx_parser import extract_and_vectorize_spectrum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: Optional list of arguments (defaults to sys.argv[1:]).

    Returns:
        The parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process a mineral EDS spectrum stored in a .docx file, "
            "vectorize it, persist it, and compare it with existing samples."
        )
    )

    parser.add_argument(
        "docx_file",
        nargs="?",
        default="data/Eds_magnetite.docx",
        help="Path to the .docx file containing the spectrum image.",
    )
    parser.add_argument(
        "--number",
        default="Magnetite - Test Sample",
        help="Name of the sample to store.",
    )
    parser.add_argument(
        "--investigador",
        default="Test System",
        help="Researcher name to store.",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        default=200,
        help="Number of points to extract from the spectrum image.",
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=0.5,
        help="Similarity threshold used when comparing against existing samples.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    """Configure the root logger for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def extract_vector(docx_file: str, vector_size: int):
    """
    Extract and vectorize the spectrum from a .docx file.

    Returns:
        A numpy array, or None if no suitable image was found.
    """
    logger.info("Extracting vector from '%s' (size=%d)", docx_file, vector_size)
    return extract_and_vectorize_spectrum(docx_file, vector_size=vector_size)


def persist_sample(session, name: str, researcher: str, vector):
    """
    Insert a new sample and its vectorized spectrum into the database.

    Returns:
        The newly created Sample ORM object.
    """
    sample = insert_sample(
        session=session,
        nombre_sample=name,
        investigador=researcher,
    )
    insert_spectrum(session, sample_id=sample.id, vector=vector)
    logger.info("Stored sample id=%s name=%r", sample.id, sample.nombre_sample)
    return sample


def report_comparisons(session, sample_id: int, threshold: float) -> None:
    """
    Compare the given sample against the rest of the database and print results.
    """
    results = compare_spectrum(
        session,
        sample_id=sample_id,
        similarity_threshold=threshold,
    )

    if not results:
        print(
            "No other samples with a spectrum were found, "
            "or none exceeded the similarity threshold."
        )
        return

    print(f"Comparison results (threshold={threshold:.2f}):")
    for result in results:
        print(
            f"  - SampleID={result.sample_id} "
            f"({result.sample_name}): similarity={result.similarity:.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """
    Run the end-to-end pipeline.

    Returns:
        Exit code: 0 on success, 1 on handled failure.
    """
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    create_tables()
    session = SessionLocal()

    try:
        # Step 1: Extract and vectorize the spectrum.
        vector = extract_vector(args.docx_file, args.vector_size)
        if vector is None:
            print(
                "No image with shape (400, 512, 3) was found in the document.",
                file=sys.stderr,
            )
            return 1

        # Step 2: Persist the sample and its spectrum.
        new_sample = persist_sample(
            session=session,
            name=args.number,
            researcher=args.investigador,
            vector=vector,
        )
        print(f"Sample '{new_sample.nombre_sample}' stored with ID={new_sample.id}")

        # Step 3: Count total samples and optionally compare.
        total_samples = count_samples(session)
        print(f"Total number of samples in the database: {total_samples}")

        if total_samples > 1:
            report_comparisons(session, new_sample.id, threshold=args.umbral)
        else:
            print("This is the first sample in the database.")

        return 0

    except Exception as exc:
        session.rollback()
        logger.exception("Failed to process the sample; transaction rolled back.")
        print(
            f"An error occurred while processing the sample: {exc}",
            file=sys.stderr,
        )
        return 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())