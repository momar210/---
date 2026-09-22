"""
python scripts\build_reference_database.py
Check if 
Samples: Number of smaples
Vectors: Numbr of Vectors
verify that every database entry contains exactly 200 values.
"""
from pathlib import Path
import numpy as np

from src.database.connection import SessionLocal
from src.database.queries import (
    create_tables,
    insert_sample,
    insert_spectrum
)
from src.parsers.docx_parser import extract_and_vectorize_spectrum


REFERENCE_DIR = Path("reference_data")


def main():

    # Create database tables
    create_tables()

    session = SessionLocal()

    try:

        docx_files = sorted(REFERENCE_DIR.glob("*.docx"))

        print(f"Found {len(docx_files)} reference DOCX files.")

        successful = 0
        failed = 0

        for docx_file in docx_files:

            print(f"Processing: {docx_file.name}")

            vector = extract_and_vectorize_spectrum(
                str(docx_file)
            )

            if vector is None:
                print("  FAILED: vector could not be extracted")
                failed += 1
                continue

            vector = np.asarray(vector, dtype=np.float32)

            # Verify vector size
            if len(vector) != 200:
                print(
                    f"  FAILED: vector has {len(vector)} "
                    "elements instead of 200"
                )
                failed += 1
                continue

            # Verify normalization
            norm = np.linalg.norm(vector)

            if not np.isclose(norm, 1.0, atol=1e-5):
                print(
                    f"  FAILED: vector norm = {norm}"
                )
                failed += 1
                continue

            # Mineral/sample name
            mineral_name = docx_file.stem

            # Insert sample
            sample = insert_sample(
                session,
                nombre_sample=mineral_name,
                investigador="Reference Dataset",
                ruta_imagen=str(docx_file)
            )

            # Insert 200-element vector
            insert_spectrum(
                session,
                sample_id=sample.id,
                vector=vector
            )

            successful += 1

            print(
                f"  OK: {mineral_name} "
                f"(200-dimensional vector)"
            )

        print("\n==============================")
        print("DATABASE BUILD COMPLETE")
        print("==============================")
        print(f"Successful: {successful}")
        print(f"Failed:     {failed}")
        print(f"Total:      {len(docx_files)}")

    finally:
        session.close()


if __name__ == "__main__":
    main()