# Mineral Identification from EDS Spectra

A modular Python system for identifying minerals from Energy-Dispersive
X-ray Spectroscopy (EDS) spectra. The pipeline extracts spectrum images
from DOCX files, vectorizes them into 200-dimensional L2-normalized
feature vectors, stores them in a SQLite reference database, and
compares unknown samples against the database using cosine similarity.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
- [Database Schema](#database-schema)
- [Comparison Metrics](#comparison-metrics)
- [Figure Generation](#figure-generation)
- [Data Sources](#data-sources)
- [Command-Line Reference](#command-line-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Traceability and Reproducibility](#traceability-and-reproducibility)
- [License](#license)

---

## Overview

The system answers a single question:

> *Given an EDS spectrum image, which mineral from a reference database
> does it most closely resemble?*

The answer is computed by:

1. Extracting the spectrum image from a `.docx` container.
2. Converting it into a fixed-length 1D feature vector.
3. Comparing that vector against every reference vector in the database.
4. Returning the top matches ranked by cosine similarity.

The design emphasizes **reproducibility**: the same input image always
produces the same vector, and the same vector always produces the same
ranking.

---

## Features

- **DOCX-native input.** Reads spectrum images directly from `.docx`
  files without manual extraction.
- **Deterministic vectorization.** A fixed pipeline (Gaussian smoothing,
  grayscale, binarization, ROI cropping, column-mean signature,
  interpolation, L2 normalization) guarantees identical output for
  identical input.
- **Four comparison metrics.** Cosine similarity (recommended),
  Pearson correlation, Euclidean distance, and Manhattan distance.
- **SQLite reference database.** A single file (`minerales_eds.db`)
  holds every sample and its vector.
- **Mineral name normalization.** Recognizes Spanish/English equivalents
  (e.g. `calcita` ↔ `calcite`).
- **Batch operations.** Build the database from a folder of DOCX files
  in one command.
- **Public reference downloaders.** Built-in scripts to fetch SFU and
  McGill EDS spectrum collections.
- **Publication-quality figures.** Automatic generation of diagrams for
  reports, slides, and papers.
- **Headless-safe.** Every figure script uses the `Agg` backend, so the
  full pipeline runs on servers without a display.

---

## Project Structure

```
project/
├── main.py                              # CLI entry point (single DOCX -> DB -> compare)
├── requirements.txt
├── requirements-dev.txt
├── README.md
│
├── data/
│   └── temp_images/                     # Legacy; no longer used (in-memory only)
│
├── external_spectra/                    # Downloaded reference collections
│   ├── SFU/
│   └── McGill/
│
├── reference_data/                      # Reference DOCX files used to build the DB
│   ├── Albite.docx
│   ├── Pyrite.docx
│   └── ...
│
├── tests/
│   └── test_data/                       # Validation DOCX files
│
├── diagrams/                            # Output figures (created on demand)
│
├── src/
│   ├── analysis/
│   │   ├── vectorize.py                 # Spectrum image -> 200D vector
│   │   └── compare.py                   # Cosine similarity, comparison API
│   │
│   ├── database/
│   │   ├── connection.py                # Engine, session factory
│   │   ├── models.py                    # ORM models (Sample, VectorizedSpectrum)
│   │   └── queries.py                   # CRUD helpers
│   │
│   └── parsers/
│       └── docx_parser.py               # DOCX -> embedded image -> vector
│
└── scripts/
    ├── build_reference_database.py      # Populate the DB from reference_data/
    ├── compare_metrics.py               # Validate cosine vs. other metrics
    ├── download_sfu_spectra.py          # Download SFU + McGill collections
    ├── generate_diagrams.py             # Documentation figures
    ├── generate_metric_comparison.py    # Metric comparison figure
    ├── generate_presentation_figures.py # Testing / objectives / vectorization
    └── generate_project_figures.py      # Mineral distribution / pipeline / etc.
```

---

## Installation

### Requirements

- Python 3.10 or newer
- pip

### Steps

```bash
git clone <repository-url>
cd <repository-directory>

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### requirements.txt

```txt
# Core numerical / scientific stack
numpy>=1.24,<2.0

# Image processing (swap for opencv-python-headless on servers)
opencv-python>=4.8,<5.0
Pillow>=10.0,<11.0

# Database / ORM
SQLAlchemy>=2.0,<3.0

# DOCX parsing
python-docx>=1.1,<2.0

# Plotting / visualization
matplotlib>=3.7,<4.0

# Testing
pytest>=7.4,<9.0
pytest-cov>=4.1,<6.0
```

### Headless environments

On servers without a display (Docker, CI, SSH), replace `opencv-python`
with the headless build:

```bash
python -m pip uninstall -y opencv-python
python -m pip install "opencv-python-headless>=4.8,<5.0"
```

All figure scripts already use the `Agg` backend, so no further changes
are needed.

---

## Quick Start

### 1. Create the database

```bash
python -c "from src.database.connection import create_tables; create_tables()"
```

This creates `minerales_eds.db` in the current working directory.

### 2. Populate it from reference DOCX files

Place your reference `.docx` files in `reference_data/`, then run:

```bash
python scripts/build_reference_database.py
```

Expected output:

```
Found 101 reference DOCX files in 'reference_data'.
Processing: Albite.docx
  [OK]      Albite (200-dimensional vector)
...
========================================
DATABASE BUILD COMPLETE
========================================
Successful: 100%
Skipped:    0%
Failed:     0%
Total:      100%
```

### 3. Identify an unknown sample

```bash
python main.py path/to/unknown_spectrum.docx
```

The script will:

1. Extract and vectorize the unknown spectrum.
2. Insert it into the database.
3. Compare it against every reference sample.
4. Print the top matches by cosine similarity.

## Pipeline

The vectorization pipeline is fixed and deterministic:

```
DOCX file
  |  extract embedded image (in memory, no temp files)
  v
Image with shape (400, 512, 3)
  |  normalize to float32 in [0, 1]
  v
Normalized image
  |  5x5 Gaussian smoothing
  v
Smoothed image
  |  BGR -> grayscale
  v
Grayscale image
  |  threshold < 0.99 -> binary mask
  v
Binary mask
  |  crop to ROI (rows 150-250, auto columns)
  v
Cropped mask
  |  column-wise mean
  v
1D spectral signature
  |  linear interpolation to 200 values
  v
200-element vector
  |  L2 normalization
  v
Final feature vector (||v|| = 1)
```

For reproducibility, this chain must remain intact. The comparison step
depends on all vectors being produced by the same pipeline.

### Key parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `vector_size` | 200 | Length of the output vector |
| `threshold` | 0.99 | Grayscale binarization cutoff |
| `row_bounds` | (150, 250) | Vertical ROI |
| `method` | "mean" | Aggregation across rows |
| `expected_shape` | (400, 512, 3) | Expected image shape |

All parameters are configurable through the public functions in
`src/analysis/vectorize.py` and `src/parsers/docx_parser.py`.

---

## Database Schema

The database is a single SQLite file (`minerales_eds.db`) with two tables.

### `samples`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER | Primary key |
| `sample_name` | STRING | Mineral / sample name |
| `researcher` | STRING | Researcher name (nullable) |
| `image_path` | STRING | Path to source file (nullable) |
| `date` | DATETIME | Creation timestamp |

### `vectorized_spectra`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER | Primary key |
| `sample_id` | INTEGER | Foreign key to `samples.id` |
| `vector` | TEXT | JSON array of 200 floats |

Relationship: **1 sample : 1 vector** (one-to-one).

### Inspecting the database

```bash
sqlite3 minerales_eds.db ".tables"
sqlite3 minerales_eds.db ".schema samples"
sqlite3 minerales_eds.db "SELECT COUNT(*) FROM samples;"
sqlite3 minerales_eds.db "SELECT id, nombre_sample FROM samples LIMIT 5;"
```

---

## Comparison Metrics

The system supports four metrics. **Cosine similarity is recommended**
for EDS spectra because it is scale invariant: two spectra of the same
mineral acquired at different intensities will still have similarity
close to 1.0.

| Metric | Range | Scale invariant? | Recommended |
|--------|-------|------------------|-------------|
| **Cosine Similarity** | [0, 1] | Yes | Yes |
| Pearson Correlation | [-1, 1] | Yes (after centering) | Alternative |
| Euclidean Distance | [0, ∞) | No | No |
| Manhattan Distance | [0, ∞) | No | No |

### Why cosine similarity

Cosine similarity measures the **angle** between two vectors:

```
                   A . B
sim(A, B) = -------------------
            ||A|| * ||B||
```

If you double the intensity of a spectrum, its direction does not
change — only its magnitude. Cosine similarity ignores magnitude, so it
returns the same value regardless of acquisition time or detector gain.
This makes it the correct choice for matching spectra whose *shape* is
diagnostic but whose *absolute intensity* is not.

### Validating this claim

Run the metric comparison script:

```bash
python scripts/compare_metrics.py --tests tests/test_data
```

It produces:

- `diagrams/metric_accuracy_comparison.png` — accuracy bar chart
- `diagrams/metric_ranking_table.png` — ranking table
- `diagrams/scale_invariance_comparison.png` — 4-panel demonstration
- `diagrams/metric_comparison_results.json` — raw numbers

---

## Figure Generation

All figures are saved to `diagrams/` unless overridden with `--output`.

| Script | Figures produced |
|--------|------------------|
| `scripts/generate_diagrams.py` | ER diagram, pipeline, cosine formula, vector interpretation, grayscale spectrum |
| `scripts/generate_metric_comparison.py` | Final metric comparison figure |
| `scripts/generate_presentation_figures.py` | Testing results, objectives, vectorization concept |
| `scripts/generate_project_figures.py` | Mineral distribution, pipeline, architecture, validation, ER, confusion matrix, spectral comparison |
| `scripts/compare_metrics.py` | Accuracy comparison, ranking table, scale invariance |

Run any of them directly:

```bash
python scripts/generate_diagrams.py
python scripts/generate_presentation_figures.py --output slides
```

List available figures without generating:

```bash
python scripts/generate_presentation_figures.py --list
```

Regenerate only some figures:

```bash
python scripts/generate_diagrams.py --only er pipeline formula
```

---

## Data Sources

The project supports two public EDS reference collections. Both are
downloaded by `scripts/download_sfu_spectra.py`.

### Simon Fraser University (SFU)

- **Source:** https://www.sfu.ca/~marshall/sem/mineral.htm
- **Format:** Individual JPEG spectrum images
- **Files:** ~79 mineral spectra
- **Local destination:** `external_spectra/SFU/`

### McGill University

- **Source:** https://www.eps.mcgill.ca/~lang/EDSSPEC/mineralfile/
- **Format:** HTML-wrapped spectra (one `.html` file per mineral)
- **Files:** ~180 HTML files, discovered dynamically from the directory index
- **Local destination:** `external_spectra/McGill/`

### Download both

```bash
python scripts/download_sfu_spectra.py --source all
```

### Download one

```bash
python scripts/download_sfu_spectra.py --source sfu
python scripts/download_sfu_spectra.py --source mcgill
```

### List the catalog without downloading

```bash
python scripts/download_sfu_spectra.py --source mcgill --list
```

### Polite mode (single-threaded, extra delay)

```bash
python scripts/download_sfu_spectra.py --workers 1 --polite-delay 2.0
```

The downloader records a SHA-256 hash for every file, enabling later
verification that the remote collections have not changed.

---

## Command-Line Reference

### `main.py` — identify one sample

```
python main.py [docx_file] [options]

  docx_file             Path to the .docx file (default: data/Eds_magnetite.docx)
  --number NAME         Sample name to store (default: "Magnetite - Test Sample")
  --investigador NAME   Researcher name (default: "Test System")
  --vector-size N       Vector length (default: 200)
  --umbral FLOAT        Similarity threshold (default: 0.5)
  --verbose, -v         Enable DEBUG logging
```

### `scripts/build_reference_database.py` — build the DB

```
python scripts/build_reference_database.py [options]

  --dir PATH              Reference directory (default: reference_data)
  --vector-size N         Expected vector length (default: 200)
  --researcher NAME       Researcher name to store (default: "Reference Dataset")
  --no-normalize-check    Skip L2 norm ~1.0 verification
  --force                 Re-import even if the sample name exists
  --reset                 Drop and recreate all tables first
  --skip-verify           Skip the final verification report
  --verbose, -v           Enable DEBUG logging
```

### `scripts/compare_metrics.py` — validate metric choice

```
python scripts/compare_metrics.py [options]

  --tests PATH            Validation .docx directory (default: tests/test_data)
  --output PATH           Output directory (default: diagrams)
  --strict                Log every incorrect prediction
  --verbose, -v           Enable DEBUG logging
```

### `scripts/download_sfu_spectra.py` — download references

```
python scripts/download_sfu_spectra.py [options]

  --source {sfu,mcgill,all}    Which collection(s) to download (default: all)
  --sfu-dir PATH               SFU output directory
  --mcgill-dir PATH            McGill output directory
  --timeout N                  Per-request timeout in seconds (default: 30)
  --retries N                  Attempts per file (default: 3)
  --retry-delay FLOAT          Backoff between retries (default: 2.0)
  --polite-delay FLOAT         Delay between requests (workers=1 only)
  --workers N                  Parallel download threads (default: 4)
  --force                      Re-download even if files exist
  --only NAME [NAME ...]       Only download listed minerals
  --list                       Print the catalog and exit
  --verbose, -v                Enable DEBUG logging
```

---

## Configuration

The database URL is resolved in this priority order:

1. `DATABASE_URL` environment variable, if set.
2. `TESTING=1` → in-memory SQLite.
3. Default → `sqlite:///minerales_eds.db`.

### Examples

**Local development (default):**

```bash
python main.py data/sample.docx
```

**Testing:**

```bash
TESTING=1 pytest
```

**Custom database location:**

```bash
export DATABASE_URL="sqlite:////absolute/path/to/minerales_eds.db"
python scripts/build_reference_database.py
```

**PostgreSQL (production):**

```bash
export DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/minerals"
python main.py data/sample.docx
```

### Environment variables

| Variable | Effect |
|----------|--------|
| `DATABASE_URL` | Full SQLAlchemy database URL (highest priority) |
| `TESTING` | Truthy value enables in-memory SQLite and disables SQL echo |
| `SQL_ECHO` | Truthy value enables SQL statement logging |

Truthy values are: `1`, `true`, `yes`, `on`, `y`, `t` (case-insensitive).

---

## Testing

Run the full test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Run a single module:

```bash
pytest tests/test_vectorize.py -v
```

Run with in-memory database:

```bash
TESTING=1 pytest
```

Tests use the in-memory SQLite backend configured in
`src/database/connection.py`. No files are written to disk during the
test run.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: libGL.so.1` | `opencv-python` on a headless server | `pip install opencv-python-headless` |
| `ModuleNotFoundError: No module named 'docx'` | Wrong package name | `pip install python-docx` (not `docx`) |
| `no such table: samples` | Tables not created | `python -c "from src.database.connection import create_tables; create_tables()"` |
| `Could not read image` in logs | Missing or corrupt DOCX | Verify the file opens in Word/LibreOffice |
| `No image with shape (400, 512, 3) found` | Image size mismatch | Inspect the DOCX with `python -c "from src.parsers.docx_parser import extract_images_from_docx; print([i.shape for i in extract_images_from_docx('file.docx')])"` |
| `cropped mask is empty` | ROI rows 150-250 contain no signal | Check the source image; adjust `row_bounds` |
| Vector norm is zero | All-black or all-white image | Verify the source image |
| Database locked | Concurrent writers | Ensure only one process writes at a time; WAL mode is already enabled for file DBs |
| Figures missing | `diagrams/` not created | Scripts create it automatically; verify write permissions |
| `AttributeError: sample_name` | Stale field name | The correct column is `sample_name` |

### Diagnostic one-liner

```bash
python -c "
from src.database.connection import SessionLocal
from src.database.models import Sample, VectorizedSpectrum
from sqlalchemy import select, func
with SessionLocal() as s:
    print('Samples:', s.scalar(select(func.count()).select_from(Sample)))
    print('Vectors:', s.scalar(select(func.count()).select_from(VectorizedSpectrum)))
"
```
---

## Traceability and Reproducibility

The system is designed so that every result can be traced to its source.

### Deterministic pipeline

The vectorization pipeline is purely deterministic:

- No random seeds are used at any stage.
- Every stage is a pure function of the previous stage.
- The same input image always produces the same vector.

### Source attribution

Every reference collection retains a link to its original source:

- SFU filenames are recorded in `MINERALS_SFU` in
  `scripts/download_sfu_spectra.py`.
- McGill files are recorded with their original HTML filenames, parsed
  from the live directory index.
- Every downloaded file's SHA-256 hash is recorded during download.

### Reference database integrity

`scripts/build_reference_database.py` performs a final verification
report:

- Sample count vs. vector count.
- Minimum and maximum vector length.
- A pass/fail verdict per vector against the expected length.

A well-formed run produces:

```
Samples: 
Vectors: 
Vector length range: [200, 200]
Expected length: 200
Result: PASS - every vector has exactly 200 values.
```

### Reproducing a published result

1. Record the SHA-256 of every reference image.
2. Pin exact dependency versions with `pip freeze > requirements.lock.txt`.
3. Save the resulting `metric_comparison_results.json` alongside any
   published figure.

Together these three artifacts fully determine a run.

---

## License

This project uses publicly available reference data from:

- Simon Fraser University (SFU) — https://www.sfu.ca/~marshall/sem/mineral.htm
- McGill University — https://www.eps.mcgill.ca/~lang/EDSSPEC/mineralfile/

Example: Quick Run
project/
├── examples/
│   ├── README.md
│   ├── sample_spectrum.docx          # One example input
│   ├── expected_output.json          # Reference output for comparison
│   └── quick_test.py                 # Self-contained example script