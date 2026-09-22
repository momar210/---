#!/usr/bin/env python3
"""
Automatic setup script for the Mineral Identification MVP.

This script:
    1. Verifies the Python version (>= 3.9).
    2. Installs dependencies from requirements.txt.
    3. Optionally populates the reference database from a folder of
       reference .docx files (default: reference_data/).
    4. Prints the next steps to run the application.

Usage:
    python setup.py
    python setup.py --yes                     # non-interactive: do everything
    python setup.py --no-populate             # install only, do not populate
    python setup.py --reference-dir PATH      # custom reference folder
    python setup.py --skip-deps               # assume deps are installed
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


# ============================================================================
# CONFIGURATION
# ============================================================================

MIN_PYTHON = (3, 9)

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_reference_database.py"
DEFAULT_REFERENCE_DIR = PROJECT_ROOT / "reference_data"

logger = logging.getLogger("setup")


# ============================================================================
# LOGGING HELPERS
# ============================================================================

def _info(msg: str) -> None:
    print(f"\n[INFO] {msg}")


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _error(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


# ============================================================================
# SHELL EXECUTION
# ============================================================================

def run_command(
    command: Sequence[str],
    description: str,
    *,
    cwd: Optional[Path] = None,
) -> bool:
    """
    Run a subprocess command and report the result.

    Args:
        command: Argument list (never a shell string, to avoid quoting bugs).
        description: Human-readable description of what is being run.
        cwd: Working directory for the command.

    Returns:
        True on exit code 0, False otherwise.
    """
    _info(f"{description}...")
    logger.debug("Running: %s (cwd=%s)", " ".join(command), cwd or Path.cwd())

    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        _ok(f"{description} completed")
        if result.stdout and logger.isEnabledFor(logging.DEBUG):
            logger.debug("stdout: %s", result.stdout.strip())
        return True
    except FileNotFoundError as exc:
        _error(f"{description} failed: command not found ({exc})")
        return False
    except subprocess.CalledProcessError as exc:
        _error(f"{description} failed with exit code {exc.returncode}")
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return False


# ============================================================================
# PYTHON VERSION CHECK
# ============================================================================

def check_python_version(min_version: tuple = MIN_PYTHON) -> bool:
    """Verify that the running Python interpreter is >= min_version."""
    version = sys.version_info
    if (version.major, version.minor) < min_version:
        _error(
            f"Python {min_version[0]}.{min_version[1]} or newer is required."
        )
        _error(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    _ok(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True


# ============================================================================
# DEPENDENCIES
# ============================================================================

def install_dependencies() -> bool:
    """Install the project dependencies from requirements.txt."""
    if not REQUIREMENTS_FILE.exists():
        _error(f"Requirements file not found: {REQUIREMENTS_FILE}")
        return False

    return run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        "Installing dependencies",
    )


# ============================================================================
# DATABASE POPULATION
# ============================================================================

def populate_database(reference_dir: Path) -> bool:
    """Run the reference database builder script."""
    if not BUILD_SCRIPT.exists():
        _error(f"Build script not found: {BUILD_SCRIPT}")
        return False

    if not reference_dir.is_dir():
        _error(f"Reference directory not found: {reference_dir}")
        return False

    docx_count = len(list(reference_dir.glob("*.docx")))
    if docx_count == 0:
        _warn(f"No .docx files found in {reference_dir}")
        return False

    _info(f"Found {docx_count} reference .docx file(s) in {reference_dir}")

    return run_command(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--dir", str(reference_dir),
        ],
        "Building the reference database",
        cwd=PROJECT_ROOT,
    )


def prompt_yes_no(question: str, *, default: bool = True) -> bool:
    """
    Prompt the user for a yes/no answer.

    Accepts: y, yes, s, si, sí (and uppercase). Empty input returns `default`.
    """
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        answer = input(question + suffix).strip().lower()
    except EOFError:
        return default

    if not answer:
        return default
    return answer in {"y", "yes", "s", "si", "sí"}


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Automatic setup for the mineral identification MVP: "
            "check Python, install dependencies, and optionally "
            "populate the reference database."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Non-interactive: assume 'yes' to all prompts.",
    )
    parser.add_argument(
        "--no-populate", action="store_true",
        help="Install dependencies only; do not populate the database.",
    )
    parser.add_argument(
        "--skip-deps", action="store_true",
        help="Assume dependencies are already installed; skip pip install.",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(DEFAULT_REFERENCE_DIR),
        help="Directory containing reference .docx files.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool = False) -> None:
    """Configure the root logger."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================================
# MAIN
# ============================================================================

def print_next_steps() -> None:
    """Print the commands to run after setup completes."""
    print("\n" + "=" * 70)
    print("Setup complete.")
    print("=" * 70)
    print("\nNext steps:")
    print("  python -m streamlit run app.py            # Launch the web application")
    print("  python scripts/build_reference_database.py  # Rebuild the database")
    print("  python main.py path/to/spectrum.docx      # Identify a single sample")
    print("  python scripts/compare_metrics.py         # Validate metric choice")
    print("\nDocumentation: see README.md for the full command reference.")


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the full setup.

    Returns:
        0 on success, 1 if any mandatory step failed.
    """
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    print("=" * 70)
    print("Mineral Identification System - MVP Setup")
    print("=" * 70)

    # --- Step 1: Python version --------------------------------------------
    if not check_python_version():
        return 1

    # --- Step 2: Dependencies ----------------------------------------------
    if args.skip_deps:
        _info("Skipping dependency installation (--skip-deps)")
    else:
        if not install_dependencies():
            _error("Dependency installation failed.")
            return 1

    # --- Step 3: Populate the database -------------------------------------
    reference_dir = Path(args.reference_dir).resolve()

    if args.no_populate:
        _info("Skipping database population (--no-populate)")
    else:
        do_populate = args.yes or prompt_yes_no(
            f"\nPopulate the reference database from '{reference_dir}'?",
            default=True,
        )
        if do_populate:
            if not populate_database(reference_dir):
                _warn(
                    "Database population failed or produced no changes. "
                    f"You can run it manually later with:\n"
                    f"    python {BUILD_SCRIPT.relative_to(PROJECT_ROOT)} "
                    f"--dir {reference_dir}"
                )
        else:
            _info(
                "Database population skipped. You can run it later with:\n"
                f"    python {BUILD_SCRIPT.relative_to(PROJECT_ROOT)}"
            )

    # --- Done --------------------------------------------------------------
    print_next_steps()
    return 0


if __name__ == "__main__":
    sys.exit(main())