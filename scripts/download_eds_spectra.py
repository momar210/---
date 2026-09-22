#!/usr/bin/env python3
"""
Script for downloading EDS spectra from public mineralogy collections.

Sources
-------
1. Simon Fraser University (SFU) — JPEG spectrum images
       https://www.sfu.ca/~marshall/sem/mineral.htm

2. McGill University — HTML-wrapped EDS spectra
       https://www.eps.mcgill.ca/~lang/EDSSPEC/mineralfile/

Traceability
------------
    - The SFU mineral list and filenames are based on the SFU mineral
      EDS spectrum collection.
    - The McGill file listing is discovered dynamically by parsing the
      Apache directory index at the URL above.
    - All downloaded files are stored locally for use as external /
      reference data in the mineral-identification system.

Purpose
-------
Download publicly available EDS spectrum files from SFU and McGill
and store them locally in per-source subdirectories.

Usage
-----
    python scripts/download_sfu_spectra.py
    python scripts/download_sfu_spectra.py --source sfu
    python scripts/download_sfu_spectra.py --source mcgill
    python scripts/download_sfu_spectra.py --source all
    python scripts/download_sfu_spectra.py --list --source mcgill
    python scripts/download_sfu_spectra.py --only Albite Pyrite -v
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Browser-like User-Agent. Some servers reject requests without one.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)

# Default network parameters.
DEFAULT_TIMEOUT = 30        # seconds per request
DEFAULT_RETRIES = 3         # attempts per file before giving up
DEFAULT_RETRY_DELAY = 2.0   # seconds between retries (exponential backoff)
DEFAULT_POLITE_DELAY = 0.5  # seconds between sequential requests
DEFAULT_WORKERS = 4         # parallel downloads

logger = logging.getLogger("eds_downloader")


# ============================================================================
# SOURCE 1 — SIMON FRASER UNIVERSITY (SFU)
# ============================================================================
# Traceability:
# The following URL is the base location of the SFU mineral EDS spectrum
# collection. Individual spectrum images are accessed by appending their
# corresponding filenames to this URL.
#
# Source:
# https://www.sfu.ca/~marshall/sem/mineral.htm

SFU_BASE_URL = "https://www.sfu.ca/~marshall/sem/"
SFU_OUTPUT_DIR = "external_spectra/SFU"

# Local filename pattern for downloaded SFU images.
SFU_LOCAL_FILENAME = "EDS_{mineral}.jpg"

# JPEG magic bytes used for response validation.
_JPEG_MAGIC = b"\xff\xd8"


# SFU mineral list. Traceability: names and filenames below were
# extracted from the SFU Mineral EDS Spectrum collection.
SFU_MINERALS: List[Tuple[str, str]] = [
    # --- SILICATES --------------------------------------------------------
    ("Adularia", "adul.jpg"),
    ("Albite", "albite.jpg"),
    ("Almandine", "Almandine.jpg"),
    ("Andalusite", "andradite.jpg"),
    ("Anorthite", "an1.jpg"),
    ("Anorthoclase", "anorthoclase.jpg"),
    ("Augite", "augite.jpg"),
    ("Axinite", "axinite.jpg"),
    ("Benitoite", "Ben2.jpg"),
    ("Beryl", "beryl.jpg"),
    ("Biotite", "Bio.jpg"),
    ("Chlorite", "Chl.jpg"),
    ("Chloritoid", "Chloritoid.jpg"),
    ("Cordierite", "Cordierite.jpg"),
    ("Diopside", "Dio6.jpg"),
    ("Epidote", "epi.jpg"),
    ("Fayalite", "Olivine.jpg"),
    ("Forsterite", "forsterite.jpg"),
    ("Glaucophane", "Glaucophane.jpg"),
    ("Grandidierite", "Grandidierite.jpg"),
    ("Hornblende", "hbl.jpg"),
    ("Hypersthene", "hypersthene.jpg"),
    ("Jadeite", "Jadeite.jpg"),
    ("Kornerupine", "Kornerupine.jpg"),
    ("Muscovite", "Muscovite.jpg"),
    ("Osumilite", "osumilite.jpg"),
    ("Paragonite", "Paragonite.jpg"),
    ("Phlogopite", "fph2.jpg"),
    ("Plagioclase", "Plag1.jpg"),
    ("Pyrope_G10", "pyropeG10.jpg"),
    ("Sapphirine", "Sapphirine.jpg"),
    ("Scapolite", "scapolite.jpg"),
    ("Sillimanite", "Sillimanite.jpg"),
    ("Staurolite", "Staurolite.jpg"),
    ("Stilpnomelane", "Stilpnomelane.jpg"),
    ("Titanite", "Titanite.jpg"),
    ("Tremolite", "Tre1.jpg"),
    ("Tourmaline", "tur.jpg"),
    ("Topaz", "Topaz.jpg"),
    ("Vesuvianite", "vesuvianite.jpg"),
    ("Willemite", "willemite.jpg"),
    ("Wollastonite", "Woll.jpg"),
    ("Zircon", "zircon.jpg"),

    # --- SULFIDES ---------------------------------------------------------
    ("Arsenopyrite", "apy.jpg"),
    ("Bismuthinite", "Bismuthinite.jpg"),
    ("Bornite", "Bornite.jpg"),
    ("Chalcocite", "chalcocite.jpg"),
    ("Chalcopyrite", "cpy.jpg"),
    ("Covellite", "CuS.jpg"),
    ("Enargite", "ena.jpg"),
    ("Galena", "galena.jpg"),
    ("Pyrrhotite", "FeS.jpg"),
    ("Pyrite", "FeS2.jpg"),
    ("Sphalerite", "ZnS.jpg"),
    ("Stibnite", "stibnite.jpg"),
    ("Stromeyerite", "Stromeyerite.jpg"),

    # --- SULFATES ---------------------------------------------------------
    ("Barite", "barite.jpg"),
    ("Celestite", "Celes.jpg"),

    # --- OXIDES -----------------------------------------------------------
    ("Brannerite", "brannerite.jpg"),
    ("Corundum", "Corundum.jpg"),
    ("Chromite", "Chromite.jpg"),
    ("Spinel", "Spinel.jpg"),
    ("Hematite", "Hematite.jpg"),
    ("Ilmenite", "ilmenite.jpg"),
    ("MnTiO3", "MnTiO3.jpg"),

    # --- CARBONATES -------------------------------------------------------
    ("Ankerite", "Ankerite.jpg"),
    ("Calcite", "cal.jpg"),
    ("Dolomite", "Dolo.jpg"),
    ("Rhodochrosite", "Rhodochrosite.jpg"),

    # --- PHOSPHATES -------------------------------------------------------
    ("Apatite", "Apa.jpg"),
    ("Beryllonite", "bep.jpg"),
    ("Monazite", "monazite.jpg"),
    ("Turquoise", "Turquoise.jpg"),

    # --- TUNGSTATES -------------------------------------------------------
    ("Scheelite", "Scheelite.jpg"),
    ("Wolframite", "Wolframite.jpg"),
    ("Wulfenite", "Wulfenite.jpg"),

    # --- ARSENIDES --------------------------------------------------------
    ("Nickeline", "NiAs.jpg"),

    # --- METALS / ALLOYS --------------------------------------------------
    ("Ag80Au20", "Ag80Au20.jpg"),
    ("Au80Ag20", "Au80Ag20.jpg"),
    ("Bismuth", "Bismuth.jpg"),
]


# ============================================================================
# SOURCE 2 — McGILL UNIVERSITY
# ============================================================================
# Traceability:
# The McGill EDS spectrum collection is hosted as an Apache directory
# index. Each mineral is represented by a small HTML file (about 1 KB)
# that embeds or links to the corresponding EDS spectrum image.
#
# Source:
# https://www.eps.mcgill.ca/~lang/EDSSPEC/mineralfile/

MCGILL_INDEX_URL = (
    "https://www.eps.mcgill.ca/~lang/EDSSPEC/mineralfile/"
)
MCGILL_OUTPUT_DIR = "external_spectra/McGill"

# Local filename pattern for downloaded McGill HTML files.
MCGILL_LOCAL_FILENAME = "EDS_{mineral}.html"


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class DownloadResult:
    """Result of a single file download attempt."""

    source: str          # "sfu" or "mcgill"
    mineral: str
    filename: str
    url: str
    status: str          # "downloaded", "skipped", or "failed"
    size_bytes: int = 0
    sha256: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in {"downloaded", "skipped"}


@dataclass
class DownloadSummary:
    """Aggregate summary of a download run."""

    source: str = ""
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[DownloadResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return self.downloaded + self.skipped

    def failures(self) -> List[DownloadResult]:
        return [r for r in self.results if r.status == "failed"]


# ============================================================================
# HTML PARSER FOR MCGILL DIRECTORY INDEX
# ============================================================================

class _IndexParser(html.parser.HTMLParser):
    """
    Extract file links from an Apache "Index of /" HTML page.

    Only entries ending with the target extension are collected. The
    parser ignores the "Parent Directory" link and any query strings.
    """

    def __init__(self, extension: str = ".html") -> None:
        super().__init__()
        self.extension = extension.lower()
        self.files: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        # Only keep entries whose href ends with the target extension.
        # Skip "Parent Directory" and query-only hrefs.
        clean = href.split("?", 1)[0]
        if clean.lower().endswith(self.extension):
            self.files.append(clean)


# ============================================================================
# CORE DOWNLOAD LOGIC
# ============================================================================

def _build_request(url: str) -> urllib.request.Request:
    """Construct an HTTP GET request with a browser-like User-Agent."""
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _sha256_of_file(path: Path, chunk_size: int = 65536) -> str:
    """Return the hex SHA-256 digest of a file on disk."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_payload(url: str, data: bytes, magic: Optional[bytes]) -> None:
    """
    Validate that the server returned the expected content.

    Raises:
        ValueError: If the payload is empty or does not start with
            the expected magic bytes.
    """
    if not data:
        raise ValueError("Empty response body")
    if magic is not None and not data.startswith(magic):
        raise ValueError(
            f"Response for {url} does not start with expected magic "
            f"{magic!r} (first bytes: {data[:4]!r})"
        )


def _download_one(
    source: str,
    mineral: str,
    remote_filename: str,
    base_url: str,
    local_filename: str,
    output_path: Path,
    *,
    timeout: int,
    retries: int,
    retry_delay: float,
    skip_existing: bool,
    magic: Optional[bytes] = None,
) -> DownloadResult:
    """
    Download a single file with retry and atomic-write semantics.

    The file is first written to a temporary path and then renamed into
    place, so a partial download never leaves a corrupted file on disk.
    """
    url = urllib.parse.urljoin(base_url, remote_filename)
    final_path = output_path / local_filename
    tmp_path = final_path.with_suffix(final_path.suffix + ".part")

    # --- Skip existing files ------------------------------------------------
    if skip_existing and final_path.exists() and final_path.stat().st_size > 0:
        return DownloadResult(
            source=source,
            mineral=mineral,
            filename=final_path.name,
            url=url,
            status="skipped",
            size_bytes=final_path.stat().st_size,
            sha256=_sha256_of_file(final_path),
        )

    last_error: Optional[str] = None

    for attempt in range(1, retries + 1):
        try:
            request = _build_request(url)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()

            _validate_payload(url, data, magic)

            # Atomic write: write to .part, then rename.
            tmp_path.write_bytes(data)
            tmp_path.replace(final_path)

            return DownloadResult(
                source=source,
                mineral=mineral,
                filename=final_path.name,
                url=url,
                status="downloaded",
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )

        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, ValueError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.debug(
                "Attempt %d/%d for %s/%s failed: %s",
                attempt, retries, source, mineral, last_error,
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(retry_delay * attempt)  # linear backoff

    return DownloadResult(
        source=source,
        mineral=mineral,
        filename=final_path.name,
        url=url,
        status="failed",
        error=last_error or "unknown error",
    )


# ============================================================================
# SFU CATALOG AND DOWNLOAD
# ============================================================================

def _sfu_catalog() -> List[Tuple[str, str]]:
    """Return the SFU mineral catalog as (mineral, remote_filename) pairs."""
    return list(SFU_MINERALS)


def download_sfu(
    output_dir: str = SFU_OUTPUT_DIR,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    polite_delay: float = DEFAULT_POLITE_DELAY,
    workers: int = 1,
    skip_existing: bool = True,
    only: Optional[Iterable[str]] = None,
) -> DownloadSummary:
    """
    Download all EDS spectra listed in the SFU mineral collection.

    Args:
        output_dir: Local directory for the downloaded images.
        timeout: Per-request timeout in seconds.
        retries: Number of attempts per file before giving up.
        retry_delay: Base delay in seconds for linear backoff.
        polite_delay: Delay between sequential requests (ignored when
            workers > 1).
        workers: Number of parallel download threads.
        skip_existing: If True, files already present are not re-downloaded.
        only: Optional iterable of mineral names (case-insensitive).

    Returns:
        A `DownloadSummary` with counts and per-file results.

    Traceability:
        Source: Simon Fraser University (SFU), Mineral EDS Spectra
        URL: https://www.sfu.ca/~marshall/sem/mineral.htm
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog = _sfu_catalog()
    if only:
        wanted = {name.lower() for name in only}
        catalog = [(m, f) for (m, f) in catalog if m.lower() in wanted]
        if not catalog:
            logger.warning("SFU: no minerals matched --only filter: %s",
                           sorted(wanted))

    return _run_download(
        source="sfu",
        catalog=catalog,
        base_url=SFU_BASE_URL,
        output_path=output_path,
        local_filename=lambda mineral, _remote: SFU_LOCAL_FILENAME.format(
            mineral=mineral
        ),
        magic=_JPEG_MAGIC,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        polite_delay=polite_delay,
        workers=workers,
        skip_existing=skip_existing,
    )


# ============================================================================
# McGILL CATALOG AND DOWNLOAD
# ============================================================================

def fetch_mcgill_catalog(timeout: int = DEFAULT_TIMEOUT) -> List[Tuple[str, str]]:
    """
    Fetch and parse the McGill directory index.

    Returns a list of (mineral_name, remote_filename) pairs, where the
    mineral name is derived from the filename by stripping the ".html"
    extension.

    Raises:
        urllib.error.URLError: If the index page cannot be fetched.
    """
    request = _build_request(MCGILL_INDEX_URL)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = _IndexParser(extension=".html")
    parser.feed(html)

    catalog: List[Tuple[str, str]] = []
    seen = set()
    for filename in parser.files:
        # Skip any href containing "/" (links to other directories).
        if "/" in filename:
            continue
        mineral = filename[:-5] if filename.lower().endswith(".html") else filename
        if mineral.lower() in seen:
            continue
        seen.add(mineral.lower())
        catalog.append((mineral, filename))

    catalog.sort(key=lambda pair: pair[0].lower())
    logger.info("McGill catalog: %d HTML files discovered.", len(catalog))
    return catalog


def download_mcgill(
    output_dir: str = MCGILL_OUTPUT_DIR,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    polite_delay: float = DEFAULT_POLITE_DELAY,
    workers: int = 1,
    skip_existing: bool = True,
    only: Optional[Iterable[str]] = None,
    catalog: Optional[List[Tuple[str, str]]] = None,
) -> DownloadSummary:
    """
    Download all HTML files listed in the McGill directory index.

    Each HTML file (about 1 KB) contains the EDS spectrum data for one
    mineral sample. The files are downloaded as-is and stored locally
    under `output_dir`.

    Args:
        output_dir: Local directory for the downloaded files.
        timeout: Per-request timeout in seconds.
        retries: Number of attempts per file before giving up.
        retry_delay: Base delay in seconds for linear backoff.
        polite_delay: Delay between sequential requests (ignored when
            workers > 1).
        workers: Number of parallel download threads.
        skip_existing: If True, files already present are not re-downloaded.
        only: Optional iterable of mineral names (case-insensitive).
        catalog: Optional override for the McGill catalog (useful for tests).

    Returns:
        A `DownloadSummary` with counts and per-file results.

    Traceability:
        Source: McGill University, EDS Spectra collection
        URL: https://www.eps.mcgill.ca/~lang/EDSSPEC/mineralfile/
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if catalog is None:
        try:
            catalog = fetch_mcgill_catalog(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - network errors vary
            logger.error("Could not fetch McGill catalog: %s", exc)
            return DownloadSummary(source="mcgill")

    if only:
        wanted = {name.lower() for name in only}
        catalog = [(m, f) for (m, f) in catalog if m.lower() in wanted]
        if not catalog:
            logger.warning("McGill: no minerals matched --only filter: %s",
                           sorted(wanted))

    return _run_download(
        source="mcgill",
        catalog=catalog,
        base_url=MCGILL_INDEX_URL,
        output_path=output_path,
        local_filename=lambda mineral, _remote: MCGILL_LOCAL_FILENAME.format(
            mineral=mineral
        ),
        magic=None,  # HTML files have no magic-byte requirement
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        polite_delay=polite_delay,
        workers=workers,
        skip_existing=skip_existing,
    )


# ============================================================================
# SHARED DOWNLOAD RUNNER
# ============================================================================

def _run_download(
    *,
    source: str,
    catalog: List[Tuple[str, str]],
    base_url: str,
    output_path: Path,
    local_filename,
    magic: Optional[bytes],
    timeout: int,
    retries: int,
    retry_delay: float,
    polite_delay: float,
    workers: int,
    skip_existing: bool,
) -> DownloadSummary:
    """
    Execute a download run for a given source.

    This is the shared engine used by both SFU and McGill downloads.
    It handles sequential vs. parallel execution, logging, and summary
    aggregation.
    """
    summary = DownloadSummary(source=source, total=len(catalog))

    logger.info("[%s] Starting download of %d files.", source.upper(), summary.total)
    logger.info("[%s] Destination: %s", source.upper(), output_path.absolute())
    if workers > 1:
        logger.info("[%s] Parallel workers: %d", source.upper(), workers)

    if workers <= 1:
        for mineral, remote in catalog:
            result = _download_one(
                source=source,
                mineral=mineral,
                remote_filename=remote,
                base_url=base_url,
                local_filename=local_filename(mineral, remote),
                output_path=output_path,
                timeout=timeout,
                retries=retries,
                retry_delay=retry_delay,
                skip_existing=skip_existing,
                magic=magic,
            )
            _log_result(result)
            summary.results.append(result)
            if polite_delay > 0 and result.status == "downloaded":
                time.sleep(polite_delay)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _download_one,
                    source=source,
                    mineral=mineral,
                    remote_filename=remote,
                    base_url=base_url,
                    local_filename=local_filename(mineral, remote),
                    output_path=output_path,
                    timeout=timeout,
                    retries=retries,
                    retry_delay=retry_delay,
                    skip_existing=skip_existing,
                    magic=magic,
                ): (mineral, remote)
                for (mineral, remote) in catalog
            }
            for future in as_completed(futures):
                result = future.result()
                _log_result(result)
                summary.results.append(result)

    for r in summary.results:
        if r.status == "downloaded":
            summary.downloaded += 1
        elif r.status == "skipped":
            summary.skipped += 1
        elif r.status == "failed":
            summary.failed += 1

    _print_summary(summary)
    return summary


# ============================================================================
# LOGGING / REPORTING
# ============================================================================

def _log_result(result: DownloadResult) -> None:
    """Log a single download result at the appropriate level."""
    if result.status == "downloaded":
        logger.info(
            "[%s][OK]     %-24s %6d B  sha256=%s",
            result.source.upper(),
            result.mineral,
            result.size_bytes,
            result.sha256[:12],
        )
    elif result.status == "skipped":
        logger.info(
            "[%s][SKIP]   %-24s (already present)",
            result.source.upper(),
            result.mineral,
        )
    else:
        logger.error(
            "[%s][FAIL]   %-24s %s",
            result.source.upper(),
            result.mineral,
            result.error,
        )


def _print_summary(summary: DownloadSummary) -> None:
    """Print a human-readable summary of a download run."""
    print(f"\n{'=' * 60}")
    print(f"[{summary.source.upper()}] Download completed:")
    print(f"  - Total:      {summary.total}")
    print(f"  - Downloaded: {summary.downloaded}")
    print(f"  - Skipped:    {summary.skipped}")
    print(f"  - Failed:     {summary.failed}")

    failures = summary.failures()
    if failures:
        print(f"\n[{summary.source.upper()}] Files with errors:")
        for r in failures:
            print(f"  - {r.mineral}: {r.error}")


def configure_logging(verbose: bool = False) -> None:
    """Configure the root logger for CLI usage."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download EDS spectra from public mineralogy collections.\n"
            "  SFU:    https://www.sfu.ca/~marshall/sem/mineral.htm\n"
            "  McGill: https://www.eps.mcgill.ca/~lang/EDSSPEC/mineralfile/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["sfu", "mcgill", "all"],
        default="all",
        help="Which source(s) to download from.",
    )
    parser.add_argument(
        "--sfu-dir", default=SFU_OUTPUT_DIR,
        help="Output directory for SFU files.",
    )
    parser.add_argument(
        "--mcgill-dir", default=MCGILL_OUTPUT_DIR,
        help="Output directory for McGill files.",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--retries", type=int, default=DEFAULT_RETRIES,
        help="Number of attempts per file before giving up.",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=DEFAULT_RETRY_DELAY,
        help="Base delay for backoff between retries.",
    )
    parser.add_argument(
        "--polite-delay", type=float, default=DEFAULT_POLITE_DELAY,
        help="Delay between sequential requests (workers=1 only).",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Number of parallel download threads. Use 1 to be polite.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="MINERAL",
        help="Only download the listed minerals (case-insensitive).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List the catalog for the selected source(s) and exit.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    return parser.parse_args(argv)


def _list_catalogs(sources: Iterable[str], timeout: int) -> None:
    """Print the catalog(s) for the requested source(s)."""
    for source in sources:
        if source == "sfu":
            catalog = _sfu_catalog()
            print(f"\n=== SFU catalog ({len(catalog)} entries) ===")
            for name, remote in catalog:
                print(f"  {name:<24} -> {SFU_BASE_URL}{remote}")
        elif source == "mcgill":
            try:
                catalog = fetch_mcgill_catalog(timeout=timeout)
                print(f"\n=== McGill catalog ({len(catalog)} entries) ===")
                for name, remote in catalog:
                    print(f"  {name:<24} -> {MCGILL_INDEX_URL}{remote}")
            except Exception as exc:  # noqa: BLE001
                print(f"\n=== McGill catalog unavailable: {exc} ===")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 0 on success, 1 if any download failed."""
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    # Determine which sources to run.
    if args.source == "all":
        sources = ["sfu", "mcgill"]
    else:
        sources = [args.source]

    if args.list:
        _list_catalogs(sources, timeout=args.timeout)
        return 0

    overall_failed = 0
    summaries: List[DownloadSummary] = []

    for source in sources:
        if source == "sfu":
            summary = download_sfu(
                output_dir=args.sfu_dir,
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
                polite_delay=args.polite_delay,
                workers=args.workers,
                skip_existing=not args.force,
                only=args.only,
            )
        else:  # mcgill
            summary = download_mcgill(
                output_dir=args.mcgill_dir,
                timeout=args.timeout,
                retries=args.retries,
                retry_delay=args.retry_delay,
                polite_delay=args.polite_delay,
                workers=args.workers,
                skip_existing=not args.force,
                only=args.only,
            )
        summaries.append(summary)
        overall_failed += summary.failed

    # Final combined report when more than one source ran.
    if len(summaries) > 1:
        print(f"\n{'=' * 60}")
        print("Overall combined report:")
        total_all = sum(s.total for s in summaries)
        downloaded_all = sum(s.downloaded for s in summaries)
        skipped_all = sum(s.skipped for s in summaries)
        print(f"  - Sources:    {len(summaries)}")
        print(f"  - Total:      {total_all}")
        print(f"  - Downloaded: {downloaded_all}")
        print(f"  - Skipped:    {skipped_all}")
        print(f"  - Failed:     {overall_failed}")
        print(f"{'=' * 60}")

    return 0 if overall_failed == 0 else 1


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())