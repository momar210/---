
# src/parsers/docx_parser.py
"""
DOCX spectrum extraction module.

Extracts and vectorizes the EDS spectrum image embedded in a DOCX file.

Pipeline
--------
    1. Open the DOCX document.
    2. Extract every embedded image.
    3. Decode each image in memory (no temporary files).
    4. Select the image whose OpenCV shape matches the expected
       spectrum dimensions, e.g. (400, 512, 3).
    5. Pass that image to `vectorize_spectrum()`.
    6. Return the resulting 200-dimensional vector.

For reproducibility, the critical chain must remain:

    DOCX -> extracted EDS image -> vectorize_spectrum()
         -> 200-element vector -> L2 normalization
         -> four similarity/distance metrics.

Main public functions:
    - extract_and_vectorize_spectrum(docx_path, ...) -> Optional[np.ndarray]
    - extract_images_from_docx(docx_path) -> List[ExtractedImage]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT

from src.analysis.vectorize import vectorize_spectrum


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Expected shape of the EDS spectrum image as returned by cv2.imread:
# (height, width, channels). This criterion must be updated if the
# source spectra ever have different dimensions.
DEFAULT_EXPECTED_SHAPE: Tuple[int, int, int] = (400, 512, 3)

# Default vector size produced by the vectorization step.
DEFAULT_VECTOR_SIZE = 200


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class ExtractedImage:
    """
    A single image extracted from a DOCX file.

    The raw bytes are kept in memory; callers can decode them with
    `cv2.imdecode` without touching the filesystem.
    """

    name: str
    content_type: str
    blob: bytes
    width: int = 0
    height: int = 0
    channels: int = 0

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the OpenCV shape tuple (height, width, channels)."""
        return (self.height, self.width, self.channels)

    def decode(self) -> Optional[np.ndarray]:
        """
        Decode the raw bytes into a BGR numpy array.

        Returns:
            A numpy array in BGR order, or None if decoding fails.
        """
        buffer = np.frombuffer(self.blob, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return image


# ============================================================================
# IMAGE EXTRACTION
# ============================================================================

def extract_images_from_docx(docx_path: str) -> List[ExtractedImage]:
    """
    Extract every embedded image from a DOCX file, in document order.

    Images are kept in memory. Nothing is written to disk. Non-image
    relationships (styles, headers, hyperlinks, etc.) are skipped.

    Args:
        docx_path: Path to the DOCX file.

    Returns:
        A list of `ExtractedImage` objects, one per embedded image.

    Raises:
        FileNotFoundError: If the DOCX file does not exist.
        docx.opc.exceptions.PackageNotFoundError: If the file is not a
            valid DOCX package.
    """
    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    document = Document(str(path))
    images: List[ExtractedImage] = []

    for rel in document.part.rels.values():
        # The canonical check is the relationship type, not the target
        # reference. This avoids picking up non-image parts that happen
        # to have "image" in their filename.
        if rel.reltype != RT.IMAGE:
            continue

        try:
            blob = rel.target_part.blob
            content_type = getattr(rel.target_part, "content_type", "")
            part_name = getattr(rel.target_part, "partname", None)
            filename = str(part_name) if part_name is not None else "image"
        except Exception as exc:  # noqa: BLE001 - malformed DOCX can raise anything
            logger.warning("Skipping unreadable image relationship: %s", exc)
            continue

        img = ExtractedImage(
            name=filename,
            content_type=content_type,
            blob=blob,
        )

        # Try to decode eagerly so we know the dimensions up front. If
        # decoding fails, we still return the record (caller decides).
        decoded = img.decode()
        if decoded is not None:
            img.height, img.width, img.channels = decoded.shape
        else:
            logger.debug("Could not decode embedded image %s", filename)

        images.append(img)

    logger.debug("Extracted %d image(s) from %s", len(images), docx_path)
    return images


# ============================================================================
# SPECTRUM SELECTION
# ============================================================================

def _select_spectrum_image(
    images: List[ExtractedImage],
    expected_shape: Tuple[int, int, int],
) -> Optional[ExtractedImage]:
    """
    Return the first image whose decoded shape matches `expected_shape`.

    Args:
        images: Images extracted from a DOCX file.
        expected_shape: Expected (height, width, channels) tuple.

    Returns:
        The matching `ExtractedImage`, or None if no match is found.
    """
    for img in images:
        if img.shape == expected_shape:
            logger.debug(
                "Selected spectrum image %s with shape %s",
                img.name, img.shape,
            )
            return img

    logger.warning(
        "No image with shape %s found. Available shapes: %s",
        expected_shape,
        [img.shape for img in images],
    )
    return None


# ============================================================================
# PUBLIC API
# ============================================================================

def extract_and_vectorize_spectrum(
    docx_path: str,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    expected_shape: Tuple[int, int, int] = DEFAULT_EXPECTED_SHAPE,
) -> Optional[np.ndarray]:
    """
    Extract and vectorize the EDS spectrum image embedded in a DOCX file.

    Processing steps:
        1. Open the DOCX document.
        2. Extract every embedded image to memory.
        3. Identify the spectrum image whose shape matches `expected_shape`.
        4. Vectorize the selected image using `vectorize_spectrum`.
        5. Return the resulting vector.

    Args:
        docx_path: Path to the DOCX file containing the EDS spectrum.
        vector_size: Number of elements in the output spectrum vector.
        expected_shape: Expected (height, width, channels) of the spectrum
            image. Default is (400, 512, 3).

    Returns:
        The L2-normalized 200-dimensional vector, or None if no image
        with the expected shape was found or if vectorization failed.

    Notes:
        The spectrum image dimensions are controlled by `expected_shape`.
        Change this argument if the source DOCX files use a different
        image size.
    """
    try:
        images = extract_images_from_docx(docx_path)
    except FileNotFoundError:
        logger.error("DOCX file does not exist: %s", docx_path)
        return None
    except Exception as exc:  # noqa: BLE001 - python-docx raises varied errors
        logger.exception("Failed to open DOCX file %s: %s", docx_path, exc)
        return None

    if not images:
        logger.warning("No embedded images found in %s", docx_path)
        return None

    selected = _select_spectrum_image(images, expected_shape)
    if selected is None:
        return None

    # Vectorize directly from the in-memory bytes. The vectorize module
    # reads from a path, so we decode once here and pass the decoded
    # array through a small adapter. This avoids writing temp files.
    decoded = selected.decode()
    if decoded is None:
        logger.error(
            "Selected image %s could not be decoded despite matching shape.",
            selected.name,
        )
        return None

    vector = _vectorize_ndarray(
        decoded,
        vector_size=vector_size,
    )

    return vector


def _vectorize_ndarray(
    image: np.ndarray,
    vector_size: int,
) -> Optional[np.ndarray]:
    """
    Internal adapter: vectorize a decoded numpy image without touching disk.

    The public `vectorize_spectrum` API takes a path. Rather than write
    the image to a temporary file, we run the same pipeline stages
    directly on the in-memory array. This keeps the pipeline identical
    while removing the I/O round trip.

    Args:
        image: Decoded BGR image.
        vector_size: Length of the output vector.

    Returns:
        L2-normalized vector, or None if any pipeline stage fails.
    """
    # Import the pipeline stages lazily so this module remains usable
    # even if a caller only needs the extraction half.
    from src.analysis.vectorize import (
        compute_signature,
        convert_to_grayscale,
        crop_mask,
        extract_mask,
        normalize_vector,
        preprocess_image,
        resize_signature,
        DEFAULT_ROW_BOUNDS,
        DEFAULT_THRESHOLD,
        DEFAULT_METHOD,
    )

    # The vectorize module expects float32 in [0, 1].
    normalized = image.astype(np.float32) / 255.0

    smoothed = preprocess_image(normalized)
    gray = convert_to_grayscale(smoothed)

    try:
        mask = extract_mask(gray, threshold=DEFAULT_THRESHOLD)
    except ValueError as exc:
        logger.error("Mask generation failed: %s", exc)
        return None

    (
        cropped_mask,
        _row_start, _row_end,
        _col_start, _col_end,
    ) = crop_mask(mask, row_bounds=DEFAULT_ROW_BOUNDS)

    if cropped_mask.size == 0:
        logger.warning("Cropped mask is empty; cannot vectorize.")
        return None

    try:
        signature = compute_signature(cropped_mask, method=DEFAULT_METHOD)
    except ValueError as exc:
        logger.error("Signature computation failed: %s", exc)
        return None

    resized = resize_signature(signature, vector_size=vector_size)
    if resized is None:
        logger.warning("Resized signature is empty.")
        return None

    vector = normalize_vector(resized)
    if vector is None:
        logger.warning("Vector normalization failed (zero norm).")
        return None

    return vector