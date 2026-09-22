# src/analysis/vectorize.py
"""
EDS spectrum vectorization utilities.

This module converts an EDS spectrum image (JPEG, PNG, or an extracted
DOCX image) into a fixed-length L2-normalized feature vector suitable
for cosine-similarity comparison.

Pipeline
--------
    DOCX
      -> extract embedded image
      -> select image with shape (400, 512, 3)
      -> normalize pixel values to [0, 1]
      -> 5 x 5 Gaussian smoothing
      -> grayscale conversion
      -> threshold < 0.99
      -> ROI: rows 150-250
      -> column-wise mean
      -> linear interpolation to 200 values
      -> L2 normalization
      -> 200-dimensional feature vector
      -> cosine / Pearson / Euclidean / Manhattan comparison

Main public functions:
    - vectorize_spectrum(image_path, ...) -> Optional[np.ndarray]
    - vectorize_pipeline(image_path, ...) -> VectorizationResult
    - read_image_float, preprocess_image, convert_to_grayscale,
      extract_mask, crop_mask, compute_signature, resize_signature,
      normalize_vector  (the individual pipeline stages)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default pipeline parameters.
DEFAULT_VECTOR_SIZE = 200
DEFAULT_THRESHOLD = 0.99
DEFAULT_ROW_BOUNDS: Tuple[int, int] = (150, 250)
DEFAULT_METHOD = "mean"

# Expected shape of the input image. Used for validation, not enforced
# as a hard requirement (some callers may provide differently shaped images).
EXPECTED_IMAGE_SHAPE: Tuple[int, int, int] = (400, 512, 3)

# Tolerance for treating vector norms as zero.
_EPS = 1e-12

# Supported aggregation methods.
_VALID_METHODS = {"mean", "max"}


# ============================================================================
# RESULT TYPE
# ============================================================================

@dataclass
class VectorizationResult:
    """
    Container for the vectorization pipeline output.

    The final `vector` is the L2-normalized 200-dimensional feature
    vector, or None if the pipeline failed. The intermediate fields are
    kept so callers can inspect or visualize each stage.
    """

    vector: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    smoothed: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    cropped_mask: Optional[np.ndarray] = None
    signature: Optional[np.ndarray] = None
    resized_signature: Optional[np.ndarray] = None

    row_bounds: Optional[Tuple[int, int]] = None
    column_bounds: Optional[Tuple[int, int]] = None

    error: Optional[str] = None
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.vector is not None and self.error is None


# ============================================================================
# STAGE 1 — READ AND NORMALIZE
# ============================================================================

def read_image_float(image_path: str) -> Optional[np.ndarray]:
    """
    Read an image and convert it to normalized float32 values in [0, 1].

    Args:
        image_path: Path to the input image.

    Returns:
        A float32 image in [0, 1] with three channels (BGR), or None if
        the image cannot be read.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("Could not read image: %s", image_path)
        return None

    # Ensure three-channel BGR.
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.astype(np.float32) / 255.0


# ============================================================================
# STAGE 2 — GAUSSIAN SMOOTHING
# ============================================================================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply 5 x 5 Gaussian smoothing.

    OpenCV determines the standard deviation automatically because
    sigmaX=0 is specified.

    Args:
        image: Normalized input image.

    Returns:
        Smoothed image (same dtype and shape).
    """
    return cv2.GaussianBlur(image, (5, 5), 0)


# ============================================================================
# STAGE 3 — GRAYSCALE
# ============================================================================

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a three-channel BGR image to a single-channel grayscale image.

    Args:
        image: Three-channel BGR image.

    Returns:
        Single-channel grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================================
# STAGE 4 — BINARY MASK
# ============================================================================

def extract_mask(gray_image: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Generate a binary mask by thresholding the grayscale image.

    Pixels with intensity strictly below `threshold` become 255 (kept);
    all others become 0 (rejected).

    Args:
        gray_image: Grayscale image with intensity values in [0, 1].
        threshold: Threshold used for binary segmentation.

    Returns:
        Binary mask with values 0 or 255, dtype uint8.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    return (gray_image < threshold).astype(np.uint8) * 255


# ============================================================================
# STAGE 5 — CROP
# ============================================================================

def crop_mask(
    mask: np.ndarray,
    row_bounds: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Crop the binary mask to the region containing the spectrum.

    Columns are restricted automatically to the range containing nonzero
    pixels. Rows are taken from `row_bounds` when supplied, otherwise
    determined automatically from nonzero rows.

    Args:
        mask: Binary spectrum mask.
        row_bounds: Optional vertical limits as (start_row, end_row).

    Returns:
        Tuple of:
            (cropped_mask, row_start, row_end, column_start, column_end)
    """
    # --- Column boundaries (always automatic) ------------------------------
    column_sum = np.sum(mask, axis=0)
    nonzero_columns = np.where(column_sum > 0)[0]

    if nonzero_columns.size == 0:
        column_start, column_end = 0, mask.shape[1]
    else:
        column_start = int(nonzero_columns[0])
        column_end = int(nonzero_columns[-1]) + 1

    # --- Row boundaries -----------------------------------------------------
    if row_bounds is not None:
        row_start, row_end = int(row_bounds[0]), int(row_bounds[1])

        # Guard against out-of-range bounds.
        row_start = max(0, min(row_start, mask.shape[0]))
        row_end = max(row_start, min(row_end, mask.shape[0]))
    else:
        row_sum = np.sum(mask, axis=1)
        nonzero_rows = np.where(row_sum > 0)[0]

        if nonzero_rows.size == 0:
            row_start, row_end = 0, mask.shape[0]
        else:
            row_start = int(nonzero_rows[0])
            row_end = int(nonzero_rows[-1]) + 1

    cropped_mask = mask[row_start:row_end, column_start:column_end]

    return cropped_mask, row_start, row_end, column_start, column_end


# ============================================================================
# STAGE 6 — 1D SPECTRAL SIGNATURE
# ============================================================================

def compute_signature(mask_cropped: np.ndarray, method: str = DEFAULT_METHOD) -> np.ndarray:
    """
    Compute the one-dimensional spectrum signature.

    Args:
        mask_cropped: Cropped binary spectrum mask.
        method: Aggregation method.
            - "mean": mean intensity across rows (default).
            - "max": maximum intensity across rows.

    Returns:
        One-dimensional spectrum signature (float64).

    Raises:
        ValueError: If `method` is not one of the valid options.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Expected one of: {sorted(_VALID_METHODS)}"
        )

    if mask_cropped.size == 0:
        return np.zeros(0, dtype=np.float64)

    if method == "max":
        return np.max(mask_cropped, axis=0).astype(np.float64)
    return np.mean(mask_cropped, axis=0).astype(np.float64)


# ============================================================================
# STAGE 7 — INTERPOLATION
# ============================================================================

def resize_signature(
    signature: np.ndarray,
    vector_size: int = DEFAULT_VECTOR_SIZE,
) -> Optional[np.ndarray]:
    """
    Interpolate the spectrum signature to a fixed vector length.

    Args:
        signature: Original 1-D spectrum signature.
        vector_size: Number of elements in the output vector.

    Returns:
        Resized float32 signature, or None if the input is empty.
    """
    if signature is None or signature.size == 0:
        return None

    if vector_size <= 0:
        raise ValueError(f"vector_size must be positive, got {vector_size}")

    # Shortcut: if the signature already has the right length,
    # no interpolation is needed.
    if signature.size == vector_size:
        return signature.astype(np.float32)

    x_old = np.arange(signature.size, dtype=np.float64)
    x_new = np.linspace(0, signature.size - 1, vector_size, dtype=np.float64)
    resized = np.interp(x_new, x_old, signature)
    return resized.astype(np.float32)


# ============================================================================
# STAGE 8 — L2 NORMALIZATION
# ============================================================================

def normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply L2 normalization to a vector.

    Args:
        vector: Input vector.

    Returns:
        L2-normalized vector, or None if the vector is None, empty,
        or has zero (or near-zero) norm.
    """
    if vector is None or vector.size == 0:
        return None

    norm = float(np.linalg.norm(vector))
    if norm < _EPS:
        logger.warning("Cannot normalize: vector norm is zero.")
        return None

    return vector / norm


# ============================================================================
# FULL PIPELINE
# ============================================================================

def vectorize_pipeline(
    image_path: str,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    row_bounds: Optional[Tuple[int, int]] = DEFAULT_ROW_BOUNDS,
    method: str = DEFAULT_METHOD,
) -> VectorizationResult:
    """
    Run the full vectorization pipeline and return every intermediate stage.

    This is the recommended entry point when you need to debug or
    visualize the pipeline. For a simple vector-only result, use
    `vectorize_spectrum`.

    Args:
        image_path: Path to the EDS spectrum image.
        vector_size: Number of elements in the final feature vector.
        threshold: Grayscale threshold for binary segmentation.
        row_bounds: Vertical region of interest as (start_row, end_row),
            or None to detect automatically.
        method: Vertical aggregation method ("mean" or "max").

    Returns:
        A `VectorizationResult` containing every intermediate array and
        the final vector. If any stage fails, `result.vector` is None
        and `result.error` describes the failure.
    """
    result = VectorizationResult()

    # --- Stage 1: read and normalize ---------------------------------------
    image = read_image_float(image_path)
    if image is None:
        result.error = f"could not read image: {image_path}"
        return result
    result.image = image

    if image.shape != EXPECTED_IMAGE_SHAPE:
        msg = (
            f"image shape {image.shape} does not match expected "
            f"{EXPECTED_IMAGE_SHAPE}; continuing anyway"
        )
        logger.info(msg)
        result.warnings.append(msg)

    # --- Stage 2: Gaussian smoothing ---------------------------------------
    result.smoothed = preprocess_image(image)

    # --- Stage 3: grayscale -------------------------------------------------
    result.gray = convert_to_grayscale(result.smoothed)

    # --- Stage 4: binary mask ----------------------------------------------
    try:
        result.mask = extract_mask(result.gray, threshold=threshold)
    except ValueError as exc:
        result.error = str(exc)
        return result

    # --- Stage 5: crop ------------------------------------------------------
    (
        result.cropped_mask,
        row_start,
        row_end,
        col_start,
        col_end,
    ) = crop_mask(result.mask, row_bounds=row_bounds)
    result.row_bounds = (row_start, row_end)
    result.column_bounds = (col_start, col_end)

    if result.cropped_mask.size == 0:
        result.error = "cropped mask is empty"
        return result

    # --- Stage 6: 1D signature ---------------------------------------------
    try:
        result.signature = compute_signature(result.cropped_mask, method=method)
    except ValueError as exc:
        result.error = str(exc)
        return result

    # --- Stage 7: resize to fixed length -----------------------------------
    result.resized_signature = resize_signature(
        result.signature, vector_size=vector_size
    )
    if result.resized_signature is None:
        result.error = "resized signature is empty"
        return result

    # --- Stage 8: L2 normalization -----------------------------------------
    result.vector = normalize_vector(result.resized_signature)
    if result.vector is None:
        result.error = "vector normalization failed (zero norm)"
        return result

    return result


def vectorize_spectrum(
    image_path: str,
    vector_size: int = DEFAULT_VECTOR_SIZE,
    threshold: float = DEFAULT_THRESHOLD,
    row_bounds: Optional[Tuple[int, int]] = DEFAULT_ROW_BOUNDS,
    method: str = DEFAULT_METHOD,
) -> Optional[np.ndarray]:
    """
    Convert an EDS spectrum image into an L2-normalized feature vector.

    This is a thin wrapper around `vectorize_pipeline` that returns
    only the final vector, for backward compatibility and simple use.

    Args:
        image_path: Path to the EDS spectrum image.
        vector_size: Number of elements in the final vector.
        threshold: Grayscale threshold for binary segmentation.
        row_bounds: Vertical region of interest (start_row, end_row),
            or None to detect automatically.
        method: Vertical aggregation method ("mean" or "max").

    Returns:
        L2-normalized feature vector of length `vector_size`, or None
        if vectorization fails.
    """
    result = vectorize_pipeline(
        image_path,
        vector_size=vector_size,
        threshold=threshold,
        row_bounds=row_bounds,
        method=method,
    )

    if result.error:
        logger.warning("Vectorization failed for %s: %s", image_path, result.error)

    return result.vector