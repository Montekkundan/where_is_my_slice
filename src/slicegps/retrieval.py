"""Image-only retrieval baselines and position-based stack evaluation.

These deliberately small baselines do not consume filenames, slice labels, or
geometry. Geometry enters only the separate scoring and ordering functions.
The mask baseline assumes that the median image border represents background;
it is a diagnostic baseline, not a validated tissue segmentation method.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


def _unit_rows(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional feature matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    # Scale before taking the norm to avoid overflow for finite large inputs.
    maximum = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(maximum == 0):
        raise ValueError(f"{name} contains an empty or zero feature vector")
    values = values / maximum
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    coordinates = np.meshgrid(
        np.linspace(0, image.shape[0] - 1, size),
        np.linspace(0, image.shape[1] - 1, size),
        indexing="ij",
    )
    return map_coordinates(image, coordinates, order=1, mode="nearest", prefilter=False)


def _prepare(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spread = float(np.ptp(image))
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError("Images must have nonconstant, finite image content")
    border = np.concatenate((image[0], image[-1], image[1:-1, 0], image[1:-1, -1]))
    background = float(np.median(border))
    foreground = np.abs(image - background) > 0.05 * spread
    if not np.any(foreground):
        raise ValueError("Image has no detectable foreground")
    low, high = np.percentile(image[foreground], (1, 99))
    if high <= low:
        # Binary images can have a constant foreground but meaningful shape.
        low, high = float(image.min()), float(image.max())
    scaled = np.clip((image - low) / (high - low), 0, 1)
    return scaled, foreground


def features(images: np.ndarray, method: str = "intensity", size: int = 32) -> np.ndarray:
    """Return L2-normalized float32 descriptors for grayscale ``(N, H, W)`` images.

    Each image is scaled using its foreground's 1st and 99th percentiles, with a
    min/max fallback for constant foreground. ``intensity`` centers the resized
    intensities; ``gradient`` centers absolute x/y gradients and their magnitude,
    making it insensitive to contrast reversal; ``mask`` resizes a binary mask
    defined by deviation from the median border. Resizing uses a fixed inclusive
    coordinate grid and linear interpolation. Constant/empty images are rejected.
    """
    if method not in {"intensity", "gradient", "mask"}:
        raise ValueError("method must be 'intensity', 'gradient', or 'mask'")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError("size must be an integer of at least 2")
    values = np.asarray(images, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or min(values.shape[1:]) < 2:
        raise ValueError("images must have shape (N, H, W) with N > 0 and H, W >= 2")
    if not np.isfinite(values).all():
        raise ValueError("images must contain only finite values")

    descriptors = []
    for image in values:
        scaled, foreground = _prepare(image)
        if method == "mask":
            descriptor = _resize(foreground.astype(np.float64), size).ravel()
        else:
            resized = _resize(scaled, size)
            if method == "intensity":
                descriptor = (resized - resized.mean()).ravel()
            else:
                dy, dx = np.gradient(resized)
                channels = np.stack((np.abs(dx), np.abs(dy), np.hypot(dx, dy)))
                channels -= channels.mean(axis=(1, 2), keepdims=True)
                descriptor = channels.ravel()
        descriptors.append(descriptor)
    return _unit_rows(np.stack(descriptors), "image descriptors").astype(np.float32)


def rank(
    query_features: np.ndarray, gallery_features: np.ndarray, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Rank image descriptors by cosine similarity, with stable gallery-order ties.

    Inputs are normalized internally. ``k`` is capped at gallery size. Returns
    ``(indices, scores)`` of shape ``(query_count, min(k, gallery_count))``;
    similarities are not calibrated probabilities or confidence estimates.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    queries = _unit_rows(query_features, "query_features")
    gallery = _unit_rows(gallery_features, "gallery_features")
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery feature dimensions must match")
    scores = np.clip(queries @ gallery.T, -1, 1)
    indices = np.argsort(-scores, axis=1, kind="stable")[:, : min(k, len(gallery))]
    return indices, np.take_along_axis(scores, indices, axis=1)


def _centers(values: np.ndarray, name: str) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 3), with N > 0")
    if not np.isfinite(centers).all():
        raise ValueError(f"{name} must contain only finite values")
    return centers


def evaluate_retrieval(
    indices: np.ndarray,
    query_centers: np.ndarray,
    gallery_centers: np.ndarray,
    tolerance_mm: float,
) -> dict[str, Any]:
    """Score ranked candidates using Euclidean center errors in a shared mm frame.

    Recall at each available k is the fraction of queries with at least one
    candidate center within tolerance. These metrics assume known crop centers
    and do not measure plane angle, fragment placement, or registration quality.
    Ground-truth centers are used here for evaluation only, never retrieval.
    """
    queries = _centers(query_centers, "query_centers")
    gallery = _centers(gallery_centers, "gallery_centers")
    ranked = np.asarray(indices)
    if (
        ranked.ndim != 2
        or ranked.shape[0] != len(queries)
        or ranked.shape[1] == 0
        or not np.issubdtype(ranked.dtype, np.integer)
    ):
        raise ValueError("indices must be an integer matrix with one nonempty row per query")
    if np.any(ranked < 0) or np.any(ranked >= len(gallery)):
        raise ValueError("indices contain an out-of-range gallery index")
    if any(len(np.unique(row)) != len(row) for row in ranked):
        raise ValueError("each ranking must contain unique gallery indices")
    if not np.isscalar(tolerance_mm) or not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError("tolerance_mm must be finite and nonnegative")

    errors = np.linalg.norm(gallery[ranked] - queries[:, None, :], axis=2)
    found = np.maximum.accumulate(errors <= tolerance_mm, axis=1)
    return {
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "k": ranked.shape[1],
        "tolerance_mm": float(tolerance_mm),
        "top1_median_error_mm": float(np.median(errors[:, 0])),
        "top1_p95_error_mm": float(np.percentile(errors[:, 0], 95)),
        "recall_at_k": {str(k + 1): float(found[:, k].mean()) for k in range(ranked.shape[1])},
        "metric_note": (
            "Known centers in a shared mm frame; center error does not score plane angle."
        ),
    }


def order_stack(
    predicted_centers: np.ndarray, direction3: Sequence[float] = (0, 1, 0)
) -> np.ndarray:
    """Return indices in ascending position along a declared cutting direction.

    The direction must be supplied in the same coordinate frame as the centers.
    It is normalized internally; ties preserve input order. This function does
    not infer cutting direction, reflections, missing tissue, or missing slices.
    """
    centers = _centers(predicted_centers, "predicted_centers")
    direction = np.asarray(direction3, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("direction3 must be a finite vector of length 3")
    maximum = float(np.max(np.abs(direction)))
    if maximum == 0:
        raise ValueError("direction3 must be nonzero")
    direction = direction / maximum
    direction /= np.linalg.norm(direction)
    return np.argsort(centers @ direction, kind="stable")


def inversion_fraction(order: Sequence[Any], truth_order: Sequence[Any]) -> float:
    """Return the exact fraction of section pairs reversed relative to truth.

    Both sequences must be nonempty permutations of the same unique, hashable
    section IDs. The result is 0 for correct order and 1 for full reversal
    (except a single-section stack, whose result is 0).
    """
    predicted = list(order)
    truth = list(truth_order)
    if not predicted or len(predicted) != len(truth):
        raise ValueError("orders must be nonempty and have equal lengths")
    try:
        predicted_ids, truth_ids = set(predicted), set(truth)
    except TypeError as error:
        raise ValueError("section IDs must be hashable scalars") from error
    if len(predicted_ids) != len(predicted) or len(truth_ids) != len(truth):
        raise ValueError("orders must contain unique section IDs")
    if predicted_ids != truth_ids:
        raise ValueError("orders must contain the same section IDs")
    if len(truth) < 2:
        return 0.0
    positions = {section_id: index for index, section_id in enumerate(truth)}
    ranks = np.asarray([positions[section_id] for section_id in predicted])
    inversions = sum(
        int(np.count_nonzero(ranks[index + 1 :] < value)) for index, value in enumerate(ranks)
    )
    return inversions / (len(truth) * (len(truth) - 1) / 2)
