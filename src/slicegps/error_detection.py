"""Calibrated image-only stack error detection, with explicit benchmark truth separation."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, label
from scipy.optimize import minimize

from .retrieval import _prepare, features

ERROR_KINDS = ("flip", "damage", "misorder_near", "misorder_far")


def _images(value, minimum=5):
    images = np.asarray(value)
    if (
        images.ndim != 3
        or len(images) < minimum
        or min(images.shape[1:]) < 8
        or not np.issubdtype(images.dtype, np.number)
        or not np.isfinite(images).all()
    ):
        raise ValueError(f"Expected at least {minimum} finite grayscale images, shape (N,H,W)")
    if np.any(np.ptp(images, axis=(1, 2)) <= 0):
        raise ValueError("Empty or constant sections require explicit missing-section handling")
    return images


def canonical_histology(images: np.ndarray) -> np.ndarray:
    """Remove bright mounting paper using pixels only, retaining the common canvas.

    A broad near-white region identifies paper-backed scans. Dark scanner frame
    and small annotation components are excluded before inversion. Dark-background
    tissue scans retain their original contrast. No publisher masks are consumed.
    """
    images = _images(images, minimum=1)
    result = []
    for image in images:
        low, high = float(image.min()), float(image.max())
        scaled = (image - low) / (high - low)
        if np.mean(scaled > 0.9) > 0.2:
            smooth = gaussian_filter(scaled, 0.6)
            mask = (smooth > 0.12) & (smooth < 0.86)
            regions, count = label(mask)
            sizes = np.bincount(regions.ravel())
            sizes[0] = 0
            border = np.unique(
                np.concatenate((regions[0], regions[-1], regions[:, 0], regions[:, -1]))
            )
            sizes[border] = 0
            keep = np.flatnonzero(sizes >= max(8, scaled.size * 0.01))
            filled = [binary_fill_holes(regions == region) for region in keep]
            # A thin scanner-frame ring must not turn the whole paper sheet into tissue.
            filled = [
                region
                for region, component in zip(filled, keep, strict=True)
                if sizes[component] / max(1, region.sum()) >= 0.3
            ]
            mask = np.logical_or.reduce(filled) if filled else np.zeros_like(mask)
            if mask.mean() >= 0.01:
                scaled = (1 - scaled) * mask
        result.append(scaled)
    return np.stack(result)


def context_features(desc: np.ndarray, flipped: np.ndarray) -> np.ndarray:
    """Local trajectory measurements on descriptors; no section metadata."""
    count = len(desc)
    neighbours = [
        np.array([j for j in range(max(0, i - 2), min(count, i + 3)) if j != i])
        for i in range(count)
    ]
    values = []
    for i, near in enumerate(neighbours):
        distances = 1 - desc[near] @ desc[i]
        flip_distances = 1 - desc[near] @ flipped[i]
        pair_distances = 1 - desc[near] @ desc[near].T
        baseline = np.median(pair_distances[np.triu_indices(len(near), 1)])
        closest = np.sort(distances)[:2].mean()
        left, right = max(0, i - 1), min(count - 1, i + 1)
        direction = desc[right] - desc[left]
        projection = float((desc[i] - desc[left]) @ direction) / (
            float(direction @ direction) + 1e-4
        )
        interpolation_error = float(np.sum((desc[i] - (desc[left] + desc[right]) / 2) ** 2))
        step = float(np.sum(direction**2))
        values.append(
            [
                distances.mean(),
                closest,
                closest - baseline,
                closest / (0.02 + baseline),
                closest - np.sort(flip_distances)[:2].mean(),
                abs(projection - 0.5),
                interpolation_error / (0.02 + step),
            ]
        )
    return np.asarray(values)


def error_features(
    images: np.ndarray, mri: np.ndarray | None = None, *, cache: dict | None = None
) -> np.ndarray:
    """See pixels in nominal stack order, never IDs, source indices, or clean copies.

    Optional MRI planes must be in expected nominal order and approximately
    aligned to the histology canvas. This is a stronger input assumption than
    arbitrary-section retrieval; results with and without MRI are kept separate.
    """
    images = _images(images)
    if mri is not None:
        mri = _images(mri)
        if mri.shape != images.shape:
            raise ValueError("Expected MRI and histology stacks must have matching shapes")
    count = len(images)
    neighbours = [
        np.array([j for j in range(max(0, i - 2), min(count, i + 3)) if j != i])
        for i in range(count)
    ]

    def cached_features(stack, method):
        if cache is None:
            return features(stack, method, size=24)
        keys = [
            (method, image.shape, image.dtype.str, hashlib.sha256(image.tobytes()).digest())
            for image in stack
        ]
        missing = {key: image for key, image in zip(keys, stack, strict=True) if key not in cache}
        if missing:
            computed = features(np.stack(list(missing.values())), method, size=24)
            cache.update(zip(missing, computed, strict=True))
        return np.stack([cache[key] for key in keys])

    columns = []
    for method in ("intensity", "gradient", "mask"):
        desc = cached_features(images, method)
        flipped = cached_features(images[:, :, ::-1], method)
        columns.append(context_features(desc, flipped))
        if mri is not None:
            reference = cached_features(mri, method)
            similarity = np.sum(desc * reference, axis=1)
            flipped_similarity = np.sum(flipped * reference, axis=1)
            columns.append(np.column_stack((1 - similarity, flipped_similarity - similarity)))
    area = np.array([_prepare(image)[1].mean() for image in images])
    reference_area = np.array([np.median(area[near]) for near in neighbours])
    columns.append(
        np.column_stack((area, area - reference_area, area / np.maximum(reference_area, 0.01)))
    )
    result = np.concatenate(columns, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite stack measurements")
    return result


def normalize_stack_features(values: np.ndarray) -> np.ndarray:
    """Robust unsupervised calibration using the current stack, assuming a clean majority.

    All input rows are observations, never a supplied clean reference. Boundary
    rows lack full context and do not determine the median or interquartile range.
    This is a transductive stack algorithm, not independent per-image inference.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 9 or not np.isfinite(values).all():
        raise ValueError("Stack normalization requires at least nine finite feature rows")
    interior = values[2:-2]
    center = np.median(interior, axis=0)
    lower, upper = np.percentile(interior, (25, 75), axis=0)
    scale = np.maximum(upper - lower, 1e-3)
    return np.clip((values - center) / scale, -10, 10)


def inject_errors(images: np.ndarray, kind: str, fraction: float, seed: int) -> tuple:
    """Corrupt a complete stack; return truth separately from the detector input.

    Near errors swap immediate neighbours. Far errors swap slots separated by
    at least a quarter of the stack. Tears remove a rectangular patch containing
    at least 5 percent of the section foreground. Error fraction is approximate
    because swaps affect two sections and small stacks require rounding.
    """
    images = _images(images, minimum=9)
    if kind not in ERROR_KINDS or not np.isfinite(fraction) or not 0 < fraction <= 0.2:
        raise ValueError("Unknown error kind or fraction outside (0, 0.2]")
    rng = np.random.default_rng(seed)
    count = len(images)
    eligible = np.arange(2, count - 2)
    number = max(1, int(round(len(eligible) * fraction)))
    observed = images.copy()
    truth = np.zeros(count, dtype=bool)
    source = np.arange(count)
    if kind.startswith("misorder"):
        available = set(eligible.tolist())
        for _ in range(max(1, number // 2)):
            possible = [
                (i, j)
                for i in sorted(available)
                for j in sorted(available)
                if j > i and (j - i == 1 if kind == "misorder_near" else j - i >= count // 4)
            ]
            if not possible:
                break
            i, j = possible[int(rng.integers(len(possible)))]
            observed[[i, j]] = observed[[j, i]]
            source[[i, j]] = source[[j, i]]
            truth[[i, j]] = True
            available.difference_update((i, j))
    else:
        for i in rng.choice(eligible, size=number, replace=False):
            if kind == "flip":
                observed[i] = np.fliplr(observed[i])
            else:
                mask = _prepare(canonical_histology(observed)[i])[1]
                height, width = images.shape[1:]
                foreground = np.argwhere(mask)
                for _ in range(100):
                    y, x = foreground[int(rng.integers(len(foreground)))]
                    dy = rng.integers(max(2, height // 5), max(3, height // 2))
                    dx = rng.integers(max(2, width // 5), max(3, width // 2))
                    y0, y1 = max(0, y - dy // 2), min(height, y + dy // 2)
                    x0, x1 = max(0, x - dx // 2), min(width, x + dx // 2)
                    if mask[y0:y1, x0:x1].sum() >= 0.05 * mask.sum():
                        background = (
                            np.quantile(images[i], 0.95)
                            if np.mean((images[i] - images[i].min()) / np.ptp(images[i]) > 0.9)
                            > 0.2
                            else np.median(
                                np.concatenate(
                                    (images[i, 0], images[i, -1], images[i, :, 0], images[i, :, -1])
                                )
                            )
                        )
                        observed[i, y0:y1, x0:x1] = background
                        break
                else:
                    raise ValueError("Could not create the declared tear")
            truth[i] = True
    if not truth.any():
        raise ValueError("Could not create the declared error")
    return observed, {"is_error": truth, "source_index": source, "kind": kind}


def classification_metrics(truth, flagged) -> dict:
    truth, flagged = np.asarray(truth), np.asarray(flagged)
    if (
        truth.shape != flagged.shape
        or truth.ndim != 1
        or truth.dtype != bool
        or flagged.dtype != bool
    ):
        raise ValueError("truth and flagged must be equally sized boolean vectors")
    tp = int(np.sum(truth & flagged))
    fp = int(np.sum(~truth & flagged))
    fn = int(np.sum(truth & ~flagged))
    tn = int(np.sum(~truth & ~flagged))
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    precision = tp / (tp + fp) if tp + fp else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(truth) if len(truth) else None,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": fpr,
        "balanced_accuracy": (recall + 1 - fpr) / 2
        if recall is not None and fpr is not None
        else None,
    }


def fit_detector(x: np.ndarray, y: np.ndarray, regularization: float = 0.01) -> dict:
    """Class-balanced regularized logistic model; all scaling fits training only."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=bool)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or len(np.unique(y)) != 2
        or not np.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError("Need finite training features, both labels, and positive regularization")
    center = np.median(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 1e-4)
    z = np.clip((x - center) / scale, -10, 10)
    design = np.column_stack((z, z * z, np.ones(len(x))))
    weights = np.where(y, 0.5 / y.sum(), 0.5 / (~y).sum())

    def objective(beta):
        logits = design @ beta
        loss = np.sum(weights * (np.logaddexp(0, logits) - y * logits))
        probability = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        gradient = design.T @ (weights * (probability - y))
        loss += regularization * np.sum(beta[:-1] ** 2) / 2
        gradient[:-1] += regularization * beta[:-1]
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(design.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not fitted.success:
        raise RuntimeError(f"Detector fit did not converge: {fitted.message}")
    return {
        "schema_version": 1,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": fitted.x.tolist(),
        "regularization": regularization,
        "score_note": "Logit score, not a calibrated probability",
    }


def detector_scores(x: np.ndarray, model: dict) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    center, scale = np.asarray(model["center"]), np.asarray(model["scale"])
    if x.ndim != 2 or x.shape[1:] != center.shape or not np.isfinite(x).all():
        raise ValueError("Detector feature contract mismatch")
    z = np.clip((x - center) / scale, -10, 10)
    return np.column_stack((z, z * z, np.ones(len(x)))) @ np.asarray(model["coefficients"])


def select_threshold(scores, truth, maximum_fpr=0.05) -> tuple[float, dict]:
    """Select on validation only, maximizing recall subject to a false-alarm cap."""
    scores, truth = np.asarray(scores), np.asarray(truth, dtype=bool)
    if (
        scores.shape != truth.shape
        or scores.ndim != 1
        or not np.isfinite(scores).all()
        or len(np.unique(truth)) != 2
        or not 0 <= maximum_fpr <= 1
    ):
        raise ValueError("Need finite validation scores, both labels and an FPR in [0,1]")
    # Strict > ensures ties at the chosen clean-score quantile do not exceed the cap.
    clean = np.sort(scores[~truth])
    allowed = int(np.floor(maximum_fpr * len(clean)))
    threshold = float(clean[max(0, len(clean) - allowed - 1)])
    return threshold, classification_metrics(truth, scores > threshold)


def spatial_splits(count: int, guard: int = 3) -> dict[str, np.ndarray]:
    """Contiguous 60/20/20 ranges with unused support gaps around both boundaries."""
    if count < 50 or guard < 2:
        raise ValueError("Need at least 50 sections and a guard of at least two")
    a, b = int(count * 0.6), int(count * 0.8)
    return {
        "train": np.arange(0, a - guard),
        "validation": np.arange(a + guard, b - guard),
        "test": np.arange(b + guard, count),
    }
