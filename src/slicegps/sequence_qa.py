"""Order evidence from changes in observed histology, without a clean reference."""

from __future__ import annotations

import hashlib

import numpy as np

from .error_detection import _images, canonical_histology
from .retrieval import _unit_rows, features


def paper_histology(images):
    """Versioned paper removal used by new experiments; old baselines stay unchanged."""
    cleaned = []
    for image in _images(images, minimum=1):
        scaled = (image - image.min()) / np.ptp(image)
        fraction = 0.0 if np.mean(scaled > 0.98) > 0.04 else 0.2
        cleaned.append(canonical_histology(image[None], paper_fraction=fraction)[0])
    return np.stack(cleaned)


def order_evidence(descriptors):
    """Measure whether exchanging each adjacent pair improves local continuity.

    Four measurements compare path length, second differences, and alignment
    with the surrounding trajectory at two radii. Both candidate slots receive
    the stronger evidence from their left and right exchange. Positive gains
    suggest review; an anatomical transition can also produce a positive gain.
    """
    desc = _unit_rows(descriptors, "sequence descriptors")
    count = len(desc)
    if count < 5:
        raise ValueError("Order evidence needs at least five sections")
    similarity = np.clip(desc @ desc.T, -1, 1)
    distance = np.maximum(0, 2 - 2 * similarity)
    pairs = []
    for j in range(count - 1):
        current = np.arange(max(0, j - 1), min(count, j + 3))
        swapped = current.copy()
        swapped[current == j], swapped[current == j + 1] = j + 1, j
        old_path = distance[current[:-1], current[1:]].sum()
        new_path = distance[swapped[:-1], swapped[1:]].sum()
        path_gain = (old_path - new_path) / max(old_path + new_path, 1e-4)
        middle = np.arange(max(1, j - 1), min(count - 1, j + 3))
        old = np.column_stack((middle - 1, middle, middle + 1))
        new = old.copy()
        new[old == j], new[old == j + 1] = j + 1, j

        def curvature(triples):
            left, center, right = triples.T
            return np.maximum(
                0,
                0.5 * distance[left, center]
                + 0.5 * distance[center, right]
                - 0.25 * distance[left, right],
            ).sum()

        old_curve, new_curve = curvature(old), curvature(new)
        curve_gain = (old_curve - new_curve) / max(old_curve + new_curve, 1e-4)
        alignment = []
        for radius in (1, 2):
            left, right = max(0, j - radius), min(count - 1, j + 1 + radius)
            product = (
                similarity[j + 1, right]
                - similarity[j + 1, left]
                - similarity[j, right]
                + similarity[j, left]
            )
            denominator = np.sqrt(distance[j, j + 1] * distance[left, right])
            alignment.append(-product / max(denominator, 1e-4))
        pairs.append([path_gain, curve_gain, *alignment])
    pair_values = np.asarray(pairs)
    result = np.empty((count, 4))
    for i in range(count):
        result[i] = pair_values[max(0, i - 1) : min(count - 1, i + 1)].max(axis=0)
    return result


def sequence_features(images, *, cache=None):
    """24 order measurements on raw and tissue-only intensity, gradient and mask."""
    images = _images(images)
    cache = {} if cache is None else cache
    transformed = []
    for image in images:
        key = ("paper_v2", image.shape, image.dtype.str, hashlib.sha256(image.tobytes()).digest())
        if key not in cache:
            cache[key] = paper_histology(image[None])[0]
        transformed.append(cache[key])
    columns = []
    for stack in (images, np.stack(transformed)):
        for method in ("intensity", "gradient", "mask"):
            keys = [
                (method, image.shape, image.dtype.str, hashlib.sha256(image.tobytes()).digest())
                for image in stack
            ]
            missing = {
                key: image for key, image in zip(keys, stack, strict=True) if key not in cache
            }
            if missing:
                encoded = features(np.stack(list(missing.values())), method, size=24)
                cache.update(zip(missing, encoded, strict=True))
            columns.append(order_evidence(np.stack([cache[key] for key in keys])))
    return np.column_stack(columns)
