"""Compare each histology slot to expected MRI alternatives without section metadata.

Local self-similarity follows the principle described by Heinrich et al. (2012),
https://doi.org/10.1016/j.media.2012.05.008. This is a compact 2D descriptor, not a
reproduction of MIND registration or the Synth-by-Reg network.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from .error_detection import _images, canonical_histology, context_features
from .retrieval import _prepare


def self_similarity(images: np.ndarray) -> np.ndarray:
    """Contrast-invariant patch comparisons, with flat background suppressed."""
    result = []
    for image in _images(images, minimum=1):
        image = np.asarray(image, dtype=np.float32)
        image = (image - image.min()) / np.ptp(image)
        image = gaussian_filter(image, 0.6)
        padded = np.pad(image, 4, mode="reflect")
        distances = []
        for radius in (1, 2, 4):
            for dy, dx in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
                shifted = padded[4 + dy : 4 + dy + len(image), 4 + dx : 4 + dx + image.shape[1]]
                distances.append(uniform_filter((image - shifted) ** 2, size=3))
        distances = np.stack(distances)
        variation = distances.mean(axis=0)
        floor = max(1e-8, float(variation.mean()) * 0.01)
        descriptor = np.exp(-(distances - distances.min(axis=0)) / np.maximum(variation, floor))
        descriptor -= descriptor.mean(axis=0)
        descriptor[:, variation < floor] = 0
        # Retain a spatial grid; no learned transform or reference histology is used.
        descriptor = descriptor[:, ::2, ::2].ravel()
        descriptor /= max(1e-8, np.linalg.norm(descriptor))
        result.append(descriptor)
    return np.stack(result)


def comparison_features(similarity: np.ndarray, flipped: np.ndarray) -> np.ndarray:
    """Contrast expected slots with alternatives and mutually beneficial exchanges."""
    similarity = np.asarray(similarity)
    count = len(similarity)
    if similarity.shape != (count, count) or flipped.shape != (count,):
        raise ValueError("Need a square candidate matrix and flipped expected-slot scores")
    values = []
    for i in range(count):
        near = np.array([j for j in range(max(0, i - 2), min(count, i + 3)) if j != i])
        diag = similarity[i, i]
        local = similarity[i, near]
        scale = max(0.01, float(np.std(similarity[i])))
        swap = similarity[i, near] + similarity[near, i] - diag - similarity[near, near]
        adjacent = np.abs(near - i) == 1
        values.append(
            [
                diag,
                (local.max() - diag) / scale,
                (similarity[i].max() - diag) / scale,
                float(np.mean(similarity[i] > diag)),
                (diag - np.median(local)) / scale,
                (flipped[i] - diag) / scale,
                swap.max() / scale,
                swap[adjacent].max() / scale,
            ]
        )
    return np.asarray(values)


def _bins(image: np.ndarray) -> np.ndarray:
    scaled = (image - image.min()) / np.ptp(image)
    return np.minimum(15, np.floor(scaled * 16)).astype(np.int32)


def _nmi(a: np.ndarray, b: np.ndarray, foreground: np.ndarray) -> float:
    joint = np.bincount((a[foreground] * 16 + b[foreground]).ravel(), minlength=256).reshape(16, 16)
    joint = joint / max(1, joint.sum())

    def entropy(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    total = entropy(joint)
    return (entropy(joint.sum(axis=0)) + entropy(joint.sum(axis=1))) / max(1e-8, total)


class PairwiseFeatures:
    """Cache image measurements, never source IDs, injected labels or clean copies."""

    def __init__(self, mri: np.ndarray):
        self.mri = _images(mri)
        self.reference = self_similarity(self.mri)
        self.quantized = np.stack([_bins(image) for image in self.mri])
        self.foreground = np.stack([_prepare(image)[1] for image in self.mri])
        self.cache = {}
        self.nmi_cache = {}

    def _encoded(self, images):
        keys = [hashlib.sha256(image.tobytes()).digest() for image in images]
        missing = {
            key: image for key, image in zip(keys, images, strict=True) if key not in self.cache
        }
        if missing:
            bank = np.stack(list(missing.values()))
            descriptors = self_similarity(bank)
            canonical = self_similarity(canonical_histology(bank))
            similarities = descriptors @ self.reference.T
            canonical_similarities = canonical @ self.reference.T
            for key, desc, canon, sim, canon_sim in zip(
                missing, descriptors, canonical, similarities, canonical_similarities, strict=True
            ):
                self.cache[key] = (desc, canon, sim, canon_sim)
        return tuple(np.stack([self.cache[key][j] for key in keys]) for j in range(4))

    def measure(self, histology: np.ndarray) -> np.ndarray:
        images = _images(histology)
        if images.shape != self.mri.shape:
            raise ValueError("Expected MRI and histology must share the same canvas and slot count")
        desc = self._encoded(images)
        flipped = self._encoded(images[:, :, ::-1])
        columns = []
        for normal, flip, similarity, flip_similarity in zip(
            desc[:2], flipped[:2], desc[2:], flipped[2:], strict=True
        ):
            flip_score = np.diag(flip_similarity)
            columns.append(comparison_features(similarity, flip_score))
            columns.append(context_features(normal, flip))
        # NMI uses an MRI-derived foreground mask, never manual labels. Only the
        # five local expected MRI candidates are needed for adjacent exchanges.
        quantized = np.stack([_bins(image) for image in images])
        scores = np.full((len(images), len(images)), np.nan)

        def cached_nmi(image, slot):
            key = (hashlib.sha256(image.tobytes()).digest(), slot)
            if key not in self.nmi_cache:
                self.nmi_cache[key] = _nmi(image, self.quantized[slot], self.foreground[slot])
            return self.nmi_cache[key]

        flip_scores = []
        for i, image in enumerate(quantized):
            for j in range(max(0, i - 2), min(len(images), i + 3)):
                scores[i, j] = cached_nmi(image, j)
            flip_scores.append(cached_nmi(image[:, ::-1], i))
        nmi_values = []
        for i in range(len(images)):
            near = np.array([j for j in range(max(0, i - 2), min(len(images), i + 3)) if j != i])
            diagonal = scores[i, i]
            swap = scores[i, near] + scores[near, i] - diagonal - scores[near, near]
            nmi_values.append(
                [diagonal, scores[i, near].max() - diagonal, flip_scores[i] - diagonal, swap.max()]
            )
        columns.append(np.asarray(nmi_values))
        result = np.column_stack(columns)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite paired image comparisons")
        return result
