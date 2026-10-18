"""Seeded pixel augmentations and shuffled-stack controls, with explicit truth.

These synthetic tears and rectangles are stress tests, not a model of real
histology damage. Truth describes operations applied, not detector outcomes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .retrieval import _prepare


def _image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.dtype.kind not in "fiu" or values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("image must be a real two-dimensional array at least 2 by 2")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("image must contain only finite values")
    _prepare(values)
    return values


def _range(value, name: str, minimum: float) -> tuple[float, float]:
    bounds = np.asarray(value, dtype=float)
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or bounds[0] < minimum
        or bounds[1] < bounds[0]
    ):
        raise ValueError(f"{name} must be an ordered finite pair >= {minimum}")
    return float(bounds[0]), float(bounds[1])


def _augment(
    image: np.ndarray,
    rng: np.random.Generator,
    gamma_range=(0.8, 1.2),
    gain_range=(0.9, 1.1),
    blur_sigma=(0.0, 0.8),
    tear_fraction: float = 0.02,
    missing_chunk_fraction: float = 0.08,
) -> tuple[np.ndarray, dict]:
    values = _image(image)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    gamma_range = _range(gamma_range, "gamma_range", 1e-6)
    gain_range = _range(gain_range, "gain_range", 1e-6)
    blur_sigma = _range(blur_sigma, "blur_sigma", 0)
    for fraction in (tear_fraction, missing_chunk_fraction):
        if not np.isscalar(fraction) or not np.isfinite(fraction) or not 0 <= fraction <= 0.4:
            raise ValueError("damage fractions must be finite and between 0 and 0.4")
    low, high = float(values.min()), float(values.max())
    gamma, gain = float(rng.uniform(*gamma_range)), float(rng.uniform(*gain_range))
    sigma = float(rng.uniform(*blur_sigma))
    scaled = np.clip(((values - low) / (high - low)) ** gamma * gain, 0, 1)
    output = gaussian_filter(scaled, sigma=sigma) * (high - low) + low
    _, foreground = _prepare(output)
    border = np.concatenate((output[0], output[-1], output[1:-1, 0], output[1:-1, -1]))
    background = float(np.median(border))
    height, width = output.shape
    cuts = []

    def cut(kind: str, row0: int, row1: int, col0: int, col1: int) -> None:
        proposal = output.copy()
        proposal[row0:row1, col0:col1] = background
        remaining = foreground & (np.abs(proposal - background) > 0.05 * (high - low))
        # Keep meaningful visible tissue, including when a cut hits a small fragment.
        if remaining.sum() >= max(1, int(np.ceil(foreground.sum() * 0.25))):
            output[:] = proposal
            cuts.append({"kind": kind, "rows": [row0, row1], "columns": [col0, col1]})

    points = np.argwhere(foreground)
    if tear_fraction > 0:
        row, col = points[rng.integers(len(points))]
        if rng.integers(2):
            thickness = max(1, int(round(width * tear_fraction)))
            start = int(np.clip(col - thickness // 2, 0, width - thickness))
            cut("vertical tear", 0, height, start, start + thickness)
        else:
            thickness = max(1, int(round(height * tear_fraction)))
            start = int(np.clip(row - thickness // 2, 0, height - thickness))
            cut("horizontal tear", start, start + thickness, 0, width)
    if missing_chunk_fraction > 0:
        row, col = points[rng.integers(len(points))]
        side = float(np.sqrt(missing_chunk_fraction))
        rows, cols = max(1, round(height * side)), max(1, round(width * side))
        row0 = int(np.clip(row - rows // 2, 0, height - rows))
        col0 = int(np.clip(col - cols // 2, 0, width - cols))
        cut("missing rectangle", row0, row0 + rows, col0, col0 + cols)
    _prepare(output)
    return output, {"gamma": gamma, "gain": gain, "blur_sigma": sigma, "cuts": cuts}


def augment_histology(
    image: np.ndarray,
    rng: np.random.Generator,
    gamma_range=(0.8, 1.2),
    gain_range=(0.9, 1.1),
    blur_sigma=(0.0, 0.8),
    tear_fraction: float = 0.02,
    missing_chunk_fraction: float = 0.08,
) -> np.ndarray:
    """Apply configurable gamma, gain, blur, tears and missing rectangles.

    Supplying the same Generator seed and pixels gives identical results. Values
    stay finite and within the input range; damage retains at least 25% of the
    original detected foreground. Set both damage fractions to zero to disable cuts.
    """
    return _augment(
        image, rng, gamma_range, gain_range, blur_sigma, tear_fraction, missing_chunk_fraction
    )[0]


def corrupt_stack(
    images: np.ndarray,
    flip_fraction: float = 0.05,
    damage_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Shuffle all sections, then flip/damage rounded fractions without replacement.

    The permutation maps shuffled input index to source index. Flip and damage
    selections are independent and may overlap. Counts round N*fraction to the
    nearest integer, with half rounded up, so small stacks may receive no damage.
    """
    values = np.asarray(images)
    if values.ndim != 3 or len(values) == 0:
        raise ValueError("images must have nonempty shape (N, H, W)")
    values = np.stack([_image(image) for image in values])
    for fraction in (flip_fraction, damage_fraction):
        if not np.isscalar(fraction) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("flip/damage fractions must be finite between zero and one")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(values))
    result = values[permutation].copy()
    flip_count = int(np.floor(len(values) * flip_fraction + 0.5))
    damage_count = int(np.floor(len(values) * damage_fraction + 0.5))
    flips = sorted(rng.choice(len(values), flip_count, replace=False).tolist())
    damage = sorted(rng.choice(len(values), damage_count, replace=False).tolist())
    operations = []
    for index in flips:
        result[index] = np.fliplr(result[index])
    for index in damage:
        result[index], operation = _augment(result[index], rng)
        operations.append({"input_index": index, **operation})
    truth = {
        "schema_version": 1,
        "seed": int(seed),
        "permutation": permutation.tolist(),
        "flipped_indices": flips,
        "damaged_indices": damage,
        "augmentations": operations,
        "fraction_rounding": "nearest count, half rounded up",
        "scope": "synthetic horizontal flips and pixel damage; no biological realism claim",
    }
    return result, truth
