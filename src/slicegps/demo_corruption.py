"""Reproducible mixed stack corruption for the demo described in project issue #10."""

from __future__ import annotations

import numpy as np

from .error_detection import _images

DEMO_KIND = "shuffle_flip"


def inject_demo_errors(images: np.ndarray, fraction: float, seed: int) -> tuple:
    """Shuffle some slots and flip others, changing approximately fraction of the stack.

    Slots are sampled uniformly without a minimum displacement. A derangement
    prevents selected shuffle slots from accidentally staying correct. Flips and
    shuffled slots are disjoint. Boundaries remain untouched because the detector
    does not score them. The supplied MRI is never corrupted or used here.
    """
    images = _images(images, minimum=9)
    if not np.isfinite(fraction) or not 0 < fraction <= 0.2:
        raise ValueError("Demo error fraction must be in (0, 0.2]")
    eligible = np.arange(2, len(images) - 2)
    number = int(round(len(eligible) * fraction))
    if number < 3:
        raise ValueError("Mixed demo needs at least three changed slots at the requested fraction")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=number, replace=False)
    # Roughly half shuffled, with at least two shuffled and one flipped slot.
    shuffle_count = max(2, number // 2)
    shuffle_slots = chosen[:shuffle_count]
    for _ in range(100):
        permutation = rng.permutation(shuffle_slots)
        if np.all(permutation != shuffle_slots):
            break
    else:
        permutation = np.roll(shuffle_slots, 1)
    source = np.arange(len(images))
    source[shuffle_slots] = permutation
    flipped = np.zeros(len(images), dtype=bool)
    flipped[chosen[shuffle_count:]] = True
    observed = images[source].copy()
    observed[flipped] = observed[flipped, :, ::-1]
    return observed, {
        "is_error": (source != np.arange(len(images))) | flipped,
        "source_index": source,
        "flipped": flipped,
        "kind": DEMO_KIND,
    }


def demo_subgroups(images, trials, seed, flags):
    """Score subgroup recall after inference; never supply this truth to the model."""
    truth = [inject_demo_errors(images, 0.05, seed + i)[1] for i in range(trials)]
    displacement = np.concatenate(
        [np.abs(t["source_index"] - np.arange(len(images)))[2:-2] for t in truth]
    )
    flipped = np.concatenate([t["flipped"][2:-2] for t in truth])
    groups = {
        "flipped": flipped,
        "shuffled": displacement > 0,
        "shuffled_to_adjacent_available_slot": displacement == 1,
    }
    result = {}
    for name, selected in groups.items():
        count = int(selected.sum())
        detected = int(np.asarray(flags)[selected].sum())
        result[name] = {
            "errors": count,
            "detected": detected,
            "recall": detected / count if count else None,
        }
    return result
