"""The issue demo changes real slots with recorded truth and no displacement filter."""

import numpy as np
import pytest

from slicegps.demo_corruption import demo_subgroups, inject_demo_errors


def stack(count=66):
    y, x = np.mgrid[:12, :12]
    return np.stack([x * (i + 1) + y * 0.1 for i in range(count)]).astype(np.float32)


def test_mixed_demo_is_exact_reproducible_and_does_not_mutate_clean_input():
    clean = stack()
    before = clean.copy()
    observed, truth = inject_demo_errors(clean, 0.05, 49301)
    again, same_truth = inject_demo_errors(clean, 0.05, 49301)
    np.testing.assert_array_equal(clean, before)
    np.testing.assert_array_equal(again, observed)
    for key in ("is_error", "source_index", "flipped"):
        np.testing.assert_array_equal(truth[key], same_truth[key])
    np.testing.assert_array_equal(np.any(clean != observed, axis=(1, 2)), truth["is_error"])
    reconstructed = clean[truth["source_index"]].copy()
    reconstructed[truth["flipped"]] = reconstructed[truth["flipped"], :, ::-1]
    np.testing.assert_array_equal(observed, reconstructed)
    shuffled = truth["source_index"] != np.arange(len(clean))
    assert shuffled.sum() == 2 and truth["flipped"].sum() == 1
    assert not np.any(shuffled & truth["flipped"])
    assert not truth["is_error"][:2].any() and not truth["is_error"][-2:].any()
    assert truth["is_error"].sum() == round((len(clean) - 4) * 0.05)


def test_uniform_shuffling_allows_immediate_neighbours_and_distant_errors():
    clean = stack()
    distances = []
    for seed in range(200):
        _, truth = inject_demo_errors(clean, 0.05, seed)
        distances.extend(np.abs(truth["source_index"] - np.arange(len(clean))))
    assert 1 in distances
    assert max(distances) >= len(clean) // 2


def test_demo_subgroups_only_measure_changed_sections_after_prediction():
    clean = stack()
    truths = [inject_demo_errors(clean, 0.05, 41 + i)[1] for i in range(2)]
    flags = np.concatenate([t["flipped"][2:-2] for t in truths])
    metrics = demo_subgroups(clean, 2, 41, flags)
    assert metrics["flipped"] == {"errors": 2, "detected": 2, "recall": 1.0}
    assert metrics["shuffled"] == {"errors": 4, "detected": 0, "recall": 0.0}
    assert metrics["shuffled_to_adjacent_available_slot"]["recall"] is None


@pytest.mark.parametrize("fraction", [0, -0.1, 0.3, float("nan")])
def test_invalid_fraction_and_unrepresentable_small_stacks_fail(fraction):
    with pytest.raises(ValueError, match="fraction"):
        inject_demo_errors(stack(), fraction, 1)
    with pytest.raises(ValueError, match="three changed slots"):
        inject_demo_errors(stack(16), 0.05, 1)
