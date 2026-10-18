"""Tests use varied image structure, not filenames or geometry, for retrieval."""

import numpy as np
import pytest

from slicegps.retrieval import (
    evaluate_retrieval,
    features,
    inversion_fraction,
    order_stack,
    rank,
)


def anatomy_patterns():
    y, x = np.mgrid[-1:1:64j, -1:1:80j]
    patterns = []
    for center_x, center_y, width in ((-0.35, -0.2, 0.32), (0.3, 0.25, 0.45), (0, -0.4, 0.2)):
        lobe = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / width**2)
        companion = 0.65 * np.exp(-((x + center_x) ** 2 / 0.12 + (y - 0.4) ** 2 / 0.06))
        tissue = (x**2 / 0.8**2 + y**2 / 0.9**2) < 1
        patterns.append(np.where(tissue, 0.15 + 0.6 * lobe + 0.25 * companion, 0))
    return np.asarray(patterns)


@pytest.mark.parametrize("method,channels", [("intensity", 1), ("gradient", 3), ("mask", 1)])
def test_features_are_finite_normalized_and_deterministic(method, channels):
    descriptor = features(anatomy_patterns(), method=method, size=24)
    assert descriptor.shape == (3, channels * 24 * 24)
    assert descriptor.dtype == np.float32
    assert np.isfinite(descriptor).all()
    np.testing.assert_allclose(np.linalg.norm(descriptor, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(descriptor, features(anatomy_patterns(), method=method, size=24))


@pytest.mark.parametrize("method", ["intensity", "gradient"])
def test_retrieves_content_after_contrast_and_noise_changes(method):
    gallery = anatomy_patterns()
    rng = np.random.default_rng(17)
    queries = 0.4 + 1.7 * gallery[[2, 0, 1]] + rng.normal(0, 0.003, gallery.shape)
    assert not np.array_equal(queries[0], gallery[2])
    indices, scores = rank(features(queries, method), features(gallery, method), k=2)
    np.testing.assert_array_equal(indices[:, 0], [2, 0, 1])
    assert np.all(scores[:, 0] > scores[:, 1])


def test_gradient_descriptor_tolerates_contrast_reversal():
    images = anatomy_patterns()
    np.testing.assert_allclose(
        features(images, "gradient"), features(1 - images, "gradient"), atol=1e-6
    )


def test_mask_baseline_uses_tissue_shape():
    images = np.zeros((3, 30, 50))
    images[0, 5:25, 5:20] = 1
    images[1, 5:25, 30:45] = 1
    images[2, 5:12, 5:45] = 1
    queries = images[[1, 2, 0]] * 0.7 + 0.1
    indices, _ = rank(features(queries, "mask"), features(images, "mask"))
    np.testing.assert_array_equal(indices[:, 0], [1, 2, 0])


@pytest.mark.parametrize("method", ["intensity", "gradient", "mask"])
@pytest.mark.parametrize("value", [0, 1, -5])
def test_constant_images_are_rejected(method, value):
    with pytest.raises(ValueError, match="nonconstant"):
        features(np.full((1, 10, 10), value), method)


@pytest.mark.parametrize("bad", [np.empty((0, 4, 4)), np.zeros((4, 4)), np.ones((1, 1, 4))])
def test_invalid_image_shapes_are_rejected(bad):
    with pytest.raises(ValueError, match="shape"):
        features(bad)


def test_nonfinite_images_and_invalid_feature_options_are_rejected():
    images = anatomy_patterns()
    images[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        features(images)
    with pytest.raises(ValueError, match="method"):
        features(anatomy_patterns(), method="untrained_model")
    with pytest.raises(ValueError, match="size"):
        features(anatomy_patterns(), size=1)


def test_rank_normalizes_features_clamps_k_and_breaks_ties_stably():
    indices, scores = rank(np.array([[5, 0]]), np.array([[2, 0], [10, 0], [0, 7]]), k=20)
    np.testing.assert_array_equal(indices, [[0, 1, 2]])
    np.testing.assert_allclose(scores, [[1, 1, 0]])


@pytest.mark.parametrize(
    "query,gallery,k",
    [
        ([[0, 0]], [[1, 0]], 1),
        ([[1, np.inf]], [[1, 0]], 1),
        ([[1, 0]], [[1, 0, 0]], 1),
        ([[1, 0]], [[1, 0]], 0),
        (np.empty((0, 2)), [[1, 0]], 1),
    ],
)
def test_invalid_rank_inputs_are_rejected(query, gallery, k):
    with pytest.raises(ValueError):
        rank(query, gallery, k)


def test_retrieval_scores_known_center_errors_at_every_returned_k():
    metrics = evaluate_retrieval(
        np.array([[0, 1], [0, 2]]),
        np.array([[0, 0, 0], [0, 3, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 3, 0]]),
        tolerance_mm=0.5,
    )
    assert metrics["top1_median_error_mm"] == 1.5
    assert metrics["top1_p95_error_mm"] == pytest.approx(2.85)
    assert metrics["recall_at_k"] == {"1": 0.5, "2": 1.0}
    assert metrics["k"] == 2
    assert "angle" in metrics["metric_note"]


@pytest.mark.parametrize("indices", [[[0, 0]], [[-1]], [[2]], [[0.5]], [[]]])
def test_invalid_rankings_cannot_be_scored(indices):
    with pytest.raises(ValueError):
        evaluate_retrieval(np.asarray(indices), np.zeros((1, 3)), np.zeros((2, 3)), 1)


def test_evaluation_rejects_nonfinite_centers_and_negative_tolerance():
    with pytest.raises(ValueError, match="finite"):
        evaluate_retrieval(np.array([[0]]), np.array([[np.nan, 0, 0]]), np.zeros((1, 3)), 1)
    with pytest.raises(ValueError, match="tolerance"):
        evaluate_retrieval(np.array([[0]]), np.zeros((1, 3)), np.zeros((1, 3)), -1)


def test_stack_order_uses_declared_oblique_direction_and_stable_ties():
    centers = np.array([[3, 0, 0], [0, 2, 0], [-1, -1, 5], [2, 1, 0]])
    np.testing.assert_array_equal(order_stack(centers, (1, 1, 0)), [2, 1, 0, 3])
    np.testing.assert_array_equal(order_stack(centers), [2, 0, 3, 1])


@pytest.mark.parametrize("direction", [(0, 0, 0), (np.inf, 0, 0), (1, 0)])
def test_invalid_stack_directions_are_rejected(direction):
    with pytest.raises(ValueError, match="direction"):
        order_stack(np.zeros((2, 3)), direction)


def test_stack_order_rejects_nonfinite_centers():
    with pytest.raises(ValueError, match="finite"):
        order_stack(np.array([[0, 0, np.nan]]))


def test_exact_inversion_fraction():
    truth = ["a", "b", "c", "d"]
    assert inversion_fraction(truth, truth) == 0
    assert inversion_fraction(truth[::-1], truth) == 1
    assert inversion_fraction(["b", "a", "d", "c"], truth) == pytest.approx(2 / 6)
    assert inversion_fraction(["a"], ["a"]) == 0


@pytest.mark.parametrize(
    "order,truth", [([], []), ([0, 0], [0, 1]), ([0, 2], [0, 1]), ([0], [0, 1])]
)
def test_invalid_order_permutations_are_rejected(order, truth):
    with pytest.raises(ValueError):
        inversion_fraction(order, truth)
