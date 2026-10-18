"""Error injection, image-only measurements, calibration and honest imbalance metrics."""

import numpy as np
import pytest

from slicegps.error_detection import (
    ERROR_KINDS,
    classification_metrics,
    detector_scores,
    error_features,
    fit_detector,
    inject_errors,
    select_threshold,
    spatial_splits,
)


def stack(count=30):
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    return np.stack(
        [
            np.exp(-((x + 0.3 - i / count * 0.5) ** 2 + (y - 0.1) ** 2) * 8)
            + 0.3 * np.exp(-((x - 0.45) ** 2 + (y + 0.4) ** 2) * 40)
            for i in range(count)
        ]
    )


@pytest.mark.parametrize("kind", ERROR_KINDS)
def test_injection_truth_is_separate_reproducible_and_exact(kind):
    clean = stack()
    original = clean.copy()
    observed, truth = inject_errors(clean, kind, 0.1, 19)
    again, same_truth = inject_errors(clean, kind, 0.1, 19)
    np.testing.assert_array_equal(clean, original)
    np.testing.assert_array_equal(observed, again)
    np.testing.assert_array_equal(truth["is_error"], same_truth["is_error"])
    changed = np.any(observed != clean, axis=(1, 2))
    np.testing.assert_array_equal(changed, truth["is_error"])
    assert not changed[:2].any() and not changed[-2:].any()
    if kind.startswith("misorder"):
        slots = np.flatnonzero(changed)
        shifts = np.abs(slots - truth["source_index"][slots])
        assert np.all(shifts == 1) if kind.endswith("near") else np.all(shifts >= len(clean) // 4)
        np.testing.assert_array_equal(observed, clean[truth["source_index"]])


def test_measurements_do_not_accept_truth_or_ids_and_require_matching_mri():
    clean = stack(10)
    local = error_features(clean)
    paired = error_features(clean, clean)
    assert local.shape == (10, 24)
    assert paired.shape == (10, 30)
    np.testing.assert_allclose(
        local, paired[:, [*range(7), *range(9, 16), *range(18, 25), 27, 28, 29]]
    )
    with pytest.raises(TypeError):
        error_features(clean, truth=np.ones(10))
    with pytest.raises(ValueError, match="matching shapes"):
        error_features(clean, clean[:-1])
    with pytest.raises(ValueError, match="finite"):
        error_features(clean * np.nan)


def test_constant_and_too_short_stacks_fail_explicitly():
    with pytest.raises(ValueError, match="constant"):
        error_features(np.ones((10, 32, 32)))
    with pytest.raises(ValueError):
        inject_errors(stack(8), "flip", 0.05, 1)


def test_all_negative_predictions_cannot_masquerade_as_high_detection_accuracy():
    truth = np.zeros(100, dtype=bool)
    truth[:5] = True
    result = classification_metrics(truth, np.zeros(100, dtype=bool))
    assert result["accuracy"] == 0.95
    assert result["balanced_accuracy"] == 0.5
    assert result["recall"] == 0
    assert result["precision"] is None


def test_threshold_handles_ties_and_validation_false_alarm_cap():
    scores = np.array([0, 0, 1, 1, 2, 3, 8, 9], dtype=float)
    truth = np.array([False] * 6 + [True] * 2)
    threshold, measured = select_threshold(scores, truth, maximum_fpr=0.2)
    assert threshold == 2
    assert measured["false_positive_rate"] <= 0.2
    assert measured["recall"] == 1


def test_fitted_model_roundtrip_and_separate_test_score_contract():
    rng = np.random.default_rng(8)
    train = rng.normal(size=(100, 3))
    labels = train[:, 0] > 0
    model = fit_detector(train, labels)
    test = rng.normal(size=(20, 3))
    scores = detector_scores(test, model)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()
    assert classification_metrics(labels, detector_scores(train, model) > 0)["recall"] > 0.9
    with pytest.raises(ValueError, match="contract"):
        detector_scores(test[:, :2], model)
    with pytest.raises(ValueError):
        fit_detector(train, np.zeros(100, dtype=bool))


def test_spatial_supports_cannot_cross_split_boundaries():
    splits = spatial_splits(344)
    assert len(splits["train"]) == 203
    assert splits["validation"][0] - splits["train"][-1] > 4
    assert splits["test"][0] - splits["validation"][-1] > 4
    for a, b in (("train", "validation"), ("validation", "test")):
        assert not set(splits[a]) & set(splits[b])


def test_paper_estimation_is_pixel_based_and_retains_canvas():
    from slicegps.error_detection import canonical_histology

    image = np.zeros((1, 64, 64))
    image[:, 5:59, 5:59] = 1
    image[:, 20:42, 18:38] = 0.4
    image[:, 12:14, 45:47] = 0.2  # Tiny writing component.
    processed = canonical_histology(image)
    assert processed.shape == image.shape
    assert processed[0, 30, 25] > 0
    assert processed[0, 10, 10] == 0
    assert processed[0, 12, 45] == 0


def test_stack_normalization_uses_only_observations_and_excludes_unsupported_edges():
    from slicegps.error_detection import normalize_stack_features

    x = np.arange(36, dtype=float).reshape(12, 3)
    normalized = normalize_stack_features(x)
    np.testing.assert_allclose(normalized, normalize_stack_features(x * [2, 3, 5] + 10))
    x[:2] += 10000
    x[-2:] -= 10000
    np.testing.assert_allclose(normalize_stack_features(x)[2:-2], normalized[2:-2])
    with pytest.raises(ValueError, match="nine"):
        normalize_stack_features(x[:8])
