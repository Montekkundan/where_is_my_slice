"""Paired comparisons must detect exchanges through image evidence, not identities."""

import json

import numpy as np
import pytest
from PIL import Image

from slicegps.error_detection import detector_scores, error_features, fit_detector
from slicegps.error_review import review_errors
from slicegps.pairwise_qa import PairwiseFeatures, comparison_features, self_similarity


def patterns():
    rng = np.random.default_rng(73)
    return rng.random((9, 24, 24), dtype=np.float32)


def test_patch_patterns_are_invariant_to_affine_intensity_and_polarity():
    images = patterns()
    descriptors = self_similarity(images)
    np.testing.assert_allclose(descriptors, self_similarity(8 - 3 * images), atol=1e-6)
    assert np.all(np.isfinite(descriptors))
    np.testing.assert_allclose(np.linalg.norm(descriptors, axis=1), 1, atol=1e-6)


def test_reciprocal_swap_gain_requires_both_candidates_to_improve():
    correct = np.eye(7)
    clean = comparison_features(correct, np.zeros(7))
    wrong = correct.copy()
    wrong[[2, 3]] = wrong[[3, 2]]
    exchanged = comparison_features(wrong, np.zeros(7))
    assert (clean[:, 7] < 0).all()
    assert (exchanged[[2, 3], 7] > 0).all()
    assert (exchanged[[0, 1, 4, 5, 6], 7] <= 0).all()


def test_paired_measurement_cache_does_not_hold_labels_and_matches_fresh_computation():
    images = patterns()
    comparator = PairwiseFeatures(images)
    first = comparator.measure(images)
    altered = images.copy()
    altered[[3, 4]] = altered[[4, 3]]
    reused = comparator.measure(altered)
    np.testing.assert_allclose(reused, PairwiseFeatures(images).measure(altered), atol=1e-7)
    assert first.shape == reused.shape == (9, 34)
    assert (reused[[3, 4], 7] > first[[3, 4], 7]).all()
    assert not any(key in vars(comparator) for key in ("truth", "labels", "ids"))
    with pytest.raises(ValueError, match="canvas"):
        comparator.measure(images[:-1])


def test_saved_paired_model_uses_same_pixels_and_features_as_benchmark(tmp_path):
    images = patterns()
    altered = images.copy()
    altered[[3, 4]] = altered[[4, 3]]
    x = np.column_stack(
        (error_features(altered, images), PairwiseFeatures(images).measure(altered))
    )
    labels = np.isin(np.arange(len(images)), [3, 4])
    model = fit_detector(x, labels)
    model.update(
        feature_contract="stack_error_pairwise_v1",
        expected_mri_required=True,
        learned_features=False,
        image_size=24,
        threshold=0.0,
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model))
    hist_paths, mri_paths = [], []
    for i, (histology, mri) in enumerate(zip(altered, images, strict=True)):
        h, m = tmp_path / f"h-{i}.tiff", tmp_path / f"m-{i}.tiff"
        Image.fromarray(histology).save(h)
        Image.fromarray(mri).save(m)
        hist_paths.append(h)
        mri_paths.append(m)
    result = review_errors(model_path, hist_paths, tmp_path / "review", mri_paths)
    np.testing.assert_allclose(
        [row["score"] for row in result["sections"]], detector_scores(x, model), atol=1e-7
    )


def test_baseline_pixel_cache_preserves_uncached_measurements():
    images = patterns()
    cache = {}
    first = error_features(images, images, cache=cache)
    np.testing.assert_allclose(first, error_features(images, images), atol=1e-8)
    changed = images.copy()
    changed[[3, 4]] = changed[[4, 3]]
    cached = error_features(changed, images, cache=cache)
    np.testing.assert_allclose(cached, error_features(changed, images), atol=1e-8)
