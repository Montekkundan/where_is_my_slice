"""Exchange evidence is geometric and observed-only, not a clean-image lookup."""

import json

import numpy as np
import pytest
from PIL import Image

from slicegps.error_detection import (
    detector_scores,
    error_features,
    fit_detector,
    normalize_stack_features,
)
from slicegps.error_review import review_errors
from slicegps.sequence_qa import order_evidence, sequence_features


def test_adjacent_exchange_improves_path_and_curvature_on_known_smooth_trajectory():
    angle = np.linspace(0, 0.8, 11)
    desc = np.column_stack((np.cos(angle), np.sin(angle)))
    clean = order_evidence(desc)
    changed = desc.copy()
    changed[[4, 5]] = changed[[5, 4]]
    exchanged = order_evidence(changed)
    assert (clean[2:-2, :2] < 0).all()
    assert (exchanged[[4, 5], :2] > 0).all()
    assert (exchanged[[4, 5], 2:] > 0).all()
    assert (exchanged[[4, 5], 1] > clean[[4, 5], 1]).all()
    # Without an absolute reference, a complete reversal is equally smooth.
    np.testing.assert_allclose(order_evidence(desc[::-1]), clean[::-1], atol=1e-10)


def test_evidence_respects_feature_basis_and_scale():
    rng = np.random.default_rng(77)
    desc = rng.normal(size=(9, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    np.testing.assert_allclose(order_evidence(desc), order_evidence(3 * desc @ q), atol=1e-12)
    with pytest.raises(ValueError, match="five"):
        order_evidence(desc[:4])
    with pytest.raises(ValueError, match="finite"):
        order_evidence(desc * np.nan)


def test_observed_pixel_cache_is_equivalent_to_fresh_measurements():
    images = np.random.default_rng(72).random((9, 24, 24), dtype=np.float32)
    cache = {}
    original = sequence_features(images, cache=cache)
    altered = images.copy()
    altered[[3, 4]] = altered[[4, 3]]
    shared = sequence_features(altered, cache=cache)
    fresh = sequence_features(altered)
    assert original.shape == shared.shape == (9, 24)
    np.testing.assert_allclose(shared, fresh, atol=1e-10)


@pytest.mark.parametrize("paired", [False, True])
def test_saved_sequence_reader_uses_benchmark_features(tmp_path, paired):
    images = np.random.default_rng(8).random((9, 24, 24), dtype=np.float32)
    altered = images.copy()
    altered[[3, 4]] = altered[[4, 3]]
    reference = images if paired else None
    x = normalize_stack_features(
        np.column_stack((error_features(altered, reference), sequence_features(altered)))
    )
    model = fit_detector(x, np.isin(np.arange(9), [3, 4]))
    model.update(
        feature_contract="stack_error_sequence_v1",
        expected_mri_required=paired,
        learned_features=False,
        stack_normalized=True,
        image_size=24,
        threshold=0.0,
    )
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    hp, mp = [], []
    for i, (histology, mri) in enumerate(zip(altered, images, strict=True)):
        h, m = tmp_path / f"h-{i}.tiff", tmp_path / f"m-{i}.tiff"
        Image.fromarray(histology).save(h)
        Image.fromarray(mri).save(m)
        hp.append(h)
        mp.append(m)
    reviewed = review_errors(path, hp, tmp_path / "review", mp if paired else None)
    np.testing.assert_allclose(
        [row["score"] for row in reviewed["sections"]], detector_scores(x, model), atol=1e-7
    )
