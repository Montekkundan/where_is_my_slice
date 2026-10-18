import numpy as np
import pytest

from slicegps.adapters import RidgeAdapter, fit_ridge
from slicegps.retrieval import rank


def _unit(values):
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_training_pairs_learn_heldout_linear_cross_modal_mapping():
    random = np.random.default_rng(42)
    train = random.normal(size=(80, 7))
    query = random.normal(size=(30, 7))
    # Orthogonal mapping preserves unit norms, so the cross-modal relation is exact.
    mapping = -np.roll(np.eye(7), 1, axis=1)
    training_targets = train @ mapping
    heldout_targets = _unit(query @ mapping)
    adapter = fit_ridge(train, training_targets, regularization=1e-4)
    prediction = adapter.transform(query)
    learned_alignment = np.sum(prediction * heldout_targets, axis=1).mean()
    raw_alignment = np.sum(_unit(query) * heldout_targets, axis=1).mean()
    assert learned_alignment > 0.9999
    assert learned_alignment > raw_alignment + 0.7
    indices, _ = rank(prediction, heldout_targets, k=1)
    np.testing.assert_array_equal(indices[:, 0], np.arange(len(query)))
    assert prediction.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(prediction, axis=1), 1, atol=1e-7)


def test_different_feature_widths_and_exact_training_only_centering():
    random = np.random.default_rng(3)
    train = random.normal(size=(30, 3)) + [0.5, 0, 0]
    basis, _ = np.linalg.qr(random.normal(size=(5, 3)))
    targets = train @ basis.T
    adapter = fit_ridge(train, targets, regularization=0.1)
    x, y = _unit(train), _unit(targets)
    np.testing.assert_allclose(adapter.input_mean, x.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(adapter.target_mean, y.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(adapter.train_centered, x - x.mean(axis=0), atol=1e-15)
    centered = x - x.mean(axis=0)
    kernel = centered @ centered.T
    scale = np.diag(kernel).mean()
    expected_weights = np.linalg.solve(kernel + 0.1 * scale * np.eye(len(x)), y - y.mean(axis=0))
    np.testing.assert_allclose(adapter.dual_weights, expected_weights, rtol=1e-10, atol=1e-12)
    assert adapter.transform(np.array([[100, 2, -1]])).shape == (1, 5)


def test_fit_and_transform_do_not_mutate_features_or_model_state():
    random = np.random.default_rng(17)
    train = random.normal(size=(12, 4))
    targets = random.normal(size=(12, 6))
    query = random.normal(size=(3, 4))
    originals = [array.copy() for array in (train, targets, query)]
    adapter = fit_ridge(train, targets)
    mean_before = adapter.input_mean.copy()
    prediction = adapter.transform(query)
    for actual, original in zip((train, targets, query), originals, strict=True):
        np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(adapter.input_mean, mean_before)
    assert np.isfinite(prediction).all()
    with pytest.raises(ValueError, match="read-only"):
        adapter.train_centered[0, 0] = 0


def test_save_load_matches_and_archive_contains_only_numeric_arrays(tmp_path):
    random = np.random.default_rng(5)
    adapter = fit_ridge(random.normal(size=(10, 4)), random.normal(size=(10, 7)), 0.3)
    query = random.normal(size=(3, 4))
    path = tmp_path / "nested" / "adapter.npz"
    adapter.save(path)
    loaded = RidgeAdapter.load(path)
    np.testing.assert_array_equal(adapter.transform(query), loaded.transform(query))
    assert loaded.regularization == 0.3
    assert loaded.kernel_scale == adapter.kernel_scale
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind in "fiu" for key in archive.files)
    with pytest.raises(ValueError, match=".npz"):
        adapter.save(tmp_path / "adapter.bin")


@pytest.mark.parametrize(
    "bad",
    [
        np.ones(4), np.empty((0, 3)), np.empty((3, 0)),
        np.array([[0, 0], [1, 1]]), np.array([[np.nan, 1], [1, 1]]),
        np.array([[np.inf, 1], [1, 1]]), np.ones((2, 2), complex),
        np.full((2, 2), "1"), np.ones((2, 2), bool),
    ],
)
def test_invalid_training_features_rejected(bad):
    with pytest.raises(ValueError):
        fit_ridge(bad, np.eye(2))
    with pytest.raises(ValueError):
        fit_ridge(np.eye(2), bad)


@pytest.mark.parametrize("regularization", [0, -1, np.nan, np.inf, True, "1", 1j])
def test_invalid_regularization_rejected(regularization):
    with pytest.raises(ValueError, match="regularization"):
        fit_ridge(np.eye(2), np.eye(2), regularization)


def test_training_requires_paired_rows_and_nonconstant_unit_inputs():
    with pytest.raises(ValueError, match="same N"):
        fit_ridge(np.eye(2), np.eye(3))
    with pytest.raises(ValueError, match="same N"):
        fit_ridge(np.ones((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError, match="vary"):
        fit_ridge(np.array([[1, 2], [2, 4]]), np.eye(2))


def test_query_validation_and_zero_prediction_have_clear_errors():
    adapter = fit_ridge(np.array([[1, 0], [-1, 0]]), np.array([[1, 0], [-1, 0]]))
    with pytest.raises(ValueError, match="dimension"):
        adapter.transform(np.ones((1, 3)))
    with pytest.raises(ValueError, match="query.*zero"):
        adapter.transform(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        adapter.transform(np.array([[np.nan, 1]]))
    with pytest.raises(ValueError, match="predicted features.*zero"):
        adapter.transform(np.array([[0, 1]]))


def test_large_finite_feature_values_are_normalized_without_overflow():
    scale = np.finfo(np.float64).max / 2
    adapter = fit_ridge(scale * np.eye(3), scale * np.eye(3))
    prediction = adapter.transform(scale * np.eye(3))
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(np.linalg.norm(prediction, axis=1), 1, atol=1e-7)


def test_loading_rejects_object_arrays_and_unknown_schema(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(path, unknown=np.array([1]))
    with pytest.raises(ValueError, match="fields"):
        RidgeAdapter.load(path)
    adapter = fit_ridge(np.eye(3), np.eye(3))
    adapter.save(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    data["input_mean"] = np.array([object()], dtype=object)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="Object arrays"):
        RidgeAdapter.load(path)


def test_loading_validates_model_shapes_and_version(tmp_path):
    adapter = fit_ridge(np.eye(3), np.eye(3))
    path = tmp_path / "adapter.npz"
    adapter.save(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    wrong_shape = {**data, "input_mean": np.ones(7)}
    np.savez(path, **wrong_shape)
    with pytest.raises(ValueError, match="means"):
        RidgeAdapter.load(path)
    np.savez(path, **{**data, "format_version": np.array(2)})
    with pytest.raises(ValueError, match="format_version"):
        RidgeAdapter.load(path)
