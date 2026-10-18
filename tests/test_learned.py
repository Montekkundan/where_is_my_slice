"""Offline protocol and leakage checks, without model downloads or human-data claims."""

import hashlib
import json
import shutil
from itertools import combinations

import numpy as np
import pytest

from slicegps import learned
from slicegps.geometry import Plane, coronal_plane
from slicegps.pipeline import create_phantom
from slicegps.retrieval import features
from slicegps.volume import load_volume, save_volume


def _array_digest(images):
    return hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()


class PixelEncoder:
    """Deterministic image-only stand-in with two descriptor representations."""

    def __init__(self, events=None):
        self.pooling = None
        self.events = [] if events is None else events

    def encode(self, images, pooling="cls"):
        self.pooling = pooling
        self.events.append(("encode", pooling, _array_digest(images)))
        intensity = features(images, "intensity", size=8)
        if pooling == "cls":
            return intensity
        assert pooling == "spatial"
        descriptors = np.concatenate((intensity, features(images, "gradient", size=8)), axis=1)
        return descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)

    @property
    def metadata(self):
        return {
            "model_id": "offline-pixel-test-encoder",
            "pooling": self.pooling,
            "training": "none; deterministic structural features",
        }


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.fixture(scope="module")
def protocol_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("learned-protocol")
    histology, mri = create_phantom(root / "phantom")
    output = root / "protocol"
    manifest = learned.prepare_protocol(
        histology, mri, output,
        train_count=16, validation_count=4, test_count=8, gallery_count=16,
        size=32, pixel_mm=1, tilt_deg=0.5,
    )
    return histology, mri, output, manifest


@pytest.fixture(scope="module")
def fitted_fixture(protocol_fixture, tmp_path_factory):
    _, _, protocol, _ = protocol_fixture
    output = tmp_path_factory.mktemp("learned-fit") / "fit"
    encoder = PixelEncoder()
    report = learned.fit_and_evaluate(
        protocol, output, encoder, regularizations=(0.01, 1.0), tolerance_mm=5,
    )
    return output, report, encoder


def test_depth_support_includes_tilted_corners_not_only_centres():
    plane = coronal_plane((0, 10, 0), (7, 9), (2, 3), (4, -3))
    grid = plane.world_grid()[1]
    expected_corners = grid[np.ix_([0, -1], [0, -1])]
    lower, upper = learned.depth_support(plane)
    assert lower == expected_corners.min()
    assert upper == expected_corners.max()
    assert lower < 10 < upper
    first = coronal_plane((0, 0, 0), (32, 32), tilt_deg=(5, 5))
    second = coronal_plane((0, 1, 0), (32, 32), tilt_deg=(5, 5))
    assert first.center_mm[1] < second.center_mm[1]
    with pytest.raises(ValueError, match="overlap"):
        learned.check_separation({
            "train": [learned.depth_support(first)],
            "test": [learned.depth_support(second)],
        })


def test_depth_support_expands_corners_by_source_interpolation_padding():
    plane = coronal_plane((0, 10, 0), (7, 9), (2, 3), (4, -3))
    grid = plane.world_grid()[1]
    assert learned.depth_support(plane, padding_mm=3) == (
        grid.min() - 3, grid.max() + 3,
    )


@pytest.mark.parametrize("padding", [-1, np.nan, np.inf, -np.inf])
def test_depth_support_rejects_negative_or_nonfinite_padding(padding):
    with pytest.raises(ValueError, match="padding"):
        learned.depth_support(coronal_plane((0, 0, 0)), padding_mm=padding)


@pytest.mark.parametrize("right", [[(1, 2)], [(0.5, 3)], [(0.25, 0.75)]])
def test_support_touching_overlap_and_nested_intervals_are_rejected(right):
    with pytest.raises(ValueError, match="overlap"):
        learned.check_separation({"train": [(0, 1)], "test": right})


def test_support_minimum_gap_checks_all_cross_split_pairs():
    groups = {"train": [(0, 1), (8, 9)], "validation": [(2, 3)], "test": [(5, 6)]}
    assert learned.check_separation(groups) == 1
    with pytest.raises(ValueError):
        learned.check_separation({"train": [(0, 1)], "test": []})


@pytest.mark.parametrize("invalid", [(np.nan, 8), (8, np.inf), (-np.inf, 8), (8, 7)])
def test_support_nonfinite_and_reversed_intervals_are_rejected(invalid):
    # A valid earlier pair must not hide a later nonfinite interval in min/max.
    with pytest.raises(ValueError):
        learned.check_separation({
            "train": [(0, 1)], "validation": [(2, 3)], "test": [invalid],
        })


def test_protocol_generates_reproducible_spatial_holdout_and_recorded_hash(protocol_fixture):
    _, _, protocol, manifest = protocol_fixture
    arrays = _arrays(protocol / "images.npz")
    assert set(arrays) == {
        "train_histology", "train_mri", "validation_histology", "test_histology", "gallery_mri",
    }
    for key, count in (
        ("train_histology", 16), ("train_mri", 16),
        ("validation_histology", 4), ("test_histology", 8), ("gallery_mri", 16),
    ):
        assert arrays[key].shape == (count, 32, 32)
        assert np.isfinite(arrays[key]).all()
    assert manifest["counts"] == {"train": 16, "validation": 4, "test": 8, "gallery": 16}
    assert manifest["images_sha256"] == hashlib.sha256(
        (protocol / "images.npz").read_bytes()
    ).hexdigest()
    assert manifest["minimum_cross_split_support_gap_mm"] > 0
    assert learned.check_separation(manifest["depth_supports_mm"]) == pytest.approx(
        manifest["minimum_cross_split_support_gap_mm"]
    )
    edges = manifest["band_edges_mm"]
    for split, planes in manifest["planes"].items():
        bands = manifest["split_bands"][split]
        for index, geometry in enumerate(planes):
            plane = Plane.from_dict(geometry)
            low, high = learned.depth_support(
                plane, padding_mm=manifest["interpolation_padding_mm"],
            )
            np.testing.assert_allclose(
                [low, high], manifest["depth_supports_mm"][split][index], atol=1e-12,
            )
            band = bands[index % len(bands)]
            assert edges[band] < low <= high < edges[band + 1]
    assert any(np.ptp(Plane.from_dict(p).world_grid()[1]) > 0 for p in manifest["planes"]["test"])
    assert "one registered volume" in manifest["scope"]


def test_coarse_grid_protocol_keeps_actual_interpolation_voxels_disjoint(
    protocol_fixture, tmp_path,
):
    histology, mri, _, _ = protocol_fixture
    affine = np.diag([1.0, 3.0, 1.0, 1.0])
    affine[:3, 3] = [-36, -132, -32]
    coarse_paths = [tmp_path / "histology.nii.gz", tmp_path / "mri.nii.gz"]
    for original, coarse in zip((histology, mri), coarse_paths, strict=True):
        data, _ = load_volume(original)
        save_volume(coarse, data, affine)
    manifest = learned.prepare_protocol(
        *coarse_paths, tmp_path / "protocol",
        train_count=16, validation_count=4, test_count=8, gallery_count=16,
        size=32, pixel_mm=1, tilt_deg=0.5,
    )
    assert manifest["interpolation_padding_mm"] == 3
    inverse = np.linalg.inv(affine)
    source_rows = {}
    actual_supports = {}
    for split, geometries in manifest["planes"].items():
        source_rows[split] = set()
        actual_supports[split] = []
        for index, geometry in enumerate(geometries):
            world = Plane.from_dict(geometry).world_grid().reshape(3, -1)
            voxel_y = (inverse[:3, :3] @ world + inverse[:3, 3, None])[1]
            # Trilinear interpolation reads the floor/ceil lattice neighbours.
            # Calculate these independently of depth_support and the manifest.
            rows = np.unique(np.concatenate((np.floor(voxel_y), np.ceil(voxel_y))))
            source_rows[split].update(rows.astype(int).tolist())
            actual_low, actual_high = 3 * rows.min() - 132, 3 * rows.max() - 132
            actual_supports[split].append((actual_low, actual_high))
            padded_low, padded_high = manifest["depth_supports_mm"][split][index]
            np.testing.assert_allclose(
                [padded_low, padded_high], [world[1].min() - 3, world[1].max() + 3],
            )
            assert padded_low <= actual_low <= actual_high <= padded_high
    for left, right in combinations(source_rows.values(), 2):
        assert left.isdisjoint(right)
    assert learned.check_separation(actual_supports) > 0

    # Distinct physical depths alone do not guarantee independent source voxels.
    nearby = [coronal_plane((0, y, 0)) for y in (1, 2)]
    assert learned.check_separation({
        "train": [learned.depth_support(nearby[0])],
        "test": [learned.depth_support(nearby[1])],
    }) == 1
    with pytest.raises(ValueError, match="overlap"):
        learned.check_separation({
            "train": [learned.depth_support(nearby[0], padding_mm=3)],
            "test": [learned.depth_support(nearby[1], padding_mm=3)],
        })


def test_seed_reproduces_arrays_and_geometry_and_allows_empty_output(protocol_fixture, tmp_path):
    histology, mri, protocol, original = protocol_fixture
    output = tmp_path / "empty_output"
    output.mkdir()
    repeated = learned.prepare_protocol(
        histology, mri, output,
        train_count=16, validation_count=4, test_count=8, gallery_count=16,
        size=32, pixel_mm=1, tilt_deg=0.5, seed=original["seed"],
    )
    first_arrays, second_arrays = _arrays(protocol / "images.npz"), _arrays(output / "images.npz")
    for key in first_arrays:
        np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
    assert repeated["planes"] == original["planes"]
    assert repeated["depth_supports_mm"] == original["depth_supports_mm"]


def test_offline_fit_outputs_validation_selection_metrics_and_numeric_models(fitted_fixture):
    output, report, _ = fitted_fixture
    assert len(report["validation"]) == 6
    assert len(report["test_results"]) == 7
    selected = report["selected_by_validation"]
    assert selected["regularization"] in (None, 0.01, 1.0)
    frozen = [row for row in report["validation"] if row["method"].startswith("dino_")]
    assert {row["method"] for row in frozen} == {"dino_cls", "dino_spatial"}
    assert all(row["regularization"] is None for row in frozen)
    best_recall = max(row["recall_at_k"]["1"] for row in report["validation"])
    best_median = min(row["top1_median_error_mm"] for row in report["validation"]
                      if row["recall_at_k"]["1"] == best_recall)
    assert selected["validation_recall_at_1"] == best_recall
    assert selected["validation_median_error_mm"] == best_median
    assert (selected["regularization"] is None) == selected["method"].startswith("dino_")
    for result in report["test_results"]:
        assert result["query_count"] == 8
        assert result["gallery_count"] == 16
        assert np.asarray(result["ranked_gallery_indices"]).shape == (8, 5)
        assert np.isfinite(result["cosine_similarities_not_confidence"]).all()
    assert (output / "matches.png").exists()
    assert (output / "metrics.json").exists()
    for pooling in ("cls", "spatial"):
        assert (output / f"adapter_{pooling}.json").exists()
        assert (output / f"features_{pooling}.npz").exists()
        assert all(array.dtype.kind in "fiu" for array in _arrays(
            output / f"adapter_{pooling}.npz"
        ).values())
    json.dumps(report, allow_nan=False)


def test_test_images_cannot_change_validation_choice_or_fitted_numeric_state(
    protocol_fixture, fitted_fixture, tmp_path,
):
    _, _, protocol, _ = protocol_fixture
    baseline_output, baseline, _ = fitted_fixture
    changed = tmp_path / "changed_protocol"
    shutil.copytree(protocol, changed)
    original = _arrays(changed / "images.npz")
    updated = {name: values.copy() for name, values in original.items()}
    updated["test_histology"] = updated["test_histology"][::-1].copy()
    assert not np.array_equal(updated["test_histology"], original["test_histology"])
    for key in set(updated) - {"test_histology"}:
        np.testing.assert_array_equal(updated[key], original[key])
    np.savez_compressed(changed / "images.npz", **updated)
    manifest = json.loads((changed / "manifest.json").read_text())
    manifest["images_sha256"] = hashlib.sha256((changed / "images.npz").read_bytes()).hexdigest()
    (changed / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "changed_fit"
    second = learned.fit_and_evaluate(
        changed, output, PixelEncoder(), regularizations=(0.01, 1.0), tolerance_mm=5,
    )
    assert second["selected_by_validation"] == baseline["selected_by_validation"]
    assert second["validation"] == baseline["validation"]
    assert second["protocol_images_sha256"] != baseline["protocol_images_sha256"]
    for pooling in ("cls", "spatial"):
        first_state = _arrays(baseline_output / f"adapter_{pooling}.npz")
        second_state = _arrays(output / f"adapter_{pooling}.npz")
        assert set(first_state) == set(second_state)
        for key in first_state:
            np.testing.assert_array_equal(first_state[key], second_state[key])
    # Image changes must still reach final scoring rather than being ignored.
    for before, after in zip(baseline["test_results"], second["test_results"], strict=True):
        assert before["method"] == after["method"]
        assert after["ranked_gallery_indices"] == before["ranked_gallery_indices"][::-1]


def test_test_geometry_changes_scoring_but_never_features_or_adapter_fit(
    protocol_fixture, fitted_fixture, tmp_path,
):
    _, _, protocol, _ = protocol_fixture
    baseline_output, baseline, _ = fitted_fixture
    changed = tmp_path / "changed_geometry"
    shutil.copytree(protocol, changed)
    manifest = json.loads((changed / "manifest.json").read_text())
    for plane in manifest["planes"]["test"]:
        plane["center_mm"][0] += 1000
    (changed / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "geometry_fit"
    report = learned.fit_and_evaluate(
        changed, output, PixelEncoder(), regularizations=(0.01, 1.0), tolerance_mm=5,
    )
    assert report["selected_by_validation"] == baseline["selected_by_validation"]
    for before, after in zip(baseline["test_results"], report["test_results"], strict=True):
        assert before["ranked_gallery_indices"] == after["ranked_gallery_indices"]
        assert after["top1_median_error_mm"] > 900
    for pooling in ("cls", "spatial"):
        for key, array in _arrays(baseline_output / f"adapter_{pooling}.npz").items():
            np.testing.assert_array_equal(array, _arrays(output / f"adapter_{pooling}.npz")[key])


def test_fit_receives_only_training_pixel_features_and_test_encoding_happens_last(
    protocol_fixture, tmp_path, monkeypatch,
):
    _, _, protocol, _ = protocol_fixture
    images = _arrays(protocol / "images.npz")
    expected = {}
    for pooling in ("cls", "spatial"):
        encoder = PixelEncoder()
        expected[pooling] = (
            encoder.encode(images["train_histology"], pooling),
            encoder.encode(images["train_mri"], pooling),
        )
    events, fitted = [], []
    actual_fit = learned.fit_ridge

    def checked_fit(inputs, targets, regularization):
        pooling = "cls" if inputs.shape[1] == 64 else "spatial"
        np.testing.assert_array_equal(inputs, expected[pooling][0])
        np.testing.assert_array_equal(targets, expected[pooling][1])
        fitted.append(pooling)
        events.append(("fit", pooling))
        return actual_fit(inputs, targets, regularization)

    monkeypatch.setattr(learned, "fit_ridge", checked_fit)
    learned.fit_and_evaluate(
        protocol, tmp_path / "verified_fit", PixelEncoder(events), regularizations=(0.01, 1.0),
    )
    assert fitted == ["cls", "cls", "spatial", "spatial"]
    test_digest = _array_digest(images["test_histology"])
    test_events = [i for i, event in enumerate(events) if event[0] == "encode"
                   and event[2] == test_digest]
    fit_events = [i for i, event in enumerate(events) if event[0] == "fit"]
    assert len(test_events) == 2
    assert min(test_events) > max(fit_events)


def test_images_hash_is_enforced_before_encoding(protocol_fixture, tmp_path):
    _, _, protocol, _ = protocol_fixture
    changed = tmp_path / "tampered"
    shutil.copytree(protocol, changed)
    with (changed / "images.npz").open("ab") as stream:
        stream.write(b"tampering")
    encoder = PixelEncoder()
    with pytest.raises(ValueError, match="hash"):
        learned.fit_and_evaluate(changed, tmp_path / "fit", encoder)
    assert not encoder.events


def test_nonfinite_image_content_is_rejected_even_with_updated_hash(protocol_fixture, tmp_path):
    _, _, protocol, _ = protocol_fixture
    changed = tmp_path / "nonfinite"
    shutil.copytree(protocol, changed)
    images = _arrays(changed / "images.npz")
    images["train_histology"][0, 0, 0] = np.nan
    np.savez_compressed(changed / "images.npz", **images)
    manifest = json.loads((changed / "manifest.json").read_text())
    manifest["images_sha256"] = hashlib.sha256((changed / "images.npz").read_bytes()).hexdigest()
    (changed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finite"):
        learned.fit_and_evaluate(changed, tmp_path / "invalid_fit", PixelEncoder())


def test_protocol_and_fit_refuse_nonempty_outputs_without_overwriting(protocol_fixture, tmp_path):
    histology, mri, protocol, _ = protocol_fixture
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "unrelated.txt"
    marker.write_text("keep this content")
    with pytest.raises(FileExistsError):
        learned.prepare_protocol(
            histology, mri, output, train_count=16, validation_count=4,
            test_count=8, gallery_count=16, size=32, pixel_mm=1, tilt_deg=0.5,
        )
    encoder = PixelEncoder()
    with pytest.raises(FileExistsError):
        learned.fit_and_evaluate(protocol, output, encoder)
    assert marker.read_text() == "keep this content"
    assert not encoder.events


def test_too_narrow_depth_bands_fail_before_creating_output(protocol_fixture, tmp_path):
    histology, mri, _, _ = protocol_fixture
    output = tmp_path / "too_narrow"
    with pytest.raises(ValueError, match="too narrow"):
        learned.prepare_protocol(
            histology, mri, output, train_count=16, validation_count=4,
            test_count=8, gallery_count=16, size=128, pixel_mm=2, tilt_deg=5,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("pixel_mm", 0), ("pixel_mm", -1), ("pixel_mm", np.nan), ("pixel_mm", np.inf),
        ("tilt_deg", -0.1), ("tilt_deg", 5.1), ("tilt_deg", np.nan), ("tilt_deg", np.inf),
        ("size", 15), ("size", np.nan), ("size", np.inf), ("size", 32.5),
        ("train_count", 3), ("validation_count", 3.5), ("test_count", 0), ("gallery_count", 2),
    ],
)
def test_invalid_protocol_parameters_have_clear_validation(
    protocol_fixture, tmp_path, option, value,
):
    histology, mri, _, _ = protocol_fixture
    options = dict(
        train_count=16, validation_count=4, test_count=8, gallery_count=16,
        size=32, pixel_mm=1, tilt_deg=0.5,
    )
    options[option] = value
    with pytest.raises(ValueError):
        learned.prepare_protocol(histology, mri, tmp_path / "invalid", **options)


@pytest.mark.parametrize("regularizations", [(), (0,), (-1,), (np.nan,), (np.inf,)])
def test_invalid_regularizations_are_rejected_before_encoding(
    protocol_fixture, tmp_path, regularizations,
):
    _, _, protocol, _ = protocol_fixture
    encoder = PixelEncoder()
    with pytest.raises(ValueError, match="regularizations"):
        learned.fit_and_evaluate(
            protocol, tmp_path / "invalid", encoder, regularizations=regularizations,
        )
    assert not encoder.events
