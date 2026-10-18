"""Synthetic offline checks for corruption, flip retrieval, and separated QA scoring."""

import json

import numpy as np
import pytest

from slicegps.augmentation import augment_histology, corrupt_stack
from slicegps.geometry import coronal_plane
from slicegps.matching import build_index
from slicegps.stack import evaluate_stack, flip_search, qa_stack


def patterns(count=6):
    images = np.zeros((count, 48, 48), dtype=np.uint8)
    images[:, 4:32, 3:10] = 180
    images[:, 31:39, 3:23] = 180
    for i in range(count):
        images[i, 5 + 5 * i : 9 + 5 * i, 28:36] = 180
    return images


def bundle(tmp_path, images):
    root = tmp_path / "bundle"
    root.mkdir()
    planes = [coronal_plane((0, i * 2, 0), images.shape[1:]).to_dict() for i in range(len(images))]
    np.savez_compressed(root / "images.npz", query_histology=images, gallery_mri=images)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "gallery_planes": planes,
                "query_planes": planes,
                "reference_space": "asymmetric phantom",
            }
        )
    )
    return root


def test_augmentation_is_seeded_finite_and_preserves_visible_tissue():
    image = patterns()[0]
    first = augment_histology(image, np.random.default_rng(9))
    second = augment_histology(image, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert first.shape == image.shape
    assert first.min() >= image.min() and first.max() <= image.max()
    assert np.ptp(first) > 0
    unchanged = augment_histology(
        image,
        np.random.default_rng(3),
        gamma_range=(1, 1),
        gain_range=(1, 1),
        blur_sigma=(0, 0),
        tear_fraction=0,
        missing_chunk_fraction=0,
    )
    np.testing.assert_array_equal(unchanged, image)


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.full((4, 4), np.nan), np.ones(4)])
def test_augmentation_rejects_invalid_images(bad):
    with pytest.raises(ValueError):
        augment_histology(bad, np.random.default_rng(1))


def test_corruption_truth_records_input_permutation_flips_and_damage():
    images = patterns()
    first, truth = corrupt_stack(images, 0.5, 0.5, seed=27)
    second, second_truth = corrupt_stack(images, 0.5, 0.5, seed=27)
    np.testing.assert_array_equal(first, second)
    assert truth == second_truth
    assert len(truth["flipped_indices"]) == 3
    assert len(truth["damaged_indices"]) == 3
    assert sorted(truth["permutation"]) == list(range(len(images)))
    assert all(operation["cuts"] for operation in truth["augmentations"])
    json.dumps(truth, allow_nan=False)
    for i, source in enumerate(truth["permutation"]):
        if i not in truth["damaged_indices"]:
            expected = (
                np.fliplr(images[source]) if i in truth["flipped_indices"] else images[source]
            )
            np.testing.assert_array_equal(first[i], expected)
    with pytest.raises(ValueError, match="fractions"):
        corrupt_stack(images, damage_fraction=1.1)


def test_flip_search_recovers_asymmetric_shuffled_images(tmp_path):
    images = patterns()
    source = bundle(tmp_path, images)
    index = tmp_path / "index"
    build_index(source, index, "intensity")
    corrupted, truth = corrupt_stack(images, 0.5, 0, seed=6)
    result = flip_search(corrupted, index, tmp_path / "matches")
    assert result["horizontal_flip_indices"] == truth["flipped_indices"]
    assert [r["candidates"][0]["gallery_index"] for r in result["matches"]] == truth["permutation"]
    assert (tmp_path / "matches/index.html").exists()
    assert "Symmetric brains" in (tmp_path / "matches/index.html").read_text()
    assert len(result["matches"]) == len(images)


def test_symmetric_images_are_marked_ambiguous(tmp_path):
    images = patterns(1)
    images = np.maximum(images, images[:, :, ::-1])
    source = bundle(tmp_path, images)
    build_index(source, tmp_path / "index", "intensity")
    result = flip_search(images, tmp_path / "index", tmp_path / "out")
    assert result["matches"][0]["flip_ambiguous"] is True
    assert result["horizontal_flip_indices"] == []


def test_qa_reports_pixel_outliers_duplicates_and_ambiguous_gaps():
    images = np.repeat(patterns(1), 10, axis=0)
    images[-1] = 0
    images[-1, 3:7, 3:7] = 180
    depths = [0, 1, 9, 2, 3, 3, 4, 5, 6, 20]
    planes = [coronal_plane((0, y, 0), (48, 48)).to_dict() for y in depths]
    result = qa_stack(images, planes, expected_spacing_mm=1)
    assert 9 in result["corruption_flagged_indices"]
    assert [4, 5] in result["duplicate_depth_pairs"]
    assert result["large_depth_gaps"]
    assert "ambiguous" in result["gap_interpretation"]
    assert 3 in result["input_depth_reversals"]
    assert 2 in result["trajectory_outlier_indices"]


def test_evaluate_scores_every_query_and_does_not_use_truth_for_matching(tmp_path):
    images = patterns()
    source = bundle(tmp_path, images)
    index = tmp_path / "index"
    build_index(source, index, "intensity")
    result = evaluate_stack(
        source, index, tmp_path / "first", flip_fraction=0.5, damage_fraction=0, seed=6
    )
    metrics = result["metrics"]
    assert metrics["evaluated_queries"] == len(images)
    assert metrics["ordering_inversion_fraction"] == 0
    assert metrics["flip_detection"]["precision"] == 1
    assert metrics["flip_detection"]["recall"] == 1
    assert metrics["corruption_detection"]["recall"] is None
    assert (tmp_path / "first/truth.json").exists()
    assert (tmp_path / "first/stack_contact_sheet.png").exists()
    manifest = json.loads((source / "manifest.json").read_text())
    for plane in manifest["query_planes"]:
        plane["center_mm"][1] *= -1
    (source / "manifest.json").write_text(json.dumps(manifest))
    changed = evaluate_stack(
        source, index, tmp_path / "changed", flip_fraction=0.5, damage_fraction=0, seed=6
    )
    assert changed["metrics"]["ordering_inversion_fraction"] == 1
    first_matches = json.loads((tmp_path / "first/retrieval/matches.json").read_text())
    changed_matches = json.loads((tmp_path / "changed/retrieval/matches.json").read_text())
    assert [r["candidates"] for r in first_matches["matches"]] == [
        r["candidates"] for r in changed_matches["matches"]
    ]
    with pytest.raises(FileExistsError):
        evaluate_stack(source, index, tmp_path / "first")


def test_flip_search_transports_float32_pixels_without_display_scaling(tmp_path):
    from PIL import Image

    class RecordingEncoder:
        pooling = None

        def __init__(self):
            self.received = []

        @property
        def metadata(self):
            return {
                "feature_fingerprint": "raw-fixture",
                "model_id": "offline-pixels",
                "model_revision": "test1",
                "preprocessing": {"pixels": "flatten"},
                "pooling": self.pooling,
            }

        def encode(self, images, pooling="cls"):
            self.pooling = pooling
            self.received.extend(np.array(image, copy=True) for image in images)
            return np.asarray(images).reshape(len(images), -1)

    ramp = np.arange(48 * 48).reshape(48, 48)
    images = (patterns(3).astype(np.float32) * 137.25 + ramp / 17 - 14.875).astype(np.float32)
    source = bundle(tmp_path, images)
    index = tmp_path / "index"
    build_index(source, index, "dino", encoder=RecordingEncoder())
    encoder = RecordingEncoder()
    result = flip_search(images, index, tmp_path / "matched", encoder=encoder)
    expected = np.concatenate((images, images[:, :, ::-1]))
    np.testing.assert_array_equal(np.stack(encoder.received), expected)
    assert result["horizontal_flip_indices"] == []
    assert [row["candidates"][0]["gallery_index"] for row in result["matches"]] == [0, 1, 2]
    for row, image in zip(result["matches"], images, strict=True):
        assert row["query_path"].endswith(".tiff")
        with Image.open(row["query_path"]) as saved:
            assert saved.mode == "F"
            np.testing.assert_array_equal(np.asarray(saved), image)
