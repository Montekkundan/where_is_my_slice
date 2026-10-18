"""Offline MRI-only gallery checks against physical-coordinate analytic volumes."""

import json
from itertools import product

import nibabel as nib
import numpy as np
import pytest

from slicegps.data import sha256
from slicegps.gallery import prepare_mri_gallery
from slicegps.geometry import Plane, coronal_plane
from slicegps.matching import build_index
from slicegps.pipeline import read_bundle
from slicegps.volume import save_volume


@pytest.fixture
def physical_mri(tmp_path):
    shape = (24, 36, 28)
    affine = np.diag([-2.0, 3.0, 1.5, 1.0])
    affine[:3, 3] = (40, -50, -12)
    indices = np.indices(shape).reshape(3, -1)
    world = affine[:3, :3] @ indices + affine[:3, 3, None]
    data = (500 + 2 * world[0] + 3 * world[1] + 5 * world[2]).reshape(shape)
    path = tmp_path / "misleading_slice_9999.nii.gz"
    save_volume(path, data, affine)
    return path


def test_default_search_bounds_are_physical_and_bundle_has_only_mri(physical_mri, tmp_path):
    output = tmp_path / "gallery"
    manifest = prepare_mri_gallery(physical_mri, output, count=4, size=12, pixel_mm=1)
    _, arrays = read_bundle(output)
    assert set(arrays) == {"gallery_mri"}
    assert arrays["gallery_mri"].shape == (4, 12, 12)
    assert arrays["gallery_mri"].dtype == np.float32
    # Flipped x, anisotropic 3mm y spacing, and translations all affect world bounds.
    np.testing.assert_allclose(manifest["source_volume"]["world_bounds_mm"],
                               [[-6, -50, -12], [40, 55, 28.5]])
    np.testing.assert_allclose(manifest["center_xz_mm"], [17, 8.25])
    np.testing.assert_allclose(manifest["depth_range_mm"], [-29, 34])
    centers = np.asarray([p["center_mm"] for p in manifest["gallery_planes"]])
    np.testing.assert_allclose(centers[:, 1], np.linspace(-29, 34, 4))
    assert manifest["gallery_candidate_indices"] == [0, 1, 2, 3]
    assert manifest["mode"] == "standalone_mri_gallery"
    assert manifest["images_sha256"] == sha256(output / "images.npz")
    assert manifest["gallery_count"] == 4
    assert manifest["exclusions"] == []
    json.dumps(manifest, allow_nan=False)


def test_oblique_candidates_match_analytical_world_field_and_explicit_product_order(
    physical_mri, tmp_path,
):
    output = tmp_path / "oblique"
    angles = (-3, 0, 3)
    manifest = prepare_mri_gallery(
        physical_mri, output, count=2, size=15, pixel_mm=1,
        center_xz=(17, 8.25), depth_range=(-20, 20), tilts=angles,
    )
    _, arrays = read_bundle(output)
    expected_order = list(product((-20, 20), angles, angles))
    assert len(manifest["gallery_planes"]) == len(expected_order) == 18
    for image, geometry, (depth, x_angle, z_angle) in zip(
        arrays["gallery_mri"], manifest["gallery_planes"], expected_order, strict=True,
    ):
        plane = Plane.from_dict(geometry)
        expected = coronal_plane((17, depth, 8.25), (15, 15), (1, 1), (x_angle, z_angle))
        np.testing.assert_allclose(plane.u, expected.u, atol=1e-12)
        np.testing.assert_allclose(plane.v, expected.v, atol=1e-12)
        grid = plane.world_grid()
        np.testing.assert_allclose(image, 500 + 2 * grid[0] + 3 * grid[1] + 5 * grid[2],
                                   atol=1e-4)


def test_exclusions_keep_actual_depth_and_original_candidate_index(tmp_path):
    data = np.zeros((16, 20, 16), dtype=np.float32)
    x, _, z = np.indices(data.shape)
    data[:, 7:12, :] = (10 + x + z)[:, 7:12, :]
    mri = tmp_path / "mri.nii.gz"
    save_volume(mri, data, np.eye(4))
    output = tmp_path / "gallery"
    manifest = prepare_mri_gallery(
        mri, output, count=5, size=8, pixel_mm=1, depth_range=(0, 19),
    )
    assert manifest["gallery_count"] == 1
    assert manifest["gallery_candidate_indices"] == [2]
    assert manifest["gallery_planes"][0]["center_mm"][1] == 9.5
    assert [entry["candidate_index"] for entry in manifest["exclusions"]] == [0, 1, 3, 4]
    np.testing.assert_allclose(
        [entry["plane"]["center_mm"][1] for entry in manifest["exclusions"]],
        [0, 4.75, 14.25, 19],
    )
    assert all("empty" in entry["reason"] for entry in manifest["exclusions"])


@pytest.mark.parametrize("value", [0, 7])
def test_empty_and_constant_mri_have_no_indexable_gallery(tmp_path, value):
    mri = tmp_path / "constant.nii.gz"
    save_volume(mri, np.full((16, 20, 16), value), np.eye(4))
    output = tmp_path / "gallery"
    with pytest.raises(ValueError, match="No nonempty, nonconstant MRI planes"):
        prepare_mri_gallery(mri, output, count=4, size=8, pixel_mm=1)
    assert not (output / "images.npz").exists()


def test_unknown_units_need_explicit_assertion(physical_mri, tmp_path):
    image = nib.load(physical_mri)
    image.header.set_xyzt_units("unknown")
    nib.save(image, physical_mri)
    with pytest.raises(ValueError, match="units"):
        prepare_mri_gallery(physical_mri, tmp_path / "bad", count=2, size=8, pixel_mm=1)
    manifest = prepare_mri_gallery(
        physical_mri, tmp_path / "asserted", count=2, size=8, pixel_mm=1, assume_mm=True,
    )
    assert manifest["source_volume"]["spatial_units"] == "unknown"
    assert manifest["assume_mm_for_unspecified_units"] is True


@pytest.mark.parametrize("units", ["meter", "micron"])
def test_non_mm_declared_units_are_rejected_even_with_assertion(physical_mri, tmp_path, units):
    image = nib.load(physical_mri)
    image.header.set_xyzt_units(units)
    nib.save(image, physical_mri)
    with pytest.raises(ValueError, match="units"):
        prepare_mri_gallery(
            physical_mri, tmp_path / "bad", count=2, size=8, pixel_mm=1, assume_mm=True,
        )


def test_bundle_builds_intensity_index_without_histology(physical_mri, tmp_path):
    bundle = tmp_path / "gallery"
    prepared = prepare_mri_gallery(physical_mri, bundle, count=3, size=12, pixel_mm=1)
    indexed = build_index(bundle, tmp_path / "index", method="intensity")
    assert indexed["gallery_planes"] == prepared["gallery_planes"]
    assert indexed["gallery_shape"] == [3, 12, 12]


def test_nonempty_output_guard_preserves_unrelated_content(physical_mri, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unrelated content")
    with pytest.raises(FileExistsError):
        prepare_mri_gallery(physical_mri, output, count=3, size=12, pixel_mm=1)
    assert marker.read_text() == "unrelated content"


@pytest.mark.parametrize(
    "changes",
    [
        {"count": -1}, {"count": 0}, {"count": True}, {"count": 2.5},
        {"size": -1}, {"size": 1}, {"size": np.nan}, {"size": 12.5},
        {"pixel_mm": -1}, {"pixel_mm": 0}, {"pixel_mm": np.nan}, {"pixel_mm": np.inf},
        {"pixel_mm": np.finfo(float).max},
        {"center_xz": (0, np.nan)}, {"center_xz": (0,)}, {"center_xz": (1j, 0)},
        {"depth_range": (1, 0)}, {"depth_range": (0, 0)}, {"depth_range": (0, np.inf)},
        {"tilts": ()}, {"tilts": (16,)}, {"tilts": (-16,)}, {"tilts": (np.nan,)},
        {"tilts": (0, 0)}, {"tilts": (1j,)},
        {"count": 10001}, {"count": 100, "tilts": tuple(np.linspace(-1, 1, 11))},
    ],
)
def test_invalid_sampling_parameters_and_limits_are_rejected(physical_mri, tmp_path, changes):
    options = {"count": 3, "size": 12, "pixel_mm": 1}
    options.update(changes)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare_mri_gallery(physical_mri, output, **options)
    assert not output.exists()
