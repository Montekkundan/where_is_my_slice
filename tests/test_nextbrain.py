import json

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from slicegps.geometry import coronal_plane
from slicegps.nextbrain import audit_case, reslice_blocks


def test_own_affines_place_shifted_rotated_pieces_and_expose_overlap():
    first = np.full((3, 1, 3), 10, dtype=np.float32)
    indices = np.indices((3, 1, 3))
    second = (20 + 2 * indices[0] + indices[2]).astype(np.float32)
    # A 90 degree y rotation places this piece at x=2..4, z=0..2.
    affine = np.array([[0, 0, 1, 2], [0, 1, 0, 0], [-1, 0, 0, 2], [0, 0, 0, 1]])
    plane = coronal_plane((2, 0, 1), (3, 5))
    image, counts = reslice_blocks([(first, np.eye(4)), (second, affine)], plane)
    expected = np.array(
        [[10, 10, 17, 25, 26], [10, 10, 16, 23, 24], [10, 10, 15, 21, 22]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(image, expected)
    np.testing.assert_array_equal(counts, np.tile([1, 1, 2, 1, 1], (3, 1)))
    assert image.dtype == np.float32
    assert counts.dtype == np.uint32


def test_missing_and_black_background_never_contribute_to_averages():
    first = np.full((3, 1, 3), 10, dtype=np.float32)
    first[0, 0, 0] = 0
    first[2, 0, 1] = 5
    second = np.full((3, 1, 3), 20, dtype=np.float32)
    second[1, 0, 1] = 0
    affine = np.array([[0, 0, 1, 2], [0, 1, 0, 0], [-1, 0, 0, 2], [0, 0, 0, 1]])
    plane = coronal_plane((2, 0, 1), (3, 5))
    image, counts = reslice_blocks([(first, np.eye(4)), (second, affine)], plane)
    assert image[0, 0] == 0 and counts[0, 0] == 0
    assert image[1, 2] == 20 and counts[1, 2] == 1
    assert image[1, 3] == 0 and counts[1, 3] == 0
    missing_image, missing_counts = reslice_blocks([(first, np.eye(4))], plane)
    assert not missing_image[:, 3:].any()
    assert not missing_counts[:, 3:].any()
    empty_image, empty_counts = reslice_blocks([], plane)
    assert not empty_image.any() and not empty_counts.any()


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, "5", 1j, np.complex64(1j), [5], True])
def test_invalid_tissue_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        reslice_blocks([], coronal_plane((0, 0, 0)), threshold)


@pytest.fixture
def mini_case(tmp_path):
    root = tmp_path / "donor"
    root.mkdir()
    (root / "block2slab_mapping.csv").write_text("BLOCK_NUM,BLOCK_NAME\n2,A1.2\n7,A1.1\n9,P3.1\n")
    dimensions = {
        "2": {"width": 12, "height": 8, "slices": 4, "orientation": "coronal"},
        "7": {"width": 3, "height": 3, "slices": 1, "orientation": "axial"},
        "9": {"width": 3, "height": 3, "slices": 1, "orientation": "sagittal"},
    }
    (root / "histologyDimensionsKey.json").write_text(json.dumps(dimensions))
    x, y, z = np.indices((20, 10, 20))
    data = (x + 2 * y + 3 * z + 20).astype(np.uint8)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = (10, -5, 7)
    mri = root / "mri_rotated" / "mri.nii.gz"
    mri.parent.mkdir()
    nib.save(nib.Nifti1Image(data, affine), mri)
    block = root / "histology" / "02"
    for kind in ("slices_HE", "slices_MRI"):
        (block / kind).mkdir(parents=True)
    # Matrix input is (row, column, zero-based section), not (x, y, section).
    matrix = np.array([[0, 1, 0, 1], [0, 0, 1, 2], [1, 0, 0, 1], [0, 0, 0, 1]])
    np.savetxt(block / "matrix.txt", matrix)
    rows, columns = np.indices((9, 13))
    for index in (0, 2):
        histology = np.full((9, 13, 3), 80 if index == 0 else 0, dtype=np.uint8)
        Image.fromarray(histology).save(block / "slices_HE" / f"slice_{index:02d}.jpg")
        reference = data[columns + 1, index + 2, rows + 1]
        Image.fromarray(reference).save(
            block / "slices_MRI" / f"slice_{index:02d}.jpg", quality=100, subsampling=0
        )
    return root


def test_audit_preserves_hierarchy_actual_dimensions_and_zero_based_geometry(mini_case):
    report = audit_case(mini_case)
    json.dumps(report, allow_nan=False)
    assert report["slabs"] == {"A1": [2, 7], "P3": [9]}
    assert report["summary"] == {
        "mapped_blocks": 3,
        "slabs": 2,
        "available_blocks": 1,
        "available_sections": 2,
        "empty_histology_sections": 1,
        "dimension_mismatch_sections": 2,
        "missing_declared_sections": 4,
    }
    assert report["blocks"][0]["available_section_indices"] == [0, 2]
    assert report["blocks"][0]["missing_declared_section_indices"] == [1, 3]
    first, second = report["sections"]
    assert first["section_index"] == 0
    assert first["actual_dimensions"] == {"width": 13, "height": 9}
    assert first["declared_dimensions"] == {"width": 12, "height": 8}
    assert first["dimensions_match"] is False
    assert not first["empty_histology"]
    assert second["empty_histology"]
    np.testing.assert_allclose(second["center_mri_voxel"], [7, 4, 5])
    np.testing.assert_allclose(second["center_ras_mm"], [24, 7, 27])
    np.testing.assert_allclose(
        second["corners_ras_mm"], [[12, 7, 11], [36, 7, 11], [36, 7, 43], [12, 7, 43]]
    )
    assert first["geometry_sanity"]["pearson_r"] > 0.99
    assert second["geometry_sanity"]["pearson_r"] > 0.99
    assert "not localization accuracy" in first["geometry_sanity"]["purpose"]
    assert report["mri"]["header_spatial_units"] == "unknown"
    assert report["mri"]["adapter_spatial_units"] == "mm"


def test_missing_mri_jpeg_does_not_prevent_section_geometry(mini_case):
    (mini_case / "histology/02/slices_MRI/slice_00.jpg").unlink()
    section = audit_case(mini_case)["sections"][0]
    assert section["geometry_sanity"] is None
    np.testing.assert_allclose(section["center_mri_voxel"], [7, 2, 5])


def test_corrupt_matrix_is_rejected(mini_case):
    np.savetxt(mini_case / "histology/02/matrix.txt", np.zeros((4, 4)))
    with pytest.raises(ValueError, match="matrix"):
        audit_case(mini_case)


def test_unknown_block_mapping_is_rejected(mini_case):
    (mini_case / "histology/99").mkdir()
    with pytest.raises(ValueError, match="Unmapped"):
        audit_case(mini_case)


def test_duplicate_mapping_is_rejected(mini_case):
    with (mini_case / "block2slab_mapping.csv").open("a") as stream:
        stream.write("2,A2.1\n")
    with pytest.raises(ValueError, match="unique"):
        audit_case(mini_case)
