import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from slicegps.geometry import Plane, coronal_plane, sample_plane


def test_world_grid_has_explicit_row_column_spacing_and_symmetric_centres():
    plane = coronal_plane((10, 20, 30), shape=(3, 5), spacing_mm=(2, 3))
    grid = plane.world_grid()
    assert grid.shape == (3, 3, 5)
    np.testing.assert_array_equal(grid[0], np.tile([4, 7, 10, 13, 16], (3, 1)))
    np.testing.assert_array_equal(grid[1], 20)
    np.testing.assert_array_equal(grid[2], np.tile([[28], [30], [32]], (1, 5)))
    np.testing.assert_array_equal(plane.normal, [0, -1, 0])
    even = coronal_plane((0, 0, 0), shape=(2, 2)).world_grid()
    np.testing.assert_allclose(even.mean(axis=(1, 2)), [0, 0, 0])
    np.testing.assert_array_equal(even[0], [[-0.5, 0.5], [-0.5, 0.5]])


def test_json_round_trip_and_immutable_canonical_fields():
    center = [1, 2, 3]
    plane = Plane(center, [1, 0, 0], [0, 0, 1], [3, 4], [0.5, 0.75])
    center[0] = 99
    assert plane.center_mm == (1, 2, 3)
    assert Plane.from_dict(json.loads(json.dumps(plane.to_dict()))) == plane
    with pytest.raises(FrozenInstanceError):
        plane.center_mm = (0, 0, 0)


def test_identity_affine_coordinate_phantom():
    x, y, z = np.indices((9, 10, 11))
    data = x + 10 * y + 100 * z
    plane = coronal_plane((4, 5, 5), shape=(3, 5))
    image, valid = sample_plane(data, np.eye(4), plane)
    expected = np.arange(2, 7)[None, :] + 50 + 100 * np.arange(4, 7)[:, None]
    np.testing.assert_array_equal(image, expected)
    assert image.dtype == np.float32
    assert valid.dtype == np.bool_
    assert valid.all()


def test_translated_anisotropic_flipped_affine_preserves_world_coordinates():
    affine = np.array([[-2, 0, 0, 30], [0, 3, 0, -5], [0, 0, 4, 7], [0, 0, 0, 1]])
    indices = np.indices((9, 10, 11)).reshape(3, -1)
    world = affine[:3, :3] @ indices + affine[:3, 3, None]
    data = (world[0] + 10 * world[1] + 100 * world[2]).reshape(9, 10, 11)
    plane = coronal_plane((22, 10, 27), shape=(3, 5), spacing_mm=(4, 2))
    image, valid = sample_plane(data, affine, plane)
    grid = plane.world_grid()
    np.testing.assert_allclose(image, grid[0] + 10 * grid[1] + 100 * grid[2])
    assert valid.all()


def test_oblique_trilinear_sampling_matches_analytical_linear_field():
    x, y, z = np.indices((32, 32, 32), dtype=np.float64)
    data = 2 * x - 3 * y + 0.7 * z + 5
    plane = coronal_plane((15, 14, 13), (11, 9), (0.7, 0.8), (13, -17))
    grid = plane.world_grid()
    expected = 2 * grid[0] - 3 * grid[1] + 0.7 * grid[2] + 5
    image, valid = sample_plane(data, np.eye(4), plane)
    np.testing.assert_allclose(image, expected, atol=1e-6)
    assert valid.all()


def test_tilt_uses_fixed_x_then_z_ras_rotations():
    plane = coronal_plane((0, 0, 0), tilt_deg=(90, 90))
    np.testing.assert_allclose(plane.u, [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(plane.v, [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(plane.normal, [0, 0, -1], atol=1e-15)


@pytest.mark.parametrize("order", [0, 1])
def test_outside_voxel_centres_are_zero_and_boundary_centres_are_valid(order):
    plane = coronal_plane((0, 1, 0), (3, 3))
    image, valid = sample_plane(np.ones((3, 3, 3)), np.eye(4), plane, order)
    expected = np.array([[False, False, False], [False, True, True], [False, True, True]])
    np.testing.assert_array_equal(valid, expected)
    np.testing.assert_array_equal(image, expected.astype(np.float32))
    # Nearest-neighbour still respects centre bounds before rounding coordinates.
    outside = coronal_plane((-0.1, 1, 1), (1, 1))
    value, mask = sample_plane(np.ones((3, 3, 3)), np.eye(4), outside, order)
    assert value[0, 0] == 0
    assert not mask[0, 0]


def test_nearest_neighbour_and_linear_interpolation_differ():
    data = np.indices((4, 4, 4))[0]
    plane = coronal_plane((1.25, 1, 1), (1, 1))
    nearest, _ = sample_plane(data, np.eye(4), plane, order=0)
    linear, _ = sample_plane(data, np.eye(4), plane, order=1)
    assert nearest[0, 0] == 1
    assert linear[0, 0] == 1.25


@pytest.mark.parametrize(
    "field,value",
    [
        ("center_mm", (np.nan, 0, 0)),
        ("center_mm", (0, 0)),
        ("center_mm", np.array([1j, 0, 0])),
        ("u", (2, 0, 0)),
        ("u", (np.inf, 0, 0)),
        ("v", (1, 0, 0)),
        ("v", (0, 0, 0)),
        ("shape", (0, 2)),
        ("shape", (-1, 2)),
        ("shape", (2.0, 2)),
        ("shape", (True, 2)),
        ("shape", (2,)),
        ("spacing_mm", (0, 1)),
        ("spacing_mm", (-1, 1)),
        ("spacing_mm", (1, np.inf)),
    ],
)
def test_invalid_plane_fields_are_rejected(field, value):
    values = coronal_plane((0, 0, 0)).to_dict()
    values[field] = value
    with pytest.raises(ValueError):
        Plane.from_dict(values)


@pytest.mark.parametrize("value", [None, {}, {"center_mm": [0, 0, 0]}, {"extra": 1}])
def test_invalid_serialized_plane_is_rejected(value):
    with pytest.raises(ValueError):
        Plane.from_dict(value)


@pytest.mark.parametrize("tilt", [(np.nan, 0), (0, np.inf), (0,), (0, 0, 0)])
def test_invalid_tilt_is_rejected(tilt):
    with pytest.raises(ValueError):
        coronal_plane((0, 0, 0), tilt_deg=tilt)


@pytest.mark.parametrize(
    "data",
    [np.ones((3, 3)), np.ones((0, 3, 3)), np.ones((3, 3, 3), complex), np.full((3, 3, 3), "a")],
)
def test_invalid_volume_is_rejected(data):
    with pytest.raises(ValueError):
        sample_plane(data, np.eye(4), coronal_plane((1, 1, 1), (1, 1)))


@pytest.mark.parametrize("order", [-1, 2, 1.0, True])
def test_unsupported_interpolation_is_rejected(order):
    with pytest.raises(ValueError):
        sample_plane(np.ones((3, 3, 3)), np.eye(4), coronal_plane((1, 1, 1)), order)


@pytest.mark.parametrize(
    "affine",
    [
        np.eye(3),
        np.eye(4, dtype=complex),
        np.zeros((4, 4)),
        np.diag([0, 1, 1, 1]),
        np.diag([1, 1, 1, 2]),
        np.diag([np.nan, 1, 1, 1]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.1, 0, 0, 1]]),
    ],
)
def test_invalid_affine_is_rejected(affine):
    with pytest.raises(ValueError):
        sample_plane(np.ones((3, 3, 3)), affine, coronal_plane((1, 1, 1)))


def test_nonfinite_sampled_values_are_rejected_without_scanning_entire_volume():
    data = np.ones((5, 5, 5))
    data[4, 4, 4] = np.nan
    image, valid = sample_plane(data, np.eye(4), coronal_plane((1, 1, 1), (1, 1)))
    assert image[0, 0] == 1 and valid[0, 0]
    with pytest.raises(ValueError, match="sampled volume values must be finite"):
        sample_plane(data, np.eye(4), coronal_plane((4, 4, 4), (1, 1)))
