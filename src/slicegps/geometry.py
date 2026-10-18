"""Sample explicitly oriented planes from a volume in RAS millimetres.

Volume axes are voxel indices, not assumed anatomical directions. The affine maps
those indices to RAS millimetres. Image rows advance along ``v`` and columns along
``u``; the plane centre is midway between the central pixels for even sizes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


def _finite_tuple(value: Any, size: int, name: str) -> tuple[float, ...]:
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex coordinates are unsupported")
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return tuple(float(number) for number in array)


@dataclass(frozen=True)
class Plane:
    """A rectangular sampling plane with orthonormal directions in RAS space."""

    center_mm: tuple[float, float, float]
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    shape: tuple[int, int]
    spacing_mm: tuple[float, float]

    def __post_init__(self) -> None:
        center = _finite_tuple(self.center_mm, 3, "center_mm")
        u = _finite_tuple(self.u, 3, "u")
        v = _finite_tuple(self.v, 3, "v")
        spacing = _finite_tuple(self.spacing_mm, 2, "spacing_mm")
        if any(number <= 0 for number in spacing):
            raise ValueError("spacing_mm must be positive")
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise ValueError("shape must contain two positive integers") from exc
        if len(shape) != 2 or any(
            isinstance(number, (bool, np.bool_)) or not isinstance(number, Integral) or number <= 0
            for number in shape
        ):
            raise ValueError("shape must contain two positive integers")
        if not np.isclose(np.linalg.norm(u), 1, rtol=0, atol=1e-6):
            raise ValueError("u must be a unit vector")
        if not np.isclose(np.linalg.norm(v), 1, rtol=0, atol=1e-6):
            raise ValueError("v must be a unit vector")
        if not np.isclose(np.dot(u, v), 0, rtol=0, atol=1e-6):
            raise ValueError("u and v must be orthogonal")
        # Canonical tuples keep a frozen Plane immutable even if callers pass lists.
        object.__setattr__(self, "center_mm", center)
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "shape", tuple(int(number) for number in shape))
        object.__setattr__(self, "spacing_mm", spacing)

    @property
    def normal(self) -> np.ndarray:
        """Return the right-handed normal ``cross(u, v)`` in RAS coordinates."""
        return np.cross(self.u, self.v)

    def world_grid(self) -> np.ndarray:
        """Return pixel-centre RAS coordinates with shape ``(3, height, width)``."""
        height, width = self.shape
        rows = (np.arange(height) - (height - 1) / 2) * self.spacing_mm[0]
        columns = (np.arange(width) - (width - 1) / 2) * self.spacing_mm[1]
        return (
            np.asarray(self.center_mm)[:, None, None]
            + np.asarray(self.v)[:, None, None] * rows[None, :, None]
            + np.asarray(self.u)[:, None, None] * columns[None, None, :]
        )

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        """Return the complete plane geometry as JSON-compatible lists."""
        return {
            "center_mm": list(self.center_mm),
            "u": list(self.u),
            "v": list(self.v),
            "shape": list(self.shape),
            "spacing_mm": list(self.spacing_mm),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Plane":
        """Load and validate geometry produced by :meth:`to_dict`."""
        if not isinstance(value, Mapping):
            raise ValueError("plane geometry must be an object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("plane geometry must contain the five Plane fields") from exc


def sample_plane(
    data3d: np.ndarray, affine4x4: np.ndarray, plane: Plane, order: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a volume at the plane's RAS pixel centres.

    Interpolation is nearest-neighbour (0) or trilinear (1). Pixels outside the
    voxel-centre bounds ``[0, size - 1]`` have value zero and a false validity mask.
    Sampled values must be finite. This function avoids a full-volume finite scan;
    data loaders should validate unsampled voxels when that guarantee is needed.
    """
    data = np.asarray(data3d)
    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("data3d must be a nonempty three-dimensional volume")
    if not (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)):
        raise ValueError("data3d must contain real numeric values")
    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, Integral)
        or order not in (0, 1)
    ):
        raise ValueError("order must be 0 or 1")
    if not isinstance(plane, Plane):
        raise ValueError("plane must be a Plane")
    try:
        if np.iscomplexobj(affine4x4):
            raise ValueError("complex affines are unsupported")
        affine = np.asarray(affine4x4, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("affine4x4 must be a finite homogeneous 4x4 affine") from exc
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], (0, 0, 0, 1), rtol=0, atol=1e-8)
    ):
        raise ValueError("affine4x4 must be a finite homogeneous 4x4 affine")
    try:
        inverse = np.linalg.inv(affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError("affine4x4 must be invertible") from exc
    if not np.isfinite(inverse).all():
        raise ValueError("affine4x4 must have a finite inverse")
    grid = plane.world_grid().reshape(3, -1)
    voxels = inverse[:3, :3] @ grid + inverse[:3, 3, None]
    valid = np.all((voxels >= 0) & (voxels <= (np.asarray(data.shape) - 1)[:, None]), axis=0)
    # scipy.ndimage does not support float16 or extended floating-point arrays.
    if np.issubdtype(data.dtype, np.floating) and data.dtype.itemsize not in (4, 8):
        data = data.astype(np.float32)
    image = map_coordinates(
        data, voxels, output=np.float32, order=int(order), mode="constant", cval=0
    )
    if not np.isfinite(image).all():
        raise ValueError("sampled volume values must be finite and representable as float32")
    image[~valid] = 0
    return image.reshape(plane.shape), valid.reshape(plane.shape)


def coronal_plane(
    center_mm: tuple[float, float, float],
    shape: tuple[int, int] = (128, 128),
    spacing_mm: tuple[float, float] = (1, 1),
    tilt_deg: tuple[float, float] = (0, 0),
) -> Plane:
    """Construct a coronal plane, optionally tilted around fixed RAS x and z axes.

    At zero tilt, columns point right (+x), rows superior (+z), and the normal is
    posterior (-y). ``tilt_deg=(x_angle, z_angle)`` applies right-handed rotation
    around x first, then around z: ``Rz @ Rx``. It is not a voxel-axis rotation.
    """
    x_angle, z_angle = np.deg2rad(_finite_tuple(tilt_deg, 2, "tilt_deg"))
    cx, sx = np.cos(x_angle), np.sin(x_angle)
    cz, sz = np.cos(z_angle), np.sin(z_angle)
    rotation_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rotation_z = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    rotation = rotation_z @ rotation_x
    return Plane(center_mm, tuple(rotation[:, 0]), tuple(rotation[:, 2]), shape, spacing_mm)
