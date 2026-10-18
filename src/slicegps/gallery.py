"""Build an MRI-only gallery in physical space for later image matching."""

from itertools import product
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .data import sha256, write_json
from .geometry import coronal_plane, sample_plane
from .retrieval import features
from .volume import audit_volume, load_volume


def _finite_pair(value, name: str) -> tuple[float, float]:
    values = np.asarray(value)
    if values.shape != (2,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain two finite real numbers")
    return tuple(float(number) for number in values)


def prepare_mri_gallery(
    mri_path: Path,
    output: Path,
    *,
    count: int = 128,
    size: int = 128,
    pixel_mm: float = 1.5,
    center_xz: tuple[float, float] | None = None,
    depth_range: tuple[float, float] | None = None,
    tilts: tuple[float, ...] = (0,),
    assume_mm: bool = False,
) -> dict:
    """Write only MRI candidate pixels and their actual RAS plane geometry.

    ``count`` is the number of equally spaced depths. For each depth, both the x
    and z tilt angles take every value in ``tilts``. Candidate order is depth,
    then x angle, then z angle, preserving the caller's tilt order. Angles are
    limited to +/-15 degrees and the requested candidate count is at most 10,000.
    Default depths span 20 to 80 percent of the MRI's physical y bounds; x/z are
    centred in its physical bounds. These are search bounds, not a tissue mask.

    Empty/constant valid samples are excluded without shifting the retained
    plane coordinates. No histology images, section labels, or paired examples
    are required. This prepares a standalone index, not a localization result.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 2:
        raise ValueError("size must be an integer >= 2")
    if (
        isinstance(pixel_mm, (bool, np.bool_))
        or not isinstance(pixel_mm, Real)
        or not np.isfinite(pixel_mm)
        or pixel_mm <= 0
    ):
        raise ValueError("pixel_mm must be finite and positive")
    try:
        field_of_view = (int(size) - 1) * float(pixel_mm)
    except OverflowError as exc:
        raise ValueError("Field of view must be finite") from exc
    if not np.isfinite(field_of_view):
        raise ValueError("Field of view must be finite")
    angles = np.asarray(tilts)
    if (
        angles.ndim != 1
        or not len(angles)
        or angles.dtype.kind not in "fiu"
        or not np.isfinite(angles).all()
        or np.any(np.abs(angles) > 15)
    ):
        raise ValueError("tilts must be nonempty finite angles between -15 and 15 degrees")
    if len(np.unique(angles)) != len(angles):
        raise ValueError("tilts must contain distinct angles")
    requested = int(count) * len(angles) ** 2
    if requested > 10_000:
        raise ValueError("Requested gallery exceeds the 10,000 plane limit")
    center_xz = _finite_pair(center_xz, "center_xz") if center_xz is not None else None
    depth_range = _finite_pair(depth_range, "depth_range") if depth_range is not None else None
    if depth_range is not None and depth_range[0] >= depth_range[1]:
        raise ValueError("depth_range must be increasing")
    mri_path, output = Path(mri_path), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    audit = audit_volume(mri_path)
    if audit["spatial_units"] != "mm" and not (
        audit["spatial_units"] == "unknown" and assume_mm
    ):
        raise ValueError("MRI requires mm units or an explicit assume_mm assertion")
    lower, upper = np.asarray(audit["world_bounds_mm"], dtype=float)
    center = (lower + upper) / 2
    x, z = center_xz if center_xz is not None else (float(center[0]), float(center[2]))
    low, high = depth_range if depth_range is not None else (
        float(0.8 * lower[1] + 0.2 * upper[1]),
        float(0.2 * lower[1] + 0.8 * upper[1]),
    )
    if high <= low:
        raise ValueError("MRI has no positive physical depth range")
    data, affine = load_volume(mri_path)
    images, planes, candidate_indices, exclusions = [], [], [], []
    candidates = product(np.linspace(low, high, int(count)), angles, angles)
    for candidate_index, (depth, angle_x, angle_z) in enumerate(candidates):
        plane = coronal_plane(
            (x, float(depth), z), (int(size), int(size)), (float(pixel_mm), float(pixel_mm)),
            (float(angle_x), float(angle_z)),
        )
        image, valid = sample_plane(data, affine, plane)
        values = image[valid]
        reason = None
        if not len(values):
            reason = "no sampled voxel coverage"
        elif not np.any(values):
            reason = "empty MRI plane"
        elif np.ptp(values) == 0:
            reason = "constant MRI plane"
        else:
            try:
                features(image[None], method="intensity")
            except ValueError as exc:
                reason = str(exc)
        if reason is not None:
            exclusions.append({
                "candidate_index": candidate_index,
                "plane": plane.to_dict(),
                "reason": reason,
                "valid_fraction": float(valid.mean()),
            })
            continue
        images.append(image)
        planes.append(plane.to_dict())
        candidate_indices.append(candidate_index)
    if not images:
        raise ValueError("No nonempty, nonconstant MRI planes; check crop and search bounds")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "images.npz", gallery_mri=np.stack(images))
    manifest = {
        "schema_version": 1,
        "mode": "standalone_mri_gallery",
        "reference_space": "source MRI physical RAS; external registration not established",
        "source_mri": str(mri_path.resolve()),
        "source_volume": audit,
        "assume_mm_for_unspecified_units": bool(assume_mm),
        "images_sha256": sha256(output / "images.npz"),
        "requested_depth_count": int(count),
        "requested_candidate_count": requested,
        "gallery_count": len(images),
        "size": int(size),
        "pixel_mm": float(pixel_mm),
        "field_of_view_between_pixel_centers_mm": field_of_view,
        "center_xz_mm": [x, z],
        "depth_range_mm": [low, high],
        "requested_tilts_deg": angles.astype(float).tolist(),
        "candidate_order": "depth, then x tilt, then z tilt; preserve supplied tilt order",
        "gallery_candidate_indices": candidate_indices,
        "gallery_planes": planes,
        "exclusions": exclusions,
        "limits": [
            "Known field of view and near-coronal search only; not unrestricted pose search",
            "Physical header coordinates do not establish cross-subject registration",
            "No histology queries, paired supervision, or localization accuracy measured",
            "Nonconstant intensity screening is not anatomical tissue segmentation",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest
