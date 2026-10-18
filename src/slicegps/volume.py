"""NIfTI loading and coordinate audits. Header agreement is not anatomical proof."""

from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np


def audit_volume(path: Path) -> dict:
    image = nib.load(path)
    if len(image.shape) != 3 or any(n <= 0 for n in image.shape):
        raise ValueError("Expected a nonempty 3D NIfTI volume")
    if image.header.get_xyzt_units()[0] not in ("mm", "unknown"):
        raise ValueError("Unsupported spatial units; convert the volume to mm first")
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("Invalid or singular image affine")
    corners = np.array(list(product(*[(0, n - 1) for n in image.shape])))
    world = nib.affines.apply_affine(affine, corners)
    return {
        "path": str(Path(path).resolve()),
        "shape": list(image.shape),
        "voxel_size_mm": list(map(float, nib.affines.voxel_sizes(affine))),
        "axis_codes": list(nib.aff2axcodes(affine)),
        "affine_voxel_to_ras_mm": affine.tolist(),
        "world_bounds_mm": [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "spatial_units": image.header.get_xyzt_units()[0],
        "units_note": "Affine values require mm units or an explicit dataset-specific mm assertion",
        "anatomical_alignment": "not established by header audit",
    }


def load_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    audit_volume(path)
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite voxel intensities: {path}")
    return data, image.affine


def save_volume(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
