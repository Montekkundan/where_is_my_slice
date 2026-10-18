"""NextBrain prepared-data geometry audits and oracle block placement.

Web JPEGs use the supplied row/column/section matrix. Cropped block NIfTIs have
their own grids and must instead use their own NIfTI affine. These transforms are
evaluation ground truth, never predictor inputs for unknown-location experiments.
"""

import csv
import json
import re
from numbers import Real
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from slicegps.geometry import Plane, sample_plane
from slicegps.volume import audit_volume, load_volume


def _matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0, 0, 0, 1), rtol=0, atol=1e-8)
    ):
        raise ValueError(f"Invalid homogeneous row-column-section matrix: {path}")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Singular histology matrix: {path}") from exc
    if not np.isfinite(inverse).all():
        raise ValueError(f"Histology matrix must have a finite inverse: {path}")
    return matrix


def _mapping(path: Path) -> dict[int, str]:
    blocks = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["BLOCK_NUM", "BLOCK_NAME"]:
            raise ValueError("Expected BLOCK_NUM,BLOCK_NAME mapping columns")
        for row in reader:
            block = int(row["BLOCK_NUM"])
            name = row["BLOCK_NAME"].strip()
            if block <= 0 or block in blocks or not name:
                raise ValueError("Block mapping requires unique positive IDs and nonempty names")
            blocks[block] = name
    if not blocks:
        raise ValueError("Block mapping is empty")
    return blocks


def _section_geometry(
    matrix: np.ndarray, mri_affine: np.ndarray, height: int, width: int, index: int
) -> dict:
    points = np.array(
        [
            [(height - 1) / 2, (width - 1) / 2, index, 1],
            [0, 0, index, 1],
            [0, width - 1, index, 1],
            [height - 1, width - 1, index, 1],
            [height - 1, 0, index, 1],
        ]
    ).T
    voxel = matrix @ points
    composite = mri_affine @ matrix
    world = composite @ points
    row_step, column_step = composite[:3, 0], composite[:3, 1]
    normal = np.cross(column_step, row_step)
    normal /= np.linalg.norm(normal)
    return {
        "center_mri_voxel": voxel[:3, 0].tolist(),
        "center_ras_mm": world[:3, 0].tolist(),
        "corners_ras_mm": world[:3, 1:].T.tolist(),
        "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "row_step_ras_mm": row_step.tolist(),
        "column_step_ras_mm": column_step.tolist(),
        "section_step_ras_mm": composite[:3, 2].tolist(),
        "normal_ras": normal.tolist(),
    }


def _geometry_sanity(path: Path, matrix: np.ndarray, data: np.ndarray, index: int) -> dict:
    """Compare a publisher MRI JPEG with MRI sampling using supplied geometry."""
    with Image.open(path) as image:
        reference = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = reference.shape
    rows, columns = np.meshgrid(np.arange(0, height, 4), np.arange(0, width, 4), indexing="ij")
    pixels = reference[::4, ::4].ravel()
    histology = np.stack(
        [rows.ravel(), columns.ravel(), np.full(rows.size, index), np.ones(rows.size)]
    )
    voxels = (matrix @ histology)[:3]
    valid = np.all((voxels >= 0) & (voxels <= (np.asarray(data.shape) - 1)[:, None]), axis=0)
    foreground = valid & (pixels > 10)
    sampled = map_coordinates(data, voxels, output=np.float32, order=0, mode="constant", cval=0)
    first, second = pixels[foreground], sampled[foreground]
    correlation = None
    reason = None
    if first.size < 3:
        reason = "fewer than three valid foreground samples"
    elif np.std(first) == 0 or np.std(second) == 0:
        reason = "constant foreground intensities"
    else:
        correlation = float(np.corrcoef(first, second)[0, 1])
    return {
        "path": str(path.resolve()),
        "actual_dimensions": {"width": width, "height": height},
        "pearson_r": correlation,
        "unavailable_reason": reason,
        "sample_count": int(first.size),
        "stride_pixels": 4,
        "foreground_threshold": 10,
        "interpolation": "nearest",
        "purpose": "geometry sanity using supplied transforms; not localization accuracy",
    }


def audit_case(root: Path) -> dict:
    """Audit a local NextBrain donor directory, including partial downloads.

    Requires the mapping CSV, dimensions JSON, MRI, and a matrix for every available
    HE JPEG stack. Report actual decoded dimensions, missing local sections,
    black/empty histology, and world geometry. Section indices are zero based.
    Supplied MRI affines are interpreted in millimetres under the NextBrain dataset
    convention, with an explicit assertion when the NIfTI units are unspecified.
    """
    root = Path(root).resolve()
    mapping = _mapping(root / "block2slab_mapping.csv")
    dimensions = json.loads((root / "histologyDimensionsKey.json").read_text())
    if not isinstance(dimensions, dict):
        raise ValueError("Histology dimensions must be an object keyed by block ID")
    mri_path = root / "mri_rotated" / "mri.nii.gz"
    mri_report = audit_volume(mri_path)
    units = nib.load(mri_path).header.get_xyzt_units()[0]
    if units not in ("unknown", "mm"):
        raise ValueError(f"NextBrain MRI must declare mm or unspecified units, found {units}")
    mri_report["header_spatial_units"] = units
    mri_report["adapter_spatial_units"] = "mm"
    mri_report["units_provenance"] = "NextBrain physical RAS convention; asserted if unspecified"
    data, affine = load_volume(mri_path)
    directories = {}
    histology_root = root / "histology"
    if histology_root.exists():
        for path in histology_root.iterdir():
            if path.is_dir() and path.name.isdecimal():
                block = int(path.name)
                if block not in mapping or block in directories:
                    raise ValueError(f"Unmapped or duplicate histology block directory: {path}")
                directories[block] = path
    slabs: dict[str, list[int]] = {}
    sections, blocks, warnings = [], [], []
    for block, name in sorted(mapping.items()):
        slab = name.split(".", 1)[0]
        slabs.setdefault(slab, []).append(block)
        declared = dimensions.get(str(block))
        if declared is not None and (
            not isinstance(declared, dict)
            or any(
                not isinstance(declared.get(key), int)
                or isinstance(declared.get(key), bool)
                or declared[key] <= 0
                for key in ("width", "height", "slices")
            )
        ):
            raise ValueError(f"Invalid declared dimensions for block {block}")
        if declared is None:
            warnings.append(f"Block {block} has no declared dimensions")
        directory = directories.get(block)
        files = {}
        if directory is not None and (directory / "slices_HE").exists():
            for path in (directory / "slices_HE").iterdir():
                match = re.fullmatch(r"slice_(\d+)\.(?:jpg|jpeg)", path.name, re.IGNORECASE)
                if match:
                    index = int(match[1])
                    if index in files:
                        raise ValueError(f"Duplicate JPEG section index in block {block}: {index}")
                    files[index] = path
        declared_indices = set(range(declared["slices"])) if declared else set()
        blocks.append(
            {
                "block_id": block,
                "block_name": name,
                "slab": slab,
                "declared": declared,
                "available_section_indices": sorted(files),
                "missing_declared_section_indices": sorted(declared_indices - files.keys()),
                "out_of_declared_range_indices": sorted(files.keys() - declared_indices)
                if declared
                else [],
            }
        )
        if not files:
            continue
        matrix_path = directory / "matrix.txt"
        matrix = _matrix(matrix_path)
        for index, path in sorted(files.items()):
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            height, width = rgb.shape[:2]
            actual = {"width": width, "height": height}
            declared_size = {key: declared[key] for key in actual} if declared else None
            dimension_match = actual == declared_size if declared else None
            foreground_fraction = float(np.any(rgb > 5, axis=2).mean())
            if dimension_match is False:
                warnings.append(
                    f"Block {block} section {index}: decoded dimensions differ from JSON"
                )
            matching_mri = directory / "slices_MRI" / path.name
            sanity = (
                _geometry_sanity(matching_mri, matrix, data, index)
                if matching_mri.exists()
                else None
            )
            if sanity and sanity["actual_dimensions"] != actual:
                warnings.append(f"Block {block} section {index}: HE and MRI JPEG dimensions differ")
            sections.append(
                {
                    "path": str(path.resolve()),
                    "block_id": block,
                    "block_name": name,
                    "slab": slab,
                    "section_index": index,
                    "actual_dimensions": actual,
                    "declared_dimensions": declared_size,
                    "dimensions_match": dimension_match,
                    "histology_foreground_fraction": foreground_fraction,
                    "empty_histology": foreground_fraction == 0,
                    "empty_detection": "all decoded RGB components <= 5",
                    "matrix_path": str(matrix_path.resolve()),
                    "matrix_row_col_section_to_mri_voxel": matrix.tolist(),
                    **_section_geometry(matrix, affine, height, width, index),
                    "geometry_sanity": sanity,
                }
            )
    return {
        "donor_root": str(root),
        "purpose": "oracle prepared-data geometry audit, not predicted placement",
        "coordinate_contract": {
            "matrix_input": ["row", "column", "zero_based_section_index", "1"],
            "world_transform": "MRI affine @ histology matrix @ pixel index",
            "units": "RAS mm",
            "block_nifti_rule": "use each NIfTI own affine; never the web JPEG matrix",
            "assembly_rule": (
                "slab membership or equal local indices do not establish a common plane"
            ),
        },
        "mri": mri_report,
        "slabs": slabs,
        "blocks": blocks,
        "sections": sections,
        "warnings": warnings,
        "summary": {
            "mapped_blocks": len(mapping),
            "slabs": len(slabs),
            "available_blocks": sum(bool(block["available_section_indices"]) for block in blocks),
            "available_sections": len(sections),
            "empty_histology_sections": sum(section["empty_histology"] for section in sections),
            "dimension_mismatch_sections": sum(
                section["dimensions_match"] is False for section in sections
            ),
            "missing_declared_sections": sum(
                len(block["missing_declared_section_indices"]) for block in blocks
            ),
        },
    }


def reslice_blocks(
    blocks: list[tuple[np.ndarray, np.ndarray]], plane: Plane, threshold: float = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Oracle-place scalar block volumes using each volume's own RAS affine.

    Only valid sampled intensities strictly above ``threshold`` contribute. The
    result averages contributing tissue values; uint32 coverage counts expose
    overlaps and zero coverage exposes missing/background regions. This does not
    estimate placement, register raw fragments, or infer a common section index.
    """
    if not isinstance(plane, Plane):
        raise ValueError("plane must be a Plane")
    if not isinstance(threshold, Real) or isinstance(threshold, (bool, np.bool_)):
        raise ValueError("threshold must be a finite nonnegative number")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative number")
    total = np.zeros(plane.shape, dtype=np.float64)
    counts = np.zeros(plane.shape, dtype=np.uint32)
    for data, affine in blocks:
        image, valid = sample_plane(data, affine, plane)
        tissue = valid & (image > threshold)
        total[tissue] += image[tissue]
        counts[tissue] += 1
    average = np.zeros(plane.shape, dtype=np.float32)
    np.divide(total, counts, out=average, where=counts > 0)
    return average, counts
