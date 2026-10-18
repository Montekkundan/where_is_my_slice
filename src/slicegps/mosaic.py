"""Known-placement preview of NextBrain blocks, not an unknown-piece solver."""

import csv
from pathlib import Path

import numpy as np

from .data import write_json
from .geometry import coronal_plane, sample_plane
from .nextbrain import reslice_blocks
from .pipeline import contact_sheet
from .volume import audit_volume, load_volume


def preview_mosaic(
    case: Path,
    block_ids: list[int],
    output: Path,
    pixel_mm: float = 0.3,
    size: int = 512,
    threshold: float = 0.02,
) -> dict:
    if len(set(block_ids)) != len(block_ids) or not block_ids:
        raise ValueError("Supply unique block IDs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    with (case / "block2slab_mapping.csv").open() as stream:
        names = {int(row["BLOCK_NUM"]): row["BLOCK_NAME"] for row in csv.DictReader(stream)}
    if any(b not in names for b in block_ids):
        raise ValueError("Unknown block ID")
    slabs = {names[b].split(".")[0] for b in block_ids}
    if len(slabs) != 1:
        raise ValueError("Mosaic pieces must belong to one gross slab")
    paths = [case / "histology_blocks" / str(b) / "HE.nii.gz" for b in block_ids]
    audits = [audit_volume(path) for path in paths]
    mri_path = case / "mri_rotated" / "mri.nii.gz"
    mri_audit = audit_volume(mri_path)
    for audit in [*audits, mri_audit]:
        if audit["spatial_units"] not in ("mm", "unknown"):
            raise ValueError("Unsupported volume units")
    lower = np.min([a["world_bounds_mm"][0] for a in audits], axis=0)
    upper = np.max([a["world_bounds_mm"][1] for a in audits], axis=0)
    ylow = max(a["world_bounds_mm"][0][1] for a in audits)
    yhigh = min(a["world_bounds_mm"][1][1] for a in audits)
    if yhigh <= ylow:
        raise ValueError("Blocks have no shared coronal coordinate bounds")
    center = (lower + upper) / 2
    center[1] = (ylow + yhigh) / 2
    plane = coronal_plane(tuple(center), shape=(size, size), spacing_mm=(pixel_mm, pixel_mm))
    blocks = [load_volume(p) for p in paths]
    image, coverage = reslice_blocks(blocks, plane, threshold=threshold)
    mri, mri_affine = load_volume(mri_path)
    mri_plane, mri_valid = sample_plane(mri, mri_affine, plane)
    pieces = [reslice_blocks([block], plane, threshold=threshold)[0] for block in blocks]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "mosaic.npz",
        histology=image,
        mri=mri_plane,
        coverage_counts=coverage,
        mri_valid=mri_valid,
    )
    contact_sheet(
        [*pieces, image, mri_plane, coverage.astype(float)],
        [
            *[f"HE {names[b]}" for b in block_ids],
            "HE assembled",
            "Same-plane MRI",
            "Tissue contribution count",
        ],
        output / "mosaic.png",
    )
    report = {
        "status": "oracle geometry preparation using supplied block affines",
        "not_a_prediction": True,
        "donor": case.name,
        "slab": next(iter(slabs)),
        "block_ids": block_ids,
        "plane": plane.to_dict(),
        "blocks": audits,
        "mri": mri_audit,
        "foreground_threshold": threshold,
        "contributing_pixels_per_block": [int(np.count_nonzero(p > threshold)) for p in pieces],
        "covered_pixels": int(np.count_nonzero(coverage)),
        "overlap_pixels": int(np.count_nonzero(coverage > 1)),
        "uncovered_mri_foreground_pixels_in_field_of_view": int(
            np.count_nonzero((mri_plane > 5) & mri_valid & (coverage == 0))
        ),
        "coverage_note": "Uncovered MRI tissue may be outside supplied pieces; not a damage score.",
        "units_assertion": "NextBrain published physical affine interpreted in mm; headers may "
        "leave units unspecified. See DATASET_LAYOUT.md and Case4 README.",
        "transform_grid": "Each block NIfTI uses its own affine. Web JPEG matrix unused.",
        "missing_slab_blocks": sorted(
            b for b, name in names.items() if name.split(".")[0] in slabs and b not in block_ids
        ),
    }
    write_json(output / "manifest.json", report)
    return {
        "output": str(output.resolve()),
        "slab": report["slab"],
        "covered_pixels": report["covered_pixels"],
        "overlap_pixels": report["overlap_pixels"],
    }
