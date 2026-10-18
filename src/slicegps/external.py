"""Image-only checks on publisher-registered real NextBrain section JPEGs.

This is retrieval within known blocks and crops, not unrestricted localization.
Supplied transforms are read only after descriptor ranking, to score predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from .data import download, sha256, write_json
from .matching import _extract, _metadata
from .nextbrain import _matrix, _section_geometry
from .retrieval import evaluate_retrieval, features, rank

CASE4_REPOSITORY = "UCL/NextBrain-Case-4"
CASE4_REVISION = "7cac312e766d603e82928845538f2a2996ec38cb"
CASE4_DONOR = "P85-18"
_SECTION = re.compile(r"slice_(\d+)\.jpg", re.IGNORECASE)


def _positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _blocks(values) -> tuple[int, ...]:
    result = tuple(_positive_int(value, "block ID") for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError("blocks must be nonempty distinct positive integers")
    return result


def _blob_sha(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_external_sections(
    root: Path,
    manifest_path: Path | None = None,
    *,
    repository: str = CASE4_REPOSITORY,
    revision: str = CASE4_REVISION,
    donor: str = CASE4_DONOR,
    blocks: tuple[int, ...] = (1, 7),
    gallery_per_block: int = 30,
    max_file_bytes: int = 25_000_000,
    max_total_bytes: int = 50_000_000,
) -> dict:
    """Discover actual pinned paths, then fetch paired JPEGs and scoring metadata.

    ``root`` is the donor directory itself. Every selected gallery section gets
    both publisher HE and MRI JPEGs so empty histology can be excluded. Existing
    files are reused only if their bytes match the pinned Git blob SHA. The total
    cap includes existing requested files, not just new transfers. The MRI is
    needed for its physical header only, never for synthesizing gallery pixels.
    """
    blocks = _blocks(blocks)
    count = _positive_int(gallery_per_block, "gallery_per_block")
    file_cap = _positive_int(max_file_bytes, "max_file_bytes")
    total_cap = _positive_int(max_total_bytes, "max_total_bytes")
    if file_cap > 25_000_000 or total_cap > 50_000_000:
        raise ValueError("External source caps cannot exceed 25 MB/file and 50 MB total")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository must be owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an immutable full Git SHA")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", donor):
        raise ValueError("donor must be one safe directory name")
    tree_url = f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1"
    request = urllib.request.Request(tree_url, headers={"User-Agent": "SliceGPS/0.1 research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        tree_bytes = response.read(10_000_001)
    if len(tree_bytes) > 10_000_000:
        raise ValueError("GitHub tree response exceeds 10 MB")
    tree = json.loads(tree_bytes)
    if tree.get("truncated") or not isinstance(tree.get("tree"), list):
        raise ValueError("A complete GitHub tree is required for path discovery")
    entries = {entry["path"]: entry for entry in tree["tree"] if entry.get("type") == "blob"}
    selected = {
        f"{donor}/mri_rotated/mri.nii.gz",
        f"{donor}/histologyDimensionsKey.json",
        f"{donor}/block2slab_mapping.csv",
    }
    section_ids = {}
    for block in blocks:
        prefix = f"{donor}/histology/{block:02d}"
        pairs = []
        for path in entries:
            if not path.startswith(prefix + "/slices_MRI/"):
                continue
            match = _SECTION.fullmatch(Path(path).name)
            paired = path.replace("/slices_MRI/", "/slices_HE/")
            if match and paired in entries:
                pairs.append((int(match[1]), path, paired))
        pairs.sort()
        if not pairs:
            raise ValueError(f"No real paired JPEG paths found in pinned tree for block {block}")
        section_ids[str(block)] = [index for index, _, _ in pairs[:count]]
        selected.add(prefix + "/matrix.txt")
        for _, mri, he in pairs[:count]:
            selected.update((mri, he))
    missing = selected - entries.keys()
    if missing:
        raise ValueError(f"Missing scoring files in pinned tree: {sorted(missing)}")
    for path in selected:
        entry = entries[path]
        if not isinstance(entry.get("size"), int) or not 0 < entry["size"] <= file_cap:
            raise ValueError(f"Source exceeds per-file byte cap: {path}")
    total = sum(entries[path]["size"] for path in selected)
    if total > total_cap:
        raise ValueError("Selected external sources exceed the total byte cap")
    root = Path(root)
    manifest_path = (
        Path(manifest_path) if manifest_path is not None else root.parent / "external_sources.json"
    )
    records = []
    result = {
        "schema_version": 1,
        "dataset": "NextBrain publisher-registered real HE/MRI section JPEGs",
        "repository": repository,
        "revision": revision,
        "donor": donor,
        "tree_url": tree_url,
        "tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
        "selected_section_indices": section_ids,
        "max_file_bytes": file_cap,
        "max_total_bytes": total_cap,
        "selected_total_bytes": total,
        "license": "CC0-1.0 (official UCL dataset deposit)",
        "data_doi": "https://doi.org/10.5522/04/24243835",
        "files": records,
    }
    for path in sorted(selected):
        entry = entries[path]
        relative = Path(path).relative_to(donor)
        destination = root / relative
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"
        if not destination.exists():
            download(url, destination, max_bytes=entry["size"])
        if destination.stat().st_size != entry["size"] or _blob_sha(destination) != entry["sha"]:
            raise ValueError(f"Pinned Git blob checksum mismatch: {destination}")
        records.append(
            {
                "path": path,
                "local_path": str(destination.resolve()),
                "url": url,
                "size_bytes": entry["size"],
                "git_blob_sha": entry["sha"],
                "sha256": sha256(destination),
            }
        )
        write_json(Path(manifest_path), result)
    return result


def _read_image(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    if min(gray.shape) < 2:
        raise ValueError("image dimensions below 2 pixels")
    if not np.any(rgb > 5):
        raise ValueError("empty image: all decoded RGB components <= 5")
    # Fix the evaluated image set across all baseline methods.
    for method in ("intensity", "gradient", "mask"):
        features(gray[None], method=method)
    return gray, gray.shape


def _local_sections(directory: Path) -> dict[int, Path]:
    sections = {}
    if directory.is_dir():
        for path in directory.iterdir():
            match = _SECTION.fullmatch(path.name)
            if match:
                index = int(match[1])
                if index in sections:
                    raise ValueError(f"Duplicate local JPEG section index: {directory}/{index}")
                sections[index] = path
    return sections


def external_check(
    root: Path,
    output: Path,
    encoder,
    *,
    blocks: tuple[int, ...] = (1, 7),
    query_per_block: int = 4,
    k: int = 5,
    tolerance_mm: float = 5,
    source_manifest: Path | None = None,
) -> dict:
    """Evaluate real HE queries against real MRI JPEGs from known donor blocks.

    All valid locally available paired sections in requested blocks form one
    pooled gallery. The first ``query_per_block`` nonempty HE sections per block
    form the query set. Section IDs and geometry are retained for scoring only.
    Inputs are already nonlinearly registered by the publisher, and candidate
    crops/block membership are known. No projection is fitted in this function.
    """
    blocks = _blocks(blocks)
    query_per_block = _positive_int(query_per_block, "query_per_block")
    k = _positive_int(k, "k")
    if (
        isinstance(tolerance_mm, (bool, np.bool_))
        or not isinstance(tolerance_mm, (int, float))
        or not np.isfinite(tolerance_mm)
        or tolerance_mm < 0
    ):
        raise ValueError("tolerance_mm must be finite and nonnegative")
    root, output = Path(root).resolve(), Path(output)
    if source_manifest is None and (root.parent / "external_sources.json").is_file():
        source_manifest = root.parent / "external_sources.json"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    declared = json.loads((root / "histologyDimensionsKey.json").read_text())
    if not isinstance(declared, dict):
        raise ValueError("histologyDimensionsKey.json must contain a block object")
    samples, exclusions, warnings = [], [], []
    for block in blocks:
        directory = root / "histology" / f"{block:02d}"
        he_files = _local_sections(directory / "slices_HE")
        mri_files = _local_sections(directory / "slices_MRI")
        for index in sorted(he_files.keys() | mri_files.keys()):
            identity = f"{root.name}/{block:02d}/{index:02d}"
            record = {"section_id": identity, "block_id": block, "section_index": index}
            if index not in he_files or index not in mri_files:
                exclusions.append({**record, "reason": "missing paired HE or MRI JPEG"})
                continue
            try:
                he, he_shape = _read_image(he_files[index])
                mri, mri_shape = _read_image(mri_files[index])
            except (ValueError, OSError) as error:
                exclusions.append({**record, "reason": str(error)})
                continue
            if he_shape != mri_shape:
                exclusions.append(
                    {
                        **record,
                        "reason": "paired HE/MRI decoded dimensions differ",
                        "he_shape": list(he_shape),
                        "mri_shape": list(mri_shape),
                    }
                )
                continue
            expected = declared.get(str(block), {})
            expected_shape = [expected.get("height"), expected.get("width")]
            matches = expected_shape == list(he_shape)
            if not matches:
                warnings.append(f"{identity}: actual JPEG dimensions differ from declared JSON")
            samples.append(
                {
                    **record,
                    "he_path": he_files[index],
                    "mri_path": mri_files[index],
                    "shape": he_shape,
                    "declared_shape": expected_shape,
                    "declared_dimensions_match": matches,
                    "he": he,
                    "mri": mri,
                }
            )
    if len(samples) < 2:
        raise ValueError("At least two nonempty paired gallery sections are required")
    query_positions = []
    for block in blocks:
        query_positions.extend(
            [i for i, sample in enumerate(samples) if sample["block_id"] == block][:query_per_block]
        )
    if not query_positions:
        raise ValueError("No nonempty real HE queries remain")
    queries = [samples[i] for i in query_positions]
    images = [sample["mri"] for sample in samples]
    query_images = [sample["he"] for sample in queries]
    predictions = {}
    for method in ("intensity", "gradient", "mask", "dino"):
        gallery_features = _extract(images, method, encoder, "spatial")
        query_features = _extract(query_images, method, encoder, "spatial")
        predictions[method] = rank(query_features, gallery_features, k=k)
    encoder_metadata = _metadata(encoder, "spatial")

    # The prediction boundary is above: no matrix/header data enter descriptors/rank.
    mri_path = root / "mri_rotated" / "mri.nii.gz"
    mri_header = nib.load(mri_path)
    units = mri_header.header.get_xyzt_units()[0]
    if units not in ("unknown", "mm"):
        raise ValueError("NextBrain MRI must declare mm or unspecified units")
    affine = np.asarray(mri_header.affine, dtype=float)
    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
        or not np.allclose(affine[3], (0, 0, 0, 1))
        or abs(np.linalg.det(affine[:3, :3])) < 1e-12
    ):
        raise ValueError("Invalid MRI physical affine")
    matrices = {
        block: _matrix(root / "histology" / f"{block:02d}" / "matrix.txt") for block in blocks
    }
    geometry = [
        _section_geometry(
            matrices[sample["block_id"]], affine, *sample["shape"], sample["section_index"]
        )
        for sample in samples
    ]
    centers = np.asarray([item["center_ras_mm"] for item in geometry])
    normals = np.asarray([item["normal_ras"] for item in geometry])
    if not np.isfinite(centers).all() or not np.isfinite(normals).all():
        raise ValueError("Section geometry must produce finite centers and normals")
    query_centers, query_normals = centers[query_positions], normals[query_positions]
    methods, rows = {}, []
    for method, (indices, scores) in predictions.items():
        metric = evaluate_retrieval(indices, query_centers, centers, tolerance_mm)
        exact = indices == np.asarray(query_positions)[:, None]
        angles = np.rad2deg(
            np.arccos(
                np.clip(np.abs(np.sum(normals[indices] * query_normals[:, None], axis=2)), 0, 1)
            )
        )
        metric.update(
            {
                "top1_exact_pair_accuracy": float(exact[:, 0].mean()),
                "exact_pair_recall_at_k": {
                    str(j + 1): float(exact[:, : j + 1].any(axis=1).mean())
                    for j in range(indices.shape[1])
                },
                "top1_median_normal_error_deg": float(np.median(angles[:, 0])),
                "top1_p95_normal_error_deg": float(np.percentile(angles[:, 0], 95)),
                "normal_error_convention": (
                    "acos(abs(dot(unit normals))); antipodal planes equivalent"
                ),
            }
        )
        methods["dino_spatial" if method == "dino" else method] = metric
        for i, sample in enumerate(queries):
            chosen = int(indices[i, 0])
            rows.append(
                {
                    "method": "dino_spatial" if method == "dino" else method,
                    "query_section_id": sample["section_id"],
                    "predicted_section_id": samples[chosen]["section_id"],
                    "exact_pair": bool(exact[i, 0]),
                    "center_error_mm": float(np.linalg.norm(centers[chosen] - query_centers[i])),
                    "normal_error_deg": float(angles[i, 0]),
                    "cosine_similarity": float(scores[i, 0]),
                    "ranked_section_ids": [samples[j]["section_id"] for j in indices[i]],
                }
            )
    provenance = None
    if source_manifest is not None:
        source_manifest = Path(source_manifest)
        source = json.loads(source_manifest.read_text())
        known = {record["path"]: record for record in source["files"]}
        used = {mri_path, root / "histologyDimensionsKey.json"}
        for sample in samples:
            used.update((sample["he_path"], sample["mri_path"]))
            used.add(root / "histology" / f"{sample['block_id']:02d}" / "matrix.txt")
        for path in used:
            relative = root.name + "/" + path.relative_to(root).as_posix()
            if relative not in known or sha256(path) != known[relative]["sha256"]:
                raise ValueError(f"External source manifest checksum mismatch: {relative}")
        provenance = {
            "path": str(source_manifest.resolve()),
            "sha256": sha256(source_manifest),
            "source": source,
        }
    section_records = []
    for i, sample in enumerate(samples):
        section_records.append(
            {
                key: sample[key]
                for key in (
                    "section_id",
                    "block_id",
                    "section_index",
                    "declared_shape",
                    "declared_dimensions_match",
                )
            }
            | {
                "actual_shape": list(sample["shape"]),
                "is_query": i in query_positions,
                "he_sha256": sha256(sample["he_path"]),
                "mri_sha256": sha256(sample["mri_path"]),
                "ground_truth": geometry[i],
            }
        )
    result = {
        "schema_version": 1,
        "evaluation": "real_publisher_registered_nextbrain_jpeg_retrieval",
        "donor": root.name,
        "donor_root": str(root),
        "query_count": len(queries),
        "gallery_count": len(samples),
        "requested_blocks": list(blocks),
        "query_selection": f"first {query_per_block} valid paired sections per requested block",
        "search_scope": (
            "pooled real MRI JPEGs from known blocks; publisher-registered crop and orientation"
        ),
        "fit": "none; frozen encoder and fixed image-only baselines",
        "geometry_use": "ground-truth scoring only, after descriptor ranking",
        "coordinate_contract": (
            "MRI affine @ supplied matrix @ (row,column,zero-based section,1); "
            "actual JPEG dimensions"
        ),
        "mri_header_spatial_units": units,
        "geometry_units": "RAS mm; dataset convention asserted when MRI header units unspecified",
        "mri_affine": affine.tolist(),
        "encoder": encoder_metadata,
        "methods": methods,
        "rows": rows,
        "sections": section_records,
        "exclusions": exclusions,
        "warnings": warnings,
        "sources": provenance,
        "limits": [
            "Real HE JPEGs are already registered to MRI by the publisher; "
            "this does not test independently positioned raw sections.",
            "Known-block/crop retrieval is easier than whole-brain arbitrary-pose localization.",
            "This donor is external to BigBrain fitting; one or two external donors "
            "do not establish population generalization.",
            "Nearby sections are strongly correlated. "
            "The gallery contains the paired MRI crop for every query.",
            "Center and normal errors do not measure in-plane deformation or registration quality.",
            "Fewer than eight queries provides only a small diagnostic check.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "ranked_section_ids": json.dumps(row["ranked_section_ids"])})
    return result
