"""Spatially separated, within-volume training and evaluation of image matching."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import kendalltau

from .adapters import fit_ridge
from .data import write_json
from .geometry import Plane, coronal_plane, sample_plane
from .pipeline import contact_sheet
from .retrieval import evaluate_retrieval, features, inversion_fraction, order_stack, rank
from .volume import audit_volume, load_volume


def _new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def depth_support(plane, padding_mm: float = 0) -> tuple[float, float]:
    """Bounding depth interval of every pixel, including the tilted image corners."""
    if not np.isfinite(padding_mm) or padding_mm < 0:
        raise ValueError("Support padding must be finite and nonnegative")
    grid = plane.world_grid()
    return float(grid[1].min() - padding_mm), float(grid[1].max() + padding_mm)


def check_separation(groups: dict[str, list[tuple[float, float]]]) -> float:
    """Reject any intersecting training/validation/test plane depth intervals."""
    for name, intervals in groups.items():
        values = np.asarray(intervals, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or not len(values)
            or not np.isfinite(values).all()
            or np.any(values[:, 1] < values[:, 0])
        ):
            raise ValueError(f"Invalid finite ordered depth intervals for {name}")
    minimum = float("inf")
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for a, b in groups[left]:
                for c, d in groups[right]:
                    gap = max(c - b, a - d)
                    if gap <= 0:
                        raise ValueError(f"Depth supports overlap between {left} and {right}")
                    minimum = min(minimum, gap)
    if not np.isfinite(minimum):
        raise ValueError("At least two nonempty spatial groups are required")
    return minimum


def prepare_protocol(
    histology_path: Path,
    mri_path: Path,
    output: Path,
    *,
    train_count: int = 192,
    validation_count: int = 24,
    test_count: int = 48,
    gallery_count: int = 128,
    size: int = 128,
    pixel_mm: float = 1.5,
    tilt_deg: float = 1.0,
    seed: int = 26018,
    assume_mm: bool = False,
) -> dict:
    """Eight depth bands, train=0/2/4/6, validation=1, final test=3/5/7.

    Plane supports remain inside their bands with a 1mm extra margin. The gallery
    covers the complete search range and is never used to fit adapter statistics.
    All splits are one registered histology volume, not independent subjects.
    """
    counts = {"train": train_count, "validation": validation_count, "test": test_count}
    if any(not isinstance(n, int) or n < 4 for n in (*counts.values(), gallery_count)):
        raise ValueError("Every image count must be an integer >=4")
    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 16
        or not np.isfinite(pixel_mm)
        or pixel_mm <= 0
    ):
        raise ValueError("Require size >=16 and finite positive pixel size")
    if not np.isfinite(tilt_deg) or not 0 <= tilt_deg <= 5:
        raise ValueError("Protocol tilt must be between 0 and 5 degrees")
    audits = [audit_volume(p) for p in (histology_path, mri_path)]
    for a in audits:
        if a["spatial_units"] != "mm" and not (a["spatial_units"] == "unknown" and assume_mm):
            raise ValueError("Require mm units or documented assume_mm assertion")
    lower = np.maximum(*(np.array(a["world_bounds_mm"][0]) for a in audits))
    upper = np.minimum(*(np.array(a["world_bounds_mm"][1]) for a in audits))
    if np.any(upper <= lower):
        raise ValueError("Volumes have no common coordinate bounds")
    center = (lower + upper) / 2
    low, high = lower[1] * 0.8 + upper[1] * 0.2, lower[1] * 0.2 + upper[1] * 0.8
    edges = np.linspace(low, high, 9)
    # |u_y| and |v_y| <= sin(max tilt), conservatively bounding both rotations.
    interpolation_padding = max(
        float(np.abs(np.array(a["affine_voxel_to_ras_mm"])[1, :3]).sum()) for a in audits
    )
    margin = (size - 1) * pixel_mm * np.sin(np.deg2rad(tilt_deg)) + interpolation_padding + 1.0
    if np.min(np.diff(edges)) <= 2 * margin:
        raise ValueError("Depth bands are too narrow for the requested oblique footprint")
    _new_directory(output)
    hist, ah = load_volume(histology_path)
    mri, am = load_volume(mri_path)
    rng = np.random.default_rng(seed)
    split_bands = {"train": [0, 2, 4, 6], "validation": [1], "test": [3, 5, 7]}
    arrays, planes, supports, exclusions = {}, {}, {}, []
    for split, bands in split_bands.items():
        h_images, m_images, split_planes, support = [], [], [], []
        attempts = 0
        while len(h_images) < counts[split]:
            if attempts > counts[split] * 20:
                raise ValueError(f"Too few tissue-containing images for {split}")
            band = bands[len(h_images) % len(bands)]
            depth = rng.uniform(edges[band] + margin, edges[band + 1] - margin)
            plane = coronal_plane(
                (center[0], depth, center[2]),
                shape=(size, size),
                spacing_mm=(pixel_mm, pixel_mm),
                tilt_deg=tuple(rng.uniform(-tilt_deg, tilt_deg, 2)),
            )
            h, vh = sample_plane(hist, ah, plane)
            m, vm = sample_plane(mri, am, plane)
            h = gaussian_filter(h, 0.5 if split != "train" else rng.uniform(0.3, 0.7))
            h[~vh] = 0
            attempts += 1
            try:
                for method in ("intensity", "gradient", "mask"):
                    features(np.stack((h, m)), method)
            except ValueError as exc:
                exclusions.append({"split": split, "reason": str(exc)})
                continue
            if np.count_nonzero(vh & vm & (h != 0)) < size * size * 0.01:
                exclusions.append({"split": split, "reason": "insufficient common tissue"})
                continue
            h_images.append(h)
            m_images.append(m)
            split_planes.append(plane.to_dict())
            support.append(depth_support(plane, interpolation_padding))
        arrays[f"{split}_histology"] = np.stack(h_images)
        # Only training MRI planes enter the fitted model.
        if split == "train":
            arrays["train_mri"] = np.stack(m_images)
        planes[split] = split_planes
        supports[split] = support
    gap = check_separation(supports)
    gallery, gallery_planes = [], []
    for depth in np.linspace(low, high, gallery_count):
        plane = coronal_plane(
            (center[0], depth, center[2]),
            shape=(size, size),
            spacing_mm=(pixel_mm, pixel_mm),
        )
        m, _ = sample_plane(mri, am, plane)
        features(m[None], "gradient")
        gallery.append(m)
        gallery_planes.append(plane.to_dict())
    arrays["gallery_mri"] = np.stack(gallery)
    np.savez_compressed(output / "images.npz", **arrays)
    digest = hashlib.sha256((output / "images.npz").read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "images_sha256": digest,
        "source_volumes": audits,
        "assume_mm": assume_mm,
        "split_bands": split_bands,
        "band_edges_mm": edges.tolist(),
        "excluded_margin_per_band_edge_mm": float(margin),
        "interpolation_padding_mm": interpolation_padding,
        "support_definition": "Pixel depth range plus conservative trilinear source-voxel extent",
        "minimum_cross_split_support_gap_mm": gap,
        "depth_supports_mm": supports,
        "planes": planes,
        "gallery_planes": gallery_planes,
        "counts": {**counts, "gallery": len(gallery)},
        "exclusions": exclusions,
        "scope": "Spatial holdout within one registered volume, known crop, near-coronal",
        "limits": [
            "No independent donor holdout",
            "No unknown fragment/pose validation",
            "BigBrain paired atlas is not donor MRI",
            "No tissue registration estimate",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _metrics(query_features, gallery_features, query_planes, gallery_planes, tolerance):
    indices, scores = rank(query_features, gallery_features, k=5)
    qc = np.array([p["center_mm"] for p in query_planes])
    gc = np.array([p["center_mm"] for p in gallery_planes])
    result = evaluate_retrieval(indices, qc, gc, tolerance)
    result["ordering_inversion_fraction"] = inversion_fraction(
        order_stack(gc[indices[:, 0]]), order_stack(qc)
    )
    result["ties_in_predicted_depth"] = int(len(qc) - len(np.unique(gc[indices[:, 0], 1])))
    predicted_depths = gc[indices[:, 0], 1]
    tau = kendalltau(qc[:, 1], predicted_depths).statistic
    result["ordering_kendall_tau_b"] = float(tau) if np.isfinite(tau) else None
    qn = np.array([Plane.from_dict(p).normal for p in query_planes])
    gn = np.array([Plane.from_dict(p).normal for p in gallery_planes])[indices[:, 0]]
    angles = np.rad2deg(np.arccos(np.clip(np.sum(qn * gn, axis=1), -1, 1)))
    result["retrieved_plane_angle_median_deg"] = float(np.median(angles))
    result["retrieved_plane_angle_p95_deg"] = float(np.percentile(angles, 95))
    result["ranked_gallery_indices"] = indices.tolist()
    result["cosine_similarities_not_confidence"] = scores.tolist()
    return result


def fit_and_evaluate(
    protocol: Path,
    output: Path,
    encoder,
    *,
    tolerance_mm: float = 5,
    regularizations: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0),
) -> dict:
    """Select pooling and ridge strength by validation only, then score locked test.

    Neither geometry nor any validation/test image is used to fit centering or
    ridge weights. Geometry enters split construction and scoring only.
    """
    if not regularizations or any(not np.isfinite(r) or r <= 0 for r in regularizations):
        raise ValueError("Provide finite positive regularizations")
    _new_directory(output)
    manifest = json.loads((protocol / "manifest.json").read_text())
    digest = hashlib.sha256((protocol / "images.npz").read_bytes()).hexdigest()
    if digest != manifest["images_sha256"]:
        raise ValueError("Protocol images differ from their recorded hash")
    check_separation(manifest["depth_supports_mm"])
    with np.load(protocol / "images.npz", allow_pickle=False) as f:
        images = {k: f[k] for k in f.files}
    gp = manifest["gallery_planes"]
    validation, selections, encoded = [], [], {}
    # Test images are not encoded/scored until every validation choice is fixed.
    for pooling in ("cls", "spatial"):
        print(f"Encoding training/validation/gallery: {pooling}", flush=True)
        feats = {
            k: encoder.encode(images[k], pooling=pooling)
            for k in ("train_histology", "train_mri", "validation_histology", "gallery_mri")
        }
        encoded[pooling] = feats
        frozen = _metrics(
            feats["validation_histology"],
            feats["gallery_mri"],
            manifest["planes"]["validation"],
            gp,
            tolerance_mm,
        )
        validation.append(
            {"method": f"dino_{pooling}", "pooling": pooling, "regularization": None, **frozen}
        )
        selections.append(
            {
                "method": f"dino_{pooling}",
                "pooling": pooling,
                "regularization": None,
                "validation_recall_at_1": frozen["recall_at_k"]["1"],
                "validation_median_error_mm": frozen["top1_median_error_mm"],
            }
        )
        candidates = []
        for reg in regularizations:
            adapter = fit_ridge(feats["train_histology"], feats["train_mri"], reg)
            metrics = _metrics(
                adapter.transform(feats["validation_histology"]),
                feats["gallery_mri"],
                manifest["planes"]["validation"],
                gp,
                tolerance_mm,
            )
            row = {
                "method": f"ridge_{pooling}",
                "pooling": pooling,
                "regularization": reg,
                **metrics,
            }
            validation.append(row)
            candidates.append((row, adapter))
        best, adapter = min(
            candidates,
            key=lambda item: (
                -item[0]["recall_at_k"]["1"],
                item[0]["top1_median_error_mm"],
                item[0]["top1_p95_error_mm"],
                item[0]["regularization"],
            ),
        )
        adapter.save(output / f"adapter_{pooling}.npz")
        sidecar = {
            "schema_version": 1,
            "encoder": encoder.metadata,
            "pooling": pooling,
            "feature_dim": feats["gallery_mri"].shape[1],
            "regularization": best["regularization"],
            "training_images_sha256": digest,
            "training_pairs": len(images["train_mri"]),
            "selection": "validation Recall@1, then median/P95 error; test never selected",
        }
        write_json(output / f"adapter_{pooling}.json", sidecar)
        selections.append(
            {
                "method": f"ridge_{pooling}",
                "pooling": pooling,
                "regularization": best["regularization"],
                "validation_recall_at_1": best["recall_at_k"]["1"],
                "validation_median_error_mm": best["top1_median_error_mm"],
            }
        )
    selected = min(
        selections,
        key=lambda row: (-row["validation_recall_at_1"], row["validation_median_error_mm"]),
    )
    write_json(output / "selection.json", {"selected": selected, "candidates": selections})
    results = []
    for method in ("intensity", "gradient", "mask"):
        results.append(
            {
                "method": method,
                **_metrics(
                    features(images["test_histology"], method),
                    features(images["gallery_mri"], method),
                    manifest["planes"]["test"],
                    gp,
                    tolerance_mm,
                ),
            }
        )
    from .adapters import RidgeAdapter

    for pooling in ("cls", "spatial"):
        print(f"Scoring locked test: {pooling}", flush=True)
        tf = encoder.encode(images["test_histology"], pooling=pooling)
        gf = encoded[pooling]["gallery_mri"]
        for name, qf in (
            (f"dino_{pooling}", tf),
            (
                f"ridge_{pooling}",
                RidgeAdapter.load(output / f"adapter_{pooling}.npz").transform(tf),
            ),
        ):
            row = {"method": name, **_metrics(qf, gf, manifest["planes"]["test"], gp, tolerance_mm)}
            results.append(row)
            if name == selected["method"]:
                chosen = range(min(6, len(tf)))
                ix = np.array(row["ranked_gallery_indices"])
                contact_sheet(
                    [
                        im
                        for i in chosen
                        for im in (images["test_histology"][i], images["gallery_mri"][ix[i, 0]])
                    ],
                    [label for i in chosen for label in (f"Test query {i}", f"MRI {ix[i, 0]}")],
                    output / "matches.png",
                )
        np.savez_compressed(
            output / f"features_{pooling}.npz", test_histology=tf, **encoded[pooling]
        )
    qc = np.array([p["center_mm"] for p in manifest["planes"]["test"]])
    gc = np.array([p["center_mm"] for p in gp])
    distances = np.linalg.norm(qc[:, None] - gc[None], axis=2)
    oracle = np.argsort(distances, axis=1)[:, :5]
    rng = np.random.default_rng(manifest["seed"] + 1)
    random_rank = np.stack([rng.permutation(len(gc))[:5] for _ in qc])
    report = {
        "schema_version": 1,
        "scope": manifest["scope"],
        "limits": manifest["limits"],
        "protocol_images_sha256": digest,
        "counts": manifest["counts"],
        "minimum_cross_split_support_gap_mm": manifest["minimum_cross_split_support_gap_mm"],
        "encoder": encoder.metadata,
        "selected_by_validation": selected,
        "validation": validation,
        "test_results": results,
        "oracle": evaluate_retrieval(oracle, qc, gc, tolerance_mm),
        "random": evaluate_retrieval(random_rank, qc, gc, tolerance_mm),
        "interpretation": "Synthetic spatial holdout, one volume; scores are similarities. "
        "Depth ordering is inferred from retrieved gallery centers; tied depths "
        "are ambiguous. Plane angles and unknown fragments remain unvalidated.",
    }
    write_json(output / "metrics.json", report)
    return report


def export_test_bundle(protocol: Path, output: Path, reference_space: str = "unspecified") -> dict:
    """Export test pixels for the reader/QA; source query poses remain scoring-only."""
    manifest = json.loads((protocol / "manifest.json").read_text())
    digest = hashlib.sha256((protocol / "images.npz").read_bytes()).hexdigest()
    if digest != manifest["images_sha256"]:
        raise ValueError("Protocol images differ from their recorded hash")
    _new_directory(output)
    with np.load(protocol / "images.npz", allow_pickle=False) as images:
        np.savez_compressed(
            output / "images.npz",
            gallery_mri=images["gallery_mri"],
            query_histology=images["test_histology"],
        )
    result = {
        "schema_version": 1,
        "reference_space": reference_space,
        "gallery_planes": manifest["gallery_planes"],
        "query_planes": manifest["planes"]["test"],
        "protocol_images_sha256": digest,
        "scope": "Test split exported for image-only inference and controlled QA scoring",
    }
    write_json(output / "manifest.json", result)
    return result
