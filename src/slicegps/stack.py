"""Image-only flip search and heuristic stack QA with separate controlled scoring."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, median_filter
from scipy.stats import kendalltau

from .augmentation import _image, corrupt_stack
from .data import write_json
from .geometry import Plane
from .matching import _empty_output, _load_index, _reader, match_images
from .pipeline import contact_sheet, read_bundle
from .retrieval import _prepare, inversion_fraction, order_stack


def _stack(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 3 or len(images) == 0:
        raise ValueError("images must have nonempty shape (N, H, W)")
    return np.stack([_image(image) for image in images])


def flip_search(
    images: np.ndarray,
    index: Path,
    output: Path,
    encoder=None,
    adapter: Path | None = None,
    k: int = 5,
) -> dict:
    """Compare both horizontal orientations for every image, using image scores only.

    A positive flipped-minus-original top-score margin suggests a horizontal
    correction. This is an uncalibrated heuristic; symmetric anatomy and poor
    retrieval can make either orientation ambiguous or wrong.
    """
    output = _empty_output(output)
    images = _stack(images)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    _load_index(Path(index))
    output.mkdir(parents=True, exist_ok=True)
    query_root = output / "query_images"
    query_root.mkdir()
    paths = []
    for flipped in (False, True):
        for i, image in enumerate(images):
            path = query_root / f"section-{i:04d}{'-flipped' if flipped else ''}.tiff"
            pixels = np.fliplr(image) if flipped else image
            # Match the raw range; percentile-scaled PNGs would be normalized twice.
            Image.fromarray(pixels.astype(np.float32)).save(path)
            paths.append(path)
    both = match_images(Path(index), paths, output / "both_orientations", adapter, encoder, k)
    selected, corrections = [], []
    for i in range(len(images)):
        original, flipped = both["matches"][i], both["matches"][i + len(images)]
        margin = (
            flipped["candidates"][0]["cosine_similarity"]
            - original["candidates"][0]["cosine_similarity"]
        )
        correction = margin > 0
        row = dict(flipped if correction else original)
        row.update(
            query_index=i,
            query_path=str(paths[i].resolve()),
            query_name=paths[i].name,
            horizontal_flip_suggested=correction,
            flip_score_margin=float(margin),
            flip_ambiguous=abs(margin) <= 0.01,
        )
        selected.append(row)
        corrections.append(correction)
    result = dict(both)
    result["matches"] = selected
    result["suggested_order"] = order_stack(
        np.array([row["candidates"][0]["plane"]["center_mm"] for row in selected])
    ).tolist()
    result["horizontal_flip_indices"] = np.flatnonzero(corrections).tolist()
    result["flip_rule"] = "flipped top cosine > original top cosine; not calibrated"
    result["flip_ambiguity_rule"] = "absolute cosine margin <= 0.01; heuristic"
    result["flip_limit"] = "symmetric anatomy can hide flips; incorrect retrieval can suggest flips"
    _, _, gallery = _load_index(Path(index))
    page = _reader(result, list(images), gallery)
    rows = "".join(
        "<tr><td>"
        + html.escape(row["query_name"])
        + "</td><td>"
        + str(row["horizontal_flip_suggested"])
        + "</td><td>"
        + f"{row['flip_score_margin']:.4f}</td><td>{row['flip_ambiguous']}</td></tr>"
        for row in selected
    )
    extra = (
        "<h2>Horizontal flip review</h2><p>Higher image cosine suggests correction. "
        "These flags are heuristic. Symmetric brains can make flips unidentifiable.</p>"
        "<table><tr><th>Input</th><th>Suggest flip</th><th>Score margin</th>"
        "<th>Ambiguous</th></tr>" + rows + "</table>"
    )
    (output / "index.html").write_text(page.replace("</main>", extra + "</main>"))
    write_json(output / "matches.json", result)
    return result


def qa_stack(
    images: np.ndarray,
    predicted_planes: list[dict],
    expected_spacing_mm: float | None = None,
    z_threshold: float = 3.5,
) -> dict:
    """Flag pixel and predicted-depth anomalies without consulting source truth.

    Robust stack medians are a diagnostic reference, not a trained damage model.
    Real anatomy, section thickness and sparse sampling can trigger these flags.
    """
    images = _stack(images)
    planes = [Plane.from_dict(plane) for plane in predicted_planes]
    if len(planes) != len(images):
        raise ValueError("predicted_planes must match image count")
    if not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("z_threshold must be positive and finite")
    measurements = []
    for image in images:
        scaled, foreground = _prepare(image)
        regions, count = label(foreground, structure=np.ones((3, 3)))
        sizes = np.bincount(regions.ravel(), minlength=count + 1)[1:]
        components = int(np.count_nonzero(sizes >= max(2, image.size * 0.001)))
        measurements.append(
            [
                float(foreground.mean()),
                float(scaled[foreground].mean()),
                float(components),
                float(sizes.max() / foreground.sum()),
            ]
        )
    measurements = np.array(measurements)
    medians = np.median(measurements, axis=0)
    mad = np.median(np.abs(measurements - medians), axis=0)
    scales = np.maximum(1.4826 * mad, [0.02, 0.05, 1.0, 0.05])
    deviations = (measurements - medians) / scales
    flags = (
        (np.abs(deviations[:, 0]) > z_threshold)
        | (np.abs(deviations[:, 1]) > z_threshold)
        | (deviations[:, 2] > z_threshold)
        | (deviations[:, 3] < -z_threshold)
    )
    depths = np.array([plane.center_mm[1] for plane in planes])
    order = np.argsort(depths, kind="stable")
    differences = np.diff(depths[order])
    positive = differences[differences > 1e-6]
    if expected_spacing_mm is None:
        spacing = float(np.median(positive)) if len(positive) else 1.0
    else:
        spacing = float(expected_spacing_mm)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("expected_spacing_mm must be finite and positive")
    window = min(5, len(images) if len(images) % 2 else len(images) - 1)
    smooth = median_filter(depths, size=max(window, 1), mode="nearest")
    gaps = [
        {"between_input_indices": [int(order[i]), int(order[i + 1])], "gap_mm": float(gap)}
        for i, gap in enumerate(differences)
        if gap > 2.5 * spacing
    ]
    return {
        "schema_version": 1,
        "query_count": len(images),
        "corruption_flagged_indices": np.flatnonzero(flags).tolist(),
        "pixel_measurement_names": [
            "foreground_fraction",
            "foreground_scaled_mean",
            "component_count",
            "largest_component_fraction",
        ],
        "pixel_measurements": measurements.tolist(),
        "pixel_robust_z": deviations.tolist(),
        "pixel_stack_medians": medians.tolist(),
        "pixel_scale_floors": [0.02, 0.05, 1.0, 0.05],
        "pixel_z_threshold": float(z_threshold),
        "predicted_depths_mm": depths.tolist(),
        "input_depth_reversals": (np.flatnonzero(np.diff(depths) < -1e-6) + 1).tolist(),
        "median_trajectory_mm": smooth.tolist(),
        "trajectory_outlier_indices": np.flatnonzero(
            np.abs(depths - smooth) > 3 * spacing
        ).tolist(),
        "trajectory_window": max(window, 1),
        "expected_spacing_mm": spacing,
        "suggested_order": order.tolist(),
        "duplicate_depth_pairs": [
            [int(order[i]), int(order[i + 1])]
            for i, difference in enumerate(differences)
            if difference <= 1e-6
        ],
        "large_depth_gaps": gaps,
        "gap_interpretation": "ambiguous: sparse sampling, matching error or missing sections",
        "scope": (
            "heuristic pixel/depth QA; anatomy and uncertain retrieval can trigger false flags"
        ),
    }


def _binary_metrics(flagged: list[int], truth: list[int], count: int) -> dict:
    predicted, actual = set(flagged), set(truth)
    true_positive = len(predicted & actual)
    false_positive, false_negative = len(predicted - actual), len(actual - predicted)
    return {
        "evaluated_queries": count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": count - len(predicted | actual),
        "precision": true_positive / len(predicted) if predicted else None,
        "recall": true_positive / len(actual) if actual else None,
    }


def evaluate_stack(
    bundle: Path,
    index: Path,
    output: Path,
    encoder=None,
    adapter: Path | None = None,
    flip_fraction: float = 0.05,
    damage_fraction: float = 0.05,
    seed: int = 0,
) -> dict:
    """Run synthetic corruption, image-only flip retrieval, QA, then truth scoring.

    Every query enters inference and every query enters scoring. Source planes
    and corruption truth are used only after image retrieval and QA complete.
    """
    output = _empty_output(output)
    manifest, arrays = read_bundle(Path(bundle))
    if "query_histology" not in arrays:
        raise ValueError("Bundle is missing query_histology")
    images, truth = corrupt_stack(arrays["query_histology"], flip_fraction, damage_fraction, seed)
    source_planes = manifest.get("query_planes")
    if not isinstance(source_planes, list) or len(source_planes) != len(images):
        raise ValueError("query_planes must match query_histology")
    source_planes = [Plane.from_dict(plane) for plane in source_planes]
    index_manifest, _, gallery = _load_index(Path(index))
    if index_manifest["reference_space"] != manifest.get("reference_space", "unspecified"):
        raise ValueError("Bundle and index reference spaces must match for scoring")
    output.mkdir(parents=True, exist_ok=True)
    found = flip_search(images, Path(index), output / "retrieval", encoder, adapter)
    predicted_planes = [row["candidates"][0]["plane"] for row in found["matches"]]
    gallery_depths = np.unique(
        [plane["center_mm"][1] for plane in index_manifest["gallery_planes"]]
    )
    spacing = float(np.median(np.diff(gallery_depths))) if len(gallery_depths) > 1 else 1.0
    qa = qa_stack(images, predicted_planes, expected_spacing_mm=spacing)
    # Truth first enters calculations here, after all inference and heuristic flags.
    true_centers = np.array([source_planes[i].center_mm for i in truth["permutation"]])
    predicted_centers = np.array([plane["center_mm"] for plane in predicted_planes])
    true_order = order_stack(true_centers)
    inversion = inversion_fraction(found["suggested_order"], true_order)
    tau = kendalltau(true_centers[:, 1], predicted_centers[:, 1]).statistic
    errors = np.linalg.norm(true_centers - predicted_centers, axis=1)
    metrics = {
        "evaluated_queries": len(images),
        "ordering_inversion_fraction": inversion,
        "input_order_inversion_fraction": inversion_fraction(np.arange(len(images)), true_order),
        "depth_kendall_tau_b": float(tau) if np.isfinite(tau) else None,
        "center_error_median_mm": float(np.median(errors)),
        "center_error_p95_mm": float(np.percentile(errors, 95)),
        "flip_detection": _binary_metrics(
            found["horizontal_flip_indices"], truth["flipped_indices"], len(images)
        ),
        "corruption_detection": _binary_metrics(
            qa["corruption_flagged_indices"], truth["damaged_indices"], len(images)
        ),
        "tie_note": (
            "inversion uses stable gallery tie order; Kendall tau-b accounts for tied depths"
        ),
    }
    chosen = range(min(len(images), 12))
    contact_sheet(
        [
            im
            for i in chosen
            for im in (images[i], gallery[found["matches"][i]["candidates"][0]["gallery_index"]])
        ],
        [
            text
            for i in chosen
            for text in (f"Input {i}", f"MRI match; flip={i in found['horizontal_flip_indices']}")
        ],
        output / "stack_contact_sheet.png",
    )
    report = {
        "schema_version": 1,
        "seed": int(seed),
        "scope": "controlled synthetic corruption of all prepared query sections",
        "reference_space": index_manifest["reference_space"],
        "truth_used_for": "scoring only; never pixel retrieval, flip selection or heuristic flags",
        "metrics": metrics,
        "qa": qa,
        "reader": "index.html",
        "limits": [
            "symmetric anatomy can make horizontal flips ambiguous",
            "cosine margins and pixel flags are uncalibrated heuristics",
            "depth sorting assumes roughly coronal sections and known reference geometry",
            "synthetic damage results do not establish real human histology accuracy",
            "large gaps are ambiguous, not automatic evidence of missing sections",
        ],
    }
    qa_rows = "".join(
        f"<tr><td>{i}</td><td>{i in qa['corruption_flagged_indices']}</td>"
        f"<td>{depth:.2f}</td><td>{i in qa['trajectory_outlier_indices']}</td></tr>"
        for i, depth in enumerate(qa["predicted_depths_mm"])
    )
    extra = (
        "<h2>Stack QA review</h2><p>Pixel and trajectory flags are heuristic. "
        "Real anatomy and wrong retrieval can trigger false alarms. Large depth gaps "
        "are ambiguous and do not establish missing tissue.</p><table><tr>"
        "<th>Input index</th><th>Pixel anomaly</th><th>Predicted depth (mm)</th>"
        "<th>Trajectory anomaly</th></tr>" + qa_rows + "</table>"
        "<h2>Synthetic control scores</h2><p>All queries are scored. "
        "These controlled results do not establish real human accuracy.</p><pre>"
        + html.escape(json.dumps(metrics, indent=2, allow_nan=False))
        + "</pre>"
    )
    page = (output / "retrieval/index.html").read_text()
    (output / "index.html").write_text(page.replace("</main>", extra + "</main>"))
    write_json(output / "truth.json", truth)
    write_json(output / "metrics.json", report)
    return report
