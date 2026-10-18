"""Controlled image retrieval smoke tests, with geometry restricted to scoring."""

from pathlib import Path

import numpy as np

from .data import write_json
from .pipeline import contact_sheet, read_bundle
from .retrieval import evaluate_retrieval, features, inversion_fraction, order_stack, rank


def benchmark(bundle: Path, output: Path, tolerance_mm: float = 5, seed: int = 91) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    manifest, images = read_bundle(bundle)
    gallery_centers = np.array([p["center_mm"] for p in manifest["gallery_planes"]])
    query_centers = np.array([p["center_mm"] for p in manifest["query_planes"]])
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(query_centers))
    query_centers = query_centers[permutation]
    queries = images["query_histology"][permutation]
    truth_order = order_stack(query_centers)
    initial_order = np.arange(len(queries))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for reference in ("histology", "mri"):
        gallery = images[f"gallery_{reference}"]
        for method in ("intensity", "gradient", "mask"):
            # Only pixels enter these functions. Centers are used below for scoring.
            indices, scores = rank(features(queries, method), features(gallery, method))
            metrics = evaluate_retrieval(indices, query_centers, gallery_centers, tolerance_mm)
            predicted = gallery_centers[indices[:, 0]]
            metrics.update(
                {
                    "reference": reference,
                    "method": method,
                    "ordering_inversion_fraction": inversion_fraction(
                        order_stack(predicted), truth_order
                    ),
                    "ties_in_predicted_depth": int(
                        len(predicted) - len(np.unique(predicted[:, 1]))
                    ),
                    "ranked_gallery_indices": indices.tolist(),
                    "cosine_similarities_not_confidence": scores.tolist(),
                }
            )
            rows.append(metrics)
            if reference == "mri" and method == "gradient":
                chosen = range(min(4, len(queries)))
                contact_sheet(
                    [im for i in chosen for im in (queries[i], gallery[indices[i, 0]])],
                    [
                        label
                        for i in chosen
                        for label in (f"Query {i}", f"MRI top1 {indices[i, 0]}")
                    ],
                    output / "mri_gradient_matches.png",
                )
    distances = np.linalg.norm(query_centers[:, None, :] - gallery_centers[None, :, :], axis=2)
    oracle = np.argsort(distances, axis=1, kind="stable")[:, : min(5, len(gallery_centers))]
    random_indices = np.stack([rng.permutation(len(gallery_centers))[:5] for _ in queries])
    report = {
        "scope": "Known-crop coronal retrieval; same-volume synthetic queries; fixed gallery",
        "not_validated": [
            "cross-donor generalization",
            "unknown fragment placement",
            "full plane pose",
            "damage or flip detection",
            "trained embeddings",
        ],
        "seed": seed,
        "query_permutation": permutation.tolist(),
        "input_manifest": str((bundle / "manifest.json").resolve()),
        "unchanged_shuffled_order_inversion_fraction": inversion_fraction(
            initial_order, truth_order
        ),
        "oracle_nearest_gallery": evaluate_retrieval(
            oracle, query_centers, gallery_centers, tolerance_mm
        ),
        "random_ranking": evaluate_retrieval(
            random_indices, query_centers, gallery_centers, tolerance_mm
        ),
        "results": rows,
        "interpretation": "Controls use the same prepared volume and crop. No model training. "
        "Correct sorting can coexist with inaccurate localization; inspect both.",
    }
    write_json(output / "metrics.json", report)
    lines = [
        "# Controlled retrieval benchmark",
        "",
        report["scope"],
        "",
        f"Center-error tolerance: {tolerance_mm:g} mm. Lower error and inversion are better.",
        "",
        "| Reference | Baseline | Median mm | P95 mm | Recall@1 | Recall@K | Inversions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        recalls = row["recall_at_k"]
        lines.append(
            f"| {row['reference']} | {row['method']} | "
            f"{row['top1_median_error_mm']:.2f} | {row['top1_p95_error_mm']:.2f} | "
            f"{recalls['1']:.1%} | {recalls[str(row['k'])]:.1%} | "
            f"{row['ordering_inversion_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"K = {rows[0]['k']} (capped by gallery size).",
            "",
            report["interpretation"],
            "",
            "Query positions are jittered and tilted, then shuffled before retrieval. "
            "Coordinates enter scoring and sorting only. Similarities are not confidence.",
            "",
            f"Unchanged shuffled-order inversion fraction: "
            f"{report['unchanged_shuffled_order_inversion_fraction']:.3f}",
            "",
            "Not validated: " + ", ".join(report["not_validated"]) + ".",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    return report
