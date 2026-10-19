"""Reproduce one saved benchmark trial through the truth-free review interface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from slicegps.data import write_json
from slicegps.demo_corruption import DEMO_KIND, inject_demo_errors
from slicegps.error_detection import ERROR_KINDS, classification_metrics, inject_errors
from slicegps.error_review import review_errors
from slicegps.sbr import load_sbr


def review_case(root, run, output, dataset, kind, trial=0, method="neighbour_images"):
    report = json.loads((run / "metrics.json").read_text())
    model_path = run / f"{method}.json"
    model = json.loads(model_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    if not 0 <= trial < report["protocol"]["trials_per_kind"]:
        raise ValueError("Trial is outside the saved experiment")
    histology, mri, ids = load_sbr(root, dataset, size=model.get("image_size", 96))
    expected_ids = report["results"][dataset]["section_ids"]
    selected = [ids.index(section_id) for section_id in expected_ids]
    histology, mri = histology[selected], mri[selected]
    demo = report["protocol"].get("corruption_protocol") == "issue_demo_v1"
    if (kind == DEMO_KIND) != demo:
        raise ValueError("Requested corruption kind does not match the saved protocol")
    observed, truth = (
        inject_demo_errors(histology, 0.05, 49301 + trial)
        if demo
        else inject_errors(histology, kind, 0.05, 48301 + ERROR_KINDS.index(kind) * 10000 + trial)
    )
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    hist_paths, mri_paths = [], []
    for i, (hist, reference) in enumerate(zip(observed, mri, strict=True)):
        h, m = inputs / f"section-{i:03d}.tiff", inputs / f"mri-{i:03d}.tiff"
        Image.fromarray(hist.astype(np.float32)).save(h)
        Image.fromarray(reference.astype(np.float32)).save(m)
        hist_paths.append(h)
        mri_paths.append(m)
    # Truth is deliberately unavailable to review_errors. Names are anonymous.
    reviewed = review_errors(
        model_path,
        hist_paths,
        output / "review",
        mri_paths if model["expected_mri_required"] else None,
    )
    with (run / f"{dataset}_{method}.csv").open() as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["kind"] == kind and int(row["trial"]) == trial
        ]
    actual = reviewed["sections"][2:-2]
    expected_scores = np.array([float(row["score"]) for row in rows])
    actual_scores = np.array([row["score"] for row in actual])
    # Floating-point matrix kernels may vary slightly with batch grouping.
    np.testing.assert_allclose(actual_scores, expected_scores, atol=1e-3, rtol=1e-5)
    np.testing.assert_array_equal(
        [row["flagged"] for row in actual], [bool(int(row["flagged"])) for row in rows]
    )
    scored_truth = truth["is_error"][2:-2]
    np.testing.assert_array_equal(scored_truth, [bool(int(row["is_error"])) for row in rows])
    result = {
        "dataset": dataset,
        "kind": kind,
        "trial": trial,
        "actual_error_indices": np.flatnonzero(truth["is_error"]).tolist(),
        "flagged_indices": reviewed["flagged_indices"],
        "metrics": classification_metrics(
            scored_truth, np.array([row["flagged"] for row in actual])
        ),
        "maximum_score_difference": float(np.max(np.abs(actual_scores - expected_scores))),
        "note": "Single trial; not an accuracy estimate. Truth is added after prediction.",
    }
    write_json(output / "ground_truth.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/synthbyreg"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("BigBrain", "Allen_labels"), default="BigBrain")
    parser.add_argument("--kind", choices=(*ERROR_KINDS, DEMO_KIND), default="flip")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=("neighbour_images", "neighbour_images_and_expected_mri"),
        default="neighbour_images",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            review_case(
                args.root, args.run, args.output, args.dataset, args.kind, args.trial, args.method
            ),
            indent=2,
        )
    )
