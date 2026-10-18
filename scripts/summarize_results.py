"""Export compact, reader-facing results from a completed experiment directory."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from slicegps.data import write_json


def read(path):
    return json.loads(path.read_text())


def aggregate(value):
    return {
        k: v
        for k, v in value.items()
        if k not in {"ranked_gallery_indices", "cosine_similarities_not_confidence"}
    }


def export(run: Path, output: Path):
    # A completion marker prevents an interrupted run becoming a published result.
    summary = read(run / "summary.json")
    retrieval = read(run / "retrieval/metrics.json")
    dual = read(run / "dual/metrics.json")
    protocol = read(run / "protocol/manifest.json")
    bigbrain = {
        "scope": retrieval["scope"],
        "counts": retrieval["counts"],
        "protocol_images_sha256": retrieval["protocol_images_sha256"],
        "seed": protocol["seed"],
        "interpolation_padding_mm": protocol["interpolation_padding_mm"],
        "minimum_cross_split_support_gap_mm": retrieval["minimum_cross_split_support_gap_mm"],
        "encoder": {
            key: retrieval["encoder"][key]
            for key in ("model_id", "model_revision", "feature_fingerprint", "package_versions")
        },
        "selected_by_validation": summary["selected_by_validation"],
        "test_results": [aggregate(value) for value in retrieval["test_results"]],
        "oracle": retrieval["oracle"],
        "random": retrieval["random"],
        "dual_contrastive_pose": {
            key: dual[key]
            for key in (
                "architecture",
                "config",
                "selected_epoch",
                "epochs_completed",
                "test_retrieval",
                "test_pose",
                "training_mean_pose_prior_baseline",
                "limitations",
            )
        },
        "limits": retrieval["limits"]
        + [
            "Exploratory results: this spatial test set was inspected during project development.",
            "Frozen spatial DINO has large tail errors; "
            "improved median does not imply reliable order.",
        ],
        "reproduce": "python scripts/run_experiments.py --output outputs/run --device auto",
    }
    write_json(output / "bigbrain.json", bigbrain)
    rows = []

    def row(dataset, method, metrics, scope):
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "queries": metrics["query_count"],
                "candidates": metrics["gallery_count"],
                "tolerance_mm": metrics["tolerance_mm"],
                "median_error_mm": metrics["top1_median_error_mm"],
                "p95_error_mm": metrics["top1_p95_error_mm"],
                "recall_at_1": metrics["recall_at_k"]["1"],
                "recall_at_5": metrics["recall_at_k"]["5"],
                "exact_pair_accuracy": metrics.get("top1_exact_pair_accuracy", ""),
                "scope": scope,
            }
        )

    for value in retrieval["test_results"]:
        row("BigBrain", value["method"], value, retrieval["scope"])
    row("BigBrain", "dual_contrastive_pose", dual["test_retrieval"], retrieval["scope"])
    donors = []
    for name in ("real_sections", "real_sections_case2"):
        raw = read(run / f"nextbrain/{name}/metrics.json")
        source = raw["sources"]["source"]
        donor = {
            key: raw[key]
            for key in (
                "donor",
                "query_count",
                "gallery_count",
                "requested_blocks",
                "query_selection",
                "search_scope",
                "fit",
                "geometry_use",
                "coordinate_contract",
                "geometry_units",
                "methods",
                "exclusions",
                "warnings",
                "limits",
            )
        }
        donor["source"] = {
            key: source[key]
            for key in (
                "repository",
                "revision",
                "donor",
                "selected_section_indices",
                "license",
                "data_doi",
            )
        }
        # Keep reproducible public provenance without machine-specific local paths.
        donor["source"]["files"] = [
            {key: record[key] for key in ("path", "url", "size_bytes", "git_blob_sha", "sha256")}
            for record in source["files"]
        ]
        donors.append(donor)
        for method, metrics in raw["methods"].items():
            row(raw["donor"], method, metrics, raw["search_scope"])
    audit = read(run / "nextbrain/audit.json")
    nextbrain = {
        "real_section_checks": donors,
        "piece_preparation": {
            "case": "P85-18",
            "preview_slab": "A5",
            "preview_blocks": [4, 27],
            "method": "Place registered NIfTI blocks using their supplied physical affines.",
            "limit": "Does not infer placement of unknown detached pieces. "
            "Piece count varies by slab.",
            "audit_summary": audit["summary"],
        },
        "comparison_limit": "Known registered block crops are much easier "
        "than BigBrain atlas retrieval.",
    }
    write_json(output / "nextbrain.json", nextbrain)
    stack = read(run / "stack/metrics.json")
    write_json(
        output / "stack.json",
        {
            key: stack[key]
            for key in (
                "schema_version",
                "seed",
                "scope",
                "reference_space",
                "truth_used_for",
                "metrics",
                "limits",
            )
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "benchmarks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for source, destination in (
        ("retrieval/matches.png", "bigbrain_matches.png"),
        ("matches/planes.svg", "bigbrain_planes.svg"),
        ("nextbrain/A5/mosaic.png", "nextbrain_A5.png"),
    ):
        shutil.copyfile(run / source, output / destination)
    print(
        f"Exported {len(rows)} benchmark rows and {len(donors)} external donor checks to {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    export(args.run, args.output)


if __name__ == "__main__":
    main()
