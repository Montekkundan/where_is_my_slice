"""Export compact, reader-facing slice-error results without model or data files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from slicegps.data import write_json


def summarize(runs: list[Path], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        report = json.loads((run / "metrics.json").read_text())
        features = (
            "pairwise"
            if report.get("pairwise_comparisons")
            else ("learned" if report["learned_features"] else "baseline")
        )
        if report.get("patch_features"):
            features = "patch"
            if report["patch_features"].get("preprocessing") == "paper_v2":
                features += "_paper"
        if report.get("sequence_features"):
            features = "sequence"
        if report.get("nonlinear_classifier"):
            features += "_mlp"
        if report.get("stack_normalized"):
            features += "_normalized"
        scope = "calibrated" if report["protocol"]["donor_calibrated"] else "independent"
        name = f"error_detection_{features}_{scope}"
        write_json(output / f"{name}.json", report)
        for dataset, result in report["results"].items():
            for method, measured in result["methods"].items():
                for kind, metrics in measured["by_kind"].items():
                    rows.append(
                        {
                            "experiment": name,
                            "dataset": dataset,
                            "method": method,
                            "error_kind": kind,
                            "unique_evaluated_sections": result["evaluated_sections"],
                            **metrics,
                        }
                    )
    with (output / "error_detection.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    summarize(args.runs, args.output)
