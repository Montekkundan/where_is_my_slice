"""Fit the paired patch matcher on the benchmark's training and validation sections."""

import argparse
from pathlib import Path

from slicegps.data import write_json
from slicegps.error_detection import spatial_splits
from slicegps.patch_matching import fit_patch_matcher
from slicegps.sbr import SBR_REVISION, load_sbr


def train(
    root,
    output,
    epochs=150,
    device="auto",
    donor_calibrated=False,
    preprocessing="canonical_v1",
):
    if output.exists():
        raise FileExistsError(output)
    donors = ("BigBrain", "Allen_labels") if donor_calibrated else ("BigBrain",)
    groups = {"train": [], "validation": []}
    splits = {}
    for donor in donors:
        histology, mri, ids = load_sbr(root, donor)
        selected = spatial_splits(len(histology))
        splits[donor] = {}
        for name in groups:
            indices = selected[name]
            groups[name].append((histology[indices], mri[indices]))
            splits[donor][name] = [ids[i] for i in indices]
    output.mkdir(parents=True)
    provenance = {
        "source_revision": SBR_REVISION,
        "donor_calibrated": donor_calibrated,
        "splits": splits,
        "test_images_used_for_fit_or_selection": False,
    }
    write_json(output / "split.json", provenance)
    matcher = fit_patch_matcher(
        groups["train"],
        groups["validation"],
        epochs=epochs,
        device=device,
        provenance=provenance,
        preprocessing=preprocessing,
    )
    matcher.save(output / "patch_encoder.safetensors")
    return matcher.metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/synthbyreg"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", default="auto", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--donor-calibrated", action="store_true")
    parser.add_argument(
        "--preprocessing", choices=("canonical_v1", "paper_v2"), default="canonical_v1"
    )
    args = parser.parse_args()
    train(
        args.root, args.output, args.epochs, args.device, args.donor_calibrated, args.preprocessing
    )
