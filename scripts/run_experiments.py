"""Reproduce the BigBrain spatial holdout, matching reader and NextBrain checks.

Run from the repository after installing .[learned]. Downloads and model caches
stay under data/. The output folder must be new or empty. DINO weights are pinned.
"""

import argparse
from pathlib import Path

from PIL import Image

from slicegps.benchmark import benchmark
from slicegps.data import fetch_bigbrain, fetch_nextbrain, write_json
from slicegps.encoders import DinoEncoder
from slicegps.learned import export_test_bundle, fit_and_evaluate, prepare_protocol
from slicegps.matching import build_index, match_images
from slicegps.mosaic import preview_mosaic
from slicegps.nextbrain import audit_case
from slicegps.pipeline import generate, read_bundle
from slicegps.stack import evaluate_stack
from slicegps.training import train_dual
from slicegps.visualization import pose_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    output, data = args.output, args.data
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    print("Downloading/checking registered BigBrain and atlas MRI", flush=True)
    fetch_bigbrain(data / "bigbrain")
    hist = data / "bigbrain/full16_400um_2009b_sym.nii.gz"
    mri = data / "bigbrain/mni_icbm152_t1_tal_nlin_sym_09b_hires.nii.gz"
    protocol = output / "protocol"
    prepare_protocol(hist, mri, protocol, seed=26018)
    encoder = DinoEncoder(data / "models", device=args.device)
    print("Fitting ridge adapters and measuring frozen features", flush=True)
    report = fit_and_evaluate(protocol, output / "retrieval", encoder)
    print("Training contrastive/pose heads with validation checkpoint selection", flush=True)
    train_dual(output / "retrieval/features_spatial.npz", protocol, output / "dual", seed=26019)
    bundle = output / "test_bundle"
    export_test_bundle(protocol, bundle, "MNI ICBM152 2009b symmetric")
    build_index(bundle, output / "index", method="dino", encoder=encoder, pooling="spatial")
    manifest, images = read_bundle(bundle)
    query_root = output / "query_images"
    query_root.mkdir()
    paths = []
    for i, image in enumerate(images["query_histology"]):
        path = query_root / f"section-{i:04d}.tiff"
        Image.fromarray(image.astype("float32")).save(path)
        paths.append(path)
    found = match_images(output / "index", paths, output / "matches", encoder=encoder)
    predictions = [row["candidates"][0]["plane"] for row in found["matches"]]
    pose_svg(manifest["query_planes"], predictions, output / "matches/planes.svg")
    evaluate_stack(bundle, output / "index", output / "stack", encoder=encoder, seed=26020)
    print("Checking NextBrain donor data and two-piece slab", flush=True)
    fetch_nextbrain(data / "nextbrain")
    case = data / "nextbrain/P85-18"
    write_json(output / "nextbrain/audit.json", audit_case(case))
    preview_mosaic(case, [4, 27], output / "nextbrain/A5")
    generate(
        case / "histology_blocks/4/HE.nii.gz",
        case / "mri_rotated/mri.nii.gz",
        output / "nextbrain/planes",
        count=16,
        pixel_mm=0.7,
        tilt_deg=1,
        reference_space="NextBrain P85-18 supplied registered geometry",
        assume_mm=True,
    )
    benchmark(output / "nextbrain/planes", output / "nextbrain/baselines", tolerance_mm=2)
    from slicegps.external import external_check, fetch_external_sections

    source_manifest = data / "nextbrain/external_sources.json"
    fetch_external_sections(case, source_manifest)
    external_check(
        case, output / "nextbrain/real_sections", encoder, source_manifest=source_manifest
    )
    print("Checking a second independent NextBrain donor", flush=True)
    case2 = data / "nextbrain_case2/P41-16"
    case2_manifest = data / "nextbrain_case2/external_sources.json"
    fetch_external_sections(
        case2,
        case2_manifest,
        repository="UCL/NextBrain-Case-2",
        revision="9c29e15e2eef6db902296f53de636d57fc5aa6a1",
        donor="P41-16",
        gallery_per_block=16,
    )
    external_check(
        case2, output / "nextbrain/real_sections_case2", encoder, source_manifest=case2_manifest
    )
    summary = {
        "selected_by_validation": report["selected_by_validation"],
        "bigbrain_test_queries": len(images["query_histology"]),
        "gallery_planes": len(manifest["gallery_planes"]),
        "readers": ["matches/index.html", "stack/index.html"],
        "scope": "Research prototype; known crops, synthetic within-volume holdout and "
        "small registered real-section external check. See individual metrics.",
    }
    write_json(output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
