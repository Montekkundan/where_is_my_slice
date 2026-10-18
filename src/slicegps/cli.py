"""Reproducible histology-to-MRI matching, evaluation and piece preparation."""

import argparse
import json
from pathlib import Path

from .benchmark import benchmark
from .data import fetch_bigbrain, fetch_nextbrain, write_json
from .pipeline import create_phantom, generate
from .volume import audit_volume


def _encoder_options(parser):
    parser.add_argument("--cache-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)


def _encoder(args):
    from .encoders import DinoEncoder

    return DinoEncoder(args.cache_dir, args.device, args.batch_size)


def _volume_pair_options(parser):
    parser.add_argument("--histology", type=Path, required=True)
    parser.add_argument("--mri", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--pixel-mm", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--assume-mm",
        action="store_true",
        help="Assert mm for unknown units only with dataset documentation",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-bigbrain", help="Download official 400um atlas-space pair")
    fetch.add_argument("--root", type=Path, default=Path("data/bigbrain"))
    fetch_nb = commands.add_parser("fetch-nextbrain", help="Download pinned 37MB Case4 subset")
    fetch_nb.add_argument("--root", type=Path, default=Path("data/nextbrain"))
    audit = commands.add_parser("audit", help="Inspect NIfTI coordinates and units")
    audit.add_argument("volumes", nargs="+", type=Path)
    audit.add_argument("--output", type=Path)
    gen = commands.add_parser(
        "generate", help="Generate paired reference planes and tilted queries"
    )
    _volume_pair_options(gen)
    gen.add_argument("--count", type=int, default=32)
    gen.add_argument(
        "--augment", action="store_true", help="Add stain/intensity and tear stress tests"
    )
    gen.add_argument("--tilt-deg", type=float, default=3)
    gen.add_argument("--reference-space", default="unspecified; verify registration")
    bench = commands.add_parser("benchmark", help="Image baselines and shuffled stack ordering")
    bench.add_argument("bundle", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--tolerance-mm", type=float, default=5)
    demo = commands.add_parser("demo", help="Offline analytic phantom end to end")
    demo.add_argument("--output", type=Path, required=True)
    nb = commands.add_parser(
        "nextbrain-audit", help="Inspect piece/slab mapping and supplied transforms"
    )
    nb.add_argument("case", type=Path)
    nb.add_argument("--output", type=Path, required=True)
    mosaic = commands.add_parser(
        "nextbrain-mosaic", help="Assemble known-affine blocks in MRI space"
    )
    mosaic.add_argument("case", type=Path)
    mosaic.add_argument("--blocks", type=int, nargs="+", required=True)
    mosaic.add_argument("--output", type=Path, required=True)
    mosaic.add_argument("--pixel-mm", type=float, default=0.3)
    mosaic.add_argument("--size", type=int, default=512)
    mosaic.add_argument("--threshold", type=float, default=0.02)
    prep = commands.add_parser(
        "learned-prepare", help="Create separated train/validation/test bands"
    )
    _volume_pair_options(prep)
    prep.add_argument("--train-count", type=int, default=192)
    prep.add_argument("--validation-count", type=int, default=24)
    prep.add_argument("--test-count", type=int, default=48)
    prep.add_argument("--gallery-count", type=int, default=128)
    prep.add_argument("--tilt-deg", type=float, default=1)
    fit = commands.add_parser("learned-fit", help="Compare frozen DINO and train paired ridge maps")
    fit.add_argument("protocol", type=Path)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--tolerance-mm", type=float, default=5)
    _encoder_options(fit)
    dual = commands.add_parser(
        "train-dual", help="Train dual contrastive and pose heads on frozen features"
    )
    dual.add_argument("features", type=Path)
    dual.add_argument("protocol", type=Path)
    dual.add_argument("--output", type=Path, required=True)
    dual.add_argument("--epochs", type=int, default=100)
    dual.add_argument("--seed", type=int, default=26019)
    dual.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    gallery = commands.add_parser(
        "mri-gallery", help="Sample MRI-only candidate planes for matching"
    )
    gallery.add_argument("mri", type=Path)
    gallery.add_argument("--output", type=Path, required=True)
    gallery.add_argument("--count", type=int, default=128)
    gallery.add_argument("--size", type=int, default=128)
    gallery.add_argument("--pixel-mm", type=float, default=1.5)
    gallery.add_argument("--center-xz", type=float, nargs=2)
    gallery.add_argument("--depth-range", type=float, nargs=2)
    gallery.add_argument("--tilts", type=float, nargs="+", default=[0])
    gallery.add_argument("--assume-mm", action="store_true")
    index = commands.add_parser(
        "index", help="Build a persistent image index from a gallery bundle"
    )
    index.add_argument("bundle", type=Path)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument(
        "--method", choices=("intensity", "gradient", "mask", "dino"), default="dino"
    )
    index.add_argument("--pooling", choices=("cls", "spatial"), default="spatial")
    index.add_argument("--model", type=Path, help="Dual-head safetensors model for both modalities")
    _encoder_options(index)
    match = commands.add_parser(
        "match", help="Retrieve MRI candidates for grayscale PNG/JPEG/TIFF sections"
    )
    match.add_argument("index", type=Path)
    match.add_argument("queries", type=Path, nargs="+")
    match.add_argument("--output", type=Path, required=True)
    match.add_argument("--adapter", type=Path)
    match.add_argument(
        "--model", type=Path, help="Must match the dual model used to build the index"
    )
    match.add_argument("--k", type=int, default=5)
    _encoder_options(match)
    stack = commands.add_parser(
        "stack-demo", help="Corrupt a prepared stack and measure heuristic QA"
    )
    stack.add_argument("bundle", type=Path)
    stack.add_argument("index", type=Path)
    stack.add_argument("--output", type=Path, required=True)
    stack.add_argument("--adapter", type=Path)
    stack.add_argument("--flip-fraction", type=float, default=0.05)
    stack.add_argument("--damage-fraction", type=float, default=0.05)
    stack.add_argument("--seed", type=int, default=26020)
    _encoder_options(stack)
    sections = commands.add_parser(
        "fetch-nextbrain-sections", help="Download pinned real HE/MRI JPEGs"
    )
    sections.add_argument("case", type=Path)
    sections.add_argument(
        "--manifest", type=Path, default=Path("data/nextbrain/external_sources.json")
    )
    external = commands.add_parser(
        "external-check", help="Evaluate fixed features on real JPEG pairs"
    )
    external.add_argument("case", type=Path)
    external.add_argument("--output", type=Path, required=True)
    external.add_argument("--source-manifest", type=Path)
    _encoder_options(external)
    fetch_sbr = commands.add_parser("fetch-sbr", help="Download pinned donor-matched paper pairs")
    fetch_sbr.add_argument("--root", type=Path, default=Path("data/synthbyreg"))
    errors = commands.add_parser("detect-errors", help="Review errors in a nominally ordered stack")
    errors.add_argument("model", type=Path)
    errors.add_argument("sections", type=Path, nargs="+")
    errors.add_argument("--mri", type=Path, nargs="+")
    errors.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-sbr":
            from .sbr import fetch_sbr

            manifest = fetch_sbr(args.root)
            result = {
                "counts": {key: len(rows) for key, rows in manifest["datasets"].items()},
                "manual_landmark_files": manifest["manual_landmark_files"],
            }
        elif args.command == "detect-errors":
            from .error_review import review_errors

            reviewed = review_errors(args.model, args.sections, args.output, args.mri)
            result = {
                "viewer": str((args.output / "index.html").resolve()),
                "flagged_indices": reviewed["flagged_indices"],
            }
        elif args.command == "fetch-bigbrain":
            result = fetch_bigbrain(args.root)
        elif args.command == "fetch-nextbrain":
            result = fetch_nextbrain(args.root)
        elif args.command == "audit":
            result = [audit_volume(p) for p in args.volumes]
            if args.output:
                write_json(args.output, result)
        elif args.command == "generate":
            result = generate(
                args.histology,
                args.mri,
                args.output,
                args.count,
                args.size,
                args.pixel_mm,
                args.tilt_deg,
                args.seed,
                args.reference_space,
                args.assume_mm,
                args.augment,
            )
            result = {"count": result["count"], "output": str(args.output.resolve())}
        elif args.command == "benchmark":
            benchmark(args.bundle, args.output, args.tolerance_mm)
            result = {"report": str((args.output / "report.md").resolve())}
        elif args.command == "demo":
            hist, mri = create_phantom(args.output / "fixture")
            generate(
                hist,
                mri,
                args.output / "planes",
                count=20,
                size=64,
                pixel_mm=1.3,
                reference_space="analytic phantom, RAS mm",
            )
            benchmark(args.output / "planes", args.output / "benchmark")
            result = {"output": str(args.output.resolve()), "human_data": False}
        elif args.command == "nextbrain-audit":
            from .nextbrain import audit_case

            result = audit_case(args.case)
            write_json(args.output, result)
            result = {"report": str(args.output.resolve()), "summary": result["summary"]}
        elif args.command == "nextbrain-mosaic":
            from .mosaic import preview_mosaic

            result = preview_mosaic(
                args.case, args.blocks, args.output, args.pixel_mm, args.size, args.threshold
            )
        elif args.command == "learned-prepare":
            from .learned import prepare_protocol

            result = prepare_protocol(
                args.histology,
                args.mri,
                args.output,
                train_count=args.train_count,
                validation_count=args.validation_count,
                test_count=args.test_count,
                gallery_count=args.gallery_count,
                size=args.size,
                pixel_mm=args.pixel_mm,
                tilt_deg=args.tilt_deg,
                seed=args.seed,
                assume_mm=args.assume_mm,
            )
            result = {"counts": result["counts"], "output": str(args.output.resolve())}
        elif args.command == "learned-fit":
            from .learned import fit_and_evaluate

            result = fit_and_evaluate(
                args.protocol, args.output, _encoder(args), tolerance_mm=args.tolerance_mm
            )
            result = {
                "selected": result["selected_by_validation"],
                "report": str((args.output / "metrics.json").resolve()),
            }
        elif args.command == "train-dual":
            from .training import train_dual

            train_dual(
                args.features,
                args.protocol,
                args.output,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
            )
            result = {"output": str(args.output.resolve())}
        elif args.command == "mri-gallery":
            from .gallery import prepare_mri_gallery

            result = prepare_mri_gallery(
                args.mri,
                args.output,
                count=args.count,
                size=args.size,
                pixel_mm=args.pixel_mm,
                center_xz=args.center_xz,
                depth_range=args.depth_range,
                tilts=tuple(args.tilts),
                assume_mm=args.assume_mm,
            )
            result = {"output": str(args.output.resolve()), "planes": len(result["gallery_planes"])}
        elif args.command == "index":
            from .matching import build_index

            build_index(
                args.bundle,
                args.output,
                args.method,
                _encoder(args) if args.method == "dino" else None,
                args.pooling,
                model=args.model,
            )
            result = {"index": str(args.output.resolve())}
        elif args.command in ("match", "stack-demo"):
            method = json.loads((args.index / "manifest.json").read_text())["method"]
            encoder = _encoder(args) if method == "dino" else None
            if args.command == "match":
                from .matching import match_images

                match_images(
                    args.index,
                    args.queries,
                    args.output,
                    args.adapter,
                    encoder,
                    args.k,
                    model=args.model,
                )
            else:
                from .stack import evaluate_stack

                evaluate_stack(
                    args.bundle,
                    args.index,
                    args.output,
                    encoder,
                    args.adapter,
                    args.flip_fraction,
                    args.damage_fraction,
                    args.seed,
                )
            result = {
                "viewer": str((args.output / "index.html").resolve()),
                "output": str(args.output.resolve()),
            }
        elif args.command == "fetch-nextbrain-sections":
            from .external import fetch_external_sections

            result = fetch_external_sections(args.case, args.manifest)
        elif args.command == "external-check":
            from .external import external_check

            result = external_check(
                args.case, args.output, _encoder(args), source_manifest=args.source_manifest
            )
            result = {"report": str((args.output / "metrics.json").resolve())}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileExistsError, FileNotFoundError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
