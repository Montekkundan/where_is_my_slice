"""Command-line entry points for reproducible prepared-data experiments."""

import argparse
import json
from pathlib import Path

from .benchmark import benchmark
from .data import fetch_bigbrain, fetch_nextbrain, write_json
from .pipeline import create_phantom, generate
from .volume import audit_volume


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-bigbrain", help="Fetch official 400um atlas-space pair")
    fetch.add_argument("--root", type=Path, default=Path("data/bigbrain"))
    fetch_nb = commands.add_parser("fetch-nextbrain", help="Fetch pinned 37MB Case4 subset")
    fetch_nb.add_argument("--root", type=Path, default=Path("data/nextbrain"))
    audit = commands.add_parser("audit", help="Inspect NIfTI coordinates and declared units")
    audit.add_argument("volumes", nargs="+", type=Path)
    audit.add_argument("--output", type=Path)
    gen = commands.add_parser("generate", help="Generate known-crop fixed-gallery plane pairs")
    gen.add_argument("--histology", type=Path, required=True)
    gen.add_argument("--mri", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--count", type=int, default=32)
    gen.add_argument("--size", type=int, default=128)
    gen.add_argument("--pixel-mm", type=float, default=1.5)
    gen.add_argument("--tilt-deg", type=float, default=3)
    gen.add_argument("--seed", type=int, default=17)
    gen.add_argument("--reference-space", default="unspecified; caller must verify registration")
    gen.add_argument(
        "--assume-mm",
        action="store_true",
        help="Assert mm for unspecified units only when dataset documentation supports it",
    )
    bench = commands.add_parser(
        "benchmark", help="Run image-only baselines and shuffled stack test"
    )
    bench.add_argument("bundle", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--tolerance-mm", type=float, default=5)
    demo = commands.add_parser("demo", help="Run offline analytic phantom end to end")
    demo.add_argument("--output", type=Path, required=True)
    nb = commands.add_parser(
        "nextbrain-audit", help="Audit web-stack pieces using supplied transforms"
    )
    nb.add_argument("case", type=Path)
    nb.add_argument("--output", type=Path, required=True)
    mosaic = commands.add_parser(
        "nextbrain-mosaic", help="Preview known-affine NIfTI blocks in MRI"
    )
    mosaic.add_argument("case", type=Path)
    mosaic.add_argument("--blocks", type=int, nargs="+", required=True)
    mosaic.add_argument("--output", type=Path, required=True)
    mosaic.add_argument("--pixel-mm", type=float, default=0.3)
    mosaic.add_argument("--size", type=int, default=512)
    mosaic.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-bigbrain":
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
            )
            result = {"count": result["count"], "output": str(args.output.resolve())}
        elif args.command == "benchmark":
            result = benchmark(args.bundle, args.output, args.tolerance_mm)
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
        else:
            from .mosaic import preview_mosaic

            result = preview_mosaic(
                args.case, args.blocks, args.output, args.pixel_mm, args.size, args.threshold
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")
