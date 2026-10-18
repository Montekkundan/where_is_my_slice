"""Pinned, donor-matched histology/MRI pairs from the Synth-by-Reg release."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .data import download, sha256, write_json

SBR_REVISION = "1b67b630cc4e8e03be210d6fb39210b722296902"
SBR_REPO = "https://github.com/acasamitjana/SynthByReg"
SBR_RAW = f"https://raw.githubusercontent.com/acasamitjana/SynthByReg/{SBR_REVISION}/"
DATASETS = ("BigBrain", "Allen_labels")


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "SliceGPS/0.1 research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("Metadata exceeds byte cap")
    return data


def listed_pairs(table: str, dataset: str, tree: list[dict]) -> list[dict]:
    """Use the authors' evaluation list, excluding unlisted and split-region files."""
    if dataset not in DATASETS:
        raise ValueError("Unknown Synth-by-Reg dataset")
    blobs = {row["path"]: row for row in tree if row["type"] == "blob"}
    pairs = []
    for row in csv.DictReader(io.StringIO(table)):
        section = row["SLICE_ID"]
        if not section.isdigit() or len(section) != 3:
            raise ValueError("Unsafe section identifier")
        paths = {
            "histology": f"data/{dataset}/histo/images_affine/slice_{section}.png",
            "mri": f"data/{dataset}/mri/images/slice_{section}.png",
        }
        if any(path not in blobs for path in paths.values()):
            raise ValueError(f"Listed pair is incomplete: {dataset}/{section}")
        pairs.append({"section_id": section, **paths})
    if not pairs or len({p["section_id"] for p in pairs}) != len(pairs):
        raise ValueError("Slice list must contain unique nonempty pairs")
    return sorted(pairs, key=lambda pair: int(pair["section_id"]))


def fetch_sbr(root: Path, workers: int = 8) -> dict:
    """Download only listed image pairs, with Git blob checks and a 50 MB cap.

    The repository contains paired images but does not include the manual
    landmark files described in the paper. Labels and landmarks are not inputs.
    """
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tree_url = (
        f"https://api.github.com/repos/acasamitjana/SynthByReg/git/trees/{SBR_REVISION}?recursive=1"
    )
    cache = root / "publisher_tree.json"
    if cache.exists():
        metadata = json.loads(cache.read_text())
        if metadata["revision"] != SBR_REVISION:
            raise ValueError("Publisher tree revision mismatch")
        tree = metadata["tree"]
    else:
        response = json.loads(_read_url(tree_url))
        if response.get("truncated"):
            raise ValueError("Publisher tree is truncated")
        tree = response["tree"]
        write_json(cache, {"revision": SBR_REVISION, "tree": tree})
    blobs = {r["path"]: r for r in tree if r["type"] == "blob"}
    tables, datasets = {}, {}
    for dataset in DATASETS:
        path = f"data/{dataset}/slice_id.txt"
        table = _read_url(SBR_RAW + path)
        expected = blobs[path]["sha"]
        actual = hashlib.sha1(f"blob {len(table)}\0".encode() + table).hexdigest()
        if actual != expected:
            raise ValueError("Publisher slice list checksum mismatch")
        tables[dataset] = table.decode()
        datasets[dataset] = listed_pairs(tables[dataset], dataset, tree)
    selected = sorted(
        {p[k] for rows in datasets.values() for p in rows for k in ("histology", "mri")}
    )
    if sum(blobs[path]["size"] for path in selected) > 50_000_000:
        raise ValueError("Paired image download exceeds 50 MB")

    def fetch(path):
        blob = blobs[path]
        destination = root / path.removeprefix("data/")
        for attempt in range(3):
            try:
                if not destination.exists():
                    download(SBR_RAW + path, destination, max_bytes=blob["size"])
                contents = destination.read_bytes()
                digest = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
                if digest != blob["sha"] or len(contents) != blob["size"]:
                    raise ValueError(f"Publisher checksum mismatch: {path}")
                return {
                    "path": path,
                    "bytes": len(contents),
                    "git_blob": digest,
                    "sha256": sha256(destination),
                }
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(fetch, selected))
    for dataset, table in tables.items():
        (root / dataset / "slice_id.txt").write_text(table)
    manifest = {
        "source": SBR_REPO,
        "revision": SBR_REVISION,
        "paper": "https://arxiv.org/abs/2107.14449",
        "pairing": "same-donor histology and resampled ex vivo MRI, publisher affine alignment",
        "datasets": datasets,
        "files": records,
        "manual_landmark_files": [r["path"] for r in tree if "landmark" in r["path"].lower()],
        "limits": [
            "Manual landmark annotations described in the paper are absent from this release",
            "Pair IDs provide correspondence and order, not naturally occurring error labels",
            "Published affine-aligned images; not an unknown-pose reconstruction test",
            "Two donors, not hundreds of independent subjects",
        ],
    }
    write_json(root / "sources.json", manifest)
    return manifest


def load_sbr(root: Path, dataset: str, size: int = 96) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a common publisher-aligned field of view; never expose IDs to a detector."""
    if dataset not in DATASETS or size < 16:
        raise ValueError("Unknown dataset or image size below 16")
    root = Path(root)
    manifest = json.loads((root / "sources.json").read_text())
    if manifest["revision"] != SBR_REVISION:
        raise ValueError("Dataset revision mismatch")
    rows = manifest["datasets"][dataset]
    images = {"histology": [], "mri": []}
    for row in rows:
        shapes = []
        for key in images:
            path = root / row[key].removeprefix("data/")
            with Image.open(path) as image:
                shapes.append(image.size)
                images[key].append(
                    np.asarray(
                        image.convert("L").resize((size, size), Image.Resampling.BILINEAR),
                        dtype=np.float32,
                    )
                    / 255
                )
        if shapes[0] != shapes[1]:
            raise ValueError("Paired canvases differ; explicit common-canvas resampling required")
    return np.stack(images["histology"]), np.stack(images["mri"]), [r["section_id"] for r in rows]
