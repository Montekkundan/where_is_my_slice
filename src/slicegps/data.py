"""Small, explicit downloads with byte caps and local provenance."""

import hashlib
import json
import shutil
import tempfile
import urllib.request
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

BIGBRAIN_BASE = (
    "https://bigbrain-ftp.loris.ca/bigbrain-ftp/BigBrainRelease.2015/"
    "3D_Volumes/MNI-ICBM152_Space/nii/"
)
BIGBRAIN_FILES = (
    "full16_400um_2009b_sym.nii.gz",
    "mni_icbm152_t1_tal_nlin_sym_09b_hires.nii.gz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def download(url: str, destination: Path, max_bytes: int) -> dict:
    """Stage a capped HTTPS download, then exclusively create its destination."""
    destination = Path(destination)
    if not url.startswith("https://") or max_bytes <= 0:
        raise ValueError("A HTTPS URL and positive byte cap are required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    request = urllib.request.Request(url, headers={"User-Agent": "SliceGPS/0.1 research"})
    temporary = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > max_bytes:
                raise ValueError(f"Download exceeds byte cap: {url}")
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Download exceeds byte cap: {url}")
                    stream.write(chunk)
        if total == 0:
            raise ValueError(f"Empty response: {url}")
        # Exclusive creation also works on external filesystems without hard links.
        with destination.open("xb") as target:
            try:
                with temporary.open("rb") as source:
                    shutil.copyfileobj(source, target)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        return {
            "url": url,
            "file": destination.name,
            "bytes": total,
            "sha256": sha256(destination),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "checksum_source": "locally computed; not a publisher checksum",
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_bigbrain(root: Path) -> dict:
    root = Path(root)
    manifest_path = root / "sources.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    known = {entry["file"]: entry for entry in previous["files"]}
    records = []
    for name in BIGBRAIN_FILES:
        destination = root / name
        if destination.exists():
            if name not in known or sha256(destination) != known[name]["sha256"]:
                raise ValueError(f"Existing data lack matching provenance: {destination}")
            record = known[name]
        else:
            record = download(BIGBRAIN_BASE + name, destination, max_bytes=200_000_000)
        records.append(record)
        known[name] = record
        # Checkpoint after each file so an interrupted second download is resumable.
        write_json(manifest_path, {"files": [known[n] for n in BIGBRAIN_FILES if n in known]})
    result = {
        "dataset": "BigBrain 2015 release plus ICBM152 T1 template",
        "reference_space": "MNI ICBM152 2009b symmetric",
        "supervision": "registered atlas correspondence, not donor-matched MRI",
        "source": BIGBRAIN_BASE,
        "files": records,
    }
    write_json(manifest_path, result)
    return result


def fetch_nextbrain(root: Path) -> dict:
    """Fetch a pinned 37MB Case4 subset, including complete two-piece slab A5."""
    manifest = json.loads(files("slicegps").joinpath("nextbrain_sample.json").read_text())
    if sum(f["size_bytes"] for f in manifest["files"]) > 50_000_000:
        raise ValueError("Sample manifest exceeds total byte cap")
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or record["size_bytes"] > 20_000_000:
            raise ValueError("Unsafe sample manifest path or file size")
        destination = root / relative
        if not destination.exists():
            download(record["url"], destination, max_bytes=record["size_bytes"])
        if (
            destination.stat().st_size != record["size_bytes"]
            or sha256(destination) != record["sha256"]
        ):
            raise ValueError(f"Sample checksum mismatch: {destination}")
    manifest["checksum_source"] = "locally computed from pinned GitHub revision"
    write_json(root / "sources.json", manifest)
    return manifest
