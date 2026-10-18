"""Generate held-out query renderings and an explicitly fixed reference gallery."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from .data import write_json
from .geometry import coronal_plane, sample_plane
from .retrieval import features
from .volume import audit_volume, load_volume, save_volume


def display_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def contact_sheet(images: list[np.ndarray], labels: list[str], output: Path) -> None:
    if len(images) != len(labels) or not images:
        raise ValueError("Provide an image and label for each tile")
    width, height = 256, 282
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), "#141820")
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(images, labels, strict=True)):
        tile = Image.fromarray(np.flipud(display_uint8(array))).convert("RGB")
        tile.thumbnail((256, 256))
        x, y = index % columns * width, index // columns * height
        canvas.paste(tile, (x, y))
        draw.text((x + 6, y + 259), label, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def create_phantom(root: Path) -> tuple[Path, Path]:
    """An asymmetric analytic fixture, clearly separated from human-brain evidence."""
    root.mkdir(parents=True, exist_ok=True)
    coords = np.indices((72, 88, 64), dtype=np.float32)
    x, y, z = coords
    tissue = ((x - 36) / 31) ** 2 + ((y - 44) / 39) ** 2 + ((z - 32) / 27) ** 2 < 1
    texture = (
        0.3 + 0.25 * np.sin(x / 4 + y / 13) ** 2 + 0.3 * np.cos(z / 5 - y / 17) ** 2 + 0.15 * x / 72
    )
    blob = np.exp(-((x - 24) ** 2 + (y - 28) ** 2 + (z - 38) ** 2) / 90)
    histology = tissue * (texture + blob)
    mri = gaussian_filter(tissue * (1.4 - texture + blob * 0.5), 0.8)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-36, -44, -32]
    histology_path, mri_path = root / "phantom_histology.nii.gz", root / "phantom_mri.nii.gz"
    save_volume(histology_path, histology, affine)
    save_volume(mri_path, mri, affine)
    return histology_path, mri_path


def generate(
    histology_path: Path,
    mri_path: Path,
    output: Path,
    count: int = 32,
    size: int = 128,
    pixel_mm: float = 1.5,
    tilt_deg: float = 3,
    seed: int = 17,
    reference_space: str = "unspecified; caller must verify registration",
    assume_mm: bool = False,
    augment: bool = False,
) -> dict:
    if count < 4 or size < 8 or pixel_mm <= 0 or not np.isfinite(pixel_mm):
        raise ValueError("Require count >=4, size >=8 and positive pixel size")
    if not np.isfinite(tilt_deg) or not 0 <= tilt_deg <= 15:
        raise ValueError("This first benchmark accepts tilt between 0 and 15 degrees")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    hist_audit, mri_audit = audit_volume(histology_path), audit_volume(mri_path)
    for audit in (hist_audit, mri_audit):
        units = audit["spatial_units"]
        if units != "mm" and not (units == "unknown" and assume_mm):
            raise ValueError("Volume units must be mm; use assume_mm only with dataset evidence")
    hist, ah = load_volume(histology_path)
    mri, am = load_volume(mri_path)
    lower = np.maximum(hist_audit["world_bounds_mm"][0], mri_audit["world_bounds_mm"][0])
    upper = np.minimum(hist_audit["world_bounds_mm"][1], mri_audit["world_bounds_mm"][1])
    if np.any(upper <= lower):
        raise ValueError("The volumes have no common world-coordinate bounds")
    center = (lower + upper) / 2
    depths = np.linspace(lower[1] * 0.8 + upper[1] * 0.2, lower[1] * 0.2 + upper[1] * 0.8, count)
    step = float(depths[1] - depths[0])
    rng = np.random.default_rng(seed)
    gallery_h, gallery_m, queries, masks, planes, query_planes = [], [], [], [], [], []
    exclusions = []
    for depth in depths:
        c = (float(center[0]), float(depth), float(center[2]))
        plane = coronal_plane(c, shape=(size, size), spacing_mm=(pixel_mm, pixel_mm))
        h, vh = sample_plane(hist, ah, plane)
        m, vm = sample_plane(mri, am, plane)
        if np.count_nonzero((h != 0) & vh & vm) < size * size * 0.01:
            exclusions.append({"depth_mm": float(depth), "reason": "insufficient tissue/coverage"})
            continue
        qcenter = (c[0], c[1] + float(rng.uniform(-0.35, 0.35) * step), c[2])
        qplane = coronal_plane(
            qcenter,
            shape=(size, size),
            spacing_mm=(pixel_mm, pixel_mm),
            tilt_deg=tuple(rng.uniform(-tilt_deg, tilt_deg, size=2)),
        )
        query, valid = sample_plane(hist, ah, qplane)
        # Render a different plane and add a modest blur, never an exact gallery PNG.
        query = gaussian_filter(query, sigma=0.5)
        query[~valid] = 0
        try:
            if augment:
                from .augmentation import augment_histology

                query = augment_histology(query, rng)
            for method in ("intensity", "gradient", "mask"):
                features(np.stack((h, m, query)), method)
        except ValueError as exc:
            exclusions.append({"depth_mm": float(depth), "reason": str(exc)})
            continue
        gallery_h.append(h)
        gallery_m.append(m)
        queries.append(query)
        masks.append(vh & vm)
        planes.append(plane.to_dict())
        query_planes.append(qplane.to_dict())
    if len(planes) < 4:
        raise ValueError("Too few tissue-containing planes; check alignment and field of view")
    np.savez_compressed(
        output / "images.npz",
        gallery_histology=np.stack(gallery_h),
        gallery_mri=np.stack(gallery_m),
        query_histology=np.stack(queries),
        gallery_valid=np.stack(masks),
    )
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "reference_space": reference_space,
        "assume_mm_for_unspecified_units": assume_mm,
        "histology": hist_audit,
        "mri": mri_audit,
        "gallery_planes": planes,
        "query_planes": query_planes,
        "count": len(planes),
        "nominal_gallery_spacing_mm": step,
        "max_query_tilt_deg_per_axis": tilt_deg,
        "query_augmentations": (
            "gamma/gain, blur, synthetic tear and missing chunk" if augment else "blur only"
        ),
        "retained_depth_gaps_mm": np.diff([p["center_mm"][1] for p in planes]).tolist(),
        "exclusions": exclusions,
        "evaluation_scope": "same-volume synthetic queries; no cross-donor validation",
        "learning": "no training or fitted preprocessing; fixed reference gallery",
        "warning": "Header overlap does not prove anatomical correspondence; review overlays.",
    }
    write_json(output / "manifest.json", manifest)
    selected = np.linspace(0, len(planes) - 1, 4, dtype=int)
    contact_sheet(
        [im for i in selected for im in (gallery_h[i], gallery_m[i], queries[i])],
        [label for i in selected for label in (f"Histology {i}", f"MRI {i}", f"Query {i}")],
        output / "paired_planes.png",
    )
    return manifest


def read_bundle(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((root / "manifest.json").read_text())
    with np.load(root / "images.npz", allow_pickle=False) as bundle:
        images = {key: bundle[key] for key in bundle.files}
    return manifest, images
