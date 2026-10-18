"""Persist image-only reference features and inspect matches against MRI planes.

Geometry is attached after ranking. Cosine similarity is not a calibrated
confidence, and sorting by MRI y assumes a roughly coronal section stack.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from .data import write_json
from .geometry import Plane
from .pipeline import read_bundle
from .retrieval import _prepare, _unit_rows, features, order_stack, rank

_METHODS = {"intensity", "gradient", "mask", "dino"}
_IDENTITY_FIELDS = ("feature_fingerprint", "model_id", "model_revision", "preprocessing", "pooling")


def _empty_output(output: Path) -> Path:
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    return output


def _images(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in "fiu" or values.ndim != 3:
        raise ValueError(f"{name} must be real grayscale images with shape (N, H, W)")
    if values.shape[0] == 0 or min(values.shape[1:]) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain nonempty, finite images at least 2 by 2")
    return values


def _metadata(encoder, pooling: str) -> dict:
    value = getattr(encoder, "metadata", None)
    if not isinstance(value, Mapping):
        raise ValueError("DINO encoder must expose its model metadata")
    metadata = dict(value)
    for field in _IDENTITY_FIELDS:
        if field not in metadata:
            raise ValueError(f"DINO metadata is missing {field}")
    for field in ("feature_fingerprint", "model_id", "model_revision"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"DINO metadata {field} must be a nonempty string")
    if not isinstance(metadata["preprocessing"], dict) or metadata["pooling"] != pooling:
        raise ValueError("DINO preprocessing/pooling metadata mismatch")
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("DINO metadata must be finite JSON data") from error


def _same_encoder(expected: dict, actual: dict) -> None:
    if any(expected.get(key) != actual.get(key) for key in _IDENTITY_FIELDS):
        raise ValueError("DINO model/preprocessing/pooling metadata mismatch")


def _dual_model(path: Path, pooling: str, encoder_metadata: dict, width: int):
    """Validate exact model provenance before either modality is projected."""
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    if (
        not isinstance(sidecar, dict)
        or sidecar.get("schema_version") != 1
        or sidecar.get("kind") != "dual_contrastive_pose"
        or sidecar.get("pooling") != pooling
        or not isinstance(sidecar.get("encoder"), dict)
    ):
        raise ValueError("Dual model provenance metadata mismatch")
    _same_encoder(encoder_metadata, sidecar["encoder"])
    architecture = sidecar.get("architecture", {})
    if (
        architecture.get("histology_input_dim") != width
        or architecture.get("mri_input_dim") != width
        or not isinstance(architecture.get("embedding_dim"), int)
        or architecture["embedding_dim"] < 1
    ):
        raise ValueError("Dual model input/output feature dimensions mismatch")
    weights_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if sidecar.get("weights_sha256") != weights_hash:
        raise ValueError("Dual model weights hash mismatch")
    from .training import DualMatcher

    matcher = DualMatcher.load(path)
    identity = {
        "weights_sha256": weights_hash,
        "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        "input_feature_dim": width,
        "embedding_dim": architecture["embedding_dim"],
        "path": str(path.resolve()),
    }
    return matcher, identity


def _same_dual(expected: dict, actual: dict) -> None:
    keys = ("weights_sha256", "sidecar_sha256", "input_feature_dim", "embedding_dim")
    if any(expected.get(key) != actual.get(key) for key in keys):
        raise ValueError("Dual model/index weights or sidecar hash mismatch")


def _projected(values, count: int, width: int) -> np.ndarray:
    normalized = _unit_rows(values, "dual model features")
    if normalized.shape != (count, width):
        raise ValueError("Dual model projected feature dimensions mismatch")
    return normalized.astype(np.float32)


def _encoder_features(images: list[np.ndarray], encoder, pooling: str) -> np.ndarray:
    # Group equal image grids for batching; paths and geometry never enter encode.
    groups: dict[tuple[int, int], list[int]] = {}
    for index, image in enumerate(images):
        groups.setdefault(image.shape, []).append(index)
    rows: list[np.ndarray | None] = [None] * len(images)
    width = None
    for positions in groups.values():
        encoded = np.asarray(encoder.encode(np.stack([images[i] for i in positions]), pooling))
        if encoded.dtype.kind not in "fiu":
            raise ValueError("DINO features must be real numeric values")
        encoded = _unit_rows(encoded, "DINO features")
        if len(encoded) != len(positions) or (width is not None and encoded.shape[1] != width):
            raise ValueError("DINO features must have one consistent descriptor per image")
        width = encoded.shape[1]
        for position, descriptor in zip(positions, encoded, strict=True):
            rows[position] = descriptor
    return np.stack(rows).astype(np.float32)


def _extract(images: list[np.ndarray], method: str, encoder, pooling: str) -> np.ndarray:
    if method == "dino":
        return _encoder_features(images, encoder, pooling)
    return np.concatenate([features(image[None], method=method) for image in images])


def _display(image: np.ndarray) -> np.ndarray:
    scaled, _ = _prepare(np.asarray(image, dtype=np.float64))
    return np.rint(scaled * 255).astype(np.uint8)


def build_index(
    bundle: Path,
    output: Path,
    method: str = "gradient",
    encoder=None,
    pooling: str = "cls",
    model: Path | None = None,
) -> dict:
    """Index a bundle's MRI pixels and retain ordered candidate plane geometry.

    DINO dependencies and weights are loaded only when the dino method is chosen;
    callers can inject an offline encoder exposing encode and metadata.
    """
    output = _empty_output(output)
    if method not in _METHODS:
        raise ValueError("method must be intensity, gradient, mask, or dino")
    if model is not None and method != "dino":
        raise ValueError("A dual model requires a DINO index")
    if method == "dino" and pooling not in {"cls", "spatial"}:
        raise ValueError("DINO pooling must be cls or spatial")
    manifest, arrays = read_bundle(Path(bundle))
    if "gallery_mri" not in arrays:
        raise ValueError("Bundle is missing gallery_mri")
    gallery = _images(arrays["gallery_mri"], "gallery_mri")
    planes = manifest.get("gallery_planes")
    if not isinstance(planes, list) or len(planes) != len(gallery):
        raise ValueError("gallery_planes must match gallery_mri image count")
    planes = [Plane.from_dict(plane).to_dict() for plane in planes]
    if any(tuple(plane["shape"]) != gallery.shape[1:] for plane in planes):
        raise ValueError("gallery plane shapes must match the image grid")
    if method == "dino" and encoder is None:
        from .encoders import DinoEncoder

        encoder = DinoEncoder(output.parent / "model-cache")
    descriptors = _extract(list(gallery), method, encoder, pooling)
    encoder_metadata = _metadata(encoder, pooling) if method == "dino" else None
    model_identity = None
    if model is not None:
        matcher, model_identity = _dual_model(
            model, pooling, encoder_metadata, descriptors.shape[1]
        )
        descriptors = _projected(
            matcher.encode_gallery(descriptors), len(gallery), model_identity["embedding_dim"]
        )
    result = {
        "schema_version": 1,
        "method": method,
        "pooling": pooling if method == "dino" else None,
        "feature_dim": descriptors.shape[1],
        "feature_spec": {"version": 1, "size": 32, "normalization": "L2 per image"},
        "encoder": encoder_metadata,
        "model": model_identity,
        "gallery_shape": list(gallery.shape),
        "gallery_planes": planes,
        "reference_space": manifest.get("reference_space", "unspecified"),
        "source_bundle": str(Path(bundle).resolve()),
        "metric": "cosine similarity; not calibrated confidence",
        "image_orientation": "array row order; no display-only flips",
    }
    displays = np.stack([_display(image) for image in gallery])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "features.npz",
        features=descriptors,
        gallery_mri=displays,
        method=np.asarray(method),
    )
    write_json(output / "manifest.json", result)
    return result


def _load_index(index: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    metadata = json.loads((index / "manifest.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported index schema_version")
    method = metadata.get("method")
    if method not in _METHODS:
        raise ValueError("Unsupported index method")
    expected_spec = {"version": 1, "size": 32, "normalization": "L2 per image"}
    if metadata.get("feature_spec") != expected_spec:
        raise ValueError("Index feature metadata mismatch")
    if method == "dino":
        if metadata.get("pooling") not in {"cls", "spatial"}:
            raise ValueError("Invalid DINO index pooling")
        if not isinstance(metadata.get("encoder"), dict):
            raise ValueError("DINO index requires encoder metadata")
    elif metadata.get("encoder") is not None or metadata.get("pooling") is not None:
        raise ValueError("Baseline index method metadata mismatch")
    with np.load(index / "features.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"features", "gallery_mri", "method"}:
            raise ValueError("Index archive fields do not match schema")
        if archive["method"].shape != () or archive["method"].item() != method:
            raise ValueError("Index archive/manifest method metadata mismatch")
        raw = archive["features"]
        if raw.dtype.kind not in "fiu":
            raise ValueError("Index features must be real numeric values")
        descriptors = _unit_rows(raw, "index features")
        gallery = _images(archive["gallery_mri"], "index gallery_mri")
    dim = metadata.get("feature_dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim != descriptors.shape[1]:
        raise ValueError("Index feature dimension metadata mismatch")
    dual = metadata.get("model")
    if dual is not None:
        if not isinstance(dual, dict) or method != "dino":
            raise ValueError("A dual model requires DINO index metadata")
        for key in ("weights_sha256", "sidecar_sha256"):
            digest = dual.get(key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("Invalid dual model hash metadata")
            if any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("Invalid dual model hash metadata")
        if (
            not isinstance(dual.get("input_feature_dim"), int)
            or dual["input_feature_dim"] < 1
            or dual.get("embedding_dim") != dim
        ):
            raise ValueError("Dual model/index feature dimension mismatch")
    baseline_width = {"intensity": 1024, "gradient": 3072, "mask": 1024}
    if method != "dino" and dim != baseline_width[method]:
        raise ValueError("Index feature method/dimension mismatch")
    if metadata.get("gallery_shape") != list(gallery.shape) or len(descriptors) != len(gallery):
        raise ValueError("Index features/images must match gallery shape metadata")
    planes = metadata.get("gallery_planes")
    if not isinstance(planes, list) or len(planes) != len(gallery):
        raise ValueError("Index gallery_planes must match image count")
    metadata["gallery_planes"] = [Plane.from_dict(plane).to_dict() for plane in planes]
    if any(tuple(plane["shape"]) != gallery.shape[1:] for plane in metadata["gallery_planes"]):
        raise ValueError("Index plane shapes must match image grids")
    return metadata, descriptors, gallery


def _thumbnail(image: np.ndarray) -> str:
    display = Image.fromarray(_display(image))
    display.thumbnail((256, 256))
    stream = io.BytesIO()
    display.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _reader(result: dict, queries: list[np.ndarray], gallery: np.ndarray) -> str:
    articles = []
    for match, image in zip(result["matches"], queries, strict=True):
        candidates = []
        for candidate in match["candidates"][:3]:
            plane = candidate["plane"]
            pose = html.escape(json.dumps({key: plane[key] for key in ("u", "v")}))
            center = ", ".join(f"{value:.2f}" for value in plane["center_mm"])
            candidates.append(
                '<figure><img alt="Candidate MRI" src="'
                + _thumbnail(gallery[candidate["gallery_index"]])
                + '"><figcaption>MRI candidate '
                + str(candidate["gallery_index"])
                + f"<br>Cosine: {candidate['cosine_similarity']:.4f}"
                + f"<br>Center RAS (mm): {center}<br>Pose: <code>{pose}</code>"
                + "</figcaption></figure>"
            )
        pose_text = ""
        if "predicted_pose" in match:
            pose = match["predicted_pose"]
            center = ", ".join(f"{value:.2f}" for value in pose["center_mm"])
            normal = ", ".join(f"{value:.4f}" for value in pose["normal_ras"])
            pose_text = (
                f"<p>Regressed center RAS (mm): {center}<br>Regressed unit normal: {normal}<br>"
                + html.escape(pose["note"])
                + "</p>"
            )
        articles.append(
            "<article><h2>Section "
            + str(match["query_index"] + 1)
            + ": "
            + html.escape(match["query_name"])
            + "</h2>"
            + pose_text
            + '<div class="row"><figure><img alt="Input section" src="'
            + _thumbnail(image)
            + '"><figcaption>Input section</figcaption></figure>'
            + "".join(candidates)
            + "</div></article>"
        )
    ordered = " → ".join(
        html.escape(result["matches"][i]["query_name"]) for i in result["suggested_order"]
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "img-src data:; style-src 'unsafe-inline'\">"
        "<title>SliceGPS section matches</title><style>"
        "body{font:16px system-ui;margin:2rem;background:#141820;color:#f4f5f7}"
        "article{border-top:1px solid #55606d;margin-top:2rem}.row{display:flex;flex-wrap:wrap}"
        "figure{margin:1rem;max-width:260px}img{max-width:256px;max-height:256px}"
        "figcaption{line-height:1.5;overflow-wrap:anywhere}code{font-size:12px}"
        "</style><main><h1>Inspect section matches</h1><p>"
        "Input sections remain in supplied order. Candidates come from image pixels only. "
        "Cosine similarity is not confidence. This prototype does not infer crop placement "
        "or cutting direction and can return an incorrect match.</p><p><strong>"
        "Suggested order along MRI y (ascending):</strong> "
        + ordered
        + "</p><p>Ordering assumes a roughly coronal stack in the indexed reference space.</p>"
        + "".join(articles)
        + "</main></html>"
    )


def match_images(
    index: Path,
    queries: list[Path],
    output: Path,
    adapter: Path | None = None,
    encoder=None,
    k: int = 5,
    model: Path | None = None,
) -> dict:
    """Rank PNG/JPEG/TIFF pixels, preserving single-band numeric TIFF values."""
    output = _empty_output(output)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if not queries:
        raise ValueError("Provide at least one query image")
    index = Path(index)
    manifest, gallery_features, gallery = _load_index(index)
    method, pooling = manifest["method"], manifest["pooling"]
    if adapter is not None and model is not None:
        raise ValueError("Ridge adapter and dual model are mutually exclusive")
    dual_identity = manifest.get("model")
    if (dual_identity is None) != (model is None):
        raise ValueError("Dual model must be supplied exactly when the index uses that model")
    matcher = None
    if model is not None:
        matcher, actual_identity = _dual_model(
            model, pooling, manifest["encoder"], dual_identity["input_feature_dim"]
        )
        _same_dual(dual_identity, actual_identity)
    if adapter is not None and method != "dino":
        raise ValueError("An adapter requires a DINO index")
    query_paths, query_images = [Path(path) for path in queries], []
    for path in query_paths:
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG", "TIFF"}:
                raise ValueError("Query images must be PNG, JPEG or TIFF")
            # Numeric grayscale is inference data, not a display conversion. In
            # particular, F TIFF retains the original float32 dynamic range.
            numeric_gray = image.mode in {"1", "L", "F", "I"} or image.mode.startswith("I;16")
            grayscale = image if numeric_gray else image.convert("L")
            pixels = np.asarray(grayscale, dtype=np.float64)
        _images(pixels[None], "query image")
        # Reject blank sections consistently even for injected encoders.
        _prepare(pixels)
        query_images.append(pixels)
    if method == "dino" and encoder is None:
        from .encoders import DinoEncoder

        encoder = DinoEncoder(index.parent / "model-cache")
    query_features = _extract(query_images, method, encoder, pooling or "cls")
    if method == "dino":
        _same_encoder(manifest["encoder"], _metadata(encoder, pooling))
    if adapter is not None:
        from .adapters import RidgeAdapter

        adapter = Path(adapter)
        sidecar = json.loads(adapter.with_suffix(".json").read_text())
        if (
            not isinstance(sidecar, dict)
            or sidecar.get("schema_version") != 1
            or sidecar.get("pooling") != pooling
            or sidecar.get("feature_dim") != manifest["feature_dim"]
            or not isinstance(sidecar.get("encoder"), dict)
        ):
            raise ValueError("Adapter provenance metadata mismatch")
        _same_encoder(manifest["encoder"], sidecar["encoder"])
        query_features = RidgeAdapter.load(adapter).transform(query_features)
    predicted_pose = None
    if matcher is not None:
        predicted_pose = matcher.predict_pose(query_features)
        centers = np.asarray(predicted_pose["centers_mm"], dtype=float)
        normals = np.asarray(predicted_pose["normals_ras"], dtype=float)
        if (
            centers.shape != (len(query_images), 3)
            or normals.shape != centers.shape
            or not np.isfinite(centers).all()
            or not np.isfinite(normals).all()
            or np.any(np.linalg.norm(normals, axis=1) < 1e-8)
        ):
            raise ValueError("Dual model produced invalid pose predictions")
        predicted_pose = {
            "centers_mm": centers,
            "normals_ras": normals / np.linalg.norm(normals, axis=1, keepdims=True),
        }
        query_features = _projected(
            matcher.transform(query_features), len(query_images), dual_identity["embedding_dim"]
        )
    indices, scores = rank(query_features, gallery_features, k=int(k))
    matches = []
    for query_index, path in enumerate(query_paths):
        matches.append(
            {
                "query_index": query_index,
                "query_path": str(path.resolve()),
                "query_name": path.name,
                "candidates": [
                    {
                        "gallery_index": int(gallery_index),
                        "cosine_similarity": float(score),
                        "plane": manifest["gallery_planes"][gallery_index],
                    }
                    for gallery_index, score in zip(
                        indices[query_index], scores[query_index], strict=True
                    )
                ],
            }
        )
    if predicted_pose is not None:
        for i, match in enumerate(matches):
            match["predicted_pose"] = {
                "center_mm": predicted_pose["centers_mm"][i].tolist(),
                "normal_ras": predicted_pose["normals_ras"][i].tolist(),
                "note": "Exploratory regression; excludes in-plane rotation; no donor validation",
            }
    centers = np.array([match["candidates"][0]["plane"]["center_mm"] for match in matches])
    result = {
        "schema_version": 1,
        "method": method,
        "pooling": pooling,
        "encoder": manifest["encoder"],
        "adapter": str(adapter.resolve()) if adapter is not None else None,
        "model": dual_identity,
        "index": str(index.resolve()),
        "reference_space": manifest["reference_space"],
        "metric": "cosine similarity; not calibrated confidence",
        "matches": matches,
        "suggested_order": order_stack(centers, (0, 1, 0)).tolist(),
        "ordering": "ascending MRI RAS y; assumes a roughly coronal stack",
    }
    reader = _reader(result, query_images, gallery)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "matches.json", result)
    (output / "index.html").write_text(reader, encoding="utf-8")
    return result
