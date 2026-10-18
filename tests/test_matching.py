"""Offline tests for persisted image-only retrieval and its static reader."""

import json

import numpy as np
import pytest
from PIL import Image

from slicegps.adapters import fit_ridge
from slicegps.geometry import coronal_plane
from slicegps.matching import build_index, match_images


def make_bundle(tmp_path):
    images = np.zeros((3, 32, 32), dtype=np.uint8)
    images[0, 4:12, 6:22] = 180
    images[1, 15:26, 2:10] = 200
    images[2, 9:27, 20:29] = 240
    root = tmp_path / "bundle"
    root.mkdir()
    np.savez_compressed(root / "images.npz", gallery_mri=images)
    planes = [coronal_plane((0, y, 0), (32, 32)).to_dict() for y in (8, -4, 2)]
    (root / "manifest.json").write_text(
        json.dumps({"gallery_planes": planes, "reference_space": "test phantom"})
    )
    return root, images


def query_files(tmp_path, images, order=(2, 0, 1)):
    paths = []
    for number, position in enumerate(order):
        path = tmp_path / f"section-{number}.png"
        Image.fromarray(images[position]).save(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("method", ["intensity", "gradient", "mask"])
def test_pixels_rank_shuffled_queries_and_attach_candidate_geometry(tmp_path, method):
    bundle, images = make_bundle(tmp_path)
    index = tmp_path / "index"
    build_index(bundle, index, method)
    queries = query_files(tmp_path, images)
    result = match_images(index, queries, tmp_path / "matches", k=2)
    assert [row["candidates"][0]["gallery_index"] for row in result["matches"]] == [2, 0, 1]
    assert result["suggested_order"] == [2, 0, 1]
    assert all(len(row["candidates"]) == 2 for row in result["matches"])
    assert all(row["candidates"][0]["cosine_similarity"] > 0.999 for row in result["matches"])
    assert json.loads((tmp_path / "matches/matches.json").read_text()) == result
    # Falsifying geometry changes ordering, never image-feature rankings.
    manifest = json.loads((index / "manifest.json").read_text())
    for plane in manifest["gallery_planes"]:
        plane["center_mm"][1] *= -1
    (index / "manifest.json").write_text(json.dumps(manifest))
    changed = match_images(index, queries, tmp_path / "other")
    assert [row["candidates"][0]["gallery_index"] for row in changed["matches"]] == [2, 0, 1]
    assert changed["suggested_order"] == [1, 0, 2]


def test_static_reader_embeds_images_and_escapes_query_names(tmp_path):
    bundle, images = make_bundle(tmp_path)
    build_index(bundle, tmp_path / "index")
    query = tmp_path / '<img src=x onerror="bad">.png'
    Image.fromarray(images[0]).save(query)
    match_images(tmp_path / "index", [query], tmp_path / "matches")
    page = (tmp_path / "matches/index.html").read_text()
    assert "&lt;img src=x onerror=&quot;bad&quot;&gt;.png" in page
    assert '<img src=x onerror="bad">' not in page
    assert "data:image/png;base64," in page
    assert "<script" not in page
    assert "Cosine similarity is not confidence" in page
    assert "Center RAS (mm)" in page


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_reject_invalid_k_before_creating_output(tmp_path, k):
    with pytest.raises(ValueError, match="positive integer"):
        match_images(tmp_path / "missing", [tmp_path / "missing.png"], tmp_path / "out", k=k)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["nan", "shape", "planes", "method", "dimension"])
def test_reject_invalid_serialized_indexes(tmp_path, mutation):
    bundle, images = make_bundle(tmp_path)
    index = tmp_path / "index"
    build_index(bundle, index)
    if mutation in {"nan", "shape"}:
        with np.load(index / "features.npz") as archive:
            descriptors, gallery = archive["features"], archive["gallery_mri"]
        if mutation == "nan":
            descriptors[0, 0] = np.nan
        else:
            gallery = gallery[:2]
        np.savez_compressed(
            index / "features.npz",
            features=descriptors,
            gallery_mri=gallery,
            method=np.asarray("gradient"),
        )
    else:
        manifest = json.loads((index / "manifest.json").read_text())
        if mutation == "planes":
            manifest["gallery_planes"] = []
        elif mutation == "method":
            manifest["method"] = "intensity"
        else:
            manifest["feature_dim"] = 9
        (index / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        match_images(index, query_files(tmp_path, images), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_output_must_be_new_or_empty_and_constant_query_rejected(tmp_path):
    bundle, images = make_bundle(tmp_path)
    with pytest.raises(FileExistsError):
        build_index(bundle, bundle)
    index = tmp_path / "index"
    build_index(bundle, index)
    with pytest.raises(FileExistsError):
        build_index(bundle, index)
    constant = tmp_path / "blank.png"
    Image.fromarray(np.zeros((12, 12), dtype=np.uint8)).save(constant)
    with pytest.raises(ValueError, match="nonconstant"):
        match_images(index, [constant], tmp_path / "out")
    with pytest.raises(ValueError, match="DINO"):
        match_images(index, query_files(tmp_path, images), tmp_path / "out", adapter=tmp_path / "a")


def test_same_width_method_change_is_rejected_and_jpeg_is_readable(tmp_path):
    bundle, images = make_bundle(tmp_path)
    index = tmp_path / "index"
    build_index(bundle, index, "intensity")
    query = tmp_path / "scanned-section.jpg"
    Image.fromarray(images[2]).save(query, quality=98)
    result = match_images(index, [query], tmp_path / "jpeg")
    assert result["matches"][0]["candidates"][0]["gallery_index"] == 2
    manifest = json.loads((index / "manifest.json").read_text())
    manifest["method"] = "mask"
    (index / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="method metadata mismatch"):
        match_images(index, [query], tmp_path / "wrong-method")


class OfflineEncoder:
    def __init__(self, fingerprint="offline1"):
        self.fingerprint = fingerprint
        self.pooling = None

    @property
    def metadata(self):
        return {
            "feature_fingerprint": self.fingerprint,
            "model_id": "offline-image-fixture",
            "model_revision": "test1",
            "preprocessing": {"pixels": "binary foreground regional sums"},
            "pooling": self.pooling,
        }

    def encode(self, images, pooling="cls"):
        self.pooling = pooling
        return np.array(
            [[im[:16, :16].sum(), im[16:, :16].sum(), im[:, 16:].sum()] for im in images],
            dtype=np.float32,
        )


def test_dino_injection_is_offline_and_checks_encoder_identity(tmp_path):
    bundle, images = make_bundle(tmp_path)
    index = tmp_path / "index"
    metadata = build_index(bundle, index, "dino", encoder=OfflineEncoder())
    assert metadata["feature_dim"] == 3
    queries = query_files(tmp_path, images)
    result = match_images(index, queries, tmp_path / "matches", encoder=OfflineEncoder())
    assert [r["candidates"][0]["gallery_index"] for r in result["matches"]] == [2, 0, 1]
    with pytest.raises(ValueError, match="metadata mismatch"):
        match_images(index, queries, tmp_path / "bad", encoder=OfflineEncoder("other"))


def test_adapter_requires_matching_model_sidecar_before_transform(tmp_path):
    bundle, images = make_bundle(tmp_path)
    index = tmp_path / "index"
    metadata = build_index(bundle, index, "dino", encoder=OfflineEncoder())
    query_encoder = OfflineEncoder()
    training = query_encoder.encode(images)
    adapter = tmp_path / "adapter.npz"
    fit_ridge(training, training).save(adapter)
    queries = query_files(tmp_path, images)
    with pytest.raises(FileNotFoundError):
        match_images(index, queries, tmp_path / "no-sidecar", adapter, OfflineEncoder())
    sidecar = {
        "schema_version": 1,
        "encoder": metadata["encoder"],
        "pooling": "cls",
        "feature_dim": 3,
    }
    adapter.with_suffix(".json").write_text(json.dumps(sidecar))
    result = match_images(index, queries, tmp_path / "good", adapter, OfflineEncoder())
    assert result["adapter"] == str(adapter.resolve())
    sidecar["encoder"]["model_revision"] = "wrong"
    adapter.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="metadata mismatch"):
        match_images(index, queries, tmp_path / "mismatch", adapter, OfflineEncoder())


class OfflineDualMatcher:
    """Different feature width ensures both heads must run for compatible ranking."""

    def __init__(self):
        self.calls = []

    def _project(self, values):
        values = np.asarray(values)
        projected = np.column_stack((values[:, 2], values[:, 0], values[:, 1], values[:, 0] * 0.5))
        return projected / np.linalg.norm(projected, axis=1, keepdims=True)

    def transform(self, values):
        self.calls.append(("histology", len(values)))
        return self._project(values)

    def encode_gallery(self, values):
        self.calls.append(("mri", len(values)))
        return self._project(values)

    def predict_pose(self, values):
        self.calls.append(("pose", len(values)))
        return {
            "centers_mm": np.asarray(values) * 10,
            "normals_ras": np.tile((0, -1, 0), (len(values), 1)),
        }


def offline_dual(tmp_path, monkeypatch, images):
    import hashlib

    from slicegps.training import DualMatcher

    encoder = OfflineEncoder()
    encoder.encode(images)
    path = tmp_path / "dual.safetensors"
    path.write_bytes(b"Offline model stub; the loader is injected and downloads nothing")
    metadata = {
        "schema_version": 1,
        "kind": "dual_contrastive_pose",
        "pooling": "cls",
        "encoder": encoder.metadata,
        "architecture": {"histology_input_dim": 3, "mri_input_dim": 3, "embedding_dim": 4},
        "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    matcher = OfflineDualMatcher()
    monkeypatch.setattr(DualMatcher, "load", lambda _: matcher)
    return path, matcher


def test_dual_model_projects_both_sides_and_attaches_predicted_pose(tmp_path, monkeypatch):
    bundle, images = make_bundle(tmp_path)
    model, matcher = offline_dual(tmp_path, monkeypatch, images)
    index = tmp_path / "index"
    metadata = build_index(bundle, index, "dino", OfflineEncoder(), "cls", model)
    assert metadata["feature_dim"] == 4
    assert metadata["model"]["input_feature_dim"] == 3
    assert len(metadata["model"]["weights_sha256"]) == 64
    assert len(metadata["model"]["sidecar_sha256"]) == 64
    result = match_images(
        index,
        query_files(tmp_path, images),
        tmp_path / "matched",
        encoder=OfflineEncoder(),
        model=model,
    )
    assert [row["candidates"][0]["gallery_index"] for row in result["matches"]] == [2, 0, 1]
    assert matcher.calls == [("mri", 3), ("pose", 3), ("histology", 3)]
    for row in result["matches"]:
        assert len(row["predicted_pose"]["center_mm"]) == 3
        assert row["predicted_pose"]["normal_ras"] == [0, -1, 0]
    assert result["model"] == metadata["model"]
    assert result["adapter"] is None
    reader = (tmp_path / "matched/index.html").read_text()
    assert "Regressed center RAS (mm)" in reader
    assert "no donor validation" in reader
    assert json.loads((tmp_path / "matched/matches.json").read_text()) == result


def test_raw_and_dual_indexes_require_matching_model_presence(tmp_path, monkeypatch):
    bundle, images = make_bundle(tmp_path)
    model, _ = offline_dual(tmp_path, monkeypatch, images)
    queries = query_files(tmp_path, images)
    raw, projected = tmp_path / "raw", tmp_path / "dual"
    build_index(bundle, raw, "dino", OfflineEncoder())
    build_index(bundle, projected, "dino", OfflineEncoder(), model=model)
    with pytest.raises(ValueError, match="exactly when"):
        match_images(raw, queries, tmp_path / "invalid", encoder=OfflineEncoder(), model=model)
    with pytest.raises(ValueError, match="exactly when"):
        match_images(projected, queries, tmp_path / "invalid", encoder=OfflineEncoder())
    with pytest.raises(ValueError, match="mutually exclusive"):
        match_images(
            projected,
            queries,
            tmp_path / "invalid",
            adapter=tmp_path / "ridge.npz",
            encoder=OfflineEncoder(),
            model=model,
        )
    with pytest.raises(ValueError, match="DINO"):
        build_index(bundle, tmp_path / "baseline", "gradient", model=model)


@pytest.mark.parametrize("mutation", ["weights", "sidecar", "encoder", "pooling"])
def test_dual_index_rejects_changed_model_or_provenance(tmp_path, monkeypatch, mutation):
    bundle, images = make_bundle(tmp_path)
    model, _ = offline_dual(tmp_path, monkeypatch, images)
    index = tmp_path / "index"
    build_index(bundle, index, "dino", OfflineEncoder(), model=model)
    if mutation == "weights":
        model.write_bytes(model.read_bytes() + b"changed")
    elif mutation == "sidecar":
        sidecar = model.with_suffix(".json")
        sidecar.write_text(sidecar.read_text() + "\n")
    else:
        sidecar = model.with_suffix(".json")
        metadata = json.loads(sidecar.read_text())
        if mutation == "encoder":
            metadata["encoder"]["feature_fingerprint"] = "other"
        else:
            metadata["pooling"] = "spatial"
        sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash|metadata mismatch"):
        match_images(
            index,
            query_files(tmp_path, images),
            tmp_path / "invalid",
            encoder=OfflineEncoder(),
            model=model,
        )
    assert not (tmp_path / "invalid").exists()


def test_dual_model_rejects_wrong_frozen_encoder_before_projection(tmp_path, monkeypatch):
    bundle, images = make_bundle(tmp_path)
    model, matcher = offline_dual(tmp_path, monkeypatch, images)
    with pytest.raises(ValueError, match="metadata mismatch"):
        build_index(bundle, tmp_path / "invalid", "dino", OfflineEncoder("wrong"), model=model)
    assert matcher.calls == []


def test_dual_model_rejects_corrupted_index_projection_dimension(tmp_path, monkeypatch):
    bundle, images = make_bundle(tmp_path)
    model, _ = offline_dual(tmp_path, monkeypatch, images)
    index = tmp_path / "index"
    build_index(bundle, index, "dino", OfflineEncoder(), model=model)
    manifest = json.loads((index / "manifest.json").read_text())
    manifest["model"]["embedding_dim"] = 128
    (index / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dimension mismatch"):
        match_images(
            index,
            query_files(tmp_path, images),
            tmp_path / "invalid",
            encoder=OfflineEncoder(),
            model=model,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.int32])
def test_numeric_tiff_preserves_raw_pixels_and_array_matching(tmp_path, dtype):
    from slicegps.retrieval import rank

    class RecordingEncoder(OfflineEncoder):
        def __init__(self):
            super().__init__()
            self.received = []

        def encode(self, images, pooling="cls"):
            self.received.extend(np.array(image, copy=True) for image in images)
            return super().encode(images, pooling)

    source, masks = make_bundle(tmp_path)
    ramp = np.arange(32 * 32).reshape(32, 32)
    if dtype == np.float32:
        images = (masks * 137.25 + ramp / 17 - 14.875).astype(dtype)
    elif dtype == np.uint16:
        images = (masks.astype(np.uint32) * 137 + ramp % 17).astype(dtype)
    else:
        # Values above 2**24 catch lossy int32 -> PIL F -> float64 conversion.
        images = (masks.astype(np.int64) * 100_000 + 100_000_001 + ramp).astype(dtype)
    np.savez_compressed(source / "images.npz", gallery_mri=images)
    index = tmp_path / "index"
    build_index(source, index, "dino", encoder=OfflineEncoder())
    order = [2, 0, 1]
    queries = []
    for number, position in enumerate(order):
        path = tmp_path / f"section-{number}.tiff"
        Image.fromarray(images[position]).save(path)
        queries.append(path)
    encoder = RecordingEncoder()
    result = match_images(index, queries, tmp_path / "matched", encoder=encoder)
    np.testing.assert_array_equal(np.stack(encoder.received), images[order])
    raw_encoder = OfflineEncoder()
    expected, scores = rank(raw_encoder.encode(images[order]), raw_encoder.encode(images))
    actual = np.array(
        [[candidate["gallery_index"] for candidate in row["candidates"]]
         for row in result["matches"]]
    )
    np.testing.assert_array_equal(actual, expected)
    actual_scores = np.array(
        [[candidate["cosine_similarity"] for candidate in row["candidates"]]
         for row in result["matches"]]
    )
    np.testing.assert_allclose(actual_scores, scores, atol=1e-7)


def test_palette_png_uses_luminance_instead_of_palette_indices(tmp_path):
    class RecordingEncoder(OfflineEncoder):
        def encode(self, images, pooling="cls"):
            self.received = np.array(images, copy=True)
            return super().encode(images, pooling)

    source, masks = make_bundle(tmp_path)
    build_index(source, tmp_path / "index", "dino", encoder=OfflineEncoder())
    pixels = (masks[0] > 0).astype(np.uint8)
    image = Image.fromarray(pixels).convert("P")
    image.putpalette([0, 0, 0, 180, 180, 180] + [0] * (768 - 6))
    query = tmp_path / "palette.png"
    image.save(query)
    encoder = RecordingEncoder()
    result = match_images(tmp_path / "index", [query], tmp_path / "matches", encoder=encoder)
    np.testing.assert_array_equal(encoder.received, masks[:1])
    assert result["matches"][0]["candidates"][0]["gallery_index"] == 0
