import hashlib
import io
import json

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from slicegps.external import external_check, fetch_external_sections
from slicegps.retrieval import features


class OfflineEncoder:
    """An image-only test double; never imports torch or downloads weights."""

    def __init__(self):
        self.calls = []
        self.metadata = {
            "feature_fingerprint": "fixture-intensity-v1",
            "model_id": "offline-test-double",
            "model_revision": "fixture",
            "preprocessing": {},
            "pooling": "spatial",
        }

    def encode(self, images, pooling):
        assert isinstance(images, np.ndarray) and images.ndim == 3
        assert pooling == "spatial"
        self.calls.append(images.copy())
        return features(images, method="intensity")


@pytest.fixture
def real_jpeg_fixture(tmp_path):
    root = tmp_path / "donor"
    root.mkdir()
    # Deliberately incorrect JSON dimensions: scoring must use decoded JPEGs.
    dimensions = {str(block): {"height": 14, "width": 20, "slices": 6} for block in (1, 7, 41)}
    (root / "histologyDimensionsKey.json").write_text(json.dumps(dimensions))
    mri = root / "mri_rotated" / "mri.nii.gz"
    mri.parent.mkdir()
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = (10, -5, 7)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), affine), mri)
    matrix = np.array([[0, 1, 0, 1], [0, 0, 1, 2], [1, 0, 0, 1], [0, 0, 0, 1]])
    rng = np.random.default_rng(41)
    for block in (1, 7, 41):
        directory = root / "histology" / f"{block:02d}"
        for kind in ("slices_HE", "slices_MRI"):
            (directory / kind).mkdir(parents=True)
        block_matrix = matrix.copy()
        block_matrix[0, 3] += block
        np.savetxt(directory / "matrix.txt", block_matrix)
        for index in range(6 if block != 41 else 1):
            image = rng.integers(20, 230, size=(15, 21), dtype=np.uint8)
            if block == 41:
                image[:] = 0
            else:
                image[[0, -1]] = 0
                image[:, [0, -1]] = 0
            for kind in ("slices_HE", "slices_MRI"):
                Image.fromarray(image).save(
                    directory / kind / f"slice_{index:02d}.jpg", quality=100, subsampling=0
                )
    return root


def test_real_jpeg_ranking_uses_eight_queries_twelve_candidates_and_actual_geometry(
    real_jpeg_fixture, tmp_path
):
    encoder = OfflineEncoder()
    output = tmp_path / "output"
    report = external_check(real_jpeg_fixture, output, encoder, blocks=(1, 7, 41), k=5)
    assert report["query_count"] == 8
    assert report["gallery_count"] == 12
    assert len(report["rows"]) == 32
    assert set(report["methods"]) == {"intensity", "gradient", "mask", "dino_spatial"}
    for name, metric in report["methods"].items():
        assert metric["query_count"] == 8 and metric["gallery_count"] == 12
        if name != "mask":
            assert metric["top1_exact_pair_accuracy"] == 1
            assert metric["top1_median_error_mm"] == 0
        assert metric["top1_median_normal_error_deg"] == 0
    section = report["sections"][0]
    assert section["actual_shape"] == [15, 21]
    assert section["declared_shape"] == [14, 20]
    # Actual center row=7, col=10. Matrix maps it to MRI voxel (12,2,8).
    np.testing.assert_allclose(section["ground_truth"]["center_ras_mm"], [34, 1, 39])
    assert section["declared_dimensions_match"] is False
    assert len(report["exclusions"]) == 1
    assert report["exclusions"][0]["section_id"] == "donor/41/00"
    assert "empty" in report["exclusions"][0]["reason"]
    assert len(report["warnings"]) == 12
    assert len(encoder.calls) == 2
    assert [len(batch) for batch in encoder.calls] == [12, 8]
    assert report == json.loads((output / "metrics.json").read_text())
    assert (output / "predictions.csv").read_text().count("\n") == 33


def test_geometry_changes_scores_without_changing_image_rankings(real_jpeg_fixture, tmp_path):
    directory = real_jpeg_fixture / "histology"
    # Deliberately wrong pair makes one query retrieve another block's image.
    target = directory / "01/slices_HE/slice_00.jpg"
    target.write_bytes((directory / "07/slices_MRI/slice_05.jpg").read_bytes())
    first = external_check(real_jpeg_fixture, tmp_path / "first", OfflineEncoder())
    matrix_path = directory / "07/matrix.txt"
    matrix = np.loadtxt(matrix_path)
    matrix[0, 3] += 100
    np.savetxt(matrix_path, matrix)
    second = external_check(real_jpeg_fixture, tmp_path / "second", OfflineEncoder())
    assert [row["ranked_section_ids"] for row in first["rows"]] == [
        row["ranked_section_ids"] for row in second["rows"]
    ]
    assert [row["cosine_similarity"] for row in first["rows"]] == [
        row["cosine_similarity"] for row in second["rows"]
    ]
    wrong_first = next(row for row in first["rows"] if row["query_section_id"] == "donor/01/00")
    wrong_second = next(row for row in second["rows"] if row["query_section_id"] == "donor/01/00")
    assert wrong_first["predicted_section_id"] == "donor/07/05"
    assert wrong_second["center_error_mm"] > wrong_first["center_error_mm"] + 100


def test_pair_dimension_mismatch_and_missing_pairs_are_explicit_exclusions(
    real_jpeg_fixture, tmp_path
):
    directory = real_jpeg_fixture / "histology/01"
    path = directory / "slices_HE/slice_00.jpg"
    with Image.open(path) as image:
        resized = image.resize((21, 16))
    resized.save(path)
    (directory / "slices_HE/slice_01.jpg").unlink()
    report = external_check(real_jpeg_fixture, tmp_path / "out", OfflineEncoder())
    reasons = {entry["section_index"]: entry["reason"] for entry in report["exclusions"]}
    assert reasons == {
        0: "paired HE/MRI decoded dimensions differ",
        1: "missing paired HE or MRI JPEG",
    }
    assert report["gallery_count"] == 10
    assert report["query_count"] == 8
    assert report["sections"][0]["section_index"] == 2


def test_output_reuse_and_small_invalid_datasets_fail(real_jpeg_fixture, tmp_path):
    output = tmp_path / "out"
    external_check(real_jpeg_fixture, output, OfflineEncoder())
    with pytest.raises(FileExistsError, match="new or empty"):
        external_check(real_jpeg_fixture, output, OfflineEncoder())
    with pytest.raises(ValueError, match="two nonempty"):
        external_check(real_jpeg_fixture, tmp_path / "empty", OfflineEncoder(), blocks=(41,))
    with pytest.raises(ValueError, match="distinct"):
        external_check(real_jpeg_fixture, tmp_path / "bad", OfflineEncoder(), blocks=(1, 1))


def test_source_manifest_is_autodiscovered_and_checked(real_jpeg_fixture, tmp_path):
    records = [
        {
            "path": real_jpeg_fixture.name + "/" + path.relative_to(real_jpeg_fixture).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in real_jpeg_fixture.rglob("*")
        if path.is_file()
    ]
    manifest = tmp_path / "external_sources.json"
    manifest.write_text(json.dumps({"files": records}))
    result = external_check(real_jpeg_fixture, tmp_path / "verified", OfflineEncoder())
    assert result["sources"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    records[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"files": records}))
    with pytest.raises(ValueError, match="manifest checksum"):
        external_check(real_jpeg_fixture, tmp_path / "tampered", OfflineEncoder())
    assert not (tmp_path / "tampered").exists()


def test_fetch_uses_discovered_pinned_paths_and_checks_blobs_before_reuse(tmp_path, monkeypatch):
    import slicegps.external as external

    donor = "P-test"
    prefix = f"{donor}/histology/01"
    blobs = {
        f"{donor}/mri_rotated/mri.nii.gz": b"fixture-mri",
        f"{donor}/histologyDimensionsKey.json": b"{}",
        f"{donor}/block2slab_mapping.csv": b"BLOCK_NUM,BLOCK_NAME\n1,A1.1\n",
        f"{prefix}/matrix.txt": b"fixture-matrix",
    }
    for index in (0, 2, 8):
        for kind in ("HE", "MRI"):
            blobs[f"{prefix}/slices_{kind}/slice_{index:02d}.jpg"] = f"{kind}{index}".encode()
    tree = {
        "truncated": False,
        "tree": [
            {
                "path": path,
                "type": "blob",
                "size": len(content),
                "sha": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
            }
            for path, content in blobs.items()
        ],
    }
    calls = []

    def tree_response(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps(tree).encode())

    def fake_download(url, destination, max_bytes):
        path = url.split("/" + external.CASE4_REVISION + "/", 1)[1]
        content = blobs[path]
        assert len(content) <= max_bytes
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    monkeypatch.setattr(external.urllib.request, "urlopen", tree_response)
    monkeypatch.setattr(external, "download", fake_download)
    root = tmp_path / donor
    manifest = tmp_path / "sources.json"
    with pytest.raises(ValueError, match="total byte cap"):
        fetch_external_sections(
            root, manifest, donor=donor, blocks=(1,), gallery_per_block=2, max_total_bytes=1
        )
    assert not root.exists()
    calls.clear()
    report = fetch_external_sections(root, manifest, donor=donor, blocks=(1,), gallery_per_block=2)
    assert calls == [
        f"https://api.github.com/repos/{external.CASE4_REPOSITORY}/git/trees/"
        f"{external.CASE4_REVISION}?recursive=1"
    ]
    assert report["selected_section_indices"] == {"1": [0, 2]}
    assert len(report["files"]) == 8
    assert all(external.CASE4_REVISION in record["url"] for record in report["files"])
    assert all(record["sha256"] for record in report["files"])
    (root / "histology/01/slices_HE/slice_00.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        fetch_external_sections(root, manifest, donor=donor, blocks=(1,), gallery_per_block=2)


def test_fetch_checks_total_budget_and_truncated_tree_before_downloading(tmp_path, monkeypatch):
    import slicegps.external as external

    monkeypatch.setattr(
        external.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b'{"truncated":true,"tree":[]}'),
    )
    with pytest.raises(ValueError, match="complete"):
        fetch_external_sections(tmp_path / "donor", tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="50 MB"):
        fetch_external_sections(
            tmp_path / "donor", tmp_path / "manifest.json", max_total_bytes=50_000_001
        )
    assert not (tmp_path / "donor").exists()
