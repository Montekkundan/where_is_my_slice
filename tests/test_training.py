"""Small CPU experiments test optimization, isolation, and safe artifact inference."""

import json

import numpy as np
import pytest

from slicegps.geometry import coronal_plane
from slicegps.training import DualMatcher, _pose_metrics, train_dual


@pytest.fixture(scope="module", autouse=True)
def single_thread_torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def make_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    latent = rng.normal(size=(42, 5)).astype(np.float32)
    hist = latent @ rng.normal(size=(5, 9)).astype(np.float32)
    mri = latent @ rng.normal(size=(5, 9)).astype(np.float32)
    train, validation, test = np.arange(24), np.arange(24, 30), np.arange(30, 36)
    path = root / "features_cls.npz"
    np.savez_compressed(
        path,
        train_histology=hist[train],
        train_mri=mri[train],
        validation_histology=hist[validation],
        test_histology=hist[test],
        gallery_mri=mri,
    )
    planes = [
        coronal_plane((0, float(row[0] * 20), 0), tilt_deg=(float(row[1]), float(row[2]))).to_dict()
        for row in latent
    ]
    manifest = {
        "schema_version": 1,
        "images_sha256": "1" * 64,
        "planes": {
            "train": [planes[i] for i in train],
            "validation": [planes[i] for i in validation],
            "test": [planes[i] for i in test],
        },
        "gallery_planes": planes,
        "scope": "Artificial paired feature fixture, not human evidence",
    }
    protocol = root / "protocol"
    protocol.mkdir()
    (protocol / "manifest.json").write_text(json.dumps(manifest))
    (root / "adapter_cls.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pooling": "cls",
                "feature_dim": 9,
                "training_images_sha256": "1" * 64,
                "encoder": {
                    "model_id": "test-only",
                    "model_revision": "not-downloadable",
                    "pooling": "cls",
                    "feature_fingerprint": "test-fixture",
                    "preprocessing": {"source": "synthetic"},
                },
            }
        )
    )
    return path, protocol, hist[train], mri[train]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("training")
    path, protocol, hist, mri = make_fixture(root / "input")
    report = train_dual(
        path,
        protocol,
        root / "model",
        epochs=35,
        seed=9,
        batch_size=24,
        validation_every=5,
        patience=20,
    )
    return root, path, protocol, hist, mri, report


def test_contrastive_training_reduces_loss_and_uses_validation_selection(trained):
    *_, report = trained
    assert (
        report["selected_full_train_contrastive_loss"]
        < report["initial_full_train_contrastive_loss"]
    )
    assert report["selected_full_train_contrastive_loss"] < 1
    best = min(
        report["validation_history"],
        key=lambda row: (-row["recall_at_k"]["1"], row["top1_median_error_mm"]),
    )
    assert report["selected_epoch"] == best["epoch"]
    assert report["test_evaluated_during_selection"] is False
    assert "test_retrieval" in report and "test_pose" in report
    assert "training_mean_pose_prior_baseline" in report
    assert any("Exploratory reuse" in text for text in report["limitations"])
    assert "in-plane rotation" in report["test_pose"]["normal_metric"]


def test_safe_roundtrip_both_modalities_pose_and_input_order(trained):
    root, _, _, hist, mri, _ = trained
    model = DualMatcher.load(root / "model/model.safetensors")
    encoded, gallery = model.transform(hist), model.encode_gallery(mri)
    assert encoded.shape == gallery.shape == (24, 128)
    assert encoded.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(gallery, axis=1), 1, atol=1e-6)
    pose = model.predict_pose(hist)
    assert pose["centers_mm"].shape == pose["normals_ras"].shape == (24, 3)
    np.testing.assert_allclose(np.linalg.norm(pose["normals_ras"], axis=1), 1, atol=1e-6)
    permutation = np.arange(len(hist))[::-1].copy()
    np.testing.assert_allclose(model.transform(hist[permutation]), encoded[permutation], atol=1e-6)
    second = root / "roundtrip/model.safetensors"
    model.save(second)
    reloaded = DualMatcher.load(second)
    np.testing.assert_array_equal(reloaded.transform(hist), encoded)
    np.testing.assert_array_equal(reloaded.encode_gallery(mri), gallery)
    for key, value in pose.items():
        np.testing.assert_array_equal(reloaded.predict_pose(hist)[key], value)
    assert reloaded.metadata["encoder"]["feature_fingerprint"] == "test-fixture"
    assert not any(parameter.requires_grad for parameter in reloaded._model.parameters())


def test_feature_and_pose_scaling_fit_training_only(trained):
    root, path, protocol, hist, mri, _ = trained
    model = DualMatcher.load(root / "model/model.safetensors")
    np.testing.assert_allclose(model._model.histology_mean.numpy(), hist.mean(axis=0))
    np.testing.assert_allclose(
        model._model.histology_std.numpy(), np.maximum(hist.std(axis=0), 1e-6)
    )
    np.testing.assert_allclose(model._model.mri_mean.numpy(), mri.mean(axis=0))
    manifest = json.loads((protocol / "manifest.json").read_text())
    centers = np.array([p["center_mm"] for p in manifest["planes"]["train"]], dtype=np.float32)
    np.testing.assert_allclose(model._model.center_mean.numpy(), centers.mean(axis=0))
    assert model.metadata["pose_scaling"]["constant_training_center_axes"] == [0, 2]
    with np.load(path, allow_pickle=False) as f:
        all_hist = np.concatenate((hist, f["validation_histology"], f["test_histology"]))
    assert not np.allclose(model._model.histology_mean.numpy(), all_hist.mean(axis=0))


def test_test_values_do_not_change_selected_weights(trained):
    torch = pytest.importorskip("torch")
    root, path, protocol, _, _, first = trained
    with np.load(path, allow_pickle=False) as f:
        arrays = {key: f[key] for key in f.files}
    arrays["test_histology"] = arrays["test_histology"] * 1.1 + 0.1
    changed = root / "changed_features.npz"
    np.savez_compressed(changed, **arrays)
    second = train_dual(
        changed,
        protocol,
        root / "changed_model",
        epochs=35,
        seed=9,
        batch_size=24,
        validation_every=5,
        patience=20,
    )
    assert second["selected_epoch"] == first["selected_epoch"]
    original = DualMatcher.load(root / "model/model.safetensors")
    altered = DualMatcher.load(root / "changed_model/model.safetensors")
    for key, value in original._model.state_dict().items():
        torch.testing.assert_close(value, altered._model.state_dict()[key], rtol=0, atol=0)


def test_weights_hash_catches_modified_artifact(trained):
    root, *_ = trained
    model = DualMatcher.load(root / "model/model.safetensors")
    target = root / "modified/model.safetensors"
    model.save(target)
    with target.open("ab") as stream:
        stream.write(b"unexpected")
    with pytest.raises(ValueError, match="hash"):
        DualMatcher.load(target)


@pytest.mark.parametrize(
    "method,width", [("transform", 9), ("encode_gallery", 9), ("predict_pose", 9)]
)
def test_inference_rejects_invalid_feature_inputs(trained, method, width):
    root, *_ = trained
    model = DualMatcher.load(root / "model/model.safetensors")
    for invalid in (
        np.zeros((1, width)),
        np.ones((2, width + 1)),
        np.full((1, width), np.nan),
        np.empty((0, width)),
    ):
        with pytest.raises(ValueError):
            getattr(model, method)(invalid)


def test_pose_metrics_measure_center_and_oriented_normal_errors():
    predicted = {
        "centers_mm": np.array([[0, 3, 4], [0, 0, 0]]),
        "normals_ras": np.array([[0, 1, 0], [1, 0, 0]]),
    }
    result = _pose_metrics(predicted, np.zeros((2, 3)), np.array([[0, -1, 0], [0, -1, 0]]))
    assert result["center_median_error_mm"] == 2.5
    assert result["normal_median_angle_deg"] == 135


@pytest.mark.parametrize(
    "options",
    [
        {"epochs": 0},
        {"batch_size": 1},
        {"temperature": 0},
        {"pose_weight": -1},
        {"patience": 0},
        {"validation_every": 0},
    ],
)
def test_invalid_training_configuration_rejected_before_io(tmp_path, options):
    with pytest.raises(ValueError):
        train_dual(tmp_path / "absent.npz", tmp_path / "absent", tmp_path / "output", **options)


@pytest.mark.parametrize("mutation", ["image_hash", "pooling", "feature_dim"])
def test_same_count_protocol_or_feature_provenance_mismatch_rejected_before_training(
    tmp_path, monkeypatch, mutation
):
    import slicegps.training as training

    path, protocol, _, _ = make_fixture(tmp_path / "input")
    if mutation == "image_hash":
        manifest_path = protocol / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["images_sha256"] = "2" * 64
        manifest_path.write_text(json.dumps(manifest))
    else:
        sidecar = path.with_name("adapter_cls.json")
        provenance = json.loads(sidecar.read_text())
        provenance[mutation] = "spatial" if mutation == "pooling" else 8
        sidecar.write_text(json.dumps(provenance))

    def unexpected_training():
        pytest.fail("Provenance must be rejected before model creation or training")

    monkeypatch.setattr(training, "_torch", unexpected_training)
    with pytest.raises(ValueError, match="provenance mismatch"):
        train_dual(path, protocol, tmp_path / "invalid", epochs=5)
    assert not (tmp_path / "invalid").exists()
