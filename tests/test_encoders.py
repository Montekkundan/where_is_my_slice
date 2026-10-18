"""No-download checks for the optional frozen encoder and its pixel contract."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from slicegps import encoders
from slicegps.encoders import MODEL_ID, MODEL_REVISION, DinoEncoder, _rgb_images
from slicegps.retrieval import _prepare


def patterns(count=5):
    y, x = np.mgrid[:24, :36]
    images = []
    for index in range(count):
        tissue = ((x - 16 - index) / 12) ** 2 + ((y - 11) / (9 - index / 2)) ** 2 < 1
        image = np.where(tissue, 0.2 + x / 90 + 0.3 * np.sin(y / 3 + index) ** 2, 0)
        images.append(image)
    return np.asarray(images)


def fake_components(torch):
    class Processor:
        def __init__(self):
            self.calls = []

        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            self.calls.append([np.array(image) for image in images])
            tensors = []
            for image in images:
                width, height = image.size
                factor = 256 / min(width, height)
                resized = image.resize(
                    (int(width * factor), int(height * factor)), Image.Resampling.BICUBIC
                )
                left, top = (resized.width - 224) // 2, (resized.height - 224) // 2
                crop = resized.crop((left, top, left + 224, top + 224))
                values = np.array(crop, dtype=np.float32) / 255
                values = (values - np.array([0.485, 0.456, 0.406])) / [0.229, 0.224, 0.225]
                tensors.append(torch.tensor(values.transpose(2, 0, 1), dtype=torch.float32))
            return {"pixel_values": torch.stack(tensors)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []
            self.mock_device = "cpu"
            self.fail_mps_batch = None
            self.mps_calls = 0

        def to(self, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            if device == "mps":
                self.mock_device = "mps"
                return self
            self.mock_device = "cpu"
            return super().to(*args, **kwargs)

        def forward(self, pixel_values):
            self.calls.append((len(pixel_values), torch.is_grad_enabled(), self.training))
            if self.mock_device == "mps":
                self.mps_calls += 1
                if self.mps_calls == self.fail_mps_batch:
                    raise RuntimeError("simulated unsupported MPS operation")
            patches = (
                torch.nn.functional.adaptive_avg_pool2d(pixel_values[:, :1], (16, 16))
                .flatten(start_dim=2)
                .transpose(1, 2)
            )
            basis = torch.arange(384, dtype=torch.float32)[None, None, :]
            patch_tokens = torch.cos(basis / 13) + patches * torch.sin(basis / 40)
            cls = patch_tokens.mean(dim=1, keepdim=True)
            return SimpleNamespace(last_hidden_state=torch.cat((cls, patch_tokens), dim=1))

    return Processor(), Model()


@pytest.fixture
def components(monkeypatch):
    torch = pytest.importorskip("torch")
    processor, model = fake_components(torch)
    monkeypatch.setattr(encoders, "_load_components", lambda _: (torch, processor, model))
    return torch, processor, model


def test_grayscale_preparation_matches_existing_foreground_scaling():
    images = patterns(2)
    rgb = _rgb_images(images)
    for source, result in zip(images, rgb, strict=True):
        scaled, _ = _prepare(source)
        expected = np.rint(scaled * 255).astype(np.uint8)
        assert result.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(result)[:, :, 0], expected)
        np.testing.assert_array_equal(np.asarray(result)[:, :, 1], expected)
        np.testing.assert_array_equal(np.asarray(result)[:, :, 2], expected)


@pytest.mark.parametrize("pooling,dimension", [("cls", 384), ("spatial", 6144)])
def test_chunking_shapes_normalization_order_and_frozen_inference(
    tmp_path, components, pooling, dimension
):
    _, processor, model = components
    images = patterns()
    encoder = DinoEncoder(tmp_path / "weights", device="cpu", batch_size=2)
    output = encoder.encode(images, pooling)
    assert output.shape == (5, dimension)
    assert output.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1, atol=1e-6)
    assert [len(call) for call in processor.calls] == [2, 2, 1]
    assert model.calls == [(2, False, False), (2, False, False), (1, False, False)]
    assert not model.weight.requires_grad
    permutation = [3, 0, 4, 1, 2]
    permuted = encoder.encode(images[permutation], pooling)
    np.testing.assert_allclose(permuted, output[permutation], atol=1e-6)
    # The stub responds to image structure; order preservation is not a constant-output test.
    assert np.max(np.abs(output[0] - output[-1])) > 1e-4


def test_spatial_pooling_preserves_row_major_patch_cells(tmp_path, components):
    torch, processor, model = components
    images = patterns(1)
    encoder = DinoEncoder(tmp_path, device="cpu")
    result = encoder.encode(images, "spatial")
    with torch.inference_mode():
        pixels = processor(images=_rgb_images(images), return_tensors="pt")["pixel_values"]
        patches = model(pixel_values=pixels).last_hidden_state[:, 1:].reshape(1, 16, 16, 384)
        expected = patches.reshape(1, 4, 4, 4, 4, 384).mean(dim=(2, 4)).reshape(1, -1).numpy()
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((1, 8, 8)),
        np.ones((2, 8, 8)),
        np.empty((0, 8, 8)),
        np.full((1, 8, 8), np.nan),
        np.ones((1, 1, 8)),
        np.ones((2, 8)),
        np.ones((1, 8, 8), dtype=complex),
    ],
)
def test_invalid_images_are_rejected_before_loading(tmp_path, monkeypatch, images):
    def unexpected_load(_):
        pytest.fail("Invalid images must not trigger dependency imports or downloads")

    monkeypatch.setattr(encoders, "_load_components", unexpected_load)
    with pytest.raises(ValueError):
        DinoEncoder(tmp_path).encode(images)


def test_metadata_fingerprint_records_revision_preprocessing_and_pooling(tmp_path, components):
    encoder = DinoEncoder(tmp_path / "cache", device="cpu")
    encoder.encode(patterns(1), "cls")
    first = encoder.metadata
    assert first["model_id"] == MODEL_ID
    assert first["model_revision"] == MODEL_REVISION
    assert first["preprocessing"]["size"] == {"shortest_edge": 256}
    assert first["preprocessing"]["crop_size"] == {"height": 224, "width": 224}
    assert first["device_used"] == "cpu"
    assert first["cache_dir"] == str((tmp_path / "cache").resolve())
    assert len(first["feature_fingerprint"]) == 64
    first["preprocessing"]["size"]["shortest_edge"] = 1
    assert encoder.metadata["preprocessing"]["size"]["shortest_edge"] == 256
    encoder.encode(patterns(1), "spatial")
    assert encoder.metadata["feature_fingerprint"] != first["feature_fingerprint"]


def test_auto_cpu_choice_is_explicit_in_metadata(tmp_path, components, monkeypatch):
    torch, _, _ = components
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    encoder = DinoEncoder(tmp_path)
    encoder.encode(patterns(1))
    assert encoder.metadata["device_used"] == "cpu"
    assert "unavailable" in encoder.metadata["cpu_fallback_reason"]


def test_auto_mps_failure_retries_all_batches_on_cpu(tmp_path, components, monkeypatch):
    torch, _, model = components
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    original_to = torch.Tensor.to

    def fake_tensor_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "mps":
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_tensor_to)
    model.fail_mps_batch = 2
    encoder = DinoEncoder(tmp_path, batch_size=2)
    output = encoder.encode(patterns())
    assert output.shape == (5, 384)
    assert [call[0] for call in model.calls] == [2, 2, 2, 2, 1]
    assert encoder.metadata["device_used"] == "cpu"
    assert "all batches retried on CPU" in encoder.metadata["cpu_fallback_reason"]
    np.testing.assert_allclose(output, encoder.encode(patterns()), atol=1e-6)


def test_loading_pins_both_components_and_uses_only_passed_cache(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    calls = []
    fake_transformers = ModuleType("transformers")

    class Factory:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
            return object()

    fake_transformers.AutoImageProcessor = Factory
    fake_transformers.AutoModel = Factory
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    cache = tmp_path / "weights"
    encoders._load_components(cache)
    assert cache.is_dir()
    for name, options in calls:
        assert name == MODEL_ID
        assert options["revision"] == MODEL_REVISION
        assert options["cache_dir"] == str(cache)
        assert options["trust_remote_code"] is False
    assert calls[0][1]["use_fast"] is False
    assert calls[0][1]["resample"] == 3
    assert calls[0][1]["image_mean"] == [0.485, 0.456, 0.406]
    assert calls[1][1]["use_safetensors"] is True


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        DinoEncoder(tmp_path, batch_size=batch_size)


def test_invalid_device_and_pooling(tmp_path):
    with pytest.raises(ValueError, match="device"):
        DinoEncoder(tmp_path, device="guess")
    with pytest.raises(ValueError, match="pooling"):
        DinoEncoder(tmp_path).encode(patterns(1), "guess")
