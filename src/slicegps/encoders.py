"""Frozen, pinned DINOv2 image features with explicit preprocessing provenance.

This module imports PyTorch and Transformers only when encoding valid images.
The encoder sees pixels only. It does not consume slice IDs, locations, or angles.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .retrieval import _prepare, _unit_rows

MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
PREPROCESSING = {
    "grayscale_scaling": "border foreground; 1st/99th percentiles; min/max fallback",
    "foreground_definition": "absolute difference from median border > 0.05 * image range",
    "rgb_conversion": "repeat grayscale in RGB; round scaled values to uint8 0..255",
    "do_resize": True,
    "size": {"shortest_edge": 256},
    "resample": 3,
    "resample_name": "PIL bicubic",
    "do_center_crop": True,
    "crop_size": {"height": 224, "width": 224},
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "do_convert_rgb": True,
    "use_fast": False,
}


def _rgb_images(images: np.ndarray) -> list[Image.Image]:
    if np.iscomplexobj(images):
        raise ValueError("images must contain real grayscale values")
    values = np.asarray(images, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or min(values.shape[1:]) < 2:
        raise ValueError("images must have shape (N, H, W) with N > 0 and H, W >= 2")
    if not np.isfinite(values).all():
        raise ValueError("images must contain only finite values")
    result = []
    for image in values:
        scaled, _ = _prepare(image)
        gray = np.rint(scaled * 255).astype(np.uint8)
        result.append(Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2)))
    return result


def _load_components(cache_dir: Path) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ImportError(
            "DINOv2 requires the optional torch and transformers dependencies"
        ) from exc
    cache_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "revision": MODEL_REVISION,
        "cache_dir": str(cache_dir),
        "trust_remote_code": False,
    }
    overrides = {
        key: value
        for key, value in PREPROCESSING.items()
        if key
        not in {"grayscale_scaling", "foreground_definition", "rgb_conversion", "resample_name"}
    }
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, **common, **overrides)
    model = AutoModel.from_pretrained(MODEL_ID, **common, use_safetensors=True)
    return torch, processor, model


class DinoEncoder:
    """Frozen ViT-S/14 descriptors from the pinned official Hugging Face weights.

    ``device='auto'`` uses MPS when available, otherwise CPU. An MPS RuntimeError
    retries the whole encoding call on CPU and records the fallback. Explicit
    ``cpu``, ``mps``, and ``cuda`` choices never silently switch devices.
    ``cls`` returns 384 values; ``spatial`` averages the 16x16 patch grid to 4x4
    and flattens row-major patch cells, returning 6144 values. Both are L2 scaled.
    The official center crop can remove peripheral anatomy, a baseline limitation.
    """

    def __init__(self, cache_dir: Path, device: str = "auto", batch_size: int = 8):
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device must be auto, cpu, mps, or cuda")
        if (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        self.cache_dir = Path(cache_dir).resolve()
        self.device = device
        self.batch_size = int(batch_size)
        self._torch = self._processor = self._model = None
        self._device_used: str | None = None
        self._fallback_reason: str | None = None
        self._pooling: str | None = None
        self._versions: dict[str, str] = {}

    @property
    def metadata(self) -> dict:
        """JSON-compatible provenance for the last encoding call (or unloaded state)."""
        contract = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "preprocessing": deepcopy(PREPROCESSING),
            "pooling": self._pooling,
            "spatial_pooling": "adaptive average 16x16 patch grid to 4x4; row-major cells",
            "normalization": "L2 per image",
            "implementation_version": 1,
        }
        fingerprint = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            **contract,
            "feature_fingerprint": fingerprint,
            "license": "Apache-2.0",
            "trust_remote_code": False,
            "weights_format": "safetensors",
            "cache_dir": str(self.cache_dir),
            "device_requested": self.device,
            "device_used": self._device_used,
            "cpu_fallback_reason": self._fallback_reason,
            "batch_size": self.batch_size,
            "package_versions": dict(self._versions),
            "training": "none; frozen pretrained model in eval/inference mode",
            "limitation": (
                "ImageNet center crop may discard peripheral anatomy; no donor fine-tuning"
            ),
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch, processor, model = _load_components(self.cache_dir)
        selected = self.device
        if selected == "auto":
            selected = "mps" if torch.backends.mps.is_available() else "cpu"
            if selected == "cpu":
                self._fallback_reason = "MPS unavailable; auto selected CPU"
        elif selected == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        elif selected == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        model.eval()
        model.requires_grad_(False)
        try:
            model.to(device=selected, dtype=torch.float32)
        except RuntimeError as exc:
            if self.device != "auto" or selected != "mps":
                raise
            selected = "cpu"
            self._fallback_reason = f"MPS model placement failed: {exc}"
            model.to(device=selected, dtype=torch.float32)
        self._torch, self._processor, self._model = torch, processor, model
        self._device_used = selected
        for package in ("torch", "transformers", "Pillow", "numpy"):
            try:
                self._versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self._versions[package] = "unavailable"

    def _encode_batches(self, images: list[Image.Image], pooling: str) -> np.ndarray:
        torch = self._torch
        descriptors = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                processed = self._processor(images=batch, return_tensors="pt")
                pixels = processed["pixel_values"]
                if tuple(pixels.shape) != (len(batch), 3, 224, 224):
                    raise ValueError("DINOv2 preprocessing must return (batch, 3, 224, 224)")
                if not bool(torch.isfinite(pixels).all()):
                    raise ValueError("DINOv2 preprocessing produced non-finite values")
                output = self._model(
                    pixel_values=pixels.to(device=self._device_used, dtype=torch.float32)
                )
                tokens = output.last_hidden_state
                if tuple(tokens.shape) != (len(batch), 257, 384):
                    raise ValueError("Expected pinned ViT-S/14 output shape (batch, 257, 384)")
                if pooling == "cls":
                    descriptor = tokens[:, 0]
                else:
                    patches = tokens[:, 1:]
                    side = math.isqrt(patches.shape[1])
                    grid = patches.reshape(len(batch), side, side, 384).permute(0, 3, 1, 2)
                    cells = torch.nn.functional.adaptive_avg_pool2d(grid, (4, 4))
                    descriptor = cells.permute(0, 2, 3, 1).flatten(start_dim=1)
                descriptors.append(descriptor.detach().to("cpu", dtype=torch.float32).numpy())
        return _unit_rows(np.concatenate(descriptors), "DINOv2 descriptors").astype(np.float32)

    def encode(self, images: np.ndarray, pooling: str = "cls") -> np.ndarray:
        """Return normalized image descriptors in input order, without geometry inputs."""
        if pooling not in {"cls", "spatial"}:
            raise ValueError("pooling must be cls or spatial")
        rgb = _rgb_images(images)
        self._ensure_loaded()
        self._pooling = pooling
        try:
            return self._encode_batches(rgb, pooling)
        except RuntimeError as exc:
            if self.device != "auto" or self._device_used != "mps":
                raise
            self._device_used = "cpu"
            self._fallback_reason = f"MPS inference failed; all batches retried on CPU: {exc}"
            self._model.to(device="cpu", dtype=self._torch.float32)
            return self._encode_batches(rgb, pooling)
