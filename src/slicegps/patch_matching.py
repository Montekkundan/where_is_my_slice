"""Small modality-specific convolutional matcher trained on paired section images.

The symmetric contrastive objective uses adjacent-plane hard negatives. This is
an independent matching experiment, not a reproduction of Synth-by-Reg.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from .data import write_json
from .error_detection import _images, canonical_histology, context_features
from .pairwise_qa import comparison_features
from .training import _device, _torch

ARCHITECTURE = "paired_patch_cnn_v1"


def _network():
    torch = _torch()

    def branch():
        layers = []
        source = 1
        for width, stride in ((16, 2), (32, 2), (32, 1)):
            layers.extend(
                [
                    torch.nn.Conv2d(source, width, 3, stride=stride, padding=1, bias=False),
                    torch.nn.InstanceNorm2d(width),
                    torch.nn.LeakyReLU(0.1),
                ]
            )
            source = width
        layers.append(torch.nn.Conv2d(32, 16, 1, bias=False))
        return torch.nn.Sequential(*layers)

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.histology = branch()
            self.mri = branch()

        def forward(self, values, modality):
            encoded = getattr(self, modality)(values).flatten(1)
            return torch.nn.functional.normalize(encoded, dim=1)

    return Network()


def prepare(images, modality, preprocessing="canonical_v1"):
    images = _images(images, minimum=1).astype(np.float32)
    if images.shape[1:] != (96, 96) or modality not in ("histology", "mri"):
        raise ValueError("Patch matcher expects 96 x 96 histology or MRI images")
    if preprocessing not in ("canonical_v1", "paper_v2"):
        raise ValueError("Unknown patch preprocessing")
    if modality == "histology":
        if preprocessing == "canonical_v1":
            images = canonical_histology(images)
        else:
            # A small mounting sheet can occupy less than 20% of the canvas.
            # Its broad almost-white plateau distinguishes it from tissue highlights.
            cleaned = []
            for image in images:
                scaled = (image - image.min()) / np.ptp(image)
                fraction = 0.0 if np.mean(scaled > 0.98) > 0.04 else 0.2
                cleaned.append(canonical_histology(image[None], paper_fraction=fraction)[0])
            images = np.stack(cleaned)
    low = images.min(axis=(1, 2), keepdims=True)
    high = images.max(axis=(1, 2), keepdims=True)
    if np.any(high <= low):
        raise ValueError("Patch matcher cannot encode empty sections")
    return np.asarray((images - low) / (high - low), dtype=np.float32)[:, None]


class PatchMatcher:
    def __init__(self, model, metadata, device="cpu", batch_size=32):
        self.device = _device(device)
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.metadata = deepcopy(metadata)
        self.batch_size = batch_size

    def encode(self, images, modality):
        torch = _torch()
        values = prepare(images, modality, self.metadata.get("preprocessing", "canonical_v1"))
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                tensor = torch.from_numpy(values[start : start + self.batch_size]).to(self.device)
                outputs.append(self.model(tensor, modality).cpu().numpy())
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite patch embeddings")
        return result

    def save(self, path):
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                key: value.detach().cpu().contiguous()
                for key, value in self.model.state_dict().items()
            },
            str(path),
        )
        metadata = deepcopy(self.metadata)
        metadata["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(path.with_suffix(".json"), metadata)

    @classmethod
    def load(cls, path, device="auto"):
        from safetensors.torch import load_file

        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata.get("architecture") != ARCHITECTURE:
            raise ValueError("Unknown patch matcher architecture")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("weights_sha256"):
            raise ValueError("Patch matcher weights hash mismatch")
        model = _network()
        model.load_state_dict(load_file(str(path)), strict=True)
        return cls(model, metadata, device)


class PatchErrorFeatures:
    """Cache only observed histology embeddings against the expected MRI gallery."""

    def __init__(self, matcher, mri):
        self.matcher = matcher
        self.reference = matcher.encode(mri, "mri")
        self.cache = {}

    def embeddings(self, images):
        keys = [hashlib.sha256(image.tobytes()).digest() for image in images]
        missing = {
            key: image for key, image in zip(keys, images, strict=True) if key not in self.cache
        }
        if missing:
            encoded = self.matcher.encode(np.stack(list(missing.values())), "histology")
            self.cache.update(zip(missing, encoded, strict=True))
        return np.stack([self.cache[key] for key in keys])

    def measure(self, images):
        if len(images) != len(self.reference):
            raise ValueError("Expected MRI and histology counts differ")
        desc = self.embeddings(images)
        flipped = self.embeddings(images[:, :, ::-1])
        similarity = desc @ self.reference.T
        flip_scores = np.sum(flipped * self.reference, axis=1)
        return np.column_stack(
            (comparison_features(similarity, flip_scores), context_features(desc, flipped))
        )


def _disjoint(train, validation):
    for modality in (0, 1):
        train_hashes = {
            hashlib.sha256(image.tobytes()).digest() for pair in train for image in pair[modality]
        }
        val_hashes = {
            hashlib.sha256(image.tobytes()).digest()
            for pair in validation
            for image in pair[modality]
        }
        if train_hashes & val_hashes:
            raise ValueError("Training and validation images overlap")


def fit_patch_matcher(
    train,
    validation,
    *,
    epochs=150,
    seed=26018,
    device="auto",
    provenance=None,
    preprocessing="canonical_v1",
):
    """Select by clean-pair validation loss; no test images are accepted by this API."""
    torch = _torch()
    if epochs < 1 or not train or len(train) != len(validation):
        raise ValueError("Need positive epochs and matching nonempty training/validation donors")
    for hist, mri in [*train, *validation]:
        if len(hist) != len(mri) or len(hist) < 2:
            raise ValueError("Need at least two corresponding images per donor and split")
    _disjoint(train, validation)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    selected_device = _device(device)
    model = _network().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    prepared = []
    for group in (train, validation):
        prepared.append(
            [
                (
                    torch.from_numpy(prepare(h, "histology", preprocessing)).to(selected_device),
                    torch.from_numpy(prepare(m, "mri", preprocessing)).to(selected_device),
                )
                for h, m in group
            ]
        )
    training, val = prepared
    temperature = 0.07

    def encode(tensor, modality):
        return torch.cat([model(tensor[i : i + 32], modality) for i in range(0, len(tensor), 32)])

    def measure():
        model.eval()
        results = []
        with torch.inference_mode():
            for h, m in val:
                a, b = encode(h, "histology"), encode(m, "mri")
                logits = a @ b.T / temperature
                labels = torch.arange(len(h), device=selected_device)
                loss = (
                    torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)
                ) / 2
                results.append(
                    {
                        "loss": float(loss.cpu()),
                        "top1": float((logits.argmax(1) == labels).float().mean().cpu()),
                    }
                )
        return results

    best_loss = math.inf
    best_state = None
    best_epoch = 0
    history = []
    patience = 0
    steps = max(1, math.ceil(sum(len(h) for h, _ in training) / 16))
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            losses = []
            for step in range(steps):
                h, m = training[int(rng.integers(len(training)))]
                batch = min(16, len(h))
                if step % 2 == 0:
                    start = int(rng.integers(len(h) - batch + 1))
                    indices = np.arange(start, start + batch)
                else:
                    indices = rng.choice(len(h), batch, replace=False)
                a, b = h[indices].clone(), m[indices].clone()
                if rng.random() < 0.5:
                    a, b = a.flip(-1), b.flip(-1)
                if rng.random() < 0.5:
                    a, b = a.flip(-2), b.flip(-2)
                for tensor in (a, b):
                    gamma = float(rng.uniform(0.8, 1.2))
                    tensor.copy_(tensor.pow(gamma))
                    tensor.add_(torch.randn_like(tensor) * 0.01).clamp_(0, 1)
                ah, bm = model(a, "histology"), model(b, "mri")
                logits = ah @ bm.T / temperature
                labels = torch.arange(batch, device=selected_device)
                loss = (
                    torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)
                ) / 2
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        if epoch % 5 == 0 or epoch == epochs:
            measured = measure()
            quality = float(np.mean([row["loss"] for row in measured]))
            history.append(
                {
                    "epoch": epoch,
                    "validation": measured,
                    "training_loss": float(np.mean(losses)) if epoch else None,
                }
            )
            top1 = [round(row["top1"], 3) for row in measured]
            print(
                f"Patch epoch {epoch}: validation loss {quality:.4f}, top1 {top1}",
                flush=True,
            )
            if quality < best_loss - 1e-5:
                best_loss, best_epoch = quality, epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= 8:
                    break
    model.load_state_dict(best_state)
    metadata = {
        "schema_version": 1,
        "architecture": ARCHITECTURE,
        "image_size": 96,
        "preprocessing": preprocessing,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selection": "Mean validation symmetric pair cross entropy",
        "temperature": temperature,
        "epochs_requested": epochs,
        "history": history,
        "training_pairs": [len(h) for h, _ in train],
        "validation_pairs": [len(h) for h, _ in validation],
        "batch_sampling": "Alternating contiguous same-donor blocks and random same-donor pairs",
        "provenance": provenance or {},
        "limitations": [
            "Two-dimensional paired-image matching; no pose or reconstruction",
            "Not a Synth-by-Reg reproduction; testing remains exploratory",
        ],
    }
    return PatchMatcher(model, metadata, selected_device)
