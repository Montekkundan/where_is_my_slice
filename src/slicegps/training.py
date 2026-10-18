"""Exploratory contrastive matching and center/normal regression on frozen features.

Training geometry supplies targets only. The inference API accepts image features,
never labels or query coordinates. A within-volume synthetic holdout is not donor
or raw-histology pose validation, particularly when BigBrain is paired to an atlas.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from .data import write_json
from .geometry import Plane
from .retrieval import evaluate_retrieval, rank


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Dual-head training requires optional torch and safetensors") from exc
    return torch


def _device(name: str) -> str:
    torch = _torch()
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if name not in {"cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    return name


def _matrix(values, name: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty real feature matrix")
    array = array.astype(np.float32)
    if not np.isfinite(array).all() or np.any(np.max(np.abs(array), axis=1) == 0):
        raise ValueError(f"{name} must contain finite nonzero feature vectors")
    if width is not None and array.shape[1] != width:
        raise ValueError(f"{name} feature width mismatch")
    return array


def _targets(planes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    parsed = [Plane.from_dict(p) for p in planes]
    if not parsed:
        raise ValueError("Plane lists must be nonempty")
    return (
        np.array([p.center_mm for p in parsed], dtype=np.float32),
        np.array([p.normal for p in parsed], dtype=np.float32),
    )


def _network(architecture: dict):
    torch = _torch()

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            hd, md = architecture["histology_input_dim"], architecture["mri_input_dim"]
            hidden, embedding = architecture["hidden_dim"], architecture["embedding_dim"]
            self.histology_hidden = torch.nn.Sequential(
                torch.nn.Linear(hd, hidden), torch.nn.GELU()
            )
            self.histology_projection = torch.nn.Linear(hidden, embedding)
            self.mri_head = torch.nn.Sequential(
                torch.nn.Linear(md, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, embedding)
            )
            self.pose_head = torch.nn.Linear(hidden, 6)
            for name, width in (
                ("histology_mean", hd),
                ("histology_std", hd),
                ("mri_mean", md),
                ("mri_std", md),
                ("center_mean", 3),
                ("center_std", 3),
                ("normal_prior", 3),
            ):
                initial = torch.ones(width) if name.endswith("std") else torch.zeros(width)
                self.register_buffer(name, initial)

        def histology(self, values):
            hidden = self.histology_hidden((values - self.histology_mean) / self.histology_std)
            embedding = torch.nn.functional.normalize(self.histology_projection(hidden), dim=1)
            return embedding, self.pose_head(hidden)

        def mri(self, values):
            return torch.nn.functional.normalize(
                self.mri_head((values - self.mri_mean) / self.mri_std), dim=1
            )

    return Network()


class DualMatcher:
    """Safe-artifact inference: histology and MRI must use their respective heads."""

    def __init__(self, model, metadata: dict, device: str = "cpu", batch_size: int = 256):
        self.device = _device(device)
        self._model = model.to(self.device).eval()
        self._model.requires_grad_(False)
        self._metadata = deepcopy(metadata)
        self.batch_size = batch_size

    @property
    def metadata(self) -> dict:
        return deepcopy(self._metadata)

    def _infer(self, values, modality: str) -> np.ndarray:
        torch = _torch()
        key = "histology_input_dim" if modality != "mri" else "mri_input_dim"
        array = _matrix(values, modality, self._metadata["architecture"][key])
        result = []
        with torch.inference_mode():
            for start in range(0, len(array), self.batch_size):
                tensor = torch.from_numpy(array[start : start + self.batch_size]).to(self.device)
                if modality == "mri":
                    output = self._model.mri(tensor)
                else:
                    embedding, pose = self._model.histology(tensor)
                    output = embedding if modality == "histology" else pose
                result.append(output.cpu().numpy())
        combined = np.concatenate(result).astype(np.float32)
        if not np.isfinite(combined).all():
            raise ValueError("Model produced non-finite predictions")
        return combined

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project histology frozen descriptors into the shared 128D retrieval space."""
        return self._infer(features, "histology")

    def encode_gallery(self, features: np.ndarray) -> np.ndarray:
        """Project MRI frozen descriptors; raw DINO gallery features are incompatible."""
        return self._infer(features, "mri")

    def predict_pose(self, features: np.ndarray) -> dict[str, np.ndarray]:
        """Predict centers and oriented unit normals, excluding in-plane rotation."""
        pose = self._infer(features, "pose")
        center_mean = self._model.center_mean.cpu().numpy()
        center_std = self._model.center_std.cpu().numpy()
        normals = pose[:, 3:]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("Model produced a degenerate plane normal")
        return {
            "centers_mm": pose[:, :3] * center_std + center_mean,
            "normals_ras": normals / norms,
        }

    def save(self, path: Path) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in self._model.state_dict().items()
        }
        save_file(state, str(path), metadata={"format": "pt"})
        metadata = self.metadata
        metadata["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(path.with_suffix(".json"), metadata)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> DualMatcher:
        from safetensors.torch import load_file

        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata.get("schema_version") != 1 or metadata.get("kind") != "dual_contrastive_pose":
            raise ValueError("Unsupported dual matcher metadata")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("weights_sha256"):
            raise ValueError("Dual matcher weights differ from their recorded hash")
        architecture = metadata["architecture"]
        if (
            any(
                not isinstance(architecture.get(k), int) or architecture[k] < 1
                for k in ("histology_input_dim", "mri_input_dim", "hidden_dim", "embedding_dim")
            )
            or architecture.get("pose_dim") != 6
        ):
            raise ValueError("Invalid dual matcher architecture")
        model = _network(architecture)
        model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        return cls(model, metadata, device)


def _retrieval(matcher, histology, gallery, query_planes, gallery_planes, tolerance):
    indices, _ = rank(matcher.transform(histology), matcher.encode_gallery(gallery), k=5)
    qc, _ = _targets(query_planes)
    gc, _ = _targets(gallery_planes)
    return evaluate_retrieval(indices, qc, gc, tolerance)


def _pose_metrics(predicted: dict, centers: np.ndarray, normals: np.ndarray) -> dict:
    error = np.linalg.norm(predicted["centers_mm"] - centers, axis=1)
    dots = np.sum(predicted["normals_ras"] * normals, axis=1)
    angle = np.rad2deg(np.arccos(np.clip(dots, -1, 1)))
    return {
        "center_median_error_mm": float(np.median(error)),
        "center_p95_error_mm": float(np.percentile(error, 95)),
        "normal_median_angle_deg": float(np.median(angle)),
        "normal_p95_angle_deg": float(np.percentile(angle, 95)),
        "normal_metric": "acos oriented unit-normal dot product; in-plane rotation unmeasured",
    }


def train_dual(
    features_path: Path,
    protocol: Path,
    output: Path,
    *,
    epochs: int = 100,
    seed: int = 26018,
    device: str = "cpu",
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    temperature: float = 0.07,
    pose_weight: float = 0.1,
    validation_every: int = 5,
    patience: int = 8,
    tolerance_mm: float = 5,
) -> dict:
    """Fit on training pairs, select by validation, save, then evaluate test once.

    Test data from earlier ridge exploration are explicitly reused exploratory
    data, not a new independent confirmation. The frozen image encoder is never
    trained. Training-only center/normal supervision updates the shared histology
    trunk and therefore its retrieval features. Inference consumes image features only.
    """
    for name, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("validation_every", validation_every),
        ("patience", patience),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if batch_size < 2:
        raise ValueError("Contrastive training needs batch_size >= 2")
    if any(not np.isfinite(v) or v <= 0 for v in (learning_rate, temperature, tolerance_mm)):
        raise ValueError("learning_rate, temperature, and tolerance_mm must be positive")
    if not np.isfinite(pose_weight) or pose_weight < 0:
        raise ValueError("pose_weight must be finite and nonnegative")
    features_path, protocol, output = Path(features_path), Path(protocol), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training output must be new or empty")
    manifest_path = protocol if protocol.is_file() else protocol / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    planes, gallery_planes = manifest["planes"], manifest["gallery_planes"]
    from .learned import check_separation

    if "depth_supports_mm" in manifest:
        check_separation(manifest["depth_supports_mm"])
    with np.load(features_path, allow_pickle=False) as archive:
        # Test feature values are deliberately not read until the checkpoint is fixed.
        train_h = _matrix(archive["train_histology"], "training histology")
        train_m = _matrix(archive["train_mri"], "training MRI")
        val_h = _matrix(archive["validation_histology"], "validation histology", train_h.shape[1])
        gallery = _matrix(archive["gallery_mri"], "gallery MRI", train_m.shape[1])
    if len(train_h) < 2 or len(train_h) != len(train_m) or len(train_h) != len(planes["train"]):
        raise ValueError("Training pairs and plane targets must have equal counts >=2")
    if len(val_h) != len(planes["validation"]) or len(gallery) != len(gallery_planes):
        raise ValueError("Validation/gallery feature and plane counts must match")
    pooling = features_path.stem.removeprefix("features_")
    encoder_metadata = None
    provenance_path = features_path.with_name(f"adapter_{pooling}.json")
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema_version") != 1
            or provenance.get("pooling") != pooling
            or provenance.get("feature_dim") != train_h.shape[1]
            or provenance.get("feature_dim") != train_m.shape[1]
            or not isinstance(provenance.get("encoder"), dict)
            or provenance["encoder"].get("pooling") != pooling
            or not manifest.get("images_sha256")
            or provenance.get("training_images_sha256") != manifest["images_sha256"]
        ):
            raise ValueError("Cached feature/protocol provenance mismatch")
        encoder_metadata = provenance["encoder"]
    centers, normals = _targets(planes["train"])
    center_mean, raw_center_std = centers.mean(axis=0), centers.std(axis=0)
    center_std = np.where(raw_center_std < 1e-6, 1, raw_center_std)
    normal_prior = normals.mean(axis=0)
    normal_prior /= np.linalg.norm(normal_prior)
    architecture = {
        "histology_input_dim": train_h.shape[1],
        "mri_input_dim": train_m.shape[1],
        "hidden_dim": 256,
        "embedding_dim": 128,
        "pose_dim": 6,
        "activation": "GELU",
    }
    torch = _torch()
    torch.manual_seed(seed)
    actual_device = _device(device)
    model = _network(architecture).to(actual_device)
    for name, array in (
        ("histology_mean", train_h.mean(axis=0)),
        ("histology_std", np.maximum(train_h.std(axis=0), 1e-6)),
        ("mri_mean", train_m.mean(axis=0)),
        ("mri_std", np.maximum(train_m.std(axis=0), 1e-6)),
        ("center_mean", center_mean),
        ("center_std", center_std),
        ("normal_prior", normal_prior),
    ):
        getattr(model, name).copy_(torch.from_numpy(array).to(actual_device))
    metadata = {
        "schema_version": 1,
        "kind": "dual_contrastive_pose",
        "architecture": architecture,
        "feature_statistics": (
            "Per-channel means/std fitted on training modality only; std floor1e-6"
        ),
        "pose_scaling": {
            "center_mean_mm": center_mean.tolist(),
            "center_std_mm": center_std.tolist(),
            "constant_training_center_axes": np.flatnonzero(raw_center_std < 1e-6).tolist(),
            "normal_target": "oriented unit RAS normal, no centering",
            "normal_prior_ras": normal_prior.tolist(),
        },
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "protocol_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "protocol_images_sha256": manifest.get("images_sha256"),
        "pooling": pooling,
        "encoder": encoder_metadata,
        "feature_provenance": (
            "Matched feature sidecar to protocol image hash, pooling, and feature width"
            if encoder_metadata is not None
            else "No encoder sidecar; generic offline features have no verified encoder identity"
        ),
        "supervision": (
            "Training paired contrastive labels plus plane center/normal targets; "
            "pose loss updates the shared histology trunk and retrieval features"
        ),
        "scope": manifest.get("scope", "Unspecified source; exploratory frozen-feature experiment"),
        "selection": (
            "Validation Recall@1, then median retrieval-center error; earliest tied checkpoint"
        ),
        "config": {
            "epochs": epochs,
            "seed": seed,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "pose_weight": pose_weight,
            "validation_every": validation_every,
            "patience_checks": patience,
            "device_requested": device,
            "device_used": actual_device,
        },
    }
    h = torch.from_numpy(train_h).to(actual_device)
    m = torch.from_numpy(train_m).to(actual_device)
    target = torch.from_numpy(
        np.concatenate(((centers - center_mean) / center_std, normals), axis=1)
    )
    target = target.to(actual_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)

    def loss_for(indices):
        eh, pose = model.histology(h[indices])
        em = model.mri(m[indices])
        similarity = eh @ em.T / temperature
        labels = torch.arange(len(indices), device=actual_device)
        contrastive = (
            torch.nn.functional.cross_entropy(similarity, labels)
            + torch.nn.functional.cross_entropy(similarity.T, labels)
        ) / 2
        pose_loss = torch.nn.functional.mse_loss(pose, target[indices])
        return contrastive + pose_weight * pose_loss, contrastive, pose_loss

    all_indices = torch.arange(len(h), device=actual_device)
    model.eval()
    with torch.inference_mode():
        initial_loss = float(loss_for(all_indices)[1].cpu())
    best_state, best_key, best_epoch, stale = None, None, None, 0
    history, validation_history = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        sums = np.zeros(3)
        for take in np.array_split(rng.permutation(len(h)), max(1, math.ceil(len(h) / batch_size))):
            indices = torch.as_tensor(take, device=actual_device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_for(indices)
            if not bool(torch.isfinite(losses[0])):
                raise ValueError("Non-finite training loss")
            losses[0].backward()
            optimizer.step()
            sums += np.array([float(loss.detach().cpu()) for loss in losses]) * len(take)
        history.append(
            {
                "epoch": epoch,
                "loss": float(sums[0] / len(h)),
                "contrastive_loss": float(sums[1] / len(h)),
                "pose_loss": float(sums[2] / len(h)),
            }
        )
        if epoch % validation_every == 0 or epoch == epochs:
            matcher = DualMatcher(model, metadata, actual_device)
            metrics = _retrieval(
                matcher, val_h, gallery, planes["validation"], gallery_planes, tolerance_mm
            )
            validation_history.append({"epoch": epoch, **metrics})
            key = (-metrics["recall_at_k"]["1"], metrics["top1_median_error_mm"])
            if best_key is None or key < best_key:
                best_key, best_epoch, stale = key, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            # The wrapper freezes inference; restore trainability before the next optimizer step.
            model.requires_grad_(True)
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        selected_contrastive = float(loss_for(all_indices)[1].cpu())
    metadata["selected_epoch"] = best_epoch
    metadata["epochs_completed"] = epoch
    metadata["test_evaluated_during_selection"] = False
    output.mkdir(parents=True, exist_ok=True)
    weights = output / "model.safetensors"
    DualMatcher(model, metadata, actual_device).save(weights)
    # Reload the persisted fixed checkpoint before reading/scoring test feature values.
    matcher = DualMatcher.load(weights, actual_device)
    with np.load(features_path, allow_pickle=False) as archive:
        test_h = _matrix(archive["test_histology"], "test histology", train_h.shape[1])
    if len(test_h) != len(planes["test"]):
        raise ValueError("Test feature and plane counts must match")
    test_centers, test_normals = _targets(planes["test"])
    predicted = matcher.predict_pose(test_h)
    prior = {
        "centers_mm": np.repeat(center_mean[None], len(test_h), axis=0),
        "normals_ras": np.repeat(normal_prior[None], len(test_h), axis=0),
    }
    report = {
        "schema_version": 1,
        "kind": "exploratory_dual_contrastive_pose",
        "weights": str(weights.resolve()),
        "architecture": architecture,
        "config": metadata["config"],
        "selected_epoch": best_epoch,
        "epochs_completed": epoch,
        "initial_full_train_contrastive_loss": initial_loss,
        "selected_full_train_contrastive_loss": selected_contrastive,
        "training_history": history,
        "validation_history": validation_history,
        "test_retrieval": _retrieval(
            matcher, test_h, gallery, planes["test"], gallery_planes, tolerance_mm
        ),
        "test_pose": _pose_metrics(predicted, test_centers, test_normals),
        "training_mean_pose_prior_baseline": _pose_metrics(prior, test_centers, test_normals),
        "selection": metadata["selection"],
        "test_evaluated_during_selection": False,
        "scope": metadata["scope"],
        "limitations": [
            "Exploratory reuse of a test set already exposed by earlier ridge experiments",
            "Same-volume synthetic query planes; no independent donor validation",
            "BigBrain atlas pairing is not donor-specific MRI correspondence",
            "Known crop and near-coronal normals; center plus normal excludes in-plane rotation",
            "No validated raw-histology or unknown-fragment pose prediction",
        ],
    }
    write_json(output / "metrics.json", report)
    return report
