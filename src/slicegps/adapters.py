"""Small image-feature adapters without torch, coordinates, or section labels.

Fit only on training pairs. Feature extraction, train/test donor separation, and
selection of regularization belong to the caller; query targets never enter fit.
"""

from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np


def _real_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not (
        np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unit_features(values: np.ndarray, name: str) -> np.ndarray:
    array = _real_array(values, name)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional feature matrix")
    # Match retrieval's cosine convention and avoid overflow for finite large inputs.
    scale = np.max(np.abs(array), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError(f"{name} contains a zero feature vector")
    scaled = array / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class RidgeAdapter:
    """A centered dual ridge map from histology to MRI feature space.

    Arrays are private model state in the statistical sense, but named fields are
    exposed for inspection. Copies are made read-only to avoid accidental changes.
    The fit cost uses an N by N kernel rather than a descriptor-width square matrix.
    """

    train_centered: np.ndarray
    dual_weights: np.ndarray
    input_mean: np.ndarray
    target_mean: np.ndarray
    regularization: float
    kernel_scale: float

    def __post_init__(self) -> None:
        arrays = {
            name: _real_array(getattr(self, name), name).copy()
            for name in ("train_centered", "dual_weights", "input_mean", "target_mean")
        }
        train, weights = arrays["train_centered"], arrays["dual_weights"]
        input_mean, target_mean = arrays["input_mean"], arrays["target_mean"]
        if train.ndim != 2 or train.shape[0] < 2 or train.shape[1] == 0:
            raise ValueError("train_centered must have shape (N, d), with N >= 2 and d > 0")
        if weights.ndim != 2 or weights.shape[0] != len(train) or weights.shape[1] == 0:
            raise ValueError("dual_weights must have shape (N, D), with D > 0")
        if input_mean.shape != (train.shape[1],) or target_mean.shape != (weights.shape[1],):
            raise ValueError("model means must match input and target feature dimensions")
        regularization = _positive_scalar(self.regularization, "regularization")
        kernel_scale = _positive_scalar(self.kernel_scale, "kernel_scale")
        if not np.allclose(train.mean(axis=0), 0, rtol=0, atol=1e-10):
            raise ValueError("train_centered must have zero column means")
        measured_scale = float(np.mean(np.einsum("ij,ij->i", train, train)))
        if not np.isclose(kernel_scale, measured_scale, rtol=1e-10, atol=0):
            raise ValueError("kernel_scale must equal the mean centered squared row norm")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "regularization", regularization)
        object.__setattr__(self, "kernel_scale", kernel_scale)

    def transform(self, query: np.ndarray) -> np.ndarray:
        """Map image descriptors ``(Q, d)`` to unit float32 MRI descriptors ``(Q, D)``.

        Uses the training means stored by fit. A zero predicted descriptor raises
        ValueError because it has no cosine direction for downstream retrieval.
        """
        values = _unit_features(query, "query")
        if values.shape[1] != self.train_centered.shape[1]:
            raise ValueError("query feature dimension must match the fitted input dimension")
        prediction = (
            (values - self.input_mean) @ self.train_centered.T @ self.dual_weights
            + self.target_mean
        )
        return _unit_features(prediction, "predicted features").astype(np.float32)

    def save(self, path: Path) -> None:
        """Save only numeric model arrays to an explicit .npz path."""
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("adapter path must end in .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                format_version=np.array(1, dtype=np.int64),
                train_centered=self.train_centered,
                dual_weights=self.dual_weights,
                input_mean=self.input_mean,
                target_mean=self.target_mean,
                regularization=np.array(self.regularization, dtype=np.float64),
                kernel_scale=np.array(self.kernel_scale, dtype=np.float64),
            )

    @classmethod
    def load(cls, path: Path) -> "RidgeAdapter":
        """Load a numeric .npz model, rejecting object arrays and unknown schemas."""
        expected = {
            "format_version", "train_centered", "dual_weights", "input_mean", "target_mean",
            "regularization", "kernel_scale",
        }
        archive = np.load(Path(path), allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("adapter must be an .npz archive")
        with archive:
            if set(archive.files) != expected:
                raise ValueError("adapter archive has missing or unexpected fields")
            version = archive["format_version"]
            if version.shape != () or version.dtype.kind not in "iu" or int(version) != 1:
                raise ValueError("unsupported adapter format_version")
            for name in ("regularization", "kernel_scale"):
                if archive[name].shape != () or archive[name].dtype.kind not in "fiu":
                    raise ValueError(f"adapter {name} must be a numeric scalar")
            return cls(
                train_centered=archive["train_centered"],
                dual_weights=archive["dual_weights"],
                input_mean=archive["input_mean"],
                target_mean=archive["target_mean"],
                regularization=float(archive["regularization"]),
                kernel_scale=float(archive["kernel_scale"]),
            )


def fit_ridge(
    histology_features: np.ndarray,
    mri_features: np.ndarray,
    regularization: float = 1.0,
) -> RidgeAdapter:
    """Fit a dual ridge map using paired training image descriptors only.

    Normalize rows of X and Y, subtract their respective training means, and set
    K = Xc @ Xc.T. Solve (K + lambda_eff I) A = Yc, where lambda_eff equals
    ``regularization * mean(diag(K))``. Regularization must be strictly positive;
    training inputs must contain variation after normalization. No sample-count
    multiplier is applied. Predictions add the training target mean before L2
    normalization. Unequal histology and MRI feature widths are supported.
    """
    regularization = _positive_scalar(regularization, "regularization")
    inputs = _unit_features(histology_features, "histology_features")
    targets = _unit_features(mri_features, "mri_features")
    if len(inputs) != len(targets) or len(inputs) < 2:
        raise ValueError("paired feature matrices must have the same N >= 2 training rows")
    input_mean, target_mean = inputs.mean(axis=0), targets.mean(axis=0)
    centered = inputs - input_mean
    kernel = centered @ centered.T
    kernel_scale = float(np.trace(kernel) / len(kernel))
    if not np.isfinite(kernel_scale) or kernel_scale <= 0:
        raise ValueError("histology training features must vary after row normalization")
    penalty = regularization * kernel_scale
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("effective regularization must be finite and positive")
    kernel.flat[:: len(kernel) + 1] += penalty
    try:
        weights = np.linalg.solve(kernel, targets - target_mean)
    except np.linalg.LinAlgError as error:
        raise ValueError("training kernel could not be solved with this regularization") from error
    return RidgeAdapter(centered, weights, input_mean, target_mean, regularization, kernel_scale)
