"""Nonlinear detector fitting and portable inference preserve the score contract."""

import json
import sys

import numpy as np
import pytest

from slicegps.error_detection import detector_scores
from slicegps.error_mlp import fit_mlp_detectors


def test_numpy_inference_has_no_torch_or_pickle_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    model = {
        "classifier": "tanh_mlp_v1",
        "center": [0, 0],
        "scale": [1, 2],
        "layers": [
            {"weight": [[1, 2], [0, 1]], "bias": [0.1, -0.2]},
            {"weight": [[0.5, -0.2]], "bias": [0.3]},
        ],
    }
    values = np.array([[1, 4], [-1, 2]])
    hidden = np.tanh(values / [1, 2] @ np.array([[1, 0], [2, 1]]) + [0.1, -0.2])
    expected = hidden @ [0.5, -0.2] + 0.3
    np.testing.assert_allclose(detector_scores(values, json.loads(json.dumps(model))), expected)
    model["layers"][1]["weight"] = [[1, 2, 3]]
    with pytest.raises(ValueError, match="layer contract"):
        detector_scores(values, model)


def test_nonlinear_candidates_learn_interactions_on_separate_test_points():
    torch = pytest.importorskip("torch")
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(28)
        train = rng.uniform(-1, 1, (256, 2))
        labels = train[:, 0] * train[:, 1] > 0
        candidates = fit_mlp_detectors(train, labels, checkpoints=(25, 200), device="cpu")
        assert [c["epochs"] for c in candidates] == [25, 200]
        test = rng.uniform(-1, 1, (256, 2))
        truth = test[:, 0] * test[:, 1] > 0
        exported = json.loads(json.dumps(candidates[-1]))
        assert np.mean((detector_scores(test, exported) > 0) == truth) > 0.9
        assert np.isfinite(detector_scores(test, exported)).all()
    finally:
        torch.set_num_threads(original)


def test_invalid_training_inputs_are_rejected():
    with pytest.raises(ValueError, match="positive epoch"):
        fit_mlp_detectors(np.ones((6, 2)), [False, True] * 3, checkpoints=(0,))
    with pytest.raises(ValueError, match="both labels"):
        fit_mlp_detectors(np.ones((6, 2)), [False] * 6)
