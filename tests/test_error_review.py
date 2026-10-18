"""Saved-model inference never consumes benchmark truth or section IDs."""

import json

import numpy as np
import pytest
from PIL import Image

from slicegps.cli import main
from slicegps.error_detection import error_features, fit_detector
from slicegps.error_review import review_errors


def fixture(tmp_path):
    rng = np.random.default_rng(91)
    images = rng.integers(0, 255, size=(9, 32, 32), dtype=np.uint8)
    paths = []
    for i, image in enumerate(images):
        path = tmp_path / f"anonymous-{i:02d}.png"
        Image.fromarray(image).save(path)
        paths.append(path)
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(x).resize((96, 96), Image.Resampling.BILINEAR), dtype=np.float32
            )
            / 255
            for x in images
        ]
    )
    x = error_features(resized)
    model = fit_detector(x, np.array([False] * 5 + [True] * 4))
    model.update(
        feature_contract="stack_error_features_v3",
        threshold=0.0,
        expected_mri_required=False,
        learned_features=False,
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model))
    return model_path, paths


def test_saved_detector_and_cli_write_truth_free_review(tmp_path, capsys):
    model, paths = fixture(tmp_path)
    first = review_errors(model, paths, tmp_path / "first")
    assert len(first["sections"]) == len(paths)
    assert first["sections"][0]["flagged"] is None
    assert first["sections"][-1]["status"] == "insufficient_context"
    assert all("is_error" not in row and "source_index" not in row for row in first["sections"])
    main(["detect-errors", str(model), *map(str, paths), "--output", str(tmp_path / "cli")])
    assert json.loads(capsys.readouterr().out)["flagged_indices"] == first["flagged_indices"]
    assert (tmp_path / "cli/index.html").exists()
    with pytest.raises(FileExistsError):
        review_errors(model, paths, tmp_path / "first")


def test_missing_mri_and_wrong_contract_are_explicit(tmp_path):
    model_path, paths = fixture(tmp_path)
    model = json.loads(model_path.read_text())
    model["expected_mri_required"] = True
    model_path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="requires the expected MRI"):
        review_errors(model_path, paths, tmp_path / "absent_mri")
    model["feature_contract"] = "unknown"
    model_path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="contract"):
        review_errors(model_path, paths, tmp_path / "unknown")


def test_float_tiff_transport_matches_png_features(tmp_path):
    model, paths = fixture(tmp_path)
    float_paths = []
    for path in paths:
        with Image.open(path) as image:
            values = (
                np.asarray(image.resize((96, 96), Image.Resampling.BILINEAR), dtype=np.float32)
                / 255
            )
        target = path.with_suffix(".tiff")
        Image.fromarray(values).save(target)
        float_paths.append(target)
    png = review_errors(model, paths, tmp_path / "png")
    tiff = review_errors(model, float_paths, tmp_path / "tiff")
    assert png["flagged_indices"] == tiff["flagged_indices"]
    np.testing.assert_allclose(
        [r["score"] for r in png["sections"]], [r["score"] for r in tiff["sections"]], atol=1e-6
    )
