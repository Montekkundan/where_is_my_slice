"""Offline CLI workflow checks using MRI-only galleries and exported section images."""

import json

import numpy as np
import pytest
from PIL import Image

from slicegps.cli import main
from slicegps.pipeline import create_phantom, generate, read_bundle
from slicegps.retrieval import _prepare


def test_help_exposes_training_matching_and_preparation(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("mri-gallery", "learned-fit", "train-dual", "stack-demo", "match"):
        assert command in help_text


def test_offline_gallery_index_and_exported_query_workflow(tmp_path, capsys):
    _, mri = create_phantom(tmp_path / "fixture")
    gallery, index, matches = (tmp_path / name for name in ("gallery", "index", "matches"))
    main(
        [
            "mri-gallery",
            str(mri),
            "--output",
            str(gallery),
            "--count",
            "8",
            "--size",
            "48",
            "--pixel-mm",
            "1.3",
        ]
    )
    main(["index", str(gallery), "--output", str(index), "--method", "intensity"])
    _, arrays = read_bundle(gallery)
    scaled, _ = _prepare(arrays["gallery_mri"][2])
    query = tmp_path / "section.png"
    Image.fromarray(np.rint(scaled * 255).astype(np.uint8)).save(query)
    main(["match", str(index), str(query), "--output", str(matches), "--k", "3"])
    report = json.loads((matches / "matches.json").read_text())
    assert report["matches"][0]["candidates"][0]["gallery_index"] == 2
    assert (matches / "index.html").exists()
    assert '"viewer"' in capsys.readouterr().out


def test_query_augmentations_are_seeded_and_preserve_first_plane(tmp_path):
    hist, mri = create_phantom(tmp_path / "fixture")
    outputs = [tmp_path / name for name in ("plain", "augmented", "repeated")]
    for output, augment in zip(outputs, (False, True, True), strict=True):
        generate(hist, mri, output, count=8, size=48, pixel_mm=1.3, seed=12, augment=augment)
    plain_manifest, plain = read_bundle(outputs[0])
    aug_manifest, augmented = read_bundle(outputs[1])
    _, repeated = read_bundle(outputs[2])
    assert plain_manifest["query_planes"][0] == aug_manifest["query_planes"][0]
    assert "tear" in aug_manifest["query_augmentations"]
    assert not np.array_equal(plain["query_histology"][0], augmented["query_histology"][0])
    np.testing.assert_array_equal(augmented["query_histology"], repeated["query_histology"])
    assert np.isfinite(augmented["query_histology"]).all()
