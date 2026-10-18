"""Integration checks for image-only matching and explicit geometry assumptions."""

import json

import nibabel as nib
import numpy as np
import pytest

from slicegps.benchmark import benchmark
from slicegps.cli import main
from slicegps.mosaic import preview_mosaic
from slicegps.pipeline import create_phantom, generate, read_bundle
from slicegps.volume import audit_volume, save_volume


def test_phantom_pipeline_and_no_label_leakage(tmp_path):
    hist, mri = create_phantom(tmp_path / "fixture")
    bundle = tmp_path / "planes"
    generate(hist, mri, bundle, count=10, size=32, pixel_mm=2.5)
    first = benchmark(bundle, tmp_path / "first")
    assert sorted(first["query_permutation"]) == list(range(10))
    assert first["unchanged_shuffled_order_inversion_fraction"] > 0.1
    assert len(first["results"]) == 6
    manifest, images = read_bundle(bundle)
    assert images["query_histology"].shape == (10, 32, 32)
    assert not np.array_equal(images["query_histology"], images["gallery_histology"])
    # Deliberately false scoring coordinates must never change pixel-based rankings.
    for plane in manifest["gallery_planes"]:
        plane["center_mm"][0] += 1000
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    second = benchmark(bundle, tmp_path / "second")
    for a, b in zip(first["results"], second["results"], strict=True):
        assert a["ranked_gallery_indices"] == b["ranked_gallery_indices"]
        assert b["top1_median_error_mm"] > 900
    assert (tmp_path / "first" / "mri_gradient_matches.png").exists()


@pytest.mark.parametrize("units", ["meter", "micron", "unknown"])
def test_generator_rejects_unsupported_units(tmp_path, units):
    hist, mri = create_phantom(tmp_path / "fixture")
    image = nib.load(hist)
    image.header.set_xyzt_units(units)
    nib.save(image, hist)
    with pytest.raises(ValueError, match="units"):
        generate(hist, mri, tmp_path / "bad", count=4)
    if units == "unknown":
        manifest = generate(hist, mri, tmp_path / "explicit", count=4, assume_mm=True)
        assert manifest["assume_mm_for_unspecified_units"] is True


def test_constant_pair_fails_with_clear_reason(tmp_path):
    hist, mri = tmp_path / "h.nii.gz", tmp_path / "m.nii.gz"
    save_volume(hist, np.ones((12, 12, 12)), np.eye(4))
    save_volume(mri, np.zeros((12, 12, 12)), np.eye(4))
    with pytest.raises(ValueError, match="Too few"):
        generate(hist, mri, tmp_path / "bad", count=5, size=16)


def test_cli_demo(tmp_path, capsys):
    main(["demo", "--output", str(tmp_path / "demo")])
    result = json.loads(capsys.readouterr().out)
    assert result["human_data"] is False
    assert (tmp_path / "demo/benchmark/report.md").exists()


def test_mosaic_uses_own_affines_and_records_slab(tmp_path):
    case = tmp_path / "donor"
    case.mkdir()
    (case / "block2slab_mapping.csv").write_text("BLOCK_NUM,BLOCK_NAME\n4,A5.1\n27,A5.2\n")
    save_volume(case / "mri_rotated/mri.nii.gz", np.ones((24, 12, 12)) * 100, np.eye(4))
    save_volume(case / "histology_blocks/4/HE.nii.gz", np.ones((8, 8, 8)), np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 10
    save_volume(case / "histology_blocks/27/HE.nii.gz", np.ones((8, 8, 8)), shifted)
    result = preview_mosaic(case, [4, 27], tmp_path / "mosaic", pixel_mm=1, size=32)
    assert result["slab"] == "A5"
    assert result["covered_pixels"] > 0
    assert result["overlap_pixels"] == 0
    report = json.loads((tmp_path / "mosaic/manifest.json").read_text())
    assert report["not_a_prediction"] is True
    assert report["missing_slab_blocks"] == []
    assert audit_volume(case / "mri_rotated/mri.nii.gz")["spatial_units"] == "mm"
