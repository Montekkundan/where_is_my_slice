"""Learned matching artifacts preserve modality, split and inference contracts."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from slicegps.error_detection import (  # noqa: E402
    detector_scores,
    error_features,
    fit_detector,
    normalize_stack_features,
)
from slicegps.error_review import review_errors  # noqa: E402
from slicegps.patch_matching import (  # noqa: E402
    ARCHITECTURE,
    PatchErrorFeatures,
    PatchMatcher,
    _network,
    fit_patch_matcher,
    prepare,
)


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def images(seed=73, count=9):
    return np.random.default_rng(seed).random((count, 96, 96), dtype=np.float32)


def matcher():
    torch.manual_seed(4)
    return PatchMatcher(_network(), {"architecture": ARCHITECTURE}, device="cpu")


def test_roundtrip_separate_heads_and_weight_integrity(tmp_path):
    original = matcher()
    values = images(count=3)
    hist, mri = original.encode(values, "histology"), original.encode(values, "mri")
    assert hist.shape == mri.shape == (3, 9216)
    assert not np.allclose(hist, mri)
    np.testing.assert_allclose(np.linalg.norm(hist, axis=1), 1, atol=1e-6)
    path = tmp_path / "encoder.safetensors"
    original.save(path)
    loaded = PatchMatcher.load(path, device="cpu")
    np.testing.assert_array_equal(loaded.encode(values, "histology"), hist)
    np.testing.assert_array_equal(loaded.encode(values, "mri"), mri)
    assert not any(p.requires_grad for p in loaded.model.parameters())
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        PatchMatcher.load(path, device="cpu")


def test_input_validation_and_train_validation_overlap():
    values = images(count=3)
    with pytest.raises(ValueError, match="96 x 96"):
        prepare(values[:, :48], "histology")
    with pytest.raises(ValueError, match="96 x 96"):
        prepare(values, "unknown")
    with pytest.raises(ValueError, match="Empty|empty"):
        prepare(np.ones_like(values), "mri")
    invalid = values.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prepare(invalid, "histology")
    with pytest.raises(ValueError, match="overlap"):
        fit_patch_matcher([(values, values)], [(values, values)], epochs=1, device="cpu")
    with pytest.raises(ValueError, match="corresponding"):
        fit_patch_matcher([(values, values[:-1])], [(values, values)], epochs=1, device="cpu")


def test_training_selects_validation_checkpoint_without_test_inputs():
    train, val = images(1, 4), images(2, 3)
    learned = fit_patch_matcher(
        [(train, 1 - train)], [(val, 1 - val)], epochs=1, device="cpu", seed=4
    )
    assert learned.metadata["training_pairs"] == [4]
    assert learned.metadata["validation_pairs"] == [3]
    assert learned.metadata["selected_epoch"] in (0, 1)
    assert len(learned.metadata["history"]) == 2
    assert np.isfinite(learned.encode(val, "histology")).all()


def test_cache_matches_fresh_observed_image_comparisons():
    values = images()
    encoder = matcher()
    comparator = PatchErrorFeatures(encoder, values)
    clean = comparator.measure(values)
    changed = values.copy()
    changed[[3, 4]] = changed[[4, 3]]
    cached = comparator.measure(changed)
    fresh = PatchErrorFeatures(encoder, values).measure(changed)
    assert clean.shape == cached.shape == (9, 15)
    np.testing.assert_allclose(cached, fresh, atol=1e-7)
    assert not any(key in vars(comparator) for key in ("truth", "labels", "ids"))
    with pytest.raises(ValueError, match="counts"):
        comparator.measure(values[:-1])


@pytest.mark.parametrize("paired", [False, True])
def test_saved_reader_parity_and_encoder_binding(tmp_path, paired):
    values = images()
    altered = values.copy()
    altered[[3, 4]] = altered[[4, 3]]
    encoder = matcher()
    path = tmp_path / "patch_encoder.safetensors"
    encoder.save(path)
    full = np.column_stack(
        (error_features(altered, values), PatchErrorFeatures(encoder, values).measure(altered))
    )
    if paired:
        x = full
    else:
        indices = [*range(7), *range(9, 16), *range(18, 25), 27, 28, 29, *range(38, 45)]
        x = full[:, indices]
    x = normalize_stack_features(x)
    model = fit_detector(x, np.isin(np.arange(9), [3, 4]))
    model.update(
        feature_contract="stack_error_patch_v1",
        expected_mri_required=paired,
        learned_features=False,
        patch_weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        stack_normalized=True,
        image_size=96,
        threshold=0.0,
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model))
    hp, mp = [], []
    for i, (h, m) in enumerate(zip(altered, values, strict=True)):
        hpath, mpath = tmp_path / f"h-{i}.tiff", tmp_path / f"m-{i}.tiff"
        Image.fromarray(h).save(hpath)
        Image.fromarray(m).save(mpath)
        hp.append(hpath)
        mp.append(mpath)
    reviewed = review_errors(model_path, hp, tmp_path / "review", mp if paired else None)
    np.testing.assert_allclose(
        [row["score"] for row in reviewed["sections"]], detector_scores(x, model), atol=1e-3
    )
    assert reviewed["flagged_indices"] == [
        i for i, score in enumerate(detector_scores(x, model)) if 2 <= i < 7 and score > 0
    ]
    other = matcher()
    with torch.no_grad():
        next(other.model.parameters()).add_(0.1)
    other.save(path)
    with pytest.raises(ValueError, match="weights differ"):
        review_errors(model_path, hp, tmp_path / "wrong", mp if paired else None)


def test_small_paper_sheet_is_removed_without_changing_dark_background_tissue():
    image = np.zeros((96, 96), dtype=np.float32)
    image[22:72, 25:70] = 1.0
    image[28:66, 31:64] = 0.55
    old = prepare(image[None], "histology", "canonical_v1")[0, 0]
    corrected = prepare(image[None], "histology", "paper_v2")[0, 0]
    assert old[24, 27] == 1
    assert corrected[24, 27] == 0
    assert corrected[40, 40] > 0
    dark = image.copy()
    dark[22:72, 25:70] = 0
    dark[28:66, 31:64] = np.linspace(0.1, 1, 64 - 31)[None]
    np.testing.assert_array_equal(
        prepare(dark[None], "histology", "canonical_v1"),
        prepare(dark[None], "histology", "paper_v2"),
    )
