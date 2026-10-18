"""Publisher list selection and paired-canvas safeguards."""

import json

import numpy as np
import pytest
from PIL import Image

from slicegps.sbr import SBR_REVISION, listed_pairs, load_sbr


def fixture_tree():
    return [
        {"type": "blob", "path": f"data/BigBrain/{kind}/slice_{i:03d}.png"}
        for i in (1, 2, 3)
        for kind in ("histo/images_affine", "mri/images")
    ]


def test_lists_define_pairs_and_order_not_directory_contents():
    rows = listed_pairs(
        "SLICE_ID,SLICE_NAME\n003,slice_003\n001,slice_001\n", "BigBrain", fixture_tree()
    )
    assert [r["section_id"] for r in rows] == ["001", "003"]
    assert all("images_affine" in r["histology"] for r in rows)


@pytest.mark.parametrize("ids", ["001\n001", "../bad", "004"])
def test_unsafe_duplicate_and_incomplete_pairs_rejected(ids):
    with pytest.raises(ValueError):
        listed_pairs("SLICE_ID\n" + ids, "BigBrain", fixture_tree())


def test_loader_refuses_differing_canvas_and_preserves_pair_order(tmp_path):
    rows = listed_pairs("SLICE_ID\n001\n003", "BigBrain", fixture_tree())
    for row in rows:
        for key in ("histology", "mri"):
            path = tmp_path / row[key].removeprefix("data/")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(path)
    (tmp_path / "sources.json").write_text(
        json.dumps({"revision": SBR_REVISION, "datasets": {"BigBrain": rows}})
    )
    hist, mri, ids = load_sbr(tmp_path, "BigBrain", 16)
    assert ids == ["001", "003"]
    np.testing.assert_array_equal(hist, mri)
    Image.new("L", (20, 16)).save(tmp_path / rows[0]["mri"].removeprefix("data/"))
    with pytest.raises(ValueError, match="canvases differ"):
        load_sbr(tmp_path, "BigBrain", 16)
