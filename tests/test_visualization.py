"""SVG geometry checks using known poses, with no browser or rendering dependencies."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from slicegps.geometry import coronal_plane
from slicegps.visualization import pose_svg

_NS = {"svg": "http://www.w3.org/2000/svg"}


def _geometry(center=(0, 0, 0), tilt=(0, 0)):
    return coronal_plane(center, (32, 32), (1, 1), tilt).to_dict()


def _elements(root, kind, css_class):
    return [element for element in root.findall(f".//svg:{kind}", _NS)
            if element.attrib.get("class") == css_class]


def test_standalone_svg_has_pose_glyphs_axes_connections_and_known_errors(tmp_path):
    path = tmp_path / "nested" / "poses.svg"
    summary = pose_svg([_geometry()], [_geometry((3, 4, 0), (60, 0))], path)
    root = ET.parse(path).getroot()
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert len(_elements(root, "polygon", "query-plane")) == 1
    assert len(_elements(root, "polygon", "predicted-plane")) == 1
    assert len(_elements(root, "circle", "query-center")) == 1
    assert len(_elements(root, "circle", "predicted-center")) == 1
    assert len(_elements(root, "line", "error-link")) == 1
    assert len(_elements(root, "line", "ras-axis")) == 3
    caption = _elements(root, "text", "error-caption")[0].text
    assert "center 5.00 mm" in caption
    assert "normal angle 60.00°" in caption
    texts = " ".join(root.itertext())
    for text in ("R (+x)", "A (+y)", "S (+z)", "Isometric", "evaluation only", "10 × 10 mm"):
        assert text in texts
    assert summary["query_count"] == 1
    assert summary["pair_errors"][0]["center_error_mm"] == 5
    assert summary["pair_errors"][0]["normal_angle_error_deg"] == pytest.approx(60)
    assert not root.findall(".//svg:script", _NS)
    assert not root.findall(".//svg:image", _NS)


def test_title_is_escaped_as_text_in_document_and_visible_heading(tmp_path):
    title = '<script>alert("x")</script> & <image href="https://example.com/a"/>'
    path = tmp_path / "safe.svg"
    pose_svg([_geometry()], [_geometry()], path, title=title)
    source = path.read_text()
    assert "&lt;script&gt;" in source
    assert "&amp;" in source
    assert "<script>" not in source
    assert "<image href=" not in source
    root = ET.fromstring(source)
    assert root.find("svg:title", _NS).text == title
    assert title in [element.text for element in root.findall(".//svg:text", _NS)]
    assert not root.findall(".//svg:script", _NS)
    assert not root.findall(".//svg:image", _NS)


def test_all_centers_are_shown_while_plane_glyphs_are_limited_to_twelve_pairs(tmp_path):
    queries = [_geometry((index, 2 * index, -index)) for index in range(20)]
    predictions = [_geometry((index, 2 * index + 1, -index), (1, -2)) for index in range(20)]
    path = tmp_path / "many.svg"
    summary = pose_svg(queries, predictions, path)
    root = ET.parse(path).getroot()
    assert len(_elements(root, "circle", "query-center")) == 20
    assert len(_elements(root, "circle", "predicted-center")) == 20
    assert len(_elements(root, "line", "error-link")) == 20
    assert len(_elements(root, "polygon", "query-plane")) == 12
    assert len(_elements(root, "polygon", "predicted-plane")) == 12
    assert len(_elements(root, "text", "error-caption")) == 12
    selected = summary["plane_glyph_pair_indices"]
    assert len(set(selected)) == 12
    assert selected[0] == 0 and selected[-1] == 19
    assert len(summary["pair_errors"]) == 20
    assert all(error["center_error_mm"] == 1 for error in summary["pair_errors"])


def test_parallel_planes_with_reversed_normal_have_zero_unoriented_angle(tmp_path):
    reversed_normal = _geometry()
    reversed_normal["u"] = [-1, 0, 0]
    result = pose_svg([_geometry()], [reversed_normal], tmp_path / "normal.svg")
    assert result["pair_errors"][0]["normal_angle_error_deg"] == 0


def test_glyph_size_is_independent_of_image_field_of_view_and_inputs_are_not_mutated(tmp_path):
    first = _geometry()
    second = {**first, "shape": [1000, 1000], "spacing_mm": [5, 5]}
    a, b = tmp_path / "small.svg", tmp_path / "large.svg"
    pose_svg([first], [first], a)
    pose_svg([second], [second], b)
    first_root, second_root = ET.parse(a).getroot(), ET.parse(b).getroot()
    first_polygon = _elements(first_root, "polygon", "query-plane")[0].attrib["points"]
    second_polygon = _elements(second_root, "polygon", "query-plane")[0].attrib["points"]
    assert first_polygon == second_polygon
    assert first["shape"] == [32, 32]
    assert second["shape"] == [1000, 1000]


@pytest.mark.parametrize(
    "queries,predictions", [([], []), ([_geometry()], []), ([], [_geometry()])],
)
def test_empty_or_mismatched_pairs_are_rejected_without_writing(tmp_path, queries, predictions):
    path = tmp_path / "invalid.svg"
    with pytest.raises(ValueError, match="nonempty lists"):
        pose_svg(queries, predictions, path)
    assert not path.exists()


@pytest.mark.parametrize(
    "field,value",
    [("center_mm", [np.nan, 0, 0]), ("center_mm", [0, np.inf, 0]),
     ("u", [2, 0, 0]), ("v", [1, 0, 0]), ("spacing_mm", [-1, 1])],
)
def test_invalid_plane_geometry_is_rejected_on_either_side(tmp_path, field, value):
    bad = {**_geometry(), field: value}
    for query, prediction in ((bad, _geometry()), (_geometry(), bad)):
        with pytest.raises(ValueError):
            pose_svg([query], [prediction], tmp_path / "invalid.svg")
    assert not (tmp_path / "invalid.svg").exists()
