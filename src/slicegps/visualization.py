"""Dependency-light SVG views of known evaluation geometry, never matcher input."""

from html import escape
from pathlib import Path

import numpy as np

from .geometry import Plane

_PROJECTION = np.array([[np.sqrt(3) / 2, -np.sqrt(3) / 2, 0], [0.5, 0.5, -1]])
_QUERY = "#38bdf8"
_PREDICTED = "#fb923c"
_GLYPH_SPAN_MM = 10.0


def _corners(plane: Plane) -> np.ndarray:
    """Fixed physical glyphs preserve pose while avoiding full-FOV occlusion."""
    offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    basis = np.stack((plane.u, plane.v))
    return np.asarray(plane.center_mm) + (_GLYPH_SPAN_MM / 2) * offsets @ basis


def pose_svg(
    query_planes: list[dict],
    predicted_planes: list[dict],
    output: Path,
    title: str = "Predicted and true planes",
) -> dict:
    """Draw all centers and at most 12 corresponding pairs of plane glyphs.

    Uses a declared isometric projection: screen x = sqrt(3)/2 * (R - A),
    screen y = (R + A)/2 - S. True and predicted normals are compared without
    sign, giving plane-angle errors in [0, 90] degrees. In-plane rotation is not
    scored. Inputs are known evaluation geometry; this function performs no image
    matching or pose prediction. The written SVG has no scripts or external assets.
    """
    if (
        not isinstance(query_planes, list)
        or not isinstance(predicted_planes, list)
        or not query_planes
        or len(query_planes) != len(predicted_planes)
    ):
        raise ValueError("Provide equally sized nonempty lists of query and predicted planes")
    if not isinstance(title, str):
        raise ValueError("title must be text")
    queries = [Plane.from_dict(plane) for plane in query_planes]
    predictions = [Plane.from_dict(plane) for plane in predicted_planes]
    count = len(queries)
    selected = np.linspace(0, count - 1, min(12, count), dtype=int).tolist()
    query_centers = np.array([plane.center_mm for plane in queries])
    predicted_centers = np.array([plane.center_mm for plane in predictions])
    query_corners = {index: _corners(queries[index]) for index in selected}
    predicted_corners = {index: _corners(predictions[index]) for index in selected}
    errors = []
    for query, prediction in zip(queries, predictions, strict=True):
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.asarray(query.center_mm) - prediction.center_mm
            maximum = float(np.max(np.abs(difference)))
            distance = maximum * float(np.linalg.norm(difference / maximum)) if maximum else 0.0
        first = query.normal / np.linalg.norm(query.normal)
        second = prediction.normal / np.linalg.norm(prediction.normal)
        angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(first, second)), 0, 1))))
        if not np.isfinite(distance):
            raise ValueError("Center errors must be representable as finite millimetres")
        errors.append({"center_error_mm": distance, "normal_angle_error_deg": angle})
    points = np.concatenate([
        query_centers, predicted_centers,
        *query_corners.values(), *predicted_corners.values(),
    ])
    origin = query_centers[0]
    with np.errstate(over="ignore", invalid="ignore"):
        projected = (points - origin) @ _PROJECTION.T
    if not np.isfinite(projected).all():
        raise ValueError("Geometry exceeds the finite SVG projection range")
    lower, upper = projected.min(axis=0), projected.max(axis=0)
    span = np.maximum(upper - lower, 1)
    scale = float(min(750 / span[0], 410 / span[1]))
    midpoint = lower / 2 + upper / 2

    def project(values):
        return ((np.asarray(values) - origin) @ _PROJECTION.T - midpoint) * scale + [435, 340]

    def polygon(values):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in project(values))

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="850" '
        'viewBox="0 0 1120 850" role="img" aria-labelledby="pose-title pose-desc">',
        f'<title id="pose-title">{escape(title)}</title>',
        '<desc id="pose-desc">Isometric evaluation view of known query and predicted plane poses. '
        'Blue marks true query geometry; orange marks predicted MRI geometry. '
        'Lines connect corresponding centers. Plane glyphs are 10 by 10 millimetres.</desc>',
        '<defs><marker id="axis-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>',
        '<rect width="1120" height="850" rx="16" fill="#101827"/>',
        '<g font-family="sans-serif" fill="#e2e8f0">',
        f'<text x="32" y="44" font-size="23" font-weight="600">{escape(title)}</text>',
        '<text x="32" y="73" font-size="14" fill="#b7c5d8">'
        'Isometric projection of RAS coordinates; known geometry for evaluation only.</text>',
        '<text x="32" y="98" font-size="13" fill="#b7c5d8">'
        'Plane glyphs are 10 × 10 mm for visibility; image fields of view are not depicted.</text>',
        '<rect x="24" y="120" width="830" height="440" rx="12" '
        'fill="#172235" stroke="#334155"/>',
    ]
    q2, p2 = project(query_centers), project(predicted_centers)
    for index, (query, prediction) in enumerate(zip(q2, p2, strict=True)):
        lines.append(
            f'<line class="error-link" data-query-index="{index}" '
            f'x1="{query[0]:.2f}" y1="{query[1]:.2f}" '
            f'x2="{prediction[0]:.2f}" y2="{prediction[1]:.2f}" '
            'stroke="#cbd5e1" stroke-width="1.2" opacity="0.55"/>'
        )
    for index in selected:
        for kind, corners, color in (
            ("query", query_corners[index], _QUERY),
            ("predicted", predicted_corners[index], _PREDICTED),
        ):
            lines.append(
                f'<polygon class="{kind}-plane" data-query-index="{index}" '
                f'points="{polygon(corners)}" fill="{color}" fill-opacity="0.15" '
                f'stroke="{color}" stroke-width="1.5"/>'
            )
    for kind, centers, color in (("query", q2, _QUERY), ("predicted", p2, _PREDICTED)):
        for index, (x, y) in enumerate(centers):
            radius = 4 if index in selected else 2.5
            lines.append(
                f'<circle class="{kind}-center" data-query-index="{index}" '
                f'cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>'
            )
    for index in selected:
        x, y = q2[index]
        lines.append(
            f'<text class="pair-label" x="{x + 7:.2f}" y="{y - 7:.2f}" '
            f'font-size="11" fill="#f8fafc">{index + 1}</text>'
        )
    lines.extend([
        '<text x="885" y="150" font-size="16" font-weight="600">Geometry</text>',
        f'<circle cx="894" cy="182" r="5" fill="{_QUERY}"/>',
        '<text x="910" y="187" font-size="13">True query</text>',
        f'<circle cx="894" cy="213" r="5" fill="{_PREDICTED}"/>',
        '<text x="910" y="218" font-size="13">Predicted MRI</text>',
        f'<text x="885" y="255" font-size="13">All {count} center pairs shown</text>',
        f'<text x="885" y="280" font-size="13">{len(selected)} plane pairs labelled</text>',
        '<text x="885" y="305" font-size="12" fill="#b7c5d8">Lines show center displacement</text>',
        '<text x="935" y="365" font-size="14">RAS axes</text>',
    ])
    axis_origin = np.array([980, 455])
    for vector, label, color, text_offset in (
        ((1, 0, 0), "R (+x)", "#f87171", (6, 5)),
        ((0, 1, 0), "A (+y)", "#4ade80", (-49, 5)),
        ((0, 0, 1), "S (+z)", "#60a5fa", (-20, -9)),
    ):
        endpoint = axis_origin + _PROJECTION @ np.array(vector) * 50
        tx, ty = endpoint + text_offset
        lines.append(
            f'<line class="ras-axis" x1="980" y1="455" x2="{endpoint[0]:.2f}" '
            f'y2="{endpoint[1]:.2f}" stroke="{color}" stroke-width="2" '
            'marker-end="url(#axis-arrow)"/>'
        )
        lines.append(f'<text x="{tx:.2f}" y="{ty:.2f}" font-size="12">{label}</text>')
    lines.extend([
        '<text x="32" y="603" font-size="15" font-weight="600">Pose errors by labelled pair</text>',
        '<text x="32" y="627" font-size="12" fill="#b7c5d8">'
        'Normal angle ignores normal sign (0 to 90°); in-plane rotation is not scored.</text>',
    ])
    for position, index in enumerate(selected):
        column, row = divmod(position, 6)
        error = errors[index]
        lines.append(
            f'<text class="error-caption" data-query-index="{index}" '
            f'x="{32 + column * 550}" y="{657 + row * 25}" font-size="13">'
            f'Pair {index + 1}: center {error["center_error_mm"]:.2f} mm, '
            f'normal angle {error["normal_angle_error_deg"]:.2f}°</text>'
        )
    lines.extend([
        '<text x="32" y="832" font-size="12" fill="#94a3b8">'
        'Geometry visualization only. It does not establish registration quality or infer a pose.'
        '</text>',
        '</g></svg>',
    ])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "path": str(output.resolve()),
        "query_count": count,
        "plane_glyph_pair_indices": selected,
        "plane_glyph_span_mm": _GLYPH_SPAN_MM,
        "projection": "isometric RAS",
        "pair_errors": errors,
        "purpose": "known geometry for evaluation only; never matcher input",
    }
