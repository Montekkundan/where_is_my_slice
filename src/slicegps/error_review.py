"""Apply a saved experimental detector to anonymized, nominally ordered sections."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import write_json
from .error_detection import (
    context_features,
    detector_scores,
    error_features,
    normalize_stack_features,
)
from .retrieval import _prepare


def review_errors(
    model_path: Path,
    sections: list[Path],
    output: Path,
    mri_sections: list[Path] | None = None,
    encoder=None,
) -> dict:
    """Write an image-based review without reading truth or extracting IDs from names."""
    model = json.loads(Path(model_path).read_text())
    contract = model.get("feature_contract")
    if contract not in (
        "stack_error_features_v3",
        "stack_error_pairwise_v1",
        "stack_error_patch_v1",
    ):
        raise ValueError("Unknown detector feature contract")
    if model["expected_mri_required"] and mri_sections is None:
        raise ValueError("This detector requires the expected MRI plane for each input slot")
    if not model["expected_mri_required"] and mri_sections is not None:
        raise ValueError("This model uses histology neighbours only; omit MRI inputs")
    if len(sections) < 5 or (mri_sections is not None and len(mri_sections) != len(sections)):
        raise ValueError("Need at least five sections and matching MRI count")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)

    image_size = int(model.get("image_size", 96))
    if contract == "stack_error_pairwise_v1" and not model["expected_mri_required"]:
        raise ValueError("Paired comparison model requires expected MRI")

    def pixels(paths):
        values = []
        for path in paths:
            with Image.open(path) as image:
                if image.mode in ("F", "I", "I;16"):
                    # Preserve scientific TIFF ranges; conversion to L clips float pixels.
                    resized = image.convert("F").resize(
                        (image_size, image_size), Image.Resampling.BILINEAR
                    )
                    values.append(np.asarray(resized, dtype=np.float32))
                else:
                    resized = image.convert("L").resize(
                        (image_size, image_size), Image.Resampling.BILINEAR
                    )
                    values.append(np.asarray(resized, dtype=np.float32) / 255)
        return np.stack(values)

    images = pixels(sections)
    mri = pixels(mri_sections) if mri_sections is not None else None
    measurements = error_features(images, mri)
    if contract == "stack_error_pairwise_v1":
        from .pairwise_qa import PairwiseFeatures

        measurements = np.column_stack((measurements, PairwiseFeatures(mri).measure(images)))
    if contract == "stack_error_patch_v1":
        from .patch_matching import PatchErrorFeatures, PatchMatcher

        weights = Path(model_path).parent / "patch_encoder.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != model.get("patch_weights_sha256"):
            raise ValueError("Detector and patch encoder weights differ")
        patch_matcher = PatchMatcher.load(weights)
        if mri is not None:
            extra = PatchErrorFeatures(patch_matcher, mri).measure(images)
        else:
            desc = patch_matcher.encode(images, "histology")
            flipped = patch_matcher.encode(images[:, :, ::-1], "histology")
            extra = context_features(desc, flipped)
        measurements = np.column_stack((measurements, extra))
    if model["learned_features"]:
        if encoder is None:
            from .encoders import DinoEncoder

            encoder = DinoEncoder(Path("data/models"))
        desc = encoder.encode(images, pooling="spatial")
        flipped = encoder.encode(images[:, :, ::-1], pooling="spatial")
        extra = context_features(desc, flipped)
        if mri is not None:
            reference = encoder.encode(mri, pooling="spatial")
            extra = np.column_stack(
                (
                    extra,
                    1 - np.sum(desc * reference, axis=1),
                    np.sum(flipped * reference, axis=1) - np.sum(desc * reference, axis=1),
                )
            )
        if encoder.metadata["model_revision"] != model["encoder_revision"]:
            raise ValueError("DINOv2 model revision mismatch")
        measurements = np.column_stack((measurements, extra))
    if model.get("stack_normalized", False):
        measurements = normalize_stack_features(measurements)
    scores = detector_scores(measurements, model)
    rows, cards = [], []
    for i, (path, score, image) in enumerate(zip(sections, scores, images, strict=True)):
        evaluated = 2 <= i < len(images) - 2
        flagged = bool(score > model["threshold"]) if evaluated else None
        rows.append(
            {
                "input_index": i,
                "name": Path(path).name,
                "score": float(score),
                "flagged": flagged,
                "status": ("review" if flagged else "unflagged")
                if evaluated
                else "insufficient_context",
            }
        )
        stream = io.BytesIO()
        Image.fromarray(np.rint(_prepare(image)[0] * 255).astype(np.uint8)).save(
            stream, format="PNG"
        )
        data = base64.b64encode(stream.getvalue()).decode()
        status = rows[-1]["status"]
        cards.append(
            f'<article class="{status}"><h3>{i}: {html.escape(Path(path).name)}</h3>'
            f'<img src="data:image/png;base64,{data}" alt="Section {i}">'
            f"<p>{status}, score {score:.3f}</p></article>"
        )
    result = {
        "schema_version": 1,
        "threshold": model["threshold"],
        "expected_mri_required": model["expected_mri_required"],
        "stack_normalized": model.get("stack_normalized", False),
        "sections": rows,
        "flagged_indices": [row["input_index"] for row in rows if row["flagged"]],
        "limitations": [
            "Experimental detector; 95 percent precision and recall have not been established",
            "Assumes nominal stack order and approximately aligned section canvases",
            "First and last two sections have insufficient context and are not classified",
            "Scores are not calibrated probabilities; unflagged does not establish correctness",
            "Empty or constant sections are rejected explicitly",
        ],
    }
    if model.get("stack_normalized", False):
        result["limitations"].append("Stack calibration assumes most supplied sections are correct")
    output.mkdir(parents=True)
    write_json(output / "errors.json", result)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Slice error review</title>'
        "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}"
        "article{border:2px solid #aaa;padding:12px;border-radius:8px}"
        ".review{border-color:#c33;background:#fff0f0}img{width:160px}h3{overflow-wrap:anywhere}"
        "</style><h1>Slice error review</h1><p>Experimental image-based flags for human review. "
        "The 95% detection target is not established. Scores are not probabilities.</p>"
        "<p>Inputs follow the supplied nominal order. Red cards need review. "
        "The first and last two sections lack enough context for classification.</p>"
        '<div class="grid">' + "".join(cards) + "</div>"
    )
    return result
