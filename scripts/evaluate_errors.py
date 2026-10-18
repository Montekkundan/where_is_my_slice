"""Evaluate slice-error detection on pinned Synth-by-Reg pairs, without NextBrain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from slicegps.data import write_json
from slicegps.error_detection import (
    ERROR_KINDS,
    classification_metrics,
    context_features,
    detector_scores,
    error_features,
    fit_detector,
    inject_errors,
    normalize_stack_features,
    select_threshold,
    spatial_splits,
)
from slicegps.sbr import SBR_REVISION, fetch_sbr, load_sbr


def make_examples(
    histology,
    mri,
    trials,
    seed,
    encoder=None,
    pairwise=False,
    stack_normalized=False,
    patch_encoder=None,
    sequence=False,
):
    """Same seeds apply to each method. Truth is consumed only after feature extraction."""
    rows, labels, kinds, trial_ids, slots = [], [], [], [], []
    descriptor_cache = {}
    measurement_cache = {}
    reference = None
    comparator = None
    if pairwise:
        from slicegps.pairwise_qa import PairwiseFeatures

        comparator = PairwiseFeatures(mri)
    patch_comparator = None
    if patch_encoder is not None:
        from slicegps.patch_matching import PatchErrorFeatures

        patch_comparator = PatchErrorFeatures(patch_encoder, mri)
    if encoder is not None:
        bank = np.concatenate((histology, histology[:, :, ::-1], mri))
        print(f"Encoding {len(bank)} pinned DINOv2 images", flush=True)
        encoded = encoder.encode(bank, pooling="spatial")
        for image, descriptor in zip(bank, encoded, strict=True):
            descriptor_cache[image.tobytes()] = descriptor
        reference = encoded[-len(mri) :]

    def embeddings(images):
        keys = [image.tobytes() for image in images]
        missing = [
            image for image, key in zip(images, keys, strict=True) if key not in descriptor_cache
        ]
        if missing:
            computed = encoder.encode(np.stack(missing), pooling="spatial")
            for image, descriptor in zip(missing, computed, strict=True):
                descriptor_cache[image.tobytes()] = descriptor
        return np.stack([descriptor_cache[key] for key in keys])

    for k, kind in enumerate(ERROR_KINDS):
        for trial in range(trials):
            observed, truth = inject_errors(histology, kind, 0.05, seed + k * 10000 + trial)
            measurements = error_features(observed, mri, cache=measurement_cache)
            if comparator is not None:
                measurements = np.column_stack((measurements, comparator.measure(observed)))
            if patch_comparator is not None:
                measurements = np.column_stack((measurements, patch_comparator.measure(observed)))
            if encoder is not None:
                desc, flipped = embeddings(observed), embeddings(observed[:, :, ::-1])
                extra = np.column_stack(
                    (
                        context_features(desc, flipped),
                        1 - np.sum(desc * reference, axis=1),
                        np.sum(flipped * reference, axis=1) - np.sum(desc * reference, axis=1),
                    )
                )
                measurements = np.column_stack((measurements, extra))
            if sequence:
                from slicegps.sequence_qa import sequence_features

                measurements = np.column_stack(
                    (measurements, sequence_features(observed, cache=measurement_cache))
                )
            if stack_normalized:
                measurements = normalize_stack_features(measurements)
            selected = np.arange(2, len(histology) - 2)
            rows.append(measurements[selected])
            labels.extend(truth["is_error"][selected])
            kinds.extend([kind] * len(selected))
            trial_ids.extend([trial] * len(selected))
            slots.extend(selected)
    return (
        np.concatenate(rows),
        np.asarray(labels, dtype=bool),
        np.asarray(kinds),
        np.asarray(trial_ids),
        np.asarray(slots),
    )


def subset_features(x, paired):
    if paired:
        return x
    indices = [*range(7), *range(9, 16), *range(18, 25), 27, 28, 29]
    if x.shape[1] == 39:
        indices += list(range(30, 37))
    if x.shape[1] == 45:
        indices += list(range(38, 45))
    if x.shape[1] == 54:
        indices += list(range(30, 54))
    return x[:, indices]


def metrics_by_kind(labels, flagged, kinds):
    return {
        kind: classification_metrics(labels[kinds == kind], flagged[kinds == kind])
        for kind in ERROR_KINDS
    }


def run(
    root: Path,
    output: Path,
    trials: int = 20,
    fetch: bool = True,
    encoder=None,
    donor_calibrated: bool = False,
    pairwise: bool = False,
    stack_normalized: bool = False,
    patch_encoder=None,
    sequence=False,
):
    if output.exists():
        raise FileExistsError(output)
    if trials < 2:
        raise ValueError("At least two corruption trials are required")
    manifest = fetch_sbr(root) if fetch else json.loads((root / "sources.json").read_text())
    if sum((pairwise, encoder is not None, patch_encoder is not None, sequence)) > 1:
        raise ValueError("Choose one feature alternative")
    image_size = 192 if pairwise else 96
    hist, mri, ids = load_sbr(root, "BigBrain", size=image_size)
    split = spatial_splits(len(hist))
    output.mkdir(parents=True)
    datasets = {"BigBrain": (hist, mri, ids)}
    splits = {"BigBrain": split}
    if donor_calibrated:
        datasets["Allen_labels"] = load_sbr(root, "Allen_labels", size=image_size)
        splits["Allen_labels"] = spatial_splits(len(datasets["Allen_labels"][0]))
    split_ids = {
        dataset: {key: [data[2][i] for i in selected] for key, selected in splits[dataset].items()}
        for dataset, data in datasets.items()
    }
    write_json(output / "split.json", split_ids)
    if patch_encoder is not None:
        expected = {
            donor: {key: value for key, value in values.items() if key in ("train", "validation")}
            for donor, values in split_ids.items()
        }
        provenance = patch_encoder.metadata["provenance"]
        if (
            provenance.get("source_revision") != SBR_REVISION
            or provenance.get("splits") != expected
        ):
            raise ValueError("Patch encoder training split differs from this benchmark")
        patch_encoder.save(output / "patch_encoder.safetensors")
    examples = {}
    for name, seed in (("train", 48101), ("validation", 48201)):
        parts = []
        for dataset, (histology, reference_mri, _) in datasets.items():
            selected = splits[dataset][name]
            print(f"Preparing {name} {dataset}: {len(selected)} unique sections", flush=True)
            parts.append(
                make_examples(
                    histology[selected],
                    reference_mri[selected],
                    trials,
                    seed,
                    encoder,
                    pairwise,
                    stack_normalized,
                    patch_encoder,
                    sequence,
                )
            )
        examples[name] = tuple(np.concatenate([part[i] for part in parts]) for i in range(5))
    x_train, y_train, *_ = examples["train"]
    x_val, y_val, kinds_val, *_ = examples["validation"]
    models, selection = {}, {}
    for paired in (False, True):
        name = "neighbour_images_and_expected_mri" if paired else "neighbour_images"
        candidates = []
        for regularization in (0.001, 0.01, 0.1, 1.0):
            model = fit_detector(subset_features(x_train, paired), y_train, regularization)
            scores = detector_scores(subset_features(x_val, paired), model)
            threshold, measured = select_threshold(scores, y_val, maximum_fpr=0.05)
            by_kind = metrics_by_kind(y_val, scores > threshold, kinds_val)
            quality = float(np.mean([r["balanced_accuracy"] for r in by_kind.values()]))
            candidates.append((quality, model, threshold, measured, by_kind))
        best = max(candidates, key=lambda item: item[0])
        _, model, threshold, measured, by_kind = best
        model.update(
            threshold=threshold,
            expected_mri_required=paired,
            feature_contract=(
                "stack_error_patch_v1"
                if patch_encoder is not None
                else (
                    "stack_error_sequence_v1"
                    if sequence
                    else "stack_error_pairwise_v1"
                    if paired and pairwise
                    else "stack_error_features_v3"
                )
            ),
            image_size=image_size,
            stack_normalized=stack_normalized,
            patch_features=patch_encoder is not None,
            sequence_features=sequence,
            patch_weights_sha256=(
                hashlib.sha256((output / "patch_encoder.safetensors").read_bytes()).hexdigest()
                if patch_encoder is not None
                else None
            ),
            learned_features=encoder is not None,
            training_dataset="BigBrain+Allen" if donor_calibrated else "BigBrain",
            source_revision=SBR_REVISION,
            encoder_revision=encoder.metadata["model_revision"] if encoder is not None else None,
        )
        models[name] = model
        selection[name] = {
            "validation": measured,
            "by_kind": by_kind,
            "regularization": model["regularization"],
        }
        write_json(output / f"{name}.json", model)
    # Models and thresholds are frozen before either test set is evaluated.
    write_json(output / "selection.json", selection)
    results = {}
    for dataset in ("BigBrain", "Allen_labels"):
        if dataset in datasets:
            histology, reference_mri, donor_ids = datasets[dataset]
            selected = splits[dataset]["test"]
            h, t = histology[selected], reference_mri[selected]
            section_ids = [donor_ids[i] for i in selected]
        else:
            h, t, section_ids = load_sbr(root, dataset, size=image_size)
        print(f"Testing {dataset}: {len(h)} unique sections", flush=True)
        x, truth, kinds, trial_ids, slots = make_examples(
            h, t, trials, 48301, encoder, pairwise, stack_normalized, patch_encoder, sequence
        )
        observed = {}
        for name, model in models.items():
            scores = detector_scores(subset_features(x, model["expected_mri_required"]), model)
            flags = scores > model["threshold"]
            by_kind = metrics_by_kind(truth, flags, kinds)
            clean_measurements = error_features(h, t)
            if pairwise:
                from slicegps.pairwise_qa import PairwiseFeatures

                clean_measurements = np.column_stack(
                    (clean_measurements, PairwiseFeatures(t).measure(h))
                )
            if patch_encoder is not None:
                from slicegps.patch_matching import PatchErrorFeatures

                clean_measurements = np.column_stack(
                    (clean_measurements, PatchErrorFeatures(patch_encoder, t).measure(h))
                )
            if encoder is not None:
                desc = encoder.encode(h, pooling="spatial")
                flipped = encoder.encode(h[:, :, ::-1], pooling="spatial")
                reference = encoder.encode(t, pooling="spatial")
                clean_measurements = np.column_stack(
                    (
                        clean_measurements,
                        context_features(desc, flipped),
                        1 - np.sum(desc * reference, axis=1),
                        np.sum(flipped * reference, axis=1) - np.sum(desc * reference, axis=1),
                    )
                )
            if sequence:
                from slicegps.sequence_qa import sequence_features

                clean_measurements = np.column_stack((clean_measurements, sequence_features(h)))
            if stack_normalized:
                clean_measurements = normalize_stack_features(clean_measurements)
            clean_scores = detector_scores(
                subset_features(clean_measurements, model["expected_mri_required"]), model
            )
            clean_flags = clean_scores[2:-2] > model["threshold"]
            observed[name] = {
                "pooled": classification_metrics(truth, flags),
                "by_kind": by_kind,
                "untouched_stack_flag_count": int(clean_flags.sum()),
                "untouched_stack_evaluated_sections": len(clean_flags),
                "untouched_stack_flag_rate": float(clean_flags.mean()),
                "target_met": all(
                    r["recall"] >= 0.95
                    and r["false_positive_rate"] <= 0.05
                    and r["precision"] is not None
                    and r["precision"] >= 0.95
                    for r in by_kind.values()
                ),
            }
            with (output / f"{dataset}_{name}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(
                    ["kind", "trial", "slot", "expected_section_id", "is_error", "score", "flagged"]
                )
                for i in range(len(truth)):
                    writer.writerow(
                        [
                            kinds[i],
                            trial_ids[i],
                            slots[i],
                            section_ids[slots[i]],
                            int(truth[i]),
                            float(scores[i]),
                            int(flags[i]),
                        ]
                    )
            print(dataset, name, json.dumps(observed[name]["pooled"]), flush=True)
        results[dataset] = {
            "all_negative_baseline": classification_metrics(truth, np.zeros_like(truth)),
            "actual_error_fraction": float(truth.mean()),
            "unique_sections": len(h),
            "evaluated_sections": len(h) - 4,
            "section_ids": section_ids,
            "methods": observed,
        }
    report = {
        "schema_version": 1,
        "source_revision": SBR_REVISION,
        "learned_features": encoder.metadata if encoder is not None else None,
        "pairwise_comparisons": pairwise,
        "sequence_features": sequence,
        "patch_features": patch_encoder.metadata if patch_encoder is not None else None,
        "stack_normalized": stack_normalized,
        "source": manifest["source"],
        "paper": manifest["paper"],
        "protocol": {
            "training": ("BigBrain+Allen" if donor_calibrated else "BigBrain")
            + " first 60 percent minus support guard",
            "validation": "Middle 20 percent of training donors with support gaps",
            "test": (
                "Both donors final 20 percent"
                if donor_calibrated
                else "BigBrain final 20 percent and entire independent Allen donor"
            ),
            "donor_calibrated": donor_calibrated,
            "guard_sections": 3,
            "context_radius_sections": 2,
            "image_size": image_size,
            "stack_normalization": "Observed stack median/IQR; boundary rows excluded"
            if stack_normalized
            else "None",
            "comparisons": (
                "Paired convolutional descriptors, reciprocal swap gains and local context"
                if patch_encoder is not None
                else (
                    "Self-similarity, NMI and reciprocal swap gains"
                    if pairwise
                    else (
                        "Adjacent exchange path, curvature and trajectory evidence"
                        if sequence
                        else "Baseline local context"
                    )
                )
            ),
            "preprocessing": (
                "Publisher affine-aligned images, common canvas retained; "
                "tissue-based tear placement"
            ),
            "selection": "Validation macro balanced accuracy; clean validation FPR <= 5 percent",
            "error_fraction_requested": 0.05,
            "trials_per_kind": trials,
            "kinds": list(ERROR_KINDS),
            "target": "Recall >=95%, precision >=95%, FPR <=5% for each error kind",
            "negative_control": "Untouched publisher stack; all-negative baseline also reported",
            "input": "Nominally ordered affine-aligned images; optional expected MRI per slot",
        },
        "selection": selection,
        "results": results,
        "limitations": [
            "Injected errors on real images, not expert-labelled naturally occurring errors",
            "Repeated trials reuse sections and are not independent subject observations",
            "Two donors only; independent donor test applies only when donor_calibrated is false",
            "Publisher affine alignment and paired MRIs already supplied",
            "Manual landmark files described in the paper are absent from the public release",
            "Slice ID differences are not converted to millimetres or angle accuracy",
            "Thresholds are frozen before tests; test findings are exploratory once inspected",
            "Near swaps exchange adjacent available sections; far swaps span at least 1/4 stack",
        ],
    }
    if stack_normalized:
        report["limitations"].append(
            "Per-stack median/IQR calibration assumes a mostly correct observed stack"
        )
    write_json(output / "metrics.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/synthbyreg"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--skip-fetch", action="store_true")
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument("--learned", action="store_true")
    methods.add_argument("--pairwise", action="store_true")
    methods.add_argument("--patch-model", type=Path)
    methods.add_argument("--sequence", action="store_true")
    parser.add_argument("--donor-calibrated", action="store_true")
    parser.add_argument("--stack-normalized", action="store_true")
    args = parser.parse_args()
    encoder = None
    if args.learned:
        from slicegps.encoders import DinoEncoder

        encoder = DinoEncoder(Path("data/models"), device="auto", batch_size=8)
    patch_encoder = None
    if args.patch_model:
        from slicegps.patch_matching import PatchMatcher

        patch_encoder = PatchMatcher.load(args.patch_model)
    run(
        args.root,
        args.output,
        args.trials,
        not args.skip_fetch,
        encoder,
        args.donor_calibrated,
        args.pairwise,
        args.stack_normalized,
        patch_encoder,
        args.sequence,
    )
