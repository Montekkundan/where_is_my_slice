# Where Is My Slice

Experimental detection of flipped, damaged and misplaced histological sections.
The goal is 95–98% reliable detection. Current results do **not** meet that goal.
NextBrain reconstruction and piece assembly are outside the current scope.

![Where's My Slice? concept demo](wheres_my_slice_search.gif)
![Stack QA concept demo](wheres_my_slice_stack.gif)

Original concept animations by Jignesh. These illustrate the intended workflow.

## Run

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,learned]'
slicegps fetch-sbr --root data/synthbyreg
python scripts/evaluate_errors.py --issue-demo --stack-normalized --output outputs/demo
python scripts/evaluate_errors.py --issue-demo --sequence --mlp-head --stack-normalized \
  --output outputs/demo-mlp
python scripts/evaluate_errors.py --output outputs/errors
pytest -q
```

The benchmark uses 344 BigBrain and 93 Allen pairs from
[Synth-by-Reg](https://arxiv.org/abs/2107.14449), with each section paired to its
**own donor's MRI**. It injects known errors, trains on separate sections and
selects thresholds on validation data. Publisher affine alignment is supplied.

Review an ordered stack using a saved model:

```bash
slicegps detect-errors outputs/errors/neighbour_images.json sections/*.png --output outputs/review
```

Open `outputs/review/index.html`. Supply files in nominal order; the first and
last two sections lack full context. Models named `neighbour_images_and_expected_mri.json`
require `--mri` followed by planes in expected slot order. Use fresh output
directories. Data, weights and full outputs stay local.

## Results

The main demo follows [issue #10](https://github.com/tfunck/bigbrainhack2026/issues/10):
randomly shuffle some sections and flip others, changing about 5% of a stack.
Shuffles can include immediate neighbours. Each experiment uses 20 fixed trials;
models and thresholds are selected on BigBrain training/validation sections,
then frozen before testing. Error locations are hidden from the detector.

| Detector | Test | Errors detected | Clean sections falsely flagged | Precision of flags |
| --- | --- | ---: | ---: | ---: |
| Image measurements + expected MRI | BigBrain | 93.3% | 13.1% | 26.7% |
| Image measurements + expected MRI | Unseen Allen | 93.8% | 14.3% | 23.6% |
| Paired MRI comparisons | BigBrain | 98.3% | 3.9% | 56.2% |
| Paired MRI comparisons | Unseen Allen | 70.0% | 10.8% | 23.3% |
| Image-only sequence features + nonlinear classifier | BigBrain | 81.7% | 7.3% | 36.3% |
| Image-only sequence features + nonlinear classifier | Unseen Allen | 92.5% | 1.2% | 77.9% |

**The 95–98% reliable detection target across donors is not met.** Paired comparisons
reach 98.3% recall on BigBrain alone, with only 56.2% of flags correct. BigBrain has 62 evaluated
test sections and Allen has 89. Repeated trials reuse them, rather than adding
donors. Calling every section clean already gives about 95% raw accuracy, so
recall, false alarms and precision must be considered together. These are
exploratory tests on two donors, with injected errors and supplied affine alignment.
The reader flags suspected errors; it does not automatically restore a stack.

Separate stress tests retain flips, damage, forced adjacent swaps and distant
swaps. The calibrated paired-comparison model detects only 17.5% of nearby
BigBrain swaps. DINOv2, a trained histology/MRI convolutional matcher and direct
sequence continuity features have not met the target. Stack calibration uses
only the observed stack's median/IQR, assuming most sections are correct.
See [all measured results](results/error_detection.csv) and [protocol JSONs](results/).

Reproduce a mixed trial through the image reader:

```bash
python scripts/review_sbr_case.py --run outputs/demo --kind shuffle_flip \
  --method neighbour_images_and_expected_mri --output outputs/demo-review
```

This verifies saved prediction parity and writes truth separately. Section IDs/order
establish injected-error truth; the detector uses pixels, without IDs or clean copies.
The paper's manual landmark files are absent from the public release used here.
Naturally occurring errors and the paper's registration model have not been validated.
Earlier atlas retrieval results remain in [benchmarks.csv](results/benchmarks.csv).

## Data and credits

Paper data are pinned to the [authors' release](https://github.com/acasamitjana/SynthByReg/tree/1b67b630cc4e8e03be210d6fb39210b722296902).
Downloads are checked against its Git blob hashes. Original data and weights are
not redistributed here. Older BigBrain-derived previews are resliced/scaled under
[CC BY-NC-SA 4.0](https://bigbrain-ftp.loris.ca/bigbrain-ftp/License.txt), crediting
[Amunts et al. (2013)](https://doi.org/10.1126/science.1235381).
Other source and model details are in [model_references.json](results/model_references.json).

Based on [Jignesh's Where Is My Slice](https://github.com/Jignesh1996/whereismyslice).
