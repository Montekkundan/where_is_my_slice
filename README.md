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
python scripts/evaluate_errors.py --output outputs/errors
python scripts/evaluate_errors.py --pairwise --donor-calibrated --stack-normalized \
  --output outputs/errors-paired
python scripts/train_patch_matcher.py --donor-calibrated --preprocessing paper_v2 \
  --output outputs/patch-matcher
python scripts/evaluate_errors.py --donor-calibrated --stack-normalized \
  --patch-model outputs/patch-matcher/patch_encoder.safetensors --output outputs/errors-learned
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

These tests use four error types and 20 trials per type. BigBrain has 62 evaluated
test sections. Independent Allen testing uses 89; models trained on both donors
have only 12 Allen test sections. Repeated trials reuse sections.

| Detector | Test | Errors detected | Clean sections falsely flagged |
| --- | --- | ---: | ---: |
| Image measurements + expected MRI | BigBrain | 77.5% | 6.5% |
| Images + stack calibration | Unseen Allen | 84.1% | 10.2% |
| Paired comparisons + stack calibration, trained on both donors | BigBrain | 74.5% | 4.7% |
| Paired comparisons + stack calibration, trained on both donors | Allen | 85.8% | 7.5% |
| Trained convolutional matcher + stack calibration, both donors | BigBrain | 79.5% | 7.1% |
| Trained convolutional matcher + stack calibration, both donors | Allen | 66.7% | 10.7% |

No method meets the target across error types and donors. The calibrated paired
model detects only 17.5% of nearby BigBrain swaps. DINOv2 features did not improve
the image/MRI baseline. The convolutional matcher uses separate histology/MRI
encoders, adjacent training negatives and clean validation pairs for checkpoint
selection. Its paper-background variant also misses most nearby BigBrain swaps. Direct
adjacent-swap continuity features raise nearby-swap recall to 30% on BigBrain,
but flag 12.6% of clean sections in that test. A nonlinear classifier reduces
pooled false alarms to 6.1% on BigBrain and 2.3% on Allen, with only 74.5% and
70.0% recall respectively. It also falls short.
Uncalibrated models can flag every Allen section; that is
a failure. Calling every section clean already gives about 95–96% raw accuracy
in the independent tests, so recall and false alarms are reported together.
Stack calibration uses the observed stack's median and interquartile range,
assuming most supplied sections are correct.

See [all per-error results](results/error_detection.csv) and the [protocol JSONs](results/).
To reproduce a saved trial through the reader, run
`python scripts/review_sbr_case.py --run outputs/errors --kind flip --output outputs/example`.
It checks predictions against the benchmark and writes truth separately.

The detector uses pixels and neighbouring images, without reading section IDs or
clean copies. IDs/order establish the injected-error ground truth for scoring.
The paper's manual landmark files are absent from the public release we used;
this tests injected errors, rather than naturally occurring ones. It does not
reproduce the paper's registration model. Earlier atlas retrieval results remain
in [benchmarks.csv](results/benchmarks.csv).

## Data and credits

Paper data are pinned to the [authors' release](https://github.com/acasamitjana/SynthByReg/tree/1b67b630cc4e8e03be210d6fb39210b722296902).
Downloads are checked against its Git blob hashes. Original data and weights are
not redistributed here. Older BigBrain-derived previews are resliced/scaled under
[CC BY-NC-SA 4.0](https://bigbrain-ftp.loris.ca/bigbrain-ftp/License.txt), crediting
[Amunts et al. (2013)](https://doi.org/10.1126/science.1235381).
Other source and model details are in [model_references.json](results/model_references.json).

Based on [Jignesh's Where Is My Slice](https://github.com/Jignesh1996/whereismyslice).
