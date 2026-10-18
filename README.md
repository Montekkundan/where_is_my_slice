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
python scripts/evaluate_errors.py --learned --output outputs/errors-learned
pytest -q
```

The benchmark uses 344 BigBrain and 93 Allen histology/MRI pairs released with
[Synth-by-Reg](https://arxiv.org/abs/2107.14449). Each histological section is paired
with a plane from the **same donor's MRI**. It injects known errors, fits a detector
on training sections, selects thresholds on validation sections and scores test
sections. Images are already approximately aligned by the publisher.

Review an ordered stack using a saved model:

```bash
slicegps detect-errors outputs/errors/neighbour_images.json sections/*.png --output outputs/review
```

Open `outputs/review/index.html`. Supply files in nominal section order; the first
and last two sections have insufficient context. Models named
`neighbour_images_and_expected_mri.json` additionally require `--mri` followed by
matching MRI planes in expected slot order. Use a new output directory each time.
Data, model weights and full outputs stay local.

## Results

These tests use the same corruption seeds, four error types and 20 trials per
type. BigBrain has 62 evaluated test sections; the independent Allen donor has 89.
Repeated trials reuse those sections and do not increase the donor count.

| Detector with expected MRI | Test donor | Errors detected | Clean sections falsely flagged |
| --- | --- | ---: | ---: |
| Image measurements + trained classifier | BigBrain | 77.5% | 6.5% |
| DINOv2 measurements + trained classifier | BigBrain | 51.5% | 9.3% |
| Image measurements + trained classifier | Unseen Allen | 100% | 100% |
| DINOv2 measurements + trained classifier | Unseen Allen | 100% | 100% |

Flagging every Allen section is a failure. Nearby swaps are also difficult:
the BigBrain image/MRI detector finds only 15% of them. A separate experiment
trained on both donors improves some Allen flip/tear checks, but still misses
most ordering errors. Its Allen test contains only 12 evaluated sections.
Calling every section clean already gives about 95–96% raw accuracy in the
independent tests, so recall and false alarms are reported together.

See [per-error results](results/error_detection.csv),
[image baseline](results/error_detection_baseline_independent.json),
[learned features](results/error_detection_learned_independent.json) and
[donor calibration](results/error_detection_learned_calibrated.json).
The detector uses pixels and neighbouring images, without reading section IDs or
clean copies. IDs/order establish the injected-error ground truth for scoring.
The paper's manual landmark files are absent from the public release we used;
these experiments do not measure naturally occurring error detection or reproduce
the paper's registration model. Earlier atlas retrieval results remain in
[benchmarks.csv](results/benchmarks.csv).

## Data and credits

Paper data are pinned to the [authors' release](https://github.com/acasamitjana/SynthByReg/tree/1b67b630cc4e8e03be210d6fb39210b722296902).
Downloads are checked against its Git blob hashes. Original data and weights are
not redistributed here. Older BigBrain-derived previews are resliced/scaled under
[CC BY-NC-SA 4.0](https://bigbrain-ftp.loris.ca/bigbrain-ftp/License.txt), crediting
[Amunts et al. (2013)](https://doi.org/10.1126/science.1235381).
Other source and model details are in [model_references.json](results/model_references.json).

Based on [Jignesh's Where Is My Slice](https://github.com/Jignesh1996/whereismyslice).
