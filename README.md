# Where Is My Slice

A research prototype for histology-to-MRI candidate retrieval, section-stack
review and NextBrain block preparation.

![Where's My Slice? concept demo](wheres_my_slice_search.gif)
![Stack QA concept demo](wheres_my_slice_stack.gif)

Original concept animations by Jignesh. They illustrate the intended workflow;
measured performance is reported below.

## Run

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,learned]'
python scripts/run_experiments.py --output outputs/run --device auto
python scripts/summarize_results.py --run outputs/run
pytest -q
```

The full run downloads public data and pinned DINOv2 weights, compares image
baselines and trained heads, checks two NextBrain donors, and writes standalone
readers to `outputs/run/matches/index.html` and `outputs/run/stack/index.html`.
Use a new output directory for each run. Data, models and full outputs stay local.
`requirements.lock` and `requirements-learned.lock` record the evaluated versions.
For an offline example without model dependencies, install `.[dev]` and run
`slicegps demo --output outputs/demo`.

To search an MRI using your own PNG, JPEG or TIFF sections:

```bash
slicegps mri-gallery brain.nii.gz --output outputs/gallery
slicegps index outputs/gallery --output outputs/index --method dino
slicegps match outputs/index sections/*.png --output outputs/matches
```

Match the candidate crop and pixel spacing to the supplied sections with
`mri-gallery --help`. This does not estimate unknown crop placement or cutting
direction. Use `--method intensity`, `gradient` or `mask` for simpler baselines;
`learned-fit`, `train-dual`, `--adapter` and `--model` expose the trained alternatives.

## Results

BigBrain: 48 near-coronal test planes, 128 MRI candidates, separated spatial bands,
known crop, 5 mm center tolerance. Frozen spatial DINO was selected on validation.

| Method | Median error | 95th percentile | Within 5 mm |
| --- | ---: | ---: | ---: |
| Intensity | 43.52 mm | 86.05 mm | 0.0% |
| Frozen spatial DINOv2 | 8.86 mm | 128.80 mm | 39.6% |
| Spatial ridge adapter | 13.26 mm | 17.52 mm | 18.8% |
| Trained contrastive + pose heads | 13.63 mm | 51.60 mm | 8.3% |

The lower median is promising, but tail errors and stack ordering remain poor.
The trained pose head does not beat a constant-normal prior. These exploratory
within-volume results do not validate arbitrary fragments, raw-section pose or
population generalization. See [all metrics](results/benchmarks.csv),
[BigBrain protocol](results/bigbrain.json), [stack scores](results/stack.json) and
[NextBrain checks](results/nextbrain.json). NextBrain uses 8 real registered HE
queries per donor, with 60 and 32 known-block MRI crops; these are small, easier
checks, and DINO does not consistently beat simple baselines.

## Data and credits

[BigBrain](https://bigbrain.loris.ca/) uses reconstructed histology registered to
an ICBM152 atlas MRI, which is not the donor's MRI. BigBrain-derived previews are
resliced/scaled under [CC BY-NC-SA 4.0](https://bigbrain-ftp.loris.ca/bigbrain-ftp/License.txt),
crediting [Amunts et al. (2013)](https://doi.org/10.1126/science.1235381).
[NextBrain](https://doi.org/10.5522/04/24243835) data are CC0. Piece counts vary by
slab; the A5 preview places two pieces using supplied affines. It does not infer
unknown piece placement. [Allen anatomy atlases](https://brain-map.org/atlases/anatomy)
are a review reference. Sources, model revision and image rights are in
[model_references.json](results/model_references.json).

Based on [Jignesh's Where Is My Slice](https://github.com/Jignesh1996/whereismyslice).
