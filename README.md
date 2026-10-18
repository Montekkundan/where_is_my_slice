# Where Is My Slice

Match human brain histology sections to MRI planes and inspect the order of a
section stack. Includes BigBrain benchmarks and NextBrain block preparation.

![Where's My Slice? concept demo](wheres_my_slice_search.gif)
![Stack QA concept demo](wheres_my_slice_stack.gif)

Original concept animations by Jignesh. Measured results are in `results/`.

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
slicegps demo --output outputs/demo
pytest -q
```

Use `slicegps --help` for data downloads, paired-plane generation, retrieval
benchmarks and NextBrain previews. Each output directory must be new or empty.

## Data

- [BigBrain](https://bigbrain.loris.ca/): reconstructed histology paired with the
  ICBM152 atlas in the same reference space. This MRI is not the donor's MRI.
- [NextBrain Case 4](https://github.com/UCL/NextBrain-Case-4): donor MRI and prepared
  histology. Read slab/block mappings; piece counts vary. NIfTI blocks use their
  own affine, while web images use the supplied row/column/section matrix.

Downloads, environments and full experiment outputs stay local. Compact benchmark
results are in `results/`. Current tests use known crops and near-coronal planes;
arbitrary detached fragments and generalization to new donors remain unvalidated.

Based on [Jignesh's Where Is My Slice](https://github.com/Jignesh1996/whereismyslice).
