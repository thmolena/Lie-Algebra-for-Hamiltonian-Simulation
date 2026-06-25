# Quarantined: synthetic / fabricated-style artifacts — DO NOT SUBMIT OR PUBLISH

These files were moved here on 2026-06-24 during a research-integrity review.
They are **not** used by the submission (`submission/main.tex` pulls every figure
from `submission/code/figures/`, all of which are produced by the real
dense-matrix + PyTorch pipeline in `submission/code/`). They are kept only so the
work is recoverable; **none of their figures or numbers may be presented as
experimental results.**

## Why these are quarantined

### `generate_liegpt_figures.py`
This is a single-qubit (`su(2)`) toy whose key "baselines" are **fabricated by
construction**, not measured from any trained model:

- **Rigged unitarity baseline** (lines ~193–196):
  ```python
  # Unconstrained: random complex matrix (simulates direct matrix pred)
  U_unc = (rng.randn(2, 2) + 1j * rng.randn(2, 2)) * 0.3
  U_unc += U_lie   # centred on correct propagator
  ```
  The "Unconstrained model" is literally injected Gaussian noise added to the
  correct propagator. Its reported unitarity violation (`outputs/generation_log.txt`:
  "Unconstrained median 1.13e+00") is an artifact of that injected noise, not the
  output of any learning method.

- **Data-efficiency "cheat"** (the script's own comment):
  ```python
  # compare on same 3-d prediction task (cheat: map 8->3 projection)
  ```
  The comparison is not apples-to-apples.

The figures it writes (`outputs/data_efficiency.png`, `noise_robustness.png`,
`unitarity_benchmark.png`, `bloch_sphere_unitarity.png`, `stability_rollout.png`,
`structure_constants.png`, `su2_basis.png`, `training_curves.png`,
`theorem1_bound.png`, `theorem2_complexity.png`, `master_comparison.png`,
`combined_summary.png`) are therefore not valid results and are quarantined with it.

### `liegpt_*.ipynb` notebooks
Notebook front-ends for the same `su(2)` toy / "LieGPT" framing. Not part of the
submission.

## What is REAL and stays in the repo
- `submission/` — the actual manuscript and its real pipeline (`submission/code/`),
  independently reproduced bit-for-bit.
- `results/*.csv` and the root `scripts/tfim_*`, `driven_tfim_*`, `*_benchmark.py`,
  `generate_paper_figures.py` — these do **real** Hamiltonian construction and
  matrix exponentiation. They are from an older "Lie GPT" version of the project
  (they target a `research_paper/` / `src/lc_qaoa` layout that no longer exists),
  so they are orphaned, but they are **not** fabricated and were left in place.

## Action item
Before any arXiv post or journal submission, delete this directory (or keep it
out of the public artifact) and ensure `README.md` / `index.html` at the repo root
do not advertise any number or figure that originates here.
