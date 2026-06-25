# lieideal-hs — Residual-Generator Trotter Compilation (RGTC)

Deterministic code artifact for *Provable operator learning of size-transferable
Trotter corrections for Hamiltonian simulation* (Molena Huynh, North Carolina
State University). The package regenerates every figure, table, and data file the
manuscript depends on from first principles, using deterministic dense-matrix
linear algebra on the transverse-field Ising model (TFIM) defined in
`common.py`. A single learned model (`learned_residual.py`) trains under a fixed
seed.

## Installation

```bash
cd submission/code
python -m pip install -e .          # installs lieideal-hs 1.0.0 and its console scripts
# or, for the exact pinned environment used to produce the reported numbers:
python -m pip install -r requirements.txt
```

Requires Python ≥ 3.10. Pinned dependencies (`requirements.txt`): numpy 2.4.2,
scipy 1.17.1, pandas 3.0.1, matplotlib 3.10.8, torch 2.10.0. Only
`learned_residual.py` requires PyTorch; the four dense-matrix figures build
without it, as `make_all.py` imports the learned experiment inside a guarded
block and prints a notice when torch is absent. The artifact runs on CPU.

## Reproduction

```bash
cd submission/code

# One command — deterministically regenerates every dataset, figure, and table:
lieideal-reproduce          # console entry point -> cli:main -> make_all.main(force=True)

# Equivalent direct invocation:
python make_all.py

# Determinism gate — runs the pipeline twice and asserts byte-identical outputs:
lieideal-verify             # console entry point -> cli:verify
```

`lieideal-reproduce` seeds all RNGs, then runs, in order, the four dense-matrix
experiments (fixed-time exact residual, compressed projected residual,
time sweep, generator structure), the resource-proxy table, the torch-based
learned-residual operator-learning experiment, and the headline comparison
figure. `lieideal-verify` runs the full pipeline twice and compares the SHA-256
of every dataset and figure across runs.

### Outputs regenerated

| Directory | Contents |
| --- | --- |
| `figures/` | `fig0_overview`, `fig1_exact_residual`, `fig2_compressed_residual`, `fig3_time_sweep`, `fig4_generator_structure`, `fig5_learned_transfer`, `fig6_headline_improvement` (each as vector PDF and PNG) |
| `tables/` | `error_summary.tex`, `projected_summary.tex`, `resource_proxy.tex`, `learned_residual_summary.tex` |
| `generated_data/` | raw CSV/JSON for every experiment, each beside a `*.meta.json` recording parameters, library versions, platform, and git commit |

### Headline numbers reproduced (verbatim from the generated data)

- **Fixed-time oracle** (`generated_data/fixed_time_errors.csv`; `J=h=1`, `t=1`,
  `r=10`): the oracle residual cancels the product-formula error to the
  double-precision floor. At `n=5`, `q=2` the baseline `1.384×10⁻²` drops to
  `1.036×10⁻¹⁴`.
- **Projected residual** (`generated_data/projected_residual_n5_q2.csv`; `n=5`,
  `q=2`): weight `w≤3` reduces the baseline `1.384×10⁻²` to `1.545×10⁻⁴`
  (89.56× improvement); `w≤4` reaches `1.850×10⁻⁷`.
- **Learned size transfer** (`generated_data/learned_residual_sizes.csv`,
  `tables/learned_residual_summary.tex`; disordered TFIM, `t=1`, `δt=0.1`):
  trained only on `n=4,5`, the network transfers to `n=10`, cutting the
  uncorrected Strang error by 58–85× (e.g. 85.1× at `n=4`, 63.1× at `n=10`),
  with held-out coefficient parity `R²=0.9999` (relative ℓ₂ error
  `1.32×10⁻²`).

## Determinism (seeds)

`determinism.py:seed_everything` fixes every source of run-to-run variation
before any experiment runs:

- `seed=42` for `random`, NumPy, and PyTorch (set by `lieideal-reproduce` and
  `lieideal-verify`).
- `SEED=20240517` inside `learned_residual.py` for the disorder sampler, the
  train/test split, and network initialization.
- All numerical-library thread counts pinned to 1
  (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`), fixing the floating-point
  reduction order.
- `SOURCE_DATE_EPOCH=1700000000` for byte-identical matplotlib PDF output.
- PyTorch deterministic algorithms enabled (`warn_only=True`), single-threaded.

Each learned network trains for full-batch Adam steps on 220 disordered
realizations at each of `n=4,5`. Figures are always drawn from the saved raw
data, never from hard-coded arrays.

### Finite-precision caveat

Only quantities at the double-precision floor can differ in their last digits
across BLAS builds: the oracle residuals (`~10⁻¹³`–`10⁻¹⁵`) and the weight-5
projected row. These are numerically zero / round-off and are flagged as such in
the manuscript. Every physically meaningful number — all baselines, the
projected `w≤3`/`w≤4` results, the learned-transfer errors, and `R²` — is stable.

## Dependencies

| Package | Pinned | Minimum (`pyproject.toml`) |
| --- | --- | --- |
| numpy | 2.4.2 | ≥ 1.24 |
| scipy | 1.17.1 | ≥ 1.10 |
| pandas | 3.0.1 | ≥ 2.0 |
| matplotlib | 3.10.8 | ≥ 3.7 |
| torch | 2.10.0 | ≥ 2.0 |

Tested with CPython 3.11.15.

## Repository map

| File | Role |
| --- | --- |
| `common.py` | Pauli operators, TFIM Hamiltonian, Trotter–Suzuki steps, exact propagator, residual factor and generator, Pauli-weight projection, spectral-norm error, deterministic I/O, figure style |
| `determinism.py` | `seed_everything`: fixes RNGs, thread counts, and the PDF timestamp |
| `overview.py` | Method-overview schematic (`fig0_overview`) |
| `fixed_time.py` | Fixed-time exact-residual experiment → Table I, `fig1` |
| `projected_residual.py` | Pauli-weight projection → Table II, `fig2` |
| `time_sweep.py` | Correction hierarchy versus time → `fig3` |
| `generator_scaling.py` | Generator structure and norm scaling → `fig4` |
| `learned_residual.py` | Translation-equivariant residual learner, size transfer → Table III, `fig5` |
| `headline.py` | Headline comparison figure → `fig6` |
| `make_all.py` | Runs all experiments and writes the resource table (Table IV) |
| `cli.py` | Console entry points `lieideal-reproduce` and `lieideal-verify` |
| `validate_submission.py` | Cross-checks the generated artifact set against the manuscript |
| `foundations.py`, `theory.py` | From-scratch theory-to-code map |

## License

MIT. See [`LICENSE`](LICENSE).
