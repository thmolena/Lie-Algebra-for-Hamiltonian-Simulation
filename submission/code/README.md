# specops-lieresidual — Residual-Generator Trotter Compilation (RGTC)

Deterministic code artifact for *The Trotter residual as an inference object: a
learnable, certified Lie-algebraic spectral truncation* (Molena Huynh, North
Carolina State University). Part of the **spectral-truncation operators
(specops)** program. The package regenerates every figure, table, and data file
the manuscript depends on from first principles, using deterministic dense-matrix
linear algebra on the transverse-field Ising model (TFIM) defined in
`common.py`. A single learned model (`learned_residual.py`) trains under a fixed
seed.

The distribution name is `specops-lieresidual`; the importable helper package is
`lieideal_hs`.

## Installation

```bash
cd submission/code
python -m pip install .              # installs specops-lieresidual 1.0.0 + console scripts
# editable, for development:
python -m pip install -e .
# or, for the exact pinned environment used to produce the reported numbers:
python -m pip install -r requirements.txt
```

Requires Python ≥ 3.9. Minimum dependencies (`pyproject.toml`): numpy ≥ 1.24,
scipy ≥ 1.10, pandas ≥ 2.0, matplotlib ≥ 3.7, torch ≥ 2.0. The pinned
environment used for the reported numbers is in `requirements.txt` (numpy 2.4.2,
scipy 1.17.1, pandas 3.0.1, matplotlib 3.10.8, torch 2.10.0). Only
`learned_residual.py` requires PyTorch; the dense-matrix figures build without
it, as `make_all.py` imports the learned experiment inside a guarded block and
prints a notice when torch is absent. The artifact runs on CPU.

## Reproduction

```bash
cd submission/code

# One command — deterministically regenerates every dataset, figure, and table:
lieresidual-reproduce       # console entry point -> lieideal_hs.reproduce:main
lieresidual-reproduce --force        # force-recompute all cached dense-matrix artifacts
lieresidual-reproduce --skip-validate  # skip the artifact-set validation step

# Backwards-compatible aliases (identical pipeline):
lieideal-reproduce          # -> cli:main -> make_all.main(force=True)

# Equivalent direct invocation:
python make_all.py

# Determinism gate — runs the pipeline twice and asserts byte-identical outputs:
lieideal-verify             # console entry point -> cli:verify
```

`lieresidual-reproduce` seeds all RNGs, then runs, in order, the four dense-matrix
experiments (fixed-time exact residual, compressed projected residual,
time sweep, generator structure), the resource-proxy table, the torch-based
learned-residual operator-learning experiment, and the main comparison
figure. `lieideal-verify` runs the full pipeline twice and compares the SHA-256
of every dataset and figure across runs.

### Outputs regenerated

| Directory | Contents |
| --- | --- |
| `figures/` | `fig0_overview`, `fig1_exact_residual`, `fig2_compressed_residual`, `fig3_time_sweep`, `fig4_generator_structure`, `fig5_learned_transfer`, `fig6_headline_improvement` (each as vector PDF and PNG) |
| `tables/` | `error_summary.tex`, `projected_summary.tex`, `resource_proxy.tex`, `learned_residual_summary.tex` |
| `generated_data/` | raw CSV/JSON for every experiment, each beside a `*.meta.json` recording parameters, library versions, platform, and git commit |

### Main numbers reproduced (verbatim from the generated data)

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
  uncorrected Strang error by 58–87× (e.g. 87.2× at `n=4`, 63.7× at `n=10`),
  with held-out coefficient parity `R²=0.9999` (relative ℓ₂ error
  `0.0075`).
- **Order generality + error–cost frontier**
  (`generated_data/order_generality.csv`, `frontier_cnot.csv`,
  `compiled_faithfulness.csv`, `xxz_generality.csv`; `tables/order_generality.tex`,
  `frontier.tex`, `xxz_generality.tex`; `figures/fig7_frontier.pdf`):
  residual locality holds at every order (the projected error collapses at weight
  ≈⌈q/2⌉+2 for `q=2,4,6`), and under the textbook CNOT cost model a *compiled*
  residual-corrected `q=4` step is Pareto-optimal — at `n=6` it reaches
  `7.95×10⁻¹⁰` at 274 CNOTs/step, in the gap between standard `q=6` (`3.50×10⁻⁹`,
  250 CNOTs) and `q=8` (`1.71×10⁻¹³`, 1250 CNOTs), i.e. **4.2–5.2× fewer two-qubit
  gates than the cheapest standard formula at matched accuracy** across `n=5,6,7`.
  The module also records the honest negative result that the *q=2* compiled
  correction does **not** reach its dense-oracle accuracy (inner-compilation
  floor), and that the XXZ chain is compressible at a larger weight threshold.

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
| `higher_order_frontier.py` | Order generality, compiled-correction faithfulness, two-qubit-gate error–cost frontier (corrected `q=4` vs standard Suzuki), XXZ generality → `fig7`, frontier/order/XXZ tables |
| `oracle_free_q4.py` | The `q=4` frontier correction is geometrically local (complete weight-≤5 local templates, Pareto-optimal, persists to `n=10`); honestly documents the finite-patch limit of small-patch tiling → `oracle_free_q4` table/data |
| `headline.py` | Main comparison figure → `fig6` |
| `make_all.py` | Runs all experiments and writes the resource table (Table IV) |
| `lieideal_hs/reproduce.py` | Console entry point `lieresidual-reproduce` (primary) |
| `cli.py` | Console entry points `lieideal-reproduce` and `lieideal-verify` (aliases) |
| `validate_submission.py` | Cross-checks the generated artifact set against the manuscript |
| `foundations.py`, `theory.py` | From-scratch theory-to-code map |

## Extend / tweak

Every experiment is a plain module with module-level constants and a
`main(force=...)` function; nothing is hidden behind a framework. The three ways
to change what runs are (1) environment variables, (2) editing the constants at
the top of a module, and (3) importing the physics primitives from `common.py`
into your own script.

### Redirect all outputs (no code edit)

```bash
export RGTC_OUTPUT_ROOT=/path/to/my_run   # figures/, tables/, generated_data/ go here
lieresidual-reproduce --force
```

`common.py` reads `RGTC_OUTPUT_ROOT`; when unset, outputs land in the manuscript's
`figures/`, `tables/`, and `generated_data/` (the dirs `main.tex` `\input`s and
`\includegraphics`es). Determinism knobs are also environment-driven:
`SOURCE_DATE_EPOCH` (PDF timestamp), `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/… (BLAS
thread count), and the seed passed by the CLI (`determinism.seed_everything`).

### Physics parameters (shared)

Every dense-matrix experiment calls into `common.py`, so the physical knobs are
uniform:

| Parameter | Where | Meaning / default |
| --- | --- | --- |
| `n_qubits` | argument to `common.global_errors`, `projected_residual_error`, per-module `for` loops | chain length; dense cost is `O(4ⁿ)`, so keep `n ≤ ~11` |
| `J`, `h` | `common.tfim_terms(n, J, h)` | TFIM `ZZ` coupling / transverse field (default `1.0`, `1.0`) |
| `t`, `r` | experiment args | total time and Trotter step count; `dt = t/r` |
| `order` (`q`) | `common.suzuki_sequence(order)`; `common.ORDERS = [1,2,4,6,8]` | product-formula order |
| `max_weight` (`w`) | `common.project_pauli_weight(K, n, w)` | Pauli-weight truncation level (the one accuracy knob) |

To sweep a different set of sizes/orders in the fixed-time table, edit the
`for n_qubits in (4, 5, 6)` / `for order in ORDERS` loops in `fixed_time.py`. The
frontier sizes are the `ns=(5, 6, 7)` default of `higher_order_frontier.frontier`;
the XXZ couplings are `Jxy=1.0, Jz=0.8, h=0.3` in `higher_order_frontier.xxz_terms`.

### Learned-residual parameters (`learned_residual.py`)

All constants live at the top of the module:

| Constant | Default | Effect |
| --- | --- | --- |
| `ORDER`, `MAX_WEIGHT` | `2`, `3` | base product-formula order and template weight (the `q=2` locality theorem fixes weight ≤ 3) |
| `TRAIN_SIZES`, `TRANSFER_SIZES` | `(4,5)`, `(4..10)` | sizes the net trains on / is evaluated on |
| `EVAL_REALIZATIONS` | `{4:40,…,10:12}` | disordered chains averaged per size |
| `N_TRAIN_REALIZATIONS` | `220` | training draws per training size |
| `DT_EVAL`, `DT_TRAIN_RANGE`, `DT_SWEEP` | `0.1`, `(0.05,0.30)`, six-point sweep | operating step size / training range / step-size sweep |
| `J_RANGE`, `H_RANGE` | `(0.5,1.5)` each | disorder distribution `U[·]` for `Jᵢ,hᵢ` |
| `PATCH_RADIUS` | `2` | site margin for oracle-free (patch) labels — larger = less finite-patch bias |
| `HIDDEN`, `EPOCHS`, `LR`, `WEIGHT_DECAY` | `64`, `4000`, `3e-3`, `1e-6` | MLP width and Adam schedule |
| `SEED` | `20240517` | disorder sampler, split, and net init |

### Add a new Hamiltonian or a new correction family

- **New Hamiltonian:** write a `terms(n, …) -> (A, B)` builder returning the two
  split generators (see `common.tfim_terms` and
  `higher_order_frontier.xxz_terms`), then feed them to
  `common.residual_factor(A, B, H, dt, order)` to get the residual `R_q`, and to
  `common.residual_generator(R)` for `K_q`. Everything downstream
  (`project_pauli_weight`, `spectral_error`) is Hamiltonian-agnostic.
- **New compression:** replace `common.project_pauli_weight` with your own map
  `K ↦ K̂` (e.g. a BCH truncation or a variational fit) and reuse the same error
  and compile helpers; `higher_order_frontier.compiled_correction` turns a term
  list into gates under the CNOT cost model.
- **New learned model:** swap the MLP in `learned_residual.py` (class defined near
  line 322) for any per-site `torch.nn.Module` with `N_FEATURES` inputs and
  `N_TEMPLATES` outputs; weight sharing across sites is what gives size transfer.

### Use the package in your own project

```python
import common
terms = common.tfim_terms(n_qubits=6, J=1.0, h=1.0)   # HamiltonianTerms(.A, .B, .H)
U, S, R, G = common.residual_factor(terms.A, terms.B, terms.H, dt=0.1, order=4)  # R = R_q
K = common.residual_generator(R)                                  # Hermitian generator K_q
K_hat = common.project_pauli_weight(K, n_qubits=6, max_weight=5)  # spectral truncation
err = common.projected_residual_error(n_qubits=6, order=4, t=1.0, r=10,
                                      max_weight=5, J=1.0, h=1.0)
```

## Cite this work

If you use this package or its results, please cite the paper (and, optionally,
the software via `CITATION.cff`):

```bibtex
@article{huynh2026lie,
  author  = {Huynh, Molena},
  title   = {The Trotter residual as an inference object: a learnable, certified Lie-algebraic spectral truncation},
  year    = {2026},
  note    = {Part of the spectral-truncation operators (specops) program},
}
```

## License

MIT. See [`LICENSE`](LICENSE).
