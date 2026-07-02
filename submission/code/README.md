# specops-lieresidual — the Trotter residual as a learnable, certified operator

Reference implementation and deterministic reproducibility package for *The Trotter
residual as an inference object: a learnable, certified Lie-algebraic spectral
truncation* (Molena Huynh, North Carolina State University). Part of the
**spectral-truncation operators (specops)** program. The distribution name is
`specops-lieresidual`; the importable helper package is `lieideal_hs`, and the
physics primitives live in `common.py`.

## Summary

Digital Hamiltonian simulation approximates the time-evolution operator
`U(t) = exp(-iHt)` by a Trotter–Suzuki product formula `S_q(δt)` of order `q`,
repeated `r` times. This work treats the *finite-step error* of that formula not as
an analytic quantity to be bounded a priori, but as a structured operator to be
compressed, certified, compiled, and learned. For an order-`q` step, the residual
factor `R_q = U(δt) S_q(δt)†` is the unique unitary left correction with
`R_q S_q = U(δt)`; its Hermitian generator `K_q = i·log R_q` is `O(δt^{q+1})` and
geometrically local. Projecting `K_q` onto the Pauli-weight filtration of the Lie
algebra of Hermitian operators is a *spectral truncation* controlled by a single
integer level `w`. The package computes every object in this pipeline — the exact
residual, its generator, the weight-`w` truncation, the certified error bound, the
first-order and symmetric gate-level compilations, the CNOT-cost error–cost
frontier, and a translation-equivariant learned model of the truncated generator —
by deterministic dense-matrix linear algebra, and regenerates every figure, table,
and number in the manuscript from first principles.

## Background & problem setting

The intended reader works in an adjacent field (numerical analysis, machine
learning for the physical sciences, or quantum information) but not in
product-formula simulation specifically. The setting is as follows.

- **Goal.** Given a many-body Hamiltonian `H` on `n` qubits (a `2^n × 2^n`
  Hermitian matrix), realize its time evolution `U(t) = exp(-iHt)` as a circuit of
  elementary gates. This is a founding application of quantum computers and a core
  subroutine of quantum phase estimation, chemistry and materials simulation, and
  Gibbs-state preparation.
- **Product formulas.** Writing `H = A + B` as a sum of two easily-exponentiated
  terms, an order-`q` product formula `S_q(δt)` approximates one short step
  `exp(-iH δt)` by an ordered product of the factors `exp(-iA·)` and `exp(-iB·)`
  (first order: `e^{-iBδt}e^{-iAδt}`; second-order Strang:
  `e^{-iBδt/2}e^{-iAδt}e^{-iBδt/2}`; higher even orders via the Suzuki recursion).
  Composing `r` steps approximates `U(t)` with a per-step error `O(δt^{q+1})`.
- **The cost that binds.** On hardware the limiting resource is the number of
  two-qubit gates (here counted as CNOTs), which sets circuit depth and therefore
  how much error accumulates. Raising the order `q` buys accuracy but multiplies
  the gate count by a large constant at every order — the elementary factor counts
  are `2, 3, 15, 75, 375` for `q = 1, 2, 4, 6, 8`. Every simulation therefore lives
  on an **error–cost frontier** that, in the standard construction, is a coarse
  staircase in the product-formula order.
- **The object of study.** All product-formula methods, and their many competitors
  (multi-product formulas, truncated Taylor series, quantum signal processing and
  qubitization, randomized formulas, Magnus-expansion methods), confront the same
  conceptual object: the finite-step *defect* between the desired unitary and the
  implemented one. This work isolates that defect as an operator and asks whether
  it can be compressed and learned so that a low-order formula reaches high-order
  accuracy without paying the high-order gate cost.

The benchmark Hamiltonian throughout is the open-boundary transverse-field Ising
model (TFIM), `H = J·Σ Z_j Z_{j+1} + h·Σ X_j`, with the natural two-term splitting
`A = J·Σ Z_j Z_{j+1}`, `B = h·Σ X_j`. It is small enough for exact dense simulation
at `n ≤ 10` yet exhibits genuine noncommuting local structure, so it serves as a
controlled laboratory for verifying the theorems and stress-testing the
compression.

## Contributions

1. **Exact residual factorization.** `R_q = U(δt) S_q(δt)†` is the unique unitary
   left correction completing `S_q` to the exact evolution, and is globally
   norm-optimal in every unitarily invariant norm.
2. **Multi-step stability, shown order-optimal.** Approximating `R_q` by `R̂_q`
   with per-step generator error `η` yields global error at most `r·η` over `r`
   steps; the linear-in-`r` estimate cannot be improved.
3. **Lie-algebraic spectral truncation.** Pauli-weight projection `Π_w K_q` is the
   unique Frobenius-optimal compressed generator, indexed by one accuracy level `w`
   that interpolates from no correction (`w = 0`) to the exact generator (`w = n`).
4. **Geometric convergence rate.** Each excluded weight level carries residual
   content of strictly higher order in `δt`, so the truncated error decays
   *geometrically* in `w` (measured per-level factors `449×` at `q = 2`, `576×` at
   `q = 4`; provable lower bound `δt^{-1} = 10` per level).
5. **A priori certificate.** The bound `r·‖K_q − Π_w K_q‖₂` is a valid global error
   guarantee at every level, computable from the generator without simulating the
   dynamics, so the minimal bandwidth `w★` for a target accuracy `ε` is read off
   *before* any simulation and grows only as `log(1/ε)`.
6. **Order-generality locality.** The leading order-`q` residual generator of the
   TFIM has Pauli weight at most `q + 2`, with an effective threshold
   `≈ ⌈q/2⌉ + 2`, so higher-order corrections stay compressible.
7. **Faithful compilation, with a symmetric repair.** The truncated correction
   compiles into mutually-commuting Pauli-rotation layers; a first-order inner
   product has an `O(δt^{2(q+1)})` floor (harmless for `q ≥ 4`, not for `q = 2`),
   and a symmetric inner compilation lowers it to `O(δt^{3(q+1)})`, e.g. repairing
   the `n = 6`, `q = 2`, `w ≤ 4` error from `2.18×10⁻⁵` to `3.76×10⁻⁷`.
8. **An advance of the error–cost frontier (the central result).** A
   spectrally-truncated fourth-order correction reaches global spectral-norm errors
   *between* standard sixth- and eighth-order Suzuki formulas at `4.2–5.2×` fewer
   two-qubit gates than the cheapest standard formula of equal accuracy
   (`n = 5–7`), placing new Pareto-optimal points on the frontier.
9. **A learned, size-transferable generator.** One translation-equivariant network,
   trained offline on four- and five-qubit disordered chains without any dense `2^n`
   propagator, predicts the truncated generator from local couplings, transfers to
   ten qubits (`R² = 0.9999` on held-out sizes), and cuts the uncorrected Strang
   error by `58–87×` under the proven stability bound.
10. **Generality beyond the TFIM and full determinism.** The mechanism extends to
    the XXZ/Heisenberg chain (larger weight threshold, `≈ 5`; `54×` reduction at
    `w ≤ 5`), and every reported number is an exact, deterministic, reproducible
    dense-matrix computation.

## Method

The pipeline, all implemented in `common.py` and `spectral_truncation.py`, is:

1. **Residual and generator.** For a chosen splitting `H = A + B`, order `q`, and
   step `δt`, form `S_q(δt)` by the Suzuki recursion, the exact `U(δt)` by dense
   matrix exponential, the residual `R_q = U S_q†`, and the Hermitian generator
   `K_q = i·log R_q` (principal branch, well defined for small `δt`).
2. **Spectral truncation.** Expand `K_q` in the Pauli basis and orthogonally
   project onto the weight-`≤ w` subspace to obtain `Π_w K_q` — the one-knob
   compression. The certified tail `‖K_q − Π_w K_q‖₂` is read directly off the
   generator.
3. **Compilation.** Decompose `Π_w K_q` into mutually-commuting Pauli-rotation
   layers and realize `exp(-i Π_w K_q)` either by a first-order product over the
   layers or by a symmetric (palindromic) product; count CNOTs under the standard
   cost model (a weight-`w` Pauli rotation costs `2(w−1)` CNOTs, a TFIM `ZZ` layer
   `2(n−1)`, single-qubit layers free).
4. **Learning.** On the disordered TFIM, a translation-equivariant multilayer
   perceptron (`12` inputs — `11` local couplings plus `δt` — through two width-`64`
   `tanh` layers to `48` weight-`≤ 3` template coefficients, weights shared across
   all sites and chain lengths) predicts the `δt^{-3}`-scaled coefficients on top of
   an analytic second-order Zassenhaus prior (delta learning). It is trained from
   fixed-size local patches, so no dense `2^n` object ever enters training.

## Main results

All values below are exact, deterministic dense-matrix computations on the
open-boundary TFIM (`J = h = 1`, `t = 1`, `r = 10` unless noted), reproduced
verbatim from the generated data.

- **Fixed-time oracle** (`generated_data/fixed_time_errors.csv`): the oracle
  residual cancels the product-formula error to the double-precision floor — at
  `n = 5`, `q = 2` the baseline `1.384×10⁻²` drops to `1.036×10⁻¹⁴`.
- **Projected residual** (`generated_data/projected_residual_n5_q2.csv`; `n = 5`,
  `q = 2`): weight `w ≤ 3` reduces the baseline `1.384×10⁻²` to `1.545×10⁻⁴`
  (`89.56×`); `w ≤ 4` reaches `1.850×10⁻⁷`.
- **Geometric spectral truncation** (`n = 6`): each retained weight level beyond the
  effective threshold multiplies the corrected error by a measured `449×` (`q = 2`)
  and `576×` (`q = 4`).
- **Certificate-driven bandwidth** (`n = 6`, `q = 2`): the a priori bound is a valid
  upper bound at every level; it selects `w★ = 4` for `ε = 10⁻⁴` (certificate
  `3.7×10⁻⁷`, achieved `3.65×10⁻⁷`) and `w★ = 5` for `ε = 10⁻⁸`.
- **Error–cost frontier** (`generated_data/frontier_cnot.csv`; `n = 6`): the
  compiled residual-corrected `q = 4`, `w ≤ 5` step reaches `7.95×10⁻¹⁰` at `274`
  CNOTs/step — in the accuracy gap between standard `q = 6` (`3.50×10⁻⁹`, `250`
  CNOTs) and `q = 8` (`1.71×10⁻¹³`, `1250` CNOTs) — i.e. **4.2–5.2× fewer two-qubit
  gates than the cheapest standard formula at matched accuracy** across `n = 5,6,7`.
  The purely-local `q = 4` correction stays Pareto-optimal to `n = 10`
  (`2.03×10⁻⁹` at `650` CNOTs).
- **Learned size transfer**
  (`generated_data/learned_residual_sizes.csv`, `tables/learned_residual_summary.tex`;
  disordered TFIM, `t = 1`, `δt = 0.1`): trained only on `n = 4,5`, the network
  transfers to `n = 10`, cutting the uncorrected Strang error by `58–87×`, with
  held-out coefficient parity `R² = 0.9999`.
- **Honest negatives.** The `q = 2` first-order compiled correction does *not* reach
  its dense-oracle accuracy (inner-compilation floor, repaired by the symmetric
  compile); correcting a second-order step is *not* competitive with simply using a
  higher-order formula at the same step size (the frontier advance is specific to
  correcting an already-high order); and small-patch tiling of the `q = 4` generator
  is not yet bulk-converged at `n = 10`.

## Significance

The finite-step error of a numerical integrator — usually treated as an analytic
object to be bounded — is here a *learnable*, geometrically local, step-size-
conditioned operator whose approximation quality is controlled *a priori* by a
proven stability bound rather than validated only a posteriori. The enabling
compression is algebraic: the correction lives in the Lie algebra of Hermitian
operators, and Pauli-weight projection is a spectral truncation of that algebra to
a finite-dimensional, symmetry-respecting hypothesis class indexed by an
interaction-range knob — the Hamiltonian-simulation counterpart of the
operator-algebraic spectral truncations of C\*-algebraic kernel learning, but
strengthened by a quantitative convergence rate, an a priori dynamical certificate,
and a measurable gate-level payoff that the static construction lacks. For
simulation practice, reaching eighth-order-class accuracy at roughly a fifth of the
gate cost is a direct reduction in circuit depth on the axis hardware actually pays
for; because the correction is a local, transferable function of the couplings, the
cost of finding it is amortized once on small chains and reused at scale.

## Installation & reproduction

Requires Python ≥ 3.9. Minimum dependencies (`pyproject.toml`): numpy ≥ 1.24,
scipy ≥ 1.10, pandas ≥ 2.0, matplotlib ≥ 3.7, torch ≥ 2.0. Only
`learned_residual.py` needs PyTorch; the dense-matrix figures build without it. The
artifact runs on CPU.

```bash
cd submission/code
pip install .                        # installs specops-lieresidual 1.0.0 + console scripts
# editable, for development:
pip install -e .
# or the exact pinned environment used for the reported numbers:
pip install -r requirements.txt      # numpy 2.4.2, scipy 1.17.1, pandas 3.0.1, matplotlib 3.10.8, torch 2.10.0
```

Reproduce every dataset, figure, and table with a single console entry point:

```bash
cd submission/code
lieresidual-reproduce                # primary entry point -> lieideal_hs.reproduce:main
lieresidual-reproduce --force        # force-recompute all cached dense-matrix artifacts
lieresidual-reproduce --skip-validate  # skip the artifact-set validation step

# Determinism gate — runs the pipeline twice and asserts byte-identical outputs:
lieideal-verify                      # console entry point -> cli:verify

# Backwards-compatible aliases (identical pipeline):
lieideal-reproduce                   # -> cli:main -> make_all.main(force=True)
python make_all.py                   # equivalent direct invocation
```

`lieresidual-reproduce` seeds all RNGs (`seed_everything`: `seed=42` for
`random`/NumPy/PyTorch; `SEED=20240517` inside `learned_residual.py`; all BLAS
thread counts pinned to 1; `SOURCE_DATE_EPOCH=1700000000` for byte-identical PDF
output), then runs the dense-matrix experiments, the learned-residual experiment,
and the figures in order. `lieideal-verify` reruns the full pipeline and compares
the SHA-256 of every dataset and figure across runs.

**Outputs.** `figures/` (each as vector PDF and PNG), `tables/` (`.tex` inputs to
`main.tex`), and `generated_data/` (raw CSV/JSON, each beside a `*.meta.json`
recording parameters, library versions, platform, and git commit).

**Finite-precision caveat.** Only quantities at the double-precision floor can differ
in their last digits across BLAS builds: the oracle residuals (`~10⁻¹³–10⁻¹⁵`) and
the weight-5 projected row. Every physically meaningful number — all baselines, the
`w ≤ 3`/`w ≤ 4` results, the learned-transfer errors, and `R²` — is stable.

### Repository map

| File | Role |
| --- | --- |
| `common.py` | Pauli operators, TFIM Hamiltonian, Trotter–Suzuki steps, exact propagator, residual factor and generator, Pauli-weight projection, spectral-norm error, deterministic I/O, figure style |
| `spectral_truncation.py` | Geometric convergence rate, certificate-driven bandwidth, first-order and symmetric compilations (`truncation_rate`, `faithful_compilation`, `certificate_selection`) |
| `higher_order_frontier.py` | Order generality, compiled-correction faithfulness, CNOT error–cost frontier (corrected `q=4` vs standard Suzuki), XXZ generality |
| `oracle_free_q4.py` | The `q=4` frontier correction is geometrically local (weight-≤5 local templates, Pareto-optimal to `n=10`); documents the finite-patch tiling limit |
| `learned_residual.py` | Translation-equivariant residual learner, size transfer |
| `fixed_time.py`, `projected_residual.py`, `time_sweep.py`, `generator_scaling.py`, `headline.py`, `overview.py` | The remaining dense-matrix experiments and schematic figures |
| `determinism.py` | `seed_everything`: fixes RNGs, thread counts, PDF timestamp |
| `make_all.py`, `lieideal_hs/reproduce.py`, `cli.py` | Reproduction drivers and console entry points |
| `validate_submission.py` | Cross-checks the generated artifact set against the manuscript |

## Extend / tweak

Every experiment is a plain module with module-level constants and a
`main(force=...)` function; nothing is hidden behind a framework.

**Redirect outputs (no code edit).** `export RGTC_OUTPUT_ROOT=/path/to/my_run`
before `lieresidual-reproduce --force`; `common.py` reads it, and `figures/`,
`tables/`, `generated_data/` land there.

**Shared physics parameters** (all routed through `common.py`):

| Parameter | Where | Meaning / default |
| --- | --- | --- |
| `n_qubits` | args to `common.global_errors`, `projected_residual_error`, module loops | chain length; dense cost `O(4ⁿ)`, keep `n ≤ ~11` |
| `J`, `h` | `common.tfim_terms(n, J, h)` | TFIM `ZZ` coupling / transverse field (default `1.0`, `1.0`) |
| `t`, `r` | experiment args | total time and Trotter step count; `δt = t/r` |
| `order` (`q`) | `common.suzuki_sequence(order)`; `common.ORDERS = [1,2,4,6,8]` | product-formula order |
| `max_weight` (`w`) | `common.project_pauli_weight(K, n, w)` | Pauli-weight truncation level (the one accuracy knob) |

Edit the `for n_qubits in (4, 5, 6)` / `for order in ORDERS` loops in `fixed_time.py`
to sweep other sizes/orders; frontier sizes default to `ns=(5, 6, 7)` in
`higher_order_frontier.frontier`; XXZ couplings are `Jxy=1.0, Jz=0.8, h=0.3` in
`higher_order_frontier.xxz_terms`.

**Learned-residual knobs** (constants at the top of `learned_residual.py`): `ORDER`,
`MAX_WEIGHT` (`2`, `3`); `TRAIN_SIZES`, `TRANSFER_SIZES` (`(4,5)`, `(4..10)`);
`N_TRAIN_REALIZATIONS` (`220`); `DT_EVAL`, `DT_TRAIN_RANGE` (`0.1`, `(0.05,0.30)`);
`J_RANGE`, `H_RANGE` (`(0.5,1.5)`); `PATCH_RADIUS` (`2` — larger reduces
finite-patch bias); `HIDDEN`, `EPOCHS`, `LR` (`64`, `4000`, `3e-3`); `SEED`
(`20240517`).

**Add a new Hamiltonian.** Write a `terms(n, …) -> (A, B)` builder (see
`common.tfim_terms`, `higher_order_frontier.xxz_terms`), feed it to
`common.residual_factor(A, B, H, dt, order)` for `R_q` and
`common.residual_generator(R)` for `K_q`; everything downstream
(`project_pauli_weight`, `spectral_error`) is Hamiltonian-agnostic.

**Add a new compression or learned model.** Replace `common.project_pauli_weight`
with any map `K ↦ K̂`, or swap the per-site MLP in `learned_residual.py` for any
`torch.nn.Module` with `N_FEATURES` inputs and `N_TEMPLATES` outputs; weight
sharing across sites is what gives size transfer.

Minimal use of the primitives:

```python
import common
terms = common.tfim_terms(n_qubits=6, J=1.0, h=1.0)                              # HamiltonianTerms(.A, .B, .H)
U, S, R, G = common.residual_factor(terms.A, terms.B, terms.H, dt=0.1, order=4)  # R = R_q
K = common.residual_generator(R)                                                 # Hermitian generator K_q
K_hat = common.project_pauli_weight(K, n_qubits=6, max_weight=5)                 # spectral truncation
err = common.projected_residual_error(n_qubits=6, order=4, t=1.0, r=10,
                                      max_weight=5, J=1.0, h=1.0)
```

## Cite this work

```bibtex
@article{huynh2026lie,
  author  = {Huynh, Molena},
  title   = {The Trotter residual as an inference object: a learnable, certified Lie-algebraic spectral truncation},
  year    = {2026},
  note    = {Part of the spectral-truncation operators (specops) program},
  url     = {https://thmolena.github.io/Lie-Algebra-for-Hamiltonian-Simulation/submission/main.pdf}
}
```

Software may additionally be cited via `CITATION.cff`.

## License

MIT. See [`LICENSE`](LICENSE).
