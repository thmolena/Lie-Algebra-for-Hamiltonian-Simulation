# Lie-Algebraic Residual Compilation for Hamiltonian Simulation

A learned, **provably-bounded** correction to Trotter–Suzuki quantum-simulation error. A
single neural network predicts the exact correction from *local* Hamiltonian couplings,
generalizes from tiny training systems to larger ones it has never seen, trains with **no
exponential-size oracle**, and inherits a closed-form error guarantee — all built on rigorous
theorems and reproduced entirely from first principles.

## Highlights

- **Out-of-distribution size transfer.** Trained only on 4–5-qubit chains, one
  translation-equivariant network corrects systems up to **10 qubits** (a 1024-dimensional
  Hilbert space) it never saw — cutting product-formula error **40–45×**, with **$R^2 = 0.9998$**
  on held-out sizes. A *proved* locality theorem explains why it generalizes.
- **No exponential oracle.** The model trains from fixed-size local patches (≤ 7 qubits),
  never a dense $2^n$ propagator, and matches the oracle-trained model — removing the
  scalability bottleneck of exact methods.
- **Machine learning with a proof.** A multi-step stability theorem guarantees global error
  grows only linearly in the model's per-step error ($r\eta$): the learned correction is
  *safe by construction*.
- **Beats the analytic baseline.** Conditioned on step size, it outperforms the leading-order
  BCH correction by **more than 10×** exactly where that correction fails.
- **Fully reproducible.** Every number is recomputed by deterministic first-principles code;
  the manuscript compiles and the pipeline regenerates every figure and table.

See the result in one figure: [`submission/figures/fig7_learned_residual.png`](submission/figures/fig7_learned_residual.png).

## Overview

Digital Hamiltonian simulation implements $U(t) = \exp(-iHt)$ by composing short
product-formula steps $S_q(\delta t)$ of order $q$. Each step carries a finite defect
controlled by nested commutators that persists at every fixed Suzuki order. This project
isolates that defect as an operator and studies it with Lie-algebraic and numerical-analysis tools.

For a product-formula step $S_q(\delta t)$ approximating $U(\delta t) = \exp(-iH\delta t)$,
define the **residual**

$$R_q(\delta t) = U(\delta t)\, S_q(\delta t)^{\dagger}.$$

The corrected step $R_q(\delta t)\, S_q(\delta t)$ equals $U(\delta t)$ exactly. The residual
is the *unique* unitary left multiplier with this property, and its Hermitian generator
$K_q = i\log R_q$ lies in the dynamical Lie algebra and is small for small $\delta t$.

The framework distinguishes two modes, and is deliberately honest about the scope of each:

- **Oracle residual** — $R_q$ computed from the exact dense matrix exponential. Not scalable;
  it establishes the exact target and the best possible one-step left correction.
- **Compressed residual** — a generator $\widehat{K}_q$ drawn from an implementable family
  (Pauli-weight projection, BCH truncation, variational compilation, or a learned model).
  This is the practical object, and its error is provably controlled.

## What this project demonstrates

- **Applied mathematics** — Lie-group/Lie-algebra structure, operator theory, optimality in
  every unitarily invariant norm, Baker–Campbell–Hausdorff and Duhamel-type perturbation
  bounds, and Frobenius-optimal subspace projection.
- **Quantum algorithms** — product-formula Hamiltonian simulation, circuit synthesis and
  compilation, and resource budgeting for fault-tolerant and NISQ settings.
- **Scientific computing** — deterministic dense-matrix experiments in which every reported
  number is recomputed from first principles, with no hand entry and no black-box model.
- **AI for quantum** — the compressed generator can be fit offline (GPU-accelerated) or
  predicted by a unitarity-preserving learned model, with global error provably linear in
  prediction error.

## Theoretical results

All results are proved in the paper ([`submission/main.tex`](submission/main.tex),
[`submission/main.pdf`](submission/main.pdf)) and verified numerically.

| Result | Statement |
| --- | --- |
| Exact factorization | $R_q = U S_q^{\dagger}$ is the unique unitary left multiplier with $R_q S_q = U$. |
| Optimality & uniqueness | $R_q$ minimizes the one-step correction error in every unitarily invariant norm. |
| Multi-step stability | An approximate residual with per-step error $\eta$ gives total error $\le r\eta$ over $r$ steps. |
| Generator-error bound | A Duhamel-type bound links spectral error to $\lVert K_q - \widehat{K}_q\rVert$. |
| Frobenius-optimal projection | Pauli-subspace projection is the unique Frobenius-optimal compressed generator. |
| TFIM structure | The leading Strang residual-generator term has Pauli weight $\le 3$, explaining why low-weight residuals work. |

## Numerical results

Deterministic dense-matrix experiments on the open-boundary transverse-field Ising model
(TFIM) for $n = 4, 5, 6$ qubits and orders $q \in \{1, 2, 4, 6, 8\}$.

- **Oracle exactness.** The corrected step cancels the product-formula defect to
  floating-point precision (spectral-norm error $\sim 10^{-15}$) at every tested order and
  system size — independent of Hilbert-space dimension.
- **Compressed residual.** A non-oracle Pauli-weight projection at $n = 5$, $q = 2$ reduces
  the global spectral-norm error from $1.384\times10^{-2}$ to $1.545\times10^{-4}$ at weight
  $w \le 3$, and to $1.850\times10^{-7}$ at weight $w \le 4$ — structured compression recovers
  most of the oracle gain using only local, low-weight terms.

Figures (`submission/figures/`): fixed-time benchmarks, time-resolved sweeps, an
improvement-ratio heatmap over the $(J, h)$ parameter grid, the projected-residual
compression curve, generator-weight distribution, and order scaling.

## Learned residual generators that transfer across system size

The locality theorem makes residual compilation a well-posed learning problem: because the
leading Strang residual generator is supported on geometrically local, weight-$\leq 3$ Pauli
strings, a single per-site network can reconstruct it, and the same network applies to a
chain of any length. We train one step-size-conditioned, translation-equivariant multilayer
perceptron to predict the local residual coefficients of $K_2$ from the local couplings of a
disordered transverse-field Ising chain — with no access to the exact propagator at inference
— and find:

- **Size transfer to n = 10.** Trained only on $n = 4, 5$ qubit chains, the same network cuts
  the uncorrected Strang spectral-norm error by $40$–$45\times$ on chains up to $n = 10$
  (a 1024-dimensional Hilbert space) that it never saw during training, with $R^2 = 0.9998$
  on held-out-size coefficients.
- **No global oracle needed.** The network trains from fixed-size local patches alone (at most
  seven qubits) — never a dense $2^n$ propagator — and matches, even slightly beats, the
  oracle-trained model. This removes the framework's central scalability limitation.
- **Beats the analytic baseline.** Conditioned on the step size, the learned correction
  outperforms the parameter-free leading-order BCH correction by more than $10\times$ at larger
  steps, where leading-order BCH degrades toward the uncorrected formula.
- **Provable control.** The total error grows linearly in the number of Trotter steps, exactly
  as the multi-step stability theorem predicts — so the learned model inherits the framework's
  error guarantee.

All error metrics are computed by the same first-principles dense-matrix code; the network
replaces only the costly map from a local Hamiltonian to its residual coefficients. See
[`submission/code/learned_residual.py`](submission/code/learned_residual.py) and
`submission/figures/fig7_learned_residual.png`.

## Repository structure

```text
submission/
  main.tex                    LaTeX source for the paper
  main.pdf                    Compiled paper
  figures/                    Publication figures (PDF + PNG)
  tables/                     LaTeX result tables
  code/
    make_all.py               Regenerates every figure and table from first principles
    validate_submission.py    Re-checks the reported numbers
    fixed_time.py time_sweep.py projected_residual.py
    parameter_heatmap.py generator_scaling.py common.py
    learned_residual.py       Operator-learning experiment (size transfer, torch)
notebooks/
  liegpt_architecture_unitarity.ipynb   Unitarity-by-construction study
  liegpt_stability.ipynb                Long-horizon rollout stability
  liegpt_efficiency_robustness.ipynb    Data efficiency and noise robustness
  *.html                                Rendered notebook outputs
scripts/                      Auxiliary experiment and plotting scripts
src/
  lc_qaoa/                    Propagators, metrics, models, experiments
results/                      CSV result tables
outputs/                     Rendered figures from the notebook studies
index.html                   Project homepage (demo)
requirements.txt
```

## Reproducing the results

All experiments are configured to run on CPU.

### Paper figures and tables

```bash
pip install -r requirements.txt

# Regenerates every figure in submission/figures/ and table in submission/tables/
cd submission/code
python make_all.py

# Re-checks the reported numbers
python validate_submission.py
```

### Notebook studies

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/liegpt_architecture_unitarity.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/liegpt_stability.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/liegpt_efficiency_robustness.ipynb
```

### Local page preview

```bash
python -m http.server 8000
```

Then open `index.html` in a browser.

## Primary materials

- [Paper (PDF)](submission/main.pdf) · [LaTeX source](submission/main.tex)
- [Project homepage](index.html)
- [Architecture & unitarity notebook](notebooks/liegpt_architecture_unitarity.html)
- [Stability notebook](notebooks/liegpt_stability.html)
- [Efficiency & robustness notebook](notebooks/liegpt_efficiency_robustness.html)
