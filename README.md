# Provable operator learning of size-transferable Trotter corrections for Hamiltonian simulation

The finite-step error of a Trotter–Suzuki product formula is the unique unitary
residual $R_q = U S_q^{\dagger}$. Its Hermitian generator $K_q = i\log R_q$ is
provably geometrically local for the transverse-field Ising model (TFIM), and a
single translation-equivariant network learns it from local couplings alone,
transferring from four- and five-qubit training chains to ten-qubit chains under
a closed-form $r\eta$ stability bound. Product-formula compilation becomes a
certified learning problem.

**Molena Huynh** · North Carolina State University · molena.huynh@jmp.com

## Summary

Digital Hamiltonian simulation realizes $U(t) = \exp(-iHt)$ by composing the
order-$q$ product-formula steps $S_q(\delta t)$ of a splitting $H = A + B$. The
implemented unitary never equals the target, and the entire defect is captured by
one object: the residual factor $R_q = U S_q^{\dagger}$, the unique unitary left
correction with $R_q S_q = U$. This work recasts the truncation error of the
integrator — conventionally an analytic quantity to be bounded — as an operator
to be learned under exact guarantees. The residual is proved optimal in every
unitarily invariant norm; Pauli-subspace projection of its generator is the
unique Frobenius-optimal compression; and approximate residuals convert per-step
error into global error linearly. For the open-boundary TFIM the leading Strang
generator is geometrically local of Pauli weight at most three, which reduces
compilation to a per-site learning problem solvable without ever forming the
dense $2^n$ propagator. Every reported number is recomputed from deterministic
dense-matrix simulation.

## Principal contributions

1. **An exact, optimal correction target.** $R_q = U S_q^{\dagger}$ is the unique
   unitary left multiplier with $R_q S_q = U$ and the global minimizer of the
   one-step error $|||U - L S_q|||$ in every unitarily invariant norm.
2. **Stability under approximation.** With per-step residual error
   $\eta = \lVert \widehat R_q - R_q \rVert_2$, the global error obeys
   $\lVert \widehat G_q^{\,r} - U(t) \rVert_2 \le r\eta$ for unitary
   $\widehat R_q$; a matching lower bound proves the linear-in-$r$ rate is
   order-optimal.
3. **Canonical, certified compression.** Pauli-subspace projection $\Pi_w K_q$ is
   the unique Frobenius-optimal compressed generator, with the direct certificate
   $\lVert e^{-iK} - e^{-i\Pi_w K} \rVert_2 \le \lVert K - \Pi_w K \rVert_F$.
4. **Provable locality for the TFIM.**
   $K_2(\delta t) = \delta t^3 K_2^{(3)} + O(\delta t^5)$ with the degree-three
   term a real combination of Pauli strings of weight at most three.
5. **A learned, size-transferable generator.** One translation-equivariant
   network predicts the step-conditioned local weight-$\le 3$ coefficients of
   $K_2$ from local couplings, trains without any dense $2^n$ oracle from
   fixed-size local patches, and transfers from $n = 4, 5$ to $n = 10$.
6. **Deterministic, falsifiable evidence.** Dense-matrix experiments, figures,
   tables, and code recompute every reported number from first principles, with
   unitarity and same-order consistency checks that expose any implementation
   error.

## Headline results

Transcribed from the manuscript ([`submission/main.tex`](submission/main.tex),
[`submission/main.pdf`](submission/main.pdf)) and the deterministic data the code
regenerates.

**Learned residual transferred across system size** (disordered TFIM,
$J_i, h_i \sim \mathcal{U}[0.5, 1.5]$, $t = 1$, $\delta t = 0.1$; trained only on
$n = 4, 5$, every $n \ge 6$ held out). Taking the analytic second-order
Zassenhaus generator as a prior and learning only the residual beyond it, the
network transfers to ten-qubit chains at $R^2 = 0.9999$ on held-out coefficients
(relative $\ell_2$ error $1.32 \times 10^{-2}$), cutting the uncorrected Strang
error by **58–85×**.

| $n$ | regime | $\epsilon_{\mathrm{Strang}}$ | $\epsilon_{\mathrm{learned}}$ | reduction | $n_{\mathrm c}$ |
| --- | --- | --- | --- | --- | --- |
| 4 | train | $1.049\times10^{-2}$ | $1.233\times10^{-4}$ | 85.1× | 40 |
| 5 | train | $1.549\times10^{-2}$ | $2.176\times10^{-4}$ | 71.2× | 40 |
| 6 | transfer | $1.844\times10^{-2}$ | $2.615\times10^{-4}$ | 70.5× | 40 |
| 7 | transfer | $2.028\times10^{-2}$ | $3.075\times10^{-4}$ | 65.9× | 40 |
| 8 | transfer | $2.433\times10^{-2}$ | $3.615\times10^{-4}$ | 67.3× | 24 |
| 9 | transfer | $3.207\times10^{-2}$ | $5.560\times10^{-4}$ | 57.7× | 16 |
| 10 | transfer | $3.668\times10^{-2}$ | $5.810\times10^{-4}$ | 63.1× | 12 |

Errors are global spectral-norm errors; reduction is
$\epsilon_{\mathrm{Strang}} / \epsilon_{\mathrm{learned}}$. The analytic prior
contributes a further $1.5$–$1.8\times$ over learning the generator without it.

**Fixed-time oracle benchmark** ($J = h = 1$, $t = 1$, $r = 10$). The oracle
residual cancels the product-formula defect to the floating-point floor: at
$n = 5$, $q = 2$ the baseline $1.384 \times 10^{-2}$ drops to
$1.036 \times 10^{-14}$.

**Projected (compressed) residual** ($n = 5$, $q = 2$). Weight $w \le 3$ reduces
the baseline $1.384 \times 10^{-2}$ to $1.545 \times 10^{-4}$ (89.56×
improvement); $w \le 4$ reaches $1.850 \times 10^{-7}$. Weight three is the first
threshold that meaningfully reduces error, exactly as the leading-support theorem
predicts.

> **Scope, stated plainly.** The reduction is measured against the order-$1/2$
> product-formula baseline; at this step size a standard higher-order Suzuki step
> is more accurate than the learned correction at comparable factor count. The
> oracle residual requires $U(\delta t)$ and is as costly as exact dense
> simulation. The learned residual removes the dense $2^n$ oracle from training,
> but evaluation here is dense (up to ten qubits), the local templates are
> TFIM-specific and leading-order, and native-gate synthesis is not attempted.
> The contribution is methodological: the integrator's truncation error is a
> learnable, geometrically local, size-transferable operator with an a priori
> stability certificate. Values near the double-precision floor ($\sim 10^{-13}$)
> are round-off, not physical error.

## Installation

```bash
pip install lieideal-hs
```

Or from source (Python ≥ 3.10):

```bash
cd submission/code
pip install -e .
```

## Reproduction

```bash
cd submission/code
lieideal-reproduce      # deterministically regenerates every figure, table, and dataset
lieideal-verify         # runs the pipeline twice and asserts byte-identical outputs
```

`lieideal-reproduce` (equivalently `python make_all.py`) seeds all RNGs, then
runs the four dense-matrix experiments, the resource-proxy table, the learned
residual operator-learning experiment, and the headline figure. The run is
deterministic: seed 42 for `random`/NumPy/PyTorch, `SEED = 20240517` inside
`learned_residual.py` for the disorder sampler, train/test split, and network
initialization; thread counts pinned to 1; and a fixed matplotlib PDF timestamp.
CPU-only. See [`submission/code/README.md`](submission/code/README.md) for the
full code-level reproduction guide.

## Repository layout

```text
submission/
  main.tex, main.pdf          manuscript source and compiled PDF
  figures/                    publication figures (PDF + PNG)
  tables/                     LaTeX result tables
  code/                       lieideal-hs package (pyproject.toml, LICENSE, README.md)
    make_all.py               regenerates every figure and table from first principles
    cli.py                    console entry points (lieideal-reproduce, lieideal-verify)
    common.py                 Pauli operators, TFIM, Trotter–Suzuki steps, residual generator
    fixed_time.py             exact-residual fixed-time benchmark
    projected_residual.py     Pauli-weight projection / compression
    time_sweep.py             correction hierarchy versus time
    generator_scaling.py      generator structure and norm scaling
    learned_residual.py       translation-equivariant residual learner (size transfer, torch)
    validate_submission.py    cross-checks the generated artifact set
    determinism.py            seed_everything: deterministic execution harness
    generated_data/           raw CSV/JSON with per-run metadata
notebooks/                    rendered architecture, stability, robustness studies
scripts/                      auxiliary experiment and plotting scripts
src/lc_qaoa/                  propagators, metrics, models, experiments
results/                      CSV result tables
index.html                   project homepage and interactive demo
website/index.html            condensed research brief
```

## Citation

```bibtex
@misc{huynh2025rgtc,
  author = {Huynh, Molena},
  title  = {Provable Operator Learning of Size-Transferable Trotter Corrections
            for Hamiltonian Simulation},
  year   = {2025},
  note   = {North Carolina State University},
  howpublished = {\url{https://github.com/thmolena/Lie-Algebra-for-Hamiltonian-Simulation}}
}
```

Molena Huynh, North Carolina State University, molena.huynh@jmp.com.

## License

MIT. See [`submission/code/LICENSE`](submission/code/LICENSE).
