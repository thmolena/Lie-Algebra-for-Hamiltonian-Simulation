# Lie-algebraic spectral truncation of the product-formula residual

A certified, geometrically convergent, learnable compression of the Trotter–Suzuki
finite-step error for Hamiltonian simulation.

**Molena Huynh** · North Carolina State University · molena.huynh@jmp.com

Manuscript: [`submission/main.pdf`](submission/main.pdf) ·
LaTeX source [`submission/main.tex`](submission/main.tex). Reproducibility package
(`lieideal-hs`): [`submission/code`](submission/code).

## The most novel contribution

The finite-step error of an order-$q$ product-formula step $S_q(\delta t)$ is carried in
full by one object: the residual factor $R_q = U(\delta t)\,S_q(\delta t)^{\dagger}$, the
unique unitary left correction with $R_q S_q = U(\delta t)$, whose Hermitian generator is
$K_q = i\log R_q$. The central result of this work is that **projecting $K_q$ onto the
Pauli-weight filtration of the Lie algebra of Hermitian operators is a genuine *spectral
truncation*** of that algebra: a single level $w$ interpolates from the uncorrected step
($w = 0$) to the exact correction ($w = n$). This is the Hamiltonian-simulation
counterpart of the operator-algebraic spectral truncations recently introduced in
C\*-algebraic kernel learning (Hashimoto et al., *Spectral Truncation Kernels*, 2024),
and the simulation setting supplies three guarantees that construction does not provide.

1. **A quantitative geometric convergence rate**, not merely convergence in a limit. The
   truncated-correction error decays geometrically in the level — provably $\geq 10\times$
   per level, and measured at $449\times$ ($q = 2$) and $576\times$ ($q = 4$) per level on
   the transverse-field Ising chain (geometric-rate theorem, Methods; Fig. 8a).
2. **An a priori dynamical certificate.** Every level carries the bound
   $\lVert \widehat G_{q,w}^{\,r} - U(t)\rVert_2 \le r\lVert K_q - \Pi_w K_q\rVert_2$,
   computable from $K_q$ without simulating the dynamics, so the truncation bandwidth is
   selected directly from the target accuracy — a map the kernel construction lacks
   (certificate corollary, Methods). The required bandwidth grows only logarithmically in $1/\varepsilon$.
3. **A gate-level realization whose structural defect is repaired by symmetry.** The naive
   first-order inner compilation of the truncated correction has an $O(\delta t^{2(q+1)})$
   error floor that fails at low base order; a **symmetric inner compilation** lowers it to
   $O(\delta t^{3(q+1)})$ and recovers the truncated-oracle accuracy at every order
   (symmetric-compilation theorem, Methods) — the simulation analogue of the positivity-restoring smoothing that the
   kernel construction needs.

The downstream payoff, which the kernel setting has no analogue of, is a measurable
advance of the product-formula **error–cost frontier**: a spectrally-truncated fourth-order
correction reaches accuracies between standard sixth- and eighth-order Suzuki formulas at
$4.2$–$5.2\times$ fewer two-qubit gates than the cheapest standard formula of matching
accuracy ($n = 5$–$7$). The truncated generator is, moreover, **learnable without the dense
propagator**: a single translation-equivariant network maps local couplings to the truncated
coefficients and, trained only on four- and five-qubit chains, transfers to ten qubits
($R^2 = 0.9999$).

## Demonstration

Install the package and reproduce the headline spectral-truncation results of the
manuscript directly from the public API.

```bash
pip install lieideal-hs            # or, from source:  cd submission/code && pip install -e .
```

```python
import spectral_truncation as st

# (1) Geometric convergence rate of the Lie-algebraic spectral truncation (spectral-truncation rate table; Fig. 8a).
rate, factors = st.truncation_rate()       # open-boundary TFIM, n = 6, t = 1, r = 10
print(factors)                             # measured per-level reduction factors:
# {'q2_factor_per_level': 449.3..., 'q4_factor_per_level': 576.4..., ...}

# (2) Symmetric inner compilation repairs the first-order floor (faithful-compilation table; Fig. 8b).
comp = st.faithful_compilation()
print(comp[["n", "q", "w", "oracle_error", "first_order_error", "symmetric_error"]])
# at n = 6, q = 2, w = 4:  oracle 3.65e-7,  first-order 2.18e-5 (fails),  symmetric 3.76e-7

# (3) Certificate-driven bandwidth selection (certificate corollary, Methods; Fig. 8c).
cert = st.certificate_selection()
print(cert[["target_eps", "w_star", "certificate_bound", "achieved_error", "bound_holds"]])
```

Every value printed is an exact, deterministic dense-matrix computation and matches the
corresponding number in [`submission/main.pdf`](submission/main.pdf).

## Full reproduction

```bash
cd submission/code
lieideal-reproduce      # deterministically regenerates every figure, table and dataset of main.tex
lieideal-verify         # reruns the pipeline and asserts byte-identical outputs
```

`lieideal-reproduce` (equivalently `python make_all.py`) seeds all generators, then runs the
deterministic dense-matrix experiments (fixed-time oracle benchmark, projected/compressed
residual, time sweep, generator structure, order generality, **spectral-truncation rate and
faithful compilation**, the oracle-free fourth-order frontier, the resource proxy), followed
by the learned residual operator-learning experiment and the headline figure. The run is
deterministic (seed 42 for `random`/NumPy/PyTorch, `SEED = 20240517` for the disorder
sampler and network initialization, single-threaded BLAS, fixed PDF timestamp); the learned
experiment reproduces byte-for-byte across environments. CPU-only. Every figure and table in
the manuscript is produced by this package and included by `\input`; no number is entered by
hand. See [`submission/code/README.md`](submission/code/README.md) for the module-level guide.

## Repository layout

```text
submission/
  main.tex, main.pdf          manuscript source and compiled PDF
  figures/                    publication figures (fig0–fig8, PDF)
  tables/                     LaTeX result tables, all produced by the package
  code/                       lieideal-hs package (pyproject.toml, LICENSE, README.md)
    common.py                 Pauli operators, TFIM, Trotter–Suzuki steps, residual generator
    spectral_truncation.py    spectral-truncation rate, certificate, symmetric compilation (Fig. 8)
    higher_order_frontier.py  order generality, error–cost frontier, XXZ generality
    projected_residual.py     Pauli-weight projection / compression
    oracle_free_q4.py         oracle-free, size-transferable fourth-order frontier correction
    learned_residual.py       translation-equivariant residual learner (size transfer, torch)
    fixed_time.py, time_sweep.py, generator_scaling.py, headline.py, overview.py
    make_all.py, cli.py       one-command reproduction and console entry points
    validate_submission.py    cross-checks citations, figures, tables and datasets
    generated_data/           raw CSV/JSON with per-run metadata
index.html                    project page foregrounding the spectral-truncation result
```

## Citation

```bibtex
@misc{huynh2025spectraltruncation,
  author = {Huynh, Molena},
  title  = {Lie-algebraic Spectral Truncation of the Product-formula Residual:
            Certified, Geometrically Convergent Corrections for Hamiltonian Simulation},
  year   = {2025},
  note   = {North Carolina State University},
  howpublished = {\url{https://github.com/thmolena/Lie-Algebra-for-Hamiltonian-Simulation}}
}
```

## License

MIT. See [`submission/code/LICENSE`](submission/code/LICENSE).
