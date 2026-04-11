# Lie GPT for Hamiltonian Simulation: Generator Prediction in Dynamical Lie Algebras Beyond Product Formulas

**Project page:** [index.html](index.html)

---

## Abstract

We introduce **Lie GPT**, a framework that reframes Hamiltonian simulation as the problem of *predicting and selecting generators within the dynamical Lie algebra* of the Hamiltonian, establishing *generator expansion* as a new axis of algorithm design beyond circuit depth and variational optimization.

The central result is a **10× improvement theorem**: for any order $k$, the Lie GPT-$k$ propagator achieves spectral error at least one order of magnitude lower than the $k$-th order Trotter–Suzuki formula for the same step size $\delta t \leq \delta t^{(k)}$, a threshold determined by the Baker–Campbell–Hausdorff prefactors of the Hamiltonian. For the four-qubit transverse-field Ising model, Lie GPT-1 satisfies this bound for $\delta t \leq 0.06$ (single-step) and for $N \geq 10$ Trotter steps at $T = 0.5$; Lie GPT-2 achieves the same improvement over Suzuki-2 for $N \geq 24$ steps.

These results demonstrate that expanding the generator set within the dynamical Lie algebra is a viable and theoretically principled pathway to improved Hamiltonian simulation without increasing circuit depth or requiring variational optimization.

---

## The Problem: Fixed-Generator Paradigm

Every standard Hamiltonian simulation method — Trotter decompositions, Suzuki formulas, QAOA, variational ansätze — fixes its generator set before building the circuit. The Baker–Campbell–Hausdorff (BCH) expansion reveals the structural cost of this choice. Multiplying $e^{-i\delta t B}e^{-i\delta t A}$ gives:

$$e^{-i\delta t B}e^{-i\delta t A} = \exp\!\Bigl[-i\delta t(A+B) + \tfrac{\delta t^2}{2}[A,B] + O(\delta t^3)\Bigr],$$

so the commutator $[A,B]$ — a member of the dynamical Lie algebra — enters unavoidably as the leading error term. Lie GPT removes this restriction by **including DLA generators explicitly in the propagator**.

---

## The Lie GPT Framework

Given $H = A + B$, the dynamical Lie algebra (DLA) is:

$$\mathcal{L}(A,B) = \mathrm{span}\{A,\,B,\,[A,B],\,[A,[A,B]],\,[B,[A,B]],\,\ldots\}.$$

Lie GPT-$k$ selects the depth-$k$ DLA truncation as its circuit generator set:

$$U_k(\boldsymbol{\theta},\delta t) = \prod_{j=1}^{d_k} e^{-i\theta_j G_j(\delta t)}, \qquad G_j \in \mathcal{L}^{(k)}(A,B).$$

| Level | Generator set | Local error |
|-------|--------------|-------------|
| **GPT-0** (= Trotter-1) | $\{A, B\}$ | $O(\delta t^2)$ |
| **GPT-1** | $\{A, B, [A,B]\}$ | $O(\delta t^3)$ |
| **GPT-2** | $\{A, B, [A,B], [A,[A,B]], [B,[A,B]]\}$ | $O(\delta t^4)$ |

**GPT-1 propagator:**
$$V_1(\delta t) = e^{-\delta t^2[A,B]/2}\,e^{-i\delta t B}\,e^{-i\delta t A}.$$

**GPT-2 propagator (corrected coefficients):**
$$V_2(\delta t) = e^{-i\alpha_3\delta t^3 C_B}\,e^{-i\alpha_2\delta t^3 C_A}\,e^{-\delta t^2[A,B]/2}\,e^{-i\delta t B}\,e^{-i\delta t A},$$
where $C_A=[A,[A,B]]$, $C_B=[B,[A,B]]$, $\alpha_2=-\tfrac{1}{6}$, $\alpha_3=-\tfrac{1}{3}$.

> **Sign correction.** Naïve BCH power-counting gives $\alpha_2=\alpha_3=+\tfrac{1}{12}$, which *reinforces* rather than cancels the third-order error. This sign error is identified and corrected in the paper; with correct coefficients, GPT-2 outperforms Suzuki-2 by $\geq10\times$ for $N\geq24$.

---

## The GPT Analogy

The name reflects a structural parallel with language model design:

| Component | GPT (language model) | Lie GPT (quantum simulation) |
|-----------|---------------------|------------------------------|
| Dynamical Lie Algebra (DLA) | Token dictionary | $\mathcal{L}(A,B)$ |
| Token | Word or sub-word | Generator $G_k$ |
| Baker–Campbell–Hausdorff (BCH) | Learned via training | BCH hierarchy (or learned from data) |
| Context | Input sequence | Commutator depth $k$ |
| Prediction | Next token | Next generator $G_{k+1}$ |
| Loss | Cross-entropy | Spectral error |
| Scaling | Model size | DLA truncation depth |

When the Baker–Campbell–Hausdorff (BCH) approach is analytically fixed, the result is the 10× improvement theorem. When the BCH approach is learned from quantum trajectory data via a GRU constrained to the DLA, the result is the stable, data-efficient learned framework of the three notebooks.

---

## The 10× Improvement Theorem

**Theorem.** Fix order $k \in \{1,2,\ldots\}$. There exists a threshold $\delta t^{(k)} > 0$ depending on the BCH prefactors of $H$ such that for all $\delta t \leq \delta t^{(k)}$:

$$\|e^{-i\delta t H} - U_k(\delta t)\|_2 \;\leq\; \tfrac{1}{10}\,\|e^{-i\delta t H} - U^{(k)}_{\mathrm{TS}}(\delta t)\|_2,$$

where $U_k$ is the Lie GPT-$k$ propagator and $U^{(k)}_{\mathrm{TS}}$ is the $k$-th order Trotter–Suzuki formula.

For $k=1$: $\delta t^{(1)} = \|[A,B]\|_2 / \bigl(10(\tfrac{1}{3}\|[A,[A,B]]\|_2 + \tfrac{2}{3}\|[B,[A,B]]\|_2)\bigr)$.

The $10\times$ improvement in one-step error propagates to $N$-step global error via the telescoping bound; the global ratio is $\approx(C_\mathrm{Trot}/C_\mathrm{GPT1})\cdot(N/T)$, growing with $N$ at fixed total time $T$.

**TFIM verification** ($n=4$, $J=1$, $h=0.5$, $T=0.5$):

| $N$ | Trotter-1 $\epsilon$ | Lie GPT-1 $\epsilon$ | Ratio | Suzuki-2 $\epsilon$ | Lie GPT-2 $\epsilon$ | Ratio |
|-----|---------------------|---------------------|-------|---------------------|---------------------|-------|
| 10  | $4.76\times10^{-2}$ | $3.89\times10^{-3}$ | **12.3×** ✓ | $9.72\times10^{-4}$ | — | — |
| 16  | $2.98\times10^{-2}$ | $1.52\times10^{-3}$ | **19.6×** | — | — | — |
| 24  | $1.98\times10^{-2}$ | $6.75\times10^{-4}$ | **29.4×** | $1.69\times10^{-4}$ | $1.58\times10^{-5}$ | **10.7×** ✓ |
| 32  | — | — | **∼39.2×** | $9.49\times10^{-5}$ | $6.66\times10^{-6}$ | **14.2×** |

✓ = 10× threshold first crossed.  Overhead: +1 circuit layer, $+2(n-1)$ two-qubit gates per step for GPT-1; all generators have Pauli weight ≤ 3.

---

## Three Jupyter Notebooks: Extending Lie GPT to Learned Baker–Campbell–Hausdorff (BCH) Approaches

The paper establishes the analytical half: BCH-optimal generator selection yields $10\times$ improvement with analytically fixed coefficients. The notebooks explore the natural generalization: what if the Baker–Campbell–Hausdorff (BCH) selection is *learned from quantum trajectory data*? A GRU is equipped with a zero-parameter **Lie Constraint Layer** that forces outputs to $\mathfrak{su}(2)$, making unitarity structurally impossible to violate.

The three notebooks form a single arc — *foundations → long-time stability → data efficiency and noise robustness* — each available as a pre-rendered HTML page and a runnable `.ipynb` source.

---

### Notebook 1 — Architecture & Unitarity Guarantee

**File:** [notebooks/liegpt_architecture_unitarity.ipynb](notebooks/liegpt_architecture_unitarity.ipynb)  
**HTML:** [notebooks/liegpt_architecture_unitarity.html](notebooks/liegpt_architecture_unitarity.html)  
**Paper sections:** §3 Framework · §4 Theory (Proposition 1)

Establishes the $\mathfrak{su}(2)$ algebra basis ($\sigma_x, \sigma_y, \sigma_z$), visualizes the Levi-Civita structure constants $[X_i,X_j]=\sum_k c_{ij}^k X_k$, and implements the Lie Constraint Layer. The GRU outputs three real scalars $\theta_i$; the layer assembles $H=\sum_i\theta_i\sigma_i$, which is Hermitian by construction, so $U = e^{-iH\Delta t}$ is always unitary (Proposition 1 proof by demonstration). Quantified over $10^5$ random Hamiltonians: Lie GPT unitarity violation $\|U^\dagger U - I\|_F \approx 10^{-16}$; soft-penalty GRU reaches $\sim10^{-2}$; unconstrained GRU reaches $\sim10^{-1}$. Bloch sphere trajectories confirm the state norm stays exactly on $S^2$.

| Figure | File | What it shows |
|--------|------|---------------|
| Fig 1 — $\mathfrak{su}(2)$ Basis | `outputs/su2_basis.png` | Real/imaginary parts of $\sigma_x,\sigma_y,\sigma_z$ — the fixed DLA basis |
| Fig 2 — Structure Constants | `outputs/structure_constants.png` | Levi-Civita heatmap confirming DLA closure |
| Fig 3 — Unitarity Benchmark | `outputs/unitarity_benchmark.png` | Lie GPT ≈ machine ε; unconstrained GRU ≈ $10^{-1}$ |
| Fig 4 — Bloch Sphere | `outputs/bloch_sphere_unitarity.png` | LieGPT stays on $S^2$; unconstrained model spirals off |

---

### Notebook 2 — Long-Time Stability (Primary Result)

**File:** [notebooks/liegpt_stability.ipynb](notebooks/liegpt_stability.ipynb)  
**HTML:** [notebooks/liegpt_stability.html](notebooks/liegpt_stability.html)  
**Paper sections:** §5 Experiments (primary) · §4 Theory (Theorem 1)

All four models (Lie GPT, unconstrained GRU, GRU+unitarity penalty, MLP) are trained on sequences of length $T=25$ and evaluated by rolling out to $T=200$ — an $8\times$ extrapolation beyond the training horizon without retraining. Lie GPT error stays bounded; the log-log slope confirms the $O(T)$ linear bound of Theorem 1. Every baseline diverges — including the GRU with a soft unitarity penalty, establishing that the hard architectural constraint is necessary, not merely convenient.

| Figure | File | What it shows |
|--------|------|---------------|
| Fig 5 — Training Curves | `outputs/training_curves.png` | Log-scale MSE for all four models over 80 epochs |
| **Fig 6 — Stability Rollout ★** | `outputs/stability_rollout.png` | **Primary result:** (A) state error mean±std, (B) unitarity violation log-scale, (C) checkpoint bar chart, (D) ratio vs Lie GPT |
| Fig 7 — Theorem 1 Bound | `outputs/theorem1_bound.png` | Empirical error vs. $T\cdot C\cdot\varepsilon$; slope ≈ 1 in log-log |

★ = primary result figure.

---

### Notebook 3 — Data Efficiency & Noise Robustness

**File:** [notebooks/liegpt_efficiency_robustness.ipynb](notebooks/liegpt_efficiency_robustness.ipynb)  
**HTML:** [notebooks/liegpt_efficiency_robustness.html](notebooks/liegpt_efficiency_robustness.html)  
**Paper sections:** §5 Experiments (supporting) · §4 Theory (Theorem 2)

Sweeps training set size $N\in\{50,100,250,500,1000,2000,5000\}$ and input noise $\sigma\in\{0,0.05,0.1,0.2,0.5\}$. The Lie Constraint Layer encodes physical structure for free: the GRU does not need to learn unitarity from data, freeing capacity for dynamics. Result: Lie GPT reaches a given accuracy threshold with $\sim3\times$ fewer trajectories. Theorem 2 (Rademacher complexity) provides the theoretical explanation: the DLA constraint reduces the effective hypothesis class by $\sqrt{k/n^2}$. Noise robustness: the hard constraint keeps unitarity violation at $\varepsilon_\text{machine}$ regardless of input noise level.

| Figure | File | What it shows |
|--------|------|---------------|
| Fig 8 — Data Efficiency | `outputs/data_efficiency.png` | Test MSE vs. $N$; Lie GPT ≈ $3\times$ more data-efficient |
| Fig 9 — Rademacher Complexity | `outputs/theorem2_complexity.png` | $\sqrt{k/n^2}$ reduction vs. unconstrained class |
| Fig 10 — Noise Robustness | `outputs/noise_robustness.png` | Lie GPT unitarity flat at $\varepsilon_\text{machine}$ at all noise levels |
| Fig 11 — Combined Summary | `outputs/combined_summary.png` | 4-panel publication summary |

---

## Repository Structure

```
research_paper/          # LaTeX source (main.tex), bibliography, figures
notebooks/               # Three Jupyter notebooks (.ipynb + pre-rendered .html)
  liegpt_architecture_unitarity.*    # Notebook 1: architecture & unitarity
  liegpt_stability.*                 # Notebook 2: long-time stability (PRIMARY)
  liegpt_efficiency_robustness.*     # Notebook 3: data efficiency & noise
src/
  lc_qaoa/               # Core analytical library
    models.py            # TwoBlockHamiltonian and Pauli operators
    propagators.py       # Trotter, Suzuki, Lie GPT-1/2 propagators
    metrics.py           # Spectral-norm and fidelity evaluation
    experiments.py       # Benchmark sweep infrastructure
    fitting.py, driven.py
scripts/                 # Experiment runners producing paper figures
  tfim_experiment.py     # Core TFIM spectral-error table
  tfim_sweep.py          # Full N-sweep at T=0.5
  generate_liegpt_figures.py   # Local error, global error, heatmap
  driven_tfim_sweep.py         # Driven (time-dependent) TFIM sweep
  plot_*.py              # Figure rendering scripts
results/                 # CSV data files from benchmark runs
outputs/                 # Generated figures (populated by running notebooks)
index.html               # Project web page
requirements.txt
```

---

## Reproducing the Results

All experiments run on CPU only. No GPU required.

### Paper benchmarks (TFIM spectral-error tables and figures)

```bash
conda activate qaoa          # or: pip install -r requirements.txt in a fresh env
export PYTHONPATH=src        # Windows: set PYTHONPATH=src

python scripts/tfim_experiment.py            # Table 2: spectral errors at N=10
python scripts/tfim_sweep.py                 # Table 3: full N-sweep
python scripts/generate_liegpt_figures.py    # Figs 1–3: local error, global error, heatmap
```

### Notebooks (unitarity, stability, data efficiency)

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/liegpt_architecture_unitarity.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  notebooks/liegpt_stability.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  notebooks/liegpt_efficiency_robustness.ipynb

# Export to HTML for offline viewing
jupyter nbconvert --to html notebooks/liegpt_architecture_unitarity.ipynb
jupyter nbconvert --to html notebooks/liegpt_stability.ipynb
jupyter nbconvert --to html notebooks/liegpt_efficiency_robustness.ipynb

# Serve the project page
python -m http.server 8000   # open http://localhost:8000
```

---

## Key Contributions

1. **Framework.** Lie GPT as a generator-prediction paradigm with a formal model hierarchy indexed by DLA truncation depth (GPT-0 = Trotter-1, GPT-1, GPT-2, …, GPT-∞ = Cartan synthesis).

2. **10× improvement theorem.** Proof that Lie GPT-$k$ achieves spectral error one order of magnitude lower than $k$-th order Trotter–Suzuki, with explicit BCH-derived thresholds $\delta t^{(k)}$.

3. **Corrected GPT-2 coefficients.** Derivation of the correct BCH coefficients $\alpha_2=-\tfrac{1}{6}$, $\alpha_3=-\tfrac{1}{3}$, identifying and fixing a sign error in naïve power-counting that caused GPT-2 to perform *worse* than Suzuki-2.

4. **TFIM and XXZ benchmarks.** Numerical validation on four-qubit TFIM and XXZ chains; consistent ordering Lie GPT-$k$ < Trotter/Suzuki-$k$ at every tested parameter point.

5. **Learned Baker–Campbell–Hausdorff (BCH) extension (notebooks).** GRU with zero-parameter Lie Constraint Layer achieves: exact unitarity ($\varepsilon_\text{machine}$ by architecture), $8\times$-extrapolation stability (Theorem 1), and $3\times$ data efficiency (Theorem 2).

---

## Connection to Prior Work

| Prior work | Core limitation | Lie GPT answer |
|---|---|---|
| First-order Trotter (GPT-0) | $\{A,B\}$ only; $[A,B]$ is pure error | GPT-1 includes $[A,B]$; 10× for $N\geq10$ |
| Suzuki-2 | Same error order as GPT-1; no DLA expansion | GPT-2 achieves $O(\delta t^4)$; 10× for $N\geq24$ |
| Cartan synthesis | Exact but not learned; full DLA decomposition required | Lie GPT interpolates: GPT-$k\to\infty$ recovers Cartan |
| Variational ansätze (QAOA) | Fixed generator family; no operator space expansion | BCH coefficients analytically fixed; no optimization |
| Neural quantum states | State space, no algebraic guarantees | Lie Constraint Layer enforces DLA membership |
| Physics-informed neural ODEs | Soft penalty; unitarity still violated during rollout | Hard architectural constraint; violation = $\varepsilon_\text{machine}$ always |

---

*Lie GPT establishes the dynamical Lie algebra as the natural design space for Hamiltonian simulation — unifying product formulas, Cartan decompositions, and variational circuits under a single generator-selection principle, and extending it naturally to learned quantum dynamics.*
