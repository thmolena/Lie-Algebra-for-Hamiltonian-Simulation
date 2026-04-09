# Research Plan: Lie-Structured Generative Models for Hamiltonian Simulation

> **Working title:** LieGPT — A Lie-Algebra–Constrained Generative Framework for Quantum Dynamics

---

## 1. Research Objective

Develop a Lie-algebra–constrained generative model for Hamiltonian simulation that **guarantees** physically valid quantum evolution — unitarity, symmetry preservation, closure under commutators — while improving long-time stability and sample efficiency over existing machine-learning and classical simulation baselines.

This project represents a deliberate pivot from the earlier Cartan-constrained QAOA direction. While Cartan-commutator-augmented QAOA introduced structured conjugator families with provable single-step error improvements, it remained tethered to the Cartan compilation paradigm. LieGPT changes the level of abstraction entirely: rather than optimizing a circuit, the model **learns the generator dynamics directly in Lie algebra space**, treating $H(t)$ as a sequence of coordinates on a structured manifold and generating physically consistent trajectories without post-hoc projection.

---

## 2. Core Thesis

> *This work introduces a structure-preserving generative modeling framework that operates directly in Lie algebra space, enabling physically consistent and computationally efficient simulation of quantum dynamics. By constructing constraints into the model architecture rather than adding them as soft penalties, LieGPT produces unitary evolution generators by design, demonstrating provable long-time stability advantages over both classical baselines and unconstrained neural network methods.*

The contribution is **not** "GPT applied to physics." It is the integration of Lie algebraic structure — closure, commutator relations, skew-Hermitian generators, exponential map — directly into the learning architecture, yielding a principled and verifiable guarantee of physical validity at every inference step.

---

## 3. Must-Have Novel Contributions

The following five contributions are non-negotiable. Each must be delivered in full. If any are weakened to approximations or dropped, the work loses its distinguishing character from existing machine-learning-for-physics literature.

### 3.1 — Operate in Lie Algebra Space, Not State Space

The model learns the Hamiltonian as a trajectory of algebra coordinates:

$$
H(t) = \sum_{i} \theta_i(t) \, X_i, \qquad X_i \in \mathfrak{g},
$$

where $\{X_i\}$ is a fixed basis of generators for the Lie algebra $\mathfrak{g}$ (e.g., Pauli matrices for $\mathfrak{su}(2)$). The learned quantities are the coefficient functions $\theta_i(t)$, not the wavefunction $|\psi(t)\rangle$ or the full matrix $U(t)$.

**Why this matters.** Existing neural-network quantum simulation methods — neural quantum states (NQS), Hamiltonian learning via energy minimization, physics-informed neural ODEs — operate in state space or wavefunction space. Operating in algebra space is a fundamentally different and more structured representation. It reduces the effective search space by the dimension of the Lie group, encodes symmetry natively, and makes physical constraints checkable in closed form.

### 3.2 — Enforce Structure by Design, Not by Penalty

The architecture must enforce:

- **Skew-Hermitian generators:** $H(t) = -H(t)^\dagger$,
- **Closure under the Lie bracket:** any commutator of learned generators remains in $\mathfrak{g}$,
- **Valid Lie algebra representation:** coordinates live in the correct subspace at all times.

These are hard constraints implemented architecturally — not soft regularization terms added to a loss function. A **Lie Constraint Layer** at the model output enforces skew-Hermitian structure and projects onto the valid generator subspace:

$$
\hat{H}(t) = \sum_i \theta_i(t) X_i, \quad \theta_i(t) \in \mathbb{R}, \quad X_i^\dagger = -X_i.
$$

The constraint is satisfied because the basis $\{X_i\}$ is fixed and skew-Hermitian by construction. The model only outputs real coefficients $\theta_i(t)$.

### 3.3 — Guarantee Physical Validity (Unitarity by Design)

The evolution operator is computed from the learned generator via the exponential map:

$$
U(t) = e^{-i H(t) \Delta t}.
$$

Because $H(t) \in \mathfrak{su}(n)$ is skew-Hermitian by the Lie Constraint Layer, the matrix exponential always produces a unitary. There is no post-hoc projection, no unitarity penalty, and no truncation error from a unitarization step. The model is structurally incapable of producing non-unitary evolution.

**This is a major differentiator.** Essentially all existing neural network methods for quantum dynamics either allow unitarity violations during training and project afterward, or add a penalty term that reduces but does not eliminate the violation. LieGPT eliminates the violation by construction.

### 3.4 — Demonstrate Long-Time Stability

The model must perform long-rollout simulations — at minimum $10\times$ the training sequence length — and show:

- Unconstrained MLP and RNN baselines accumulate unbounded unitarity error and state-prediction error.
- LieGPT maintains stable rollout error for the entire extended trajectory.

This is the single most publishable experimental result in the paper. It requires no specialized quantum hardware background to evaluate and communicates the core advantage immediately.

### 3.5 — Show a Measurable Advantage from Lie Structure

At least one of the following must hold with statistical significance:

| Claim | Evidence required |
|---|---|
| Better data efficiency | LieGPT trained on $N$ examples matches the accuracy of an unconstrained model trained on $\alpha N$, with $\alpha > 1$ |
| Better long-time stability | Rollout error at step $T_{\max}$ is smaller by a statistically significant margin |
| Lower simulation error at fixed computation | Propagator error $\|U_\text{pred} - U_\text{true}\|$ is smaller at matched compute cost |

---

## 4. Problem Setup

### 4.1 Lie Algebras

| Algebra | Dimension | System |
|---|---|---|
| $\mathfrak{su}(2)$ | 3 generators | Single qubit (primary target) |
| $\mathfrak{su}(4)$ | 15 generators | Two-qubit (optional extension) |

Start with $\mathfrak{su}(2)$. The Pauli matrices $\{iX, iY, iZ\}$ form a basis of skew-Hermitian generators and the structure constants are known exactly. This keeps all matrix operations at $2 \times 2$, making CPU-only experiments fast and reproducible.

### 4.2 Hamiltonians

Target time-dependent Hamiltonians of the form:

$$
H(t) = a(t) \, \sigma_x + b(t) \, \sigma_y + c(t) \, \sigma_z,
$$

where $a(t), b(t), c(t)$ are smooth scalar functions (sinusoids, polynomials, or random smooth draws). The model task reduces to learning the three scalar trajectories $(\theta_1(t), \theta_2(t), \theta_3(t))$.

### 4.3 Systems

- **Primary:** single qubit, $d = 2$, all matrix exponentials are $2 \times 2$.
- **Secondary (optional):** two qubits, $d = 4$.

---

## 5. Model Design

### 5.1 Representation

The model inputs a sequence of past Hamiltonian coefficients and outputs future coefficients:

$$
(\theta_1(t-T), \ldots, \theta_k(t-T)), \; \ldots, \; (\theta_1(t), \ldots, \theta_k(t)) \;\longrightarrow\; (\theta_1(t+\Delta t), \ldots, \theta_k(t+\Delta t)).
$$

For $\mathfrak{su}(2)$, $k = 3$. The generator at each step is formed as $H(t) = \sum_{i=1}^k \theta_i(t) X_i$, with $\{X_i\}$ precomputed and fixed.

### 5.2 Architecture (CPU-Friendly)

| Component | Specification |
|---|---|
| Model type | Small Transformer or GRU |
| Layers | 2–4 |
| Hidden size | 64–128 |
| Sequence length | $\leq 50$ |
| Output activation | Linear (real coefficients only) |

### 5.3 Lie Constraint Layer (Required)

The Lie Constraint Layer is the architectural core of LieGPT:

1. Model outputs $k$ raw scalars $\hat{\theta}_1(t), \ldots, \hat{\theta}_k(t)$.
2. Layer assembles: $\hat{H}(t) = \sum_i \hat{\theta}_i(t) X_i$.
3. By construction, $\hat{H}(t)$ is skew-Hermitian and lies in $\mathfrak{g}$.
4. No projection, no clamping, no penalty needed.

This layer has **zero learnable parameters**. Its role is purely structural: to guarantee the physical validity of every output before any downstream computation.

### 5.4 Optional: Commutator-Aware Features (Highest Novelty)

Augment the attention mechanism with structure-constant information. For $\mathfrak{su}(2)$:

$$
[X_i, X_j] = \sum_k c_{ij}^k X_k,
$$

where $c_{ij}^k$ are the Lie algebra structure constants (Levi-Civita symbols for $\mathfrak{su}(2)$). These can be used as a fixed inductive bias in attention weights, yielding **commutator attention** — attention that respects the algebraic mixing prescribed by the Lie bracket rather than pure dot-product similarity. This is a standalone architectural innovation.

---

## 6. Evolution Pipeline

1. **Predict generator:** $\hat{H}(t) = \sum_i \hat{\theta}_i(t) X_i \in \mathfrak{g}$.
2. **Compute unitary:** $U(t) = e^{-i \hat{H}(t) \Delta t}$ via `scipy.linalg.expm`.
3. **Update state:** $|\psi(t + \Delta t)\rangle = U(t) |\psi(t)\rangle$.
4. **Rollout:** repeat for all $T$ steps.

For $2 \times 2$ matrices, the matrix exponential admits the closed-form Bloch-vector rotation:

$$
e^{-i(\theta_1 X + \theta_2 Y + \theta_3 Z)\Delta t} = \cos(\|\theta\|\Delta t) \, I - i \sin(\|\theta\|\Delta t) \, \hat{\theta} \cdot \vec{\sigma},
$$

which replaces `scipy.linalg.expm` for single-qubit experiments, eliminating all numerical matrix exponentiation error.

---

## 7. Dataset Generation

Generate fully synthetic, noise-free data for exact supervised training and evaluation.

### 7.1 Hamiltonian Trajectories

For each trajectory, draw random smooth coefficient functions $a(t), b(t), c(t)$ as sums of sinusoids with random frequencies and amplitudes, or as polynomial splines. Evaluate at $T$ discrete timesteps of size $\Delta t$.

### 7.2 Outputs

For each trajectory, compute:

| Quantity | Description | Method |
|---|---|---|
| $H(t_k)$ | Generator matrix at each step | From coefficients |
| $U(t_k)$ | Unitary at each step | `scipy.linalg.expm` |
| $|\psi(t_k)\rangle$ | Evolved state | $\psi_{k+1} = U_k \psi_k$ |

### 7.3 Dataset Scale

| Scale | Purpose |
|---|---|
| 100 trajectories | Low-data regime for data-efficiency benchmarks |
| 1,000 trajectories | Development and ablations |
| 10,000 trajectories | Full training |

---

## 8. Baselines

### 8.1 Classical Baselines

| Method | Description |
|---|---|
| Exact matrix exponential | $U(t) = e^{-iH(t)\Delta t}$ at each step; oracle bound |
| First-order Trotter | $e^{-i(A+B)\Delta t} \approx e^{-iA\Delta t} e^{-iB\Delta t}$ |
| Second-order Suzuki | Symmetric Trotter with $O(\Delta t^3)$ local error |

### 8.2 Machine Learning Baselines

| Method | Unconstrained |
|---|---|
| MLP (no memory, no physics) | Yes |
| GRU (no constraints) | Yes |
| GRU + soft unitarity penalty $\|U^\dagger U - I\|$ | Soft only |
| **LieGPT (ours)** | Hard constraint by design |

The soft-penalty GRU is a critical baseline because it shows that adding a loss term is not sufficient to achieve the stability and unitarity guarantees that LieGPT provides by construction.

---

## 9. Evaluation Metrics

### 9.1 State Prediction Error

$$
\varepsilon_\psi(t) = \|\psi_\text{true}(t) - \psi_\text{pred}(t)\|_2.
$$

### 9.2 Unitarity Violation (Critical)

$$
\varepsilon_U(t) = \|U(t)^\dagger U(t) - I\|_F.
$$

For LieGPT this is zero (to floating-point precision) at every step by the Lie Constraint Layer. For unconstrained baselines, this grows during rollout.

### 9.3 Long-Time Rollout Stability (Most Important)

Plot $\varepsilon_\psi(t)$ and $\varepsilon_U(t)$ as a function of rollout step from $t = 0$ to $t = T_\text{max} \gg T_\text{train}$. The key result: unconstrained baselines diverge; LieGPT remains stable.

### 9.4 Data Efficiency

Train all models on $N \in \{100, 500, 1000, 5000, 10000\}$ and plot test error vs. $N$. LieGPT should reach a given error threshold with fewer samples due to the reduced effective hypothesis space imposed by the Lie constraints.

### 9.5 Propagator Fidelity

$$
\mathcal{F}(t) = \frac{1}{d^2} \left|\operatorname{Tr}(U_\text{true}(t)^\dagger \, U_\text{pred}(t))\right|^2.
$$

This is directly analogous to the operator-matching loss used in the prior Cartan-QAOA work, providing a natural bridge between the two research directions.

---

## 10. Experiments

### Experiment 1 — Short-Time Accuracy

Evaluate all baselines on in-distribution sequences. Confirms that constraints do not hurt accuracy.

### Experiment 2 — Long-Time Stability (Key Result)

Extend all trained models to $T_\text{max} = 5T_\text{train}$ without retraining. Plot rollout error curves showing baselines diverging and LieGPT remaining stable. This is the primary publication figure.

### Experiment 3 — Data Efficiency

Train across $N \in \{100, 500, 1000, 5000\}$. Plot test error vs. $\log N$, demonstrating that the Lie constraint reduces the required training set size.

### Experiment 4 — Noise Robustness

Inject Gaussian noise into training Hamiltonian coefficients. Under noise, the constrained model degrades gracefully; unconstrained models violate unitarity more severely.

### Experiment 5 (Optional) — Commutator Attention Ablation

Compare LieGPT with standard dot-product attention vs. commutator attention (structure-constant-based mixing). Isolates the contribution of commutator-aware architecture.

---

## 11. Theoretical Component

### Proposition 1 (Required): Lie Structure Guarantees Unitarity

**Statement.** Let $\mathfrak{g} = \mathfrak{su}(n)$ and $H(t) \in \mathfrak{g}$ for all $t$. Define $U(t) = e^{-iH(t)\Delta t}$. Then $U(t) \in \mathrm{SU}(n)$, i.e., $U(t)^\dagger U(t) = I$ and $\det(U(t)) = 1$.

*Proof sketch.* The exponential map $\exp: \mathfrak{su}(n) \to \mathrm{SU}(n)$ is the standard Lie group–algebra correspondence. $H(t)$ skew-Hermitian implies $-iH(t)$ is Hermitian, so $e^{-iH(t)\Delta t} \in \mathrm{U}(n)$. The trace condition $\operatorname{Tr}(H(t)) = 0$ gives $\det(e^{-iH(t)\Delta t}) = 1$, hence $U(t) \in \mathrm{SU}(n)$.

### Theorem 1 (Target): Long-Time Boundedness Under Exact Lie Constraint

**Statement.** Let the model predict coefficients $\hat\theta(t)$ with bounded error $|\hat\theta_i(t) - \theta_i^*(t)| \leq \epsilon$ for all $t, i$. Then:

$$
\|\psi_\text{pred}(T) - \psi_\text{true}(T)\|_2 \leq T \cdot C \cdot \epsilon,
$$

for a constant $C$ depending on $\Delta t$ and the algebra basis norms, and moreover $\varepsilon_U(t) = 0$ for all $t$.

*Argument.* Each $U_\text{pred}(t) = e^{-i\hat H(t)\Delta t}$ is exactly unitary by Proposition 1. The state error telescopes: each step contributes at most $\|U_t^\text{pred} - U_t^\text{true}\| \lesssim k\epsilon\Delta t\|X_\text{max}\|$. Unconstrained models accumulate an additional unitarity-violation term at each step that is unbounded.

### Theorem 2 (Target): Reduced Hypothesis Space from Lie Constraints

**Statement.** The Rademacher complexity of LieGPT's output class scales as $O(\sqrt{k/n^2})$ relative to the unconstrained matrix prediction class, where $k = \dim(\mathfrak{g})$ and $n^2$ is the number of real degrees of freedom in an unconstrained $n \times n$ complex matrix.

*Argument.* Standard learning-theoretic analysis of restricted output classes. For $\mathfrak{su}(2)$ with $k=3$ and $n=2$ (so $n^2 = 4$), this gives a factor-of-$\sqrt{3/4}$ reduction in effective search space — modest but rigorously derived, and exact for small systems.

---

## 12. Implementation Plan

### 12.1 Tools

| Component | Tool |
|---|---|
| Array operations | NumPy |
| Matrix exponential | `scipy.linalg.expm` (or analytic Bloch formula for $2 \times 2$) |
| Model training | PyTorch (CPU) |
| Dataset generation | NumPy + custom scripts |

### 12.2 Compute Constraints

- Matrix size: $2 \times 2$ (primary) or $4 \times 4$ (optional).
- Batch size: 8–32.
- Precompute all basis matrices $\{X_i\}$ once at initialization.
- Model parameter count: target $\sim 10^4$–$10^5$.
- Training rollout length $\leq 50$; stability evaluation rollout up to 500.

### 12.3 Proposed Source Layout

```
src/
  liegpt/
    __init__.py
    basis.py          # Precomputed Lie algebra bases, structure constants
    constraint.py     # Lie Constraint Layer
    models.py         # Transformer / GRU sequence models
    evolution.py      # Evolution pipeline: expm, state update, rollout
    dataset.py        # Synthetic trajectory generation
    metrics.py        # State error, unitarity violation, fidelity
    train.py          # Training loop
scripts/
  generate_data.py
  train_liegpt.py
  evaluate_baselines.py
  plot_stability.py
  plot_data_efficiency.py
```

---

## 13. Paper Structure

| Section | Content |
|---|---|
| Introduction | Problem, gap in prior work, core claims |
| Background | Lie groups and algebras, exponential map, quantum dynamics, prior ML methods |
| Method | LieGPT architecture, Lie Constraint Layer, evolution pipeline |
| Theory | Proposition 1 and Theorems 1–2 |
| Experiments | Datasets, baselines, all five experiments |
| Results | Stability figures, data efficiency plots, ablations |
| Discussion | Scope, limitations, connection to prior Cartan-QAOA direction |
| Conclusion | Summary, future work (commutator attention, $\mathfrak{su}(4)$, noise models) |

---

## 14. Title Options

**Primary:**
> Lie-Structured Generative Models for Hamiltonian Simulation

**Alternative:**
> Algebra-Constrained Transformers for Quantum Dynamics

**Working Name:** LieGPT (used in abstract as: *"We refer to this model as LieGPT."*)

The paper uses the formal title. The name is useful for recall but must not be the primary framing — the framing is always a new structure-preserving learning paradigm grounded in Lie theory.

---

## 15. Positioning Against Prior Work

| Prior work | Core limitation | This work's answer |
|---|---|---|
| Fixed-depth Cartan simulation (PRL 129, 070501) | Not learned; deterministic compilation | Learnable, generalizes across Hamiltonians |
| Neural quantum states (NQS, VMC) | Operates in state space; no algebraic guarantees | Operates in algebra space; unitarity by design |
| Physics-informed neural ODEs | Symmetry via penalty terms | Constraints built into architecture |
| QAOA / variational ansatze | Ground state or short-time approximation | Learns full time-dependent generator trajectories |
| Unconstrained sequence models (RNN, Transformer on physics) | No structural enforcement; diverges at long rollout | Lie Constraint Layer guarantees valid generators |
| Cartan-constrained commutator-augmented QAOA (prior work in this repo) | Structured but operates in circuit/conjugator space | Generative model operating in full algebra space |

The prior Cartan-QAOA direction developed in this repository is the **starting point and motivation**, not a competitor. Its key lessons carry forward:

1. **Commutator structure provides provable advantage** at fixed depth — LieGPT generalizes this by operating in the full Lie algebra rather than adding one commutator direction by hand.
2. **Cartan decomposition identifies the relevant algebra** — LieGPT does not require a Cartan decomposition, but benefits from working with well-understood, finite-dimensional algebras.
3. **Orbit alignment and propagator fidelity metrics** from prior experiments carry directly into LieGPT's evaluation via the $\mathcal{F}(t)$ propagator fidelity metric.
4. **TFIM and driven TFIM benchmarks** explored in prior sweep scripts provide natural test cases; single-qubit reductions fall directly within the $\mathfrak{su}(2)$ scope of this plan.

---

## 16. Final Differentiation Checklist

Before submission, every item must be answered **YES**:

| Requirement | Status |
|---|---|
| Model operates in Lie algebra space (not state space) | To implement |
| Physical constraints enforced by architecture (not penalty) | To implement |
| Unitarity guaranteed exactly at every step | By Lie Constraint Layer construction |
| Long-time rollout stability demonstrated over baselines | To demonstrate experimentally |
| Clear quantitative improvement over ML baselines | To demonstrate experimentally |
| At least one formal theoretical result | To prove |
| Work is not "GPT applied to physics" — it is a new learning paradigm | To articulate in introduction |

---

## 17. Summary

> The core contribution of LieGPT is the identification of **Lie algebra space as the correct representation space for learned quantum dynamics**, and the construction of a model that operates natively in that space with hard physical constraints built in by design. This makes every output unitary, reduces the effective hypothesis space by the ratio of algebra dimension to full matrix space dimension, and provides provable long-time stability that unconstrained models cannot match.

The work generalizes the central lesson of the Cartan-QAOA direction — that Lie structure provides a principled and exploitable advantage over naive parameterizations — into a fully learned framework applicable across Hamiltonian families, without requiring a fixed Cartan compilation step.

---

*Research plan last updated: April 2026.*