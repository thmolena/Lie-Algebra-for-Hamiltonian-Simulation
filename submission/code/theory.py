"""From-principles guide for the RGTC Hamiltonian-simulation code artifact.

This module replaces the former text notes with an importable guide that lives
inside the code folder. It is intentionally written from the smallest objects up
to the manuscript figures, so a reader can connect each claim in main.tex to the
Python functions that regenerate it.

Computational problem
---------------------
A local quantum system with n qubits has a state vector in a complex vector
space of dimension d = 2**n. Exact time evolution is

    U(t) = exp(-i H t),

where H is a Hermitian Hamiltonian. Directly storing U(t) costs O(4**n) memory,
which is why Hamiltonian simulation is a core problem in quantum computation:
the interesting algorithmic question is how to approximate U(t) using local,
implementable pieces without paying the dense-matrix cost.

Linear algebra primitives
-------------------------
common.py builds the Pauli matrices I, X, Y and Z, tensor products, Pauli
strings, Pauli weights, and the transverse-field Ising Hamiltonian

    H = A + B,
    A = J sum_j Z_j Z_{j+1},
    B = h sum_j X_j.

These are the only ingredients needed for the dense reference experiments.

Product formulas
----------------
The Lie-Trotter and Strang/Suzuki steps approximate exp(-i(A+B)dt) by products
of exp(-iA tau) and exp(-iB tau). In the code, suzuki_sequence() constructs the
factor list and product_formula() multiplies the dense factors. The global error
reported in the paper is the spectral norm of U(t) - S(dt)**r, computed by
spectral_error().

Residual-generator Trotter compilation
--------------------------------------
For a product-formula step S_q(dt), define

    R_q(dt) = U(dt) S_q(dt)^dagger,
    K_q(dt) = i log R_q(dt).

R_q is the unique left factor that makes R_q S_q exactly equal to U(dt) in exact
arithmetic. K_q is its Hermitian residual generator. The code computes it with
residual_factor() and residual_generator(), then studies which lower-complexity
approximations keep the correction useful.

Stability and approximation
---------------------------
If an approximate residual has per-step operator error eta, the telescoping
identity gives an r-step error bounded linearly by r eta for unitary corrected
steps. That bound is why the project can learn or project K_q and still make a
certified statement about the full simulation interval.

Machine learning component
--------------------------
learned_residual.py treats the local Pauli coefficients of the Strang residual
generator as supervised targets. A translation-equivariant network maps local
couplings to local weight-at-most-three coefficients. Training uses fixed seeds;
evaluation checks transfer from small chains to held-out larger chains.

Figures and tables
------------------
make_all.py regenerates every manuscript artifact:

* fig0_overview: method schematic.
* fig1_exact_residual: exact residual cancellation.
* fig2_compressed_residual and fig3_time_sweep: Pauli-weight compression.
* fig4_generator_structure: Pauli-mass and generator-norm scaling.
* fig5_learned_transfer and fig6_headline_improvement: learned generator.
* tables/*.tex and generated_data/*: the source data consumed by main.tex.

Run from submission/code:

    export PYTHONPATH=.
    python make_all.py
    python validate_submission.py
"""


GUIDE = __doc__


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
