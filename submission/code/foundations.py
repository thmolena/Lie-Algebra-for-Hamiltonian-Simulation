"""From-scratch foundations for the RGTC Hamiltonian-simulation artifact.

This module is intentionally prose-heavy Python.  It replaces standalone text
notes with an importable, executable guide that keeps the code folder compliant
with the no-extra-text-file submission rule while still explaining the artifact
from the basic theory of computation, linear algebra, quantum simulation and
machine learning.
"""

from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Computation problem",
        "An n-qubit Hamiltonian acts on a vector space of dimension 2**n.  "
        "The exact propagator U(t)=expm(-i H t) is the object a quantum "
        "computer would implement efficiently for local H, while a classical "
        "dense simulation grows exponentially.  The scripts therefore use small "
        "n as an exact laboratory and never present dense exponentiation as the "
        "scalable algorithm.",
    ),
    (
        "Operator basis",
        "The Pauli strings form an orthogonal basis for Hermitian operators.  "
        "The helper functions in common.py build single-site Paulis, tensor "
        "products, Pauli strings, Pauli weights and Pauli-basis projections.  "
        "Weight counts how many qubits an operator touches, so it is the code's "
        "locality measure.",
    ),
    (
        "Hamiltonian model",
        "The benchmark Hamiltonian is the transverse-field Ising model "
        "H=A+B with nearest-neighbour ZZ coupling and single-site X fields.  "
        "A and B are each easy to exponentiate but do not commute, which is why "
        "product formulas have non-zero error.",
    ),
    (
        "Product formulas",
        "product_formula() implements Lie-Trotter, Strang and recursive Suzuki "
        "steps.  The local error scales as dt**(q+1) for order q, but the gate "
        "factor count grows with q.  The manuscript figures compare accuracy "
        "against this resource proxy rather than claiming cost-free high order.",
    ),
    (
        "Residual generator",
        "For a product-formula step S_q(dt), residual_factor() computes "
        "R_q=U(dt) S_q(dt)^dagger.  The corrected step R_q S_q is exactly U at "
        "the dense oracle scale.  residual_generator() computes K_q=i log R_q, "
        "the Hermitian generator that the compression and learning experiments "
        "try to approximate.",
    ),
    (
        "Stability certificate",
        "If an approximate residual differs from the exact residual by eta per "
        "step, telescoping the product gives an r*eta global error bound for "
        "unitary corrections.  This is why approximating K_q is a controlled "
        "computation rather than a heuristic.",
    ),
    (
        "Compression and locality",
        "projected_residual.py expands K_q in the Pauli basis and keeps only "
        "low-weight strings.  For the TFIM, the leading Strang residual is "
        "supported by local commutator structure, which explains the sharp gain "
        "when weight-three terms are included.",
    ),
    (
        "Machine learning",
        "learned_residual.py treats local couplings as features and the local "
        "weight-three residual coefficients as labels.  A small seeded network "
        "learns a translation-equivariant map from local Hamiltonian data to "
        "the correction generator, then transfers to larger held-out chains.  "
        "The dense oracle supplies labels only for this controlled small-scale "
        "study.",
    ),
    (
        "Reproduction path",
        "Run python make_all.py from submission/code to regenerate tables, "
        "figures and machine-readable generated_data.  Run python "
        "validate_submission.py to check that main.tex references the generated "
        "artifacts.  The install metadata in pyproject.toml declares the Python "
        "dependencies; no requirements text file is needed.",
    ),
)


def iter_foundations() -> tuple[tuple[str, str], ...]:
    """Return the ordered foundations guide as (heading, body) pairs."""

    return FOUNDATION_SECTIONS


def print_foundations() -> None:
    """Print a readable theory-to-code map for command-line inspection."""

    for index, (heading, body) in enumerate(FOUNDATION_SECTIONS, start=1):
        print(f"{index}. {heading}\n{body}\n")


if __name__ == "__main__":
    print_foundations()
