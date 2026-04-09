from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .models import Array, TwoBlockHamiltonian


def commutator(left: Array, right: Array) -> Array:
    return left @ right - right @ left


def normalize_hermitian(operator: Array) -> Array:
    scale = float(np.linalg.norm(operator, ord="fro"))
    if scale < 1e-12:
        return operator.copy()
    return operator / scale


def evolution_unitary(hermitian: Array, time: float) -> Array:
    """Compute exp(-i time H) for Hermitian H using eigendecomposition."""

    hermitian_part = 0.5 * (hermitian + hermitian.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_part)
    phases = np.exp(-1j * time * eigenvalues)
    return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T


def exact_propagator(hamiltonian: Array, time: float) -> Array:
    return evolution_unitary(hamiltonian, time)


def alternating_step(model: TwoBlockHamiltonian, time: float) -> Array:
    return evolution_unitary(model.B, time) @ evolution_unitary(model.A, time)


def first_order_trotter(model: TwoBlockHamiltonian, total_time: float, n_steps: int) -> Array:
    dt = total_time / n_steps
    step = alternating_step(model, dt)
    result = np.eye(model.dimension, dtype=complex)
    for _ in range(n_steps):
        result = step @ result
    return result


def second_order_suzuki(model: TwoBlockHamiltonian, total_time: float, n_steps: int) -> Array:
    dt = total_time / n_steps
    half_a = evolution_unitary(model.A, 0.5 * dt)
    full_b = evolution_unitary(model.B, dt)
    step = half_a @ full_b @ half_a
    result = np.eye(model.dimension, dtype=complex)
    for _ in range(n_steps):
        result = step @ result
    return result


def lc_qaoa_single_step(model: TwoBlockHamiltonian, time: float) -> Array:
    c = 1j * commutator(model.A, model.B)
    correction = evolution_unitary(c, -0.5 * (time**2))
    return correction @ alternating_step(model, time)


def lc_qaoa_repeated(model: TwoBlockHamiltonian, total_time: float, n_steps: int) -> Array:
    dt = total_time / n_steps
    step = lc_qaoa_single_step(model, dt)
    result = np.eye(model.dimension, dtype=complex)
    for _ in range(n_steps):
        result = step @ result
    return result


def commuting_projection(model: TwoBlockHamiltonian, operator: Array) -> Array:
    """Project onto the commuting Cartan surrogate spanned by the A-block basis.

    For TFIM the ZZ terms commute, so the A-block term basis provides a more
    faithful surrogate of the Cartan commuting sector than projecting onto the
    full A sum alone.
    """

    if model.a_terms:
        projection = np.zeros_like(operator)
        for basis_term in model.a_terms:
            denominator = np.trace(basis_term.conj().T @ basis_term)
            if abs(denominator) < 1e-12:
                continue
            coefficient = np.trace(basis_term.conj().T @ operator) / denominator
            projection += coefficient * basis_term
        return projection

    numerator = np.trace(model.A.conj().T @ operator)
    denominator = np.trace(model.A.conj().T @ model.A)
    if abs(denominator) < 1e-12:
        return np.zeros_like(operator)
    coefficient = numerator / denominator
    return coefficient * model.A


def cartan_single_generator(model: TwoBlockHamiltonian) -> Array:
    return normalize_hermitian(1j * commutator(model.A, model.B))


def restricted_cartan_baseline_step(model: TwoBlockHamiltonian, time: float) -> Array:
    """A small executable surrogate for a fixed-depth Cartan baseline.

    The circuit uses a single fixed-depth conjugator direction and projects the
    conjugated Hamiltonian onto the commuting A block, then conjugates back.
    This is intended only as a benchmarkable stand-in for the PRL-style Cartan
    pipeline on small TFIM instances.
    """

    generator = cartan_single_generator(model)
    cartan_target = normalize_hermitian(model.A)

    def objective(theta: float) -> float:
        k_theta = evolution_unitary(generator, theta)
        aligned = k_theta.conj().T @ cartan_target @ k_theta
        overlap = np.trace(aligned.conj().T @ model.hamiltonian)
        return float(-np.real(overlap))

    optimum = minimize_scalar(objective, bounds=(-np.pi, np.pi), method="bounded")
    theta_star = float(optimum.x)
    k_star = evolution_unitary(generator, theta_star)
    conjugated = k_star @ model.hamiltonian @ k_star.conj().T
    h0 = commuting_projection(model, conjugated)
    return k_star.conj().T @ evolution_unitary(h0, time) @ k_star


def restricted_cartan_baseline_repeated(model: TwoBlockHamiltonian, total_time: float, n_steps: int) -> Array:
    dt = total_time / n_steps
    step = restricted_cartan_baseline_step(model, dt)
    result = np.eye(model.dimension, dtype=complex)
    for _ in range(n_steps):
        result = step @ result
    return result
