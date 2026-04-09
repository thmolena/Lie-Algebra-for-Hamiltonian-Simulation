from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .metrics import normalized_trace_fidelity
from .models import Array, TwoBlockHamiltonian
from .propagators import commutator, commuting_projection, evolution_unitary, exact_propagator, normalize_hermitian


ALT_MAXITER = 180
PRL_MAXITER = 160
CQ_MAXITER = 180


@dataclass(frozen=True)
class FitResult:
    parameters: Array
    unitary: Array
    loss: float


def _loss_from_unitary(target: Array, approx: Array) -> float:
    return 1.0 - normalized_trace_fidelity(target, approx)


def _cartan_generator_pool(model: TwoBlockHamiltonian) -> tuple[Array, Array, Array]:
    c = normalize_hermitian(1j * commutator(model.A, model.B))
    d_a = normalize_hermitian(commutator(model.A, commutator(model.A, model.B)))
    d_b = normalize_hermitian(commutator(model.B, commutator(model.A, model.B)))
    return c, d_a, d_b


def _extended_cartan_generator_pool(model: TwoBlockHamiltonian) -> tuple[Array, ...]:
    c, d_a, d_b = _cartan_generator_pool(model)
    d_mix = normalize_hermitian(1j * commutator(d_a, d_b))
    return c, d_a, d_b, d_mix


def _commuting_basis(model: TwoBlockHamiltonian) -> tuple[Array, ...]:
    return model.a_terms if model.a_terms else (model.A,)


def _operator_from_coefficients(basis: tuple[Array, ...], coefficients: Array) -> Array:
    operator = np.zeros_like(basis[0])
    for coefficient, basis_term in zip(coefficients, basis):
        operator = operator + float(coefficient) * basis_term
    return operator


def _basis_projection_coefficients(operator: Array, basis: tuple[Array, ...]) -> Array:
    coefficients = []
    for basis_term in basis:
        denominator = np.real(np.trace(basis_term.conj().T @ basis_term))
        if abs(denominator) < 1e-12:
            coefficients.append(0.0)
            continue
        numerator = np.real(np.trace(basis_term.conj().T @ operator))
        coefficients.append(float(numerator / denominator))
    return np.asarray(coefficients, dtype=float)


def _cartan_target_from_spectrum(model: TwoBlockHamiltonian) -> Array:
    basis = _commuting_basis(model)
    if len(basis) == 1:
        return basis[0]

    basis_diagonals = np.stack([np.real(np.diag(term)) for term in basis], axis=1)
    target_spectrum = np.sort(np.linalg.eigvalsh(model.hamiltonian))
    initial = _basis_projection_coefficients(model.hamiltonian, basis)

    def objective(coefficients: Array) -> float:
        commuting_spectrum = np.sort(basis_diagonals @ coefficients)
        spectral_loss = float(np.linalg.norm(commuting_spectrum - target_spectrum))
        regularization = 1e-3 * float(np.linalg.norm(coefficients))
        return spectral_loss + regularization

    optimum = minimize(objective, initial, method="Powell", options={"maxiter": ALT_MAXITER, "xtol": 1e-4, "ftol": 1e-6})
    return _operator_from_coefficients(basis, np.asarray(optimum.x, dtype=float))


def _compose_conjugator(model: TwoBlockHamiltonian, parameters: Array) -> Array:
    generator_pool = _extended_cartan_generator_pool(model)
    parameter_stride = len(generator_pool)
    result = np.eye(model.dimension, dtype=complex)
    for layer in range(len(parameters) // parameter_stride):
        layer_unitary = np.eye(model.dimension, dtype=complex)
        for offset, generator in enumerate(generator_pool):
            angle = float(parameters[parameter_stride * layer + offset])
            layer_unitary = evolution_unitary(generator, angle) @ layer_unitary
        result = layer_unitary @ result
    return result


def _orbit_alignment_loss(conjugated: Array, cartan_target: Array) -> float:
    return float(np.linalg.norm(conjugated - cartan_target, ord="fro"))


def alternating_qaoa_unitary(model: TwoBlockHamiltonian, total_time: float, parameters: Array) -> Array:
    layer_count = len(parameters) // 2
    result = np.eye(model.dimension, dtype=complex)
    for layer in range(layer_count):
        alpha = float(parameters[2 * layer])
        beta = float(parameters[2 * layer + 1])
        layer_unitary = evolution_unitary(model.B, beta) @ evolution_unitary(model.A, alpha)
        result = layer_unitary @ result
    return result


def train_alternating_qaoa(model: TwoBlockHamiltonian, total_time: float, layer_count: int) -> FitResult:
    target = exact_propagator(model.hamiltonian, total_time)
    initial = np.tile(np.array([total_time / layer_count, total_time / layer_count]), layer_count)

    def objective(parameters: Array) -> float:
        approx = alternating_qaoa_unitary(model, total_time, parameters)
        return _loss_from_unitary(target, approx)

    optimum = minimize(objective, initial, method="Powell", options={"maxiter": 200, "xtol": 1e-4, "ftol": 1e-6})
    best_parameters = np.asarray(optimum.x, dtype=float)
    best_unitary = alternating_qaoa_unitary(model, total_time, best_parameters)
    return FitResult(parameters=best_parameters, unitary=best_unitary, loss=float(optimum.fun))


def prl_cartan_baseline_unitary(model: TwoBlockHamiltonian, total_time: float, parameters: Array) -> Array:
    cartan_target = _cartan_target_from_spectrum(model)
    conjugator = _compose_conjugator(model, parameters)
    return conjugator.conj().T @ evolution_unitary(cartan_target, total_time) @ conjugator


def train_prl_cartan_baseline(model: TwoBlockHamiltonian, total_time: float, layer_count: int) -> FitResult:
    target = exact_propagator(model.hamiltonian, total_time)
    cartan_target = _cartan_target_from_spectrum(model)
    generator_pool = _extended_cartan_generator_pool(model)
    initial = np.zeros(len(generator_pool) * layer_count, dtype=float)

    def objective(parameters: Array) -> float:
        k_theta = _compose_conjugator(model, parameters)
        conjugated = k_theta @ model.hamiltonian @ k_theta.conj().T
        return _orbit_alignment_loss(conjugated, cartan_target)

    optimum = minimize(objective, initial, method="Powell", options={"maxiter": PRL_MAXITER, "xtol": 1e-4, "ftol": 1e-6})
    best_parameters = np.asarray(optimum.x, dtype=float)
    best_unitary = prl_cartan_baseline_unitary(model, total_time, best_parameters)
    return FitResult(parameters=best_parameters, unitary=best_unitary, loss=_loss_from_unitary(target, best_unitary))


def cartan_constrained_qaoa_unitary(model: TwoBlockHamiltonian, total_time: float, parameters: Array) -> Array:
    k_theta = _compose_conjugator(model, parameters)
    conjugated = k_theta @ model.hamiltonian @ k_theta.conj().T
    h0 = commuting_projection(model, conjugated)
    return k_theta.conj().T @ evolution_unitary(h0, total_time) @ k_theta


def train_cartan_constrained_qaoa(model: TwoBlockHamiltonian, total_time: float, layer_count: int) -> FitResult:
    target = exact_propagator(model.hamiltonian, total_time)
    generator_pool = _extended_cartan_generator_pool(model)
    initial = train_prl_cartan_baseline(model, total_time, layer_count).parameters
    cartan_target = _cartan_target_from_spectrum(model)

    def objective(parameters: Array) -> float:
        unitary = cartan_constrained_qaoa_unitary(model, total_time, parameters)
        operator_loss = _loss_from_unitary(target, unitary)
        k_theta = _compose_conjugator(model, parameters)
        conjugated = k_theta @ model.hamiltonian @ k_theta.conj().T
        alignment_loss = _orbit_alignment_loss(conjugated, cartan_target)
        residual = conjugated - commuting_projection(model, conjugated)
        cartan_loss = float(np.linalg.norm(residual, ord="fro"))
        return operator_loss + 0.02 * alignment_loss + 0.02 * cartan_loss

    optimum = minimize(objective, initial, method="Powell", options={"maxiter": CQ_MAXITER, "xtol": 1e-4, "ftol": 1e-6})
    best_parameters = np.asarray(optimum.x, dtype=float)
    best_unitary = cartan_constrained_qaoa_unitary(model, total_time, best_parameters)
    return FitResult(parameters=best_parameters, unitary=best_unitary, loss=float(optimum.fun))