from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitting import train_alternating_qaoa, train_cartan_constrained_qaoa, train_prl_cartan_baseline
from .metrics import normalized_trace_fidelity, operator_spectral_error
from .models import TwoBlockHamiltonian, driven_tfim_hamiltonian
from .propagators import exact_propagator, first_order_trotter, lc_qaoa_repeated, restricted_cartan_baseline_repeated, second_order_suzuki


FIT_GENERATOR_COUNT = 4


@dataclass(frozen=True)
class DrivenSimulationResult:
    method: str
    resolution: int
    gate_proxy: int
    spectral_error: float
    trace_fidelity: float


def midpoint_driven_tfim_model(n_qubits: int, coupling_j: float, field_h: float, amplitude: float, omega: float, midpoint_time: float) -> TwoBlockHamiltonian:
    return driven_tfim_hamiltonian(
        n_qubits=n_qubits,
        coupling_j=coupling_j,
        field_h=field_h,
        amplitude=amplitude,
        omega=omega,
        time=midpoint_time,
    )


def exact_driven_tfim_propagator(
    n_qubits: int,
    coupling_j: float,
    field_h: float,
    amplitude: float,
    omega: float,
    total_time: float,
    reference_steps: int = 256,
) -> np.ndarray:
    dt = total_time / reference_steps
    result = np.eye(2**n_qubits, dtype=complex)
    for step in range(reference_steps):
        midpoint = (step + 0.5) * dt
        model = midpoint_driven_tfim_model(n_qubits, coupling_j, field_h, amplitude, omega, midpoint)
        result = exact_propagator(model.hamiltonian, dt) @ result
    return result


def approximate_driven_tfim(
    method: str,
    n_qubits: int,
    coupling_j: float,
    field_h: float,
    amplitude: float,
    omega: float,
    total_time: float,
    resolution: int,
    fit_layer_count: int = 1,
) -> tuple[np.ndarray, int]:
    dt = total_time / resolution
    result = np.eye(2**n_qubits, dtype=complex)

    if method == "trotter1":
        gate_proxy = 2 * resolution
    elif method == "suzuki2":
        gate_proxy = 3 * resolution
    elif method == "cartan_restricted":
        gate_proxy = 3 * resolution
    elif method == "lc_qaoa":
        gate_proxy = 3 * resolution
    elif method == "alt_qaoa_fit":
        gate_proxy = 2 * fit_layer_count * resolution
    elif method == "prl_cartan_fit":
        gate_proxy = (FIT_GENERATOR_COUNT * fit_layer_count + 1) * resolution
    elif method == "cartan_qaoa_fit":
        gate_proxy = (FIT_GENERATOR_COUNT * fit_layer_count + 1) * resolution
    else:
        raise ValueError(f"Unknown method: {method}")

    for step in range(resolution):
        midpoint = (step + 0.5) * dt
        model = midpoint_driven_tfim_model(n_qubits, coupling_j, field_h, amplitude, omega, midpoint)
        if method == "trotter1":
            step_unitary = first_order_trotter(model, dt, 1)
        elif method == "suzuki2":
            step_unitary = second_order_suzuki(model, dt, 1)
        elif method == "cartan_restricted":
            step_unitary = restricted_cartan_baseline_repeated(model, dt, 1)
        elif method == "lc_qaoa":
            step_unitary = lc_qaoa_repeated(model, dt, 1)
        elif method == "alt_qaoa_fit":
            step_unitary = train_alternating_qaoa(model, dt, fit_layer_count).unitary
        elif method == "prl_cartan_fit":
            step_unitary = train_prl_cartan_baseline(model, dt, fit_layer_count).unitary
        else:
            step_unitary = train_cartan_constrained_qaoa(model, dt, fit_layer_count).unitary
        result = step_unitary @ result

    return result, gate_proxy


def benchmark_driven_tfim(
    method: str,
    n_qubits: int,
    coupling_j: float,
    field_h: float,
    amplitude: float,
    omega: float,
    total_time: float,
    resolution: int,
    reference_steps: int = 256,
    fit_layer_count: int = 1,
) -> DrivenSimulationResult:
    target = exact_driven_tfim_propagator(n_qubits, coupling_j, field_h, amplitude, omega, total_time, reference_steps=reference_steps)
    approx, gate_proxy = approximate_driven_tfim(method, n_qubits, coupling_j, field_h, amplitude, omega, total_time, resolution, fit_layer_count=fit_layer_count)
    return DrivenSimulationResult(
        method=method,
        resolution=resolution,
        gate_proxy=gate_proxy,
        spectral_error=operator_spectral_error(target, approx),
        trace_fidelity=normalized_trace_fidelity(target, approx),
    )