from __future__ import annotations

from dataclasses import dataclass

from .fitting import train_alternating_qaoa, train_cartan_constrained_qaoa, train_prl_cartan_baseline
from .metrics import normalized_trace_fidelity, operator_spectral_error
from .models import TwoBlockHamiltonian
from .propagators import (
    exact_propagator,
    first_order_trotter,
    lc_qaoa_repeated,
    restricted_cartan_baseline_repeated,
    second_order_suzuki,
)


FIT_GENERATOR_COUNT = 4


@dataclass(frozen=True)
class SimulationResult:
    method: str
    n_steps: int
    spectral_error: float
    trace_fidelity: float
    gate_proxy: int


def benchmark_method(model: TwoBlockHamiltonian, total_time: float, n_steps: int, method: str) -> SimulationResult:
    target = exact_propagator(model.hamiltonian, total_time)
    if method == "trotter1":
        approx = first_order_trotter(model, total_time, n_steps)
        gate_proxy = 2 * n_steps
    elif method == "suzuki2":
        approx = second_order_suzuki(model, total_time, n_steps)
        gate_proxy = 3 * n_steps
    elif method == "cartan_restricted":
        approx = restricted_cartan_baseline_repeated(model, total_time, n_steps)
        gate_proxy = 3 * n_steps
    elif method == "lc_qaoa":
        approx = lc_qaoa_repeated(model, total_time, n_steps)
        gate_proxy = 3 * n_steps
    elif method == "alt_qaoa_fit":
        fit = train_alternating_qaoa(model, total_time, n_steps)
        approx = fit.unitary
        gate_proxy = 2 * n_steps
    elif method == "prl_cartan_fit":
        fit = train_prl_cartan_baseline(model, total_time, n_steps)
        approx = fit.unitary
        gate_proxy = FIT_GENERATOR_COUNT * n_steps + 1
    elif method == "cartan_qaoa_fit":
        fit = train_cartan_constrained_qaoa(model, total_time, n_steps)
        approx = fit.unitary
        gate_proxy = FIT_GENERATOR_COUNT * n_steps + 1
    else:
        raise ValueError(f"Unknown method: {method}")

    return SimulationResult(
        method=method,
        n_steps=n_steps,
        spectral_error=operator_spectral_error(target, approx),
        trace_fidelity=normalized_trace_fidelity(target, approx),
        gate_proxy=gate_proxy,
    )
