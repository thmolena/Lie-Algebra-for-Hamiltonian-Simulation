from __future__ import annotations

import csv
import warnings
from pathlib import Path

from lc_qaoa.experiments import benchmark_method
from lc_qaoa.models import tfim_hamiltonian


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    output_path = Path("results") / "tfim_baseline_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = tfim_hamiltonian(n_qubits=4, coupling_j=1.0, field_h=1.0)
    total_time = 1.0
    methods = ["trotter1", "suzuki2", "cartan_restricted", "lc_qaoa", "alt_qaoa_fit", "prl_cartan_fit", "cartan_qaoa_fit"]
    step_counts = [1, 2, 4, 8]

    rows: list[dict[str, float | int | str]] = []
    for method in methods:
        for n_steps in step_counts:
            result = benchmark_method(model, total_time, n_steps, method)
            rows.append(
                {
                    "method": result.method,
                    "n_steps": result.n_steps,
                    "gate_proxy": result.gate_proxy,
                    "spectral_error": result.spectral_error,
                    "trace_fidelity": result.trace_fidelity,
                }
            )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()