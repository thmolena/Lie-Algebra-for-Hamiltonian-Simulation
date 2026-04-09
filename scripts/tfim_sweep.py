from __future__ import annotations

import csv
import warnings
from pathlib import Path

from lc_qaoa.experiments import benchmark_method
from lc_qaoa.models import tfim_hamiltonian


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    output_path = Path("results") / "tfim_sweep_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qubit_counts = [4]
    couplings = [0.5, 1.0]
    fields = [0.5, 1.0, 2.0]
    total_times = [1.0, 2.0]
    methods = ["trotter1", "suzuki2", "cartan_restricted", "lc_qaoa", "alt_qaoa_fit", "prl_cartan_fit", "cartan_qaoa_fit"]
    resolutions = [1, 2, 4]

    rows: list[dict[str, float | int | str]] = []
    for n_qubits in qubit_counts:
        for coupling_j in couplings:
            for field_h in fields:
                model = tfim_hamiltonian(n_qubits=n_qubits, coupling_j=coupling_j, field_h=field_h)
                for total_time in total_times:
                    for method in methods:
                        for resolution in resolutions:
                            result = benchmark_method(model, total_time, resolution, method)
                            rows.append(
                                {
                                    "model": model.name,
                                    "n_qubits": n_qubits,
                                    "coupling_j": coupling_j,
                                    "field_h": field_h,
                                    "total_time": total_time,
                                    "method": result.method,
                                    "resolution": result.n_steps,
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