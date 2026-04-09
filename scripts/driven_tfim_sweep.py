from __future__ import annotations

import csv
import warnings
from pathlib import Path

from lc_qaoa.driven import benchmark_driven_tfim


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    output_path = Path("results") / "driven_tfim_sweep_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = ["trotter1", "suzuki2", "cartan_restricted", "lc_qaoa", "alt_qaoa_fit", "prl_cartan_fit", "cartan_qaoa_fit"]
    amplitudes = [0.5, 1.0]
    omegas = [1.0]
    total_times = [1.0, 2.0]
    resolutions = [2, 4]

    rows: list[dict[str, float | int | str]] = []
    for amplitude in amplitudes:
        for omega in omegas:
            for total_time in total_times:
                for method in methods:
                    for resolution in resolutions:
                        result = benchmark_driven_tfim(
                            method=method,
                            n_qubits=4,
                            coupling_j=1.0,
                            field_h=1.0,
                            amplitude=amplitude,
                            omega=omega,
                            total_time=total_time,
                            resolution=resolution,
                            fit_layer_count=1,
                        )
                        rows.append(
                            {
                                "method": result.method,
                                "amplitude": amplitude,
                                "omega": omega,
                                "total_time": total_time,
                                "resolution": result.resolution,
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