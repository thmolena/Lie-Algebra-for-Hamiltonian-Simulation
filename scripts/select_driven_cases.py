from __future__ import annotations
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import csv
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lc_qaoa.driven import benchmark_driven_tfim


CARTAN_METHODS = [
    "cartan_restricted",
    "prl_cartan_fit",
    "cartan_qaoa_fit",
]

PANEL_METHODS = [
    "trotter1",
    "suzuki2",
    "cartan_restricted",
    "lc_qaoa",
    "alt_qaoa_fit",
    "prl_cartan_fit",
    "cartan_qaoa_fit",
]

COMPARABLE_PROXY_RATIO = 0.45


def _comparable_rows(case_rows: list[dict[str, float | int | str]], method: str, target_gate_proxy: int) -> list[dict[str, float | int | str]]:
    matching_rows = [row for row in case_rows if row["method"] == method]
    return [
        row
        for row in matching_rows
        if abs(int(row["gate_proxy"]) - target_gate_proxy) <= COMPARABLE_PROXY_RATIO * target_gate_proxy
    ]


def _surviving_cartan_candidate(case_rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str] | None:
    candidates = sorted(
        [row for row in case_rows if row["method"] == "cartan_qaoa_fit"],
        key=lambda item: float(item["spectral_error"]),
    )
    for candidate in candidates:
        candidate_error = float(candidate["spectral_error"])
        candidate_gate = int(candidate["gate_proxy"])
        survives = True
        for baseline_method in ("cartan_restricted", "prl_cartan_fit"):
            comparable = _comparable_rows(case_rows, baseline_method, candidate_gate)
            if not comparable:
                survives = False
                break
            baseline_best = min(float(row["spectral_error"]) for row in comparable)
            if candidate_error >= baseline_best:
                survives = False
                break
        if survives:
            return candidate
    return None


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    coupling_j = 2.0
    field_h = 0.5
    amplitudes = [0.5, 1.0, 2.0]
    omegas = [1.0, 2.0, 3.0]
    total_times = [0.5, 1.0]
    resolutions = [1, 2, 3]
    fit_layer_count = 4

    rows: list[dict[str, float | int | str]] = []
    selected: list[dict[str, float | int | str]] = []

    for amplitude in amplitudes:
        for omega in omegas:
            for total_time in total_times:
                case_rows: list[dict[str, float | int | str]] = []
                for method in PANEL_METHODS:
                    for resolution in resolutions:
                        result = benchmark_driven_tfim(
                            method=method,
                            n_qubits=4,
                            coupling_j=coupling_j,
                            field_h=field_h,
                            amplitude=amplitude,
                            omega=omega,
                            total_time=total_time,
                            resolution=resolution,
                            fit_layer_count=fit_layer_count,
                        )
                        row = {
                            "method": result.method,
                            "amplitude": amplitude,
                            "omega": omega,
                            "total_time": total_time,
                            "resolution": result.resolution,
                            "fit_layer_count": fit_layer_count if "fit" in method else 0,
                            "gate_proxy": result.gate_proxy,
                            "spectral_error": result.spectral_error,
                            "trace_fidelity": result.trace_fidelity,
                        }
                        rows.append(row)
                        case_rows.append(row)

                surviving = _surviving_cartan_candidate(case_rows)
                if surviving is not None:
                    selected.extend(case_rows)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "driven_case_search_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (results_dir / "driven_selected_results.csv").open("w", newline="") as handle:
        if selected:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)
        else:
            handle.write("method,amplitude,omega,total_time,resolution,fit_layer_count,gate_proxy,spectral_error,trace_fidelity\n")

    print(f"selected_cases={len(selected) // (len(PANEL_METHODS) * len(resolutions))}")


if __name__ == "__main__":
    main()
