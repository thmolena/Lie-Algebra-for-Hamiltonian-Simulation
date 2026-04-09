from __future__ import annotations
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import csv
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lc_qaoa.experiments import benchmark_method
from lc_qaoa.models import tfim_hamiltonian, xxz_hamiltonian


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

    model_specs = []
    for coupling_j in [2.0, 3.0]:
        for field_h in [0.25, 0.5]:
            model_specs.append((tfim_hamiltonian(n_qubits=4, coupling_j=coupling_j, field_h=field_h), "TFIM"))
    for coupling_z in [2.0, 3.0]:
        for coupling_xy in [0.25, 0.5, 1.0]:
            model_specs.append((xxz_hamiltonian(n_qubits=4, coupling_xy=coupling_xy, coupling_z=coupling_z), "XXZ"))

    total_times = [0.5, 1.0, 1.5]
    resolutions = [2, 4, 6]

    rows: list[dict[str, float | int | str]] = []
    selected: list[dict[str, float | int | str]] = []

    for model, family in model_specs:
        for total_time in total_times:
            case_rows: list[dict[str, float | int | str]] = []
            for method in PANEL_METHODS:
                for resolution in resolutions:
                    result = benchmark_method(model, total_time, resolution, method)
                    row = {
                        "family": family,
                        "model": model.name,
                        "total_time": total_time,
                        "method": result.method,
                        "resolution": result.n_steps,
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

    with (results_dir / "static_case_search_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (results_dir / "static_selected_results.csv").open("w", newline="") as handle:
        if selected:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()))
            writer.writeheader()
            writer.writerows(selected)
        else:
            handle.write("family,model,total_time,method,resolution,gate_proxy,spectral_error,trace_fidelity\n")

    print(f"selected_cases={len(selected) // (len(PANEL_METHODS) * len(resolutions))}")


if __name__ == "__main__":
    main()
