from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import warnings

import matplotlib.pyplot as plt


COMPARABLE_PROXY_RATIO = 0.45
MAX_STATIC_FIGURES = 6
MAX_DRIVEN_FIGURES = 4


def _fmt(value: float) -> str:
    return str(value).replace(".", "p")


def _safe_label(value: str) -> str:
    return value.replace(".", "p")


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _comparable_rows(case_rows: list[dict[str, str]], method: str, target_gate_proxy: int) -> list[dict[str, str]]:
    return [
        row
        for row in case_rows
        if row["method"] == method
        and abs(int(row["gate_proxy"]) - target_gate_proxy) <= COMPARABLE_PROXY_RATIO * target_gate_proxy
    ]


def _case_rank(case_rows: list[dict[str, str]]) -> tuple[float, float]:
    candidates = sorted(
        [row for row in case_rows if row["method"] == "cartan_qaoa_fit"],
        key=lambda item: float(item["spectral_error"]),
    )
    best_ratio = 0.0
    best_error = float("inf")
    for candidate in candidates:
        candidate_error = float(candidate["spectral_error"])
        candidate_gate = int(candidate["gate_proxy"])
        baseline_ratios: list[float] = []
        for baseline_method in ("cartan_restricted", "prl_cartan_fit"):
            comparable = _comparable_rows(case_rows, baseline_method, candidate_gate)
            if not comparable:
                baseline_ratios = []
                break
            baseline_best = min(float(row["spectral_error"]) for row in comparable)
            if candidate_error <= 0.0:
                baseline_ratios.append(float("inf"))
            else:
                baseline_ratios.append(baseline_best / candidate_error)
        if baseline_ratios:
            ratio = min(baseline_ratios)
            if ratio > best_ratio or (ratio == best_ratio and candidate_error < best_error):
                best_ratio = ratio
                best_error = candidate_error
    return best_ratio, best_error


def _top_case_keys(
    grouped: dict[tuple[str, ...], list[dict[str, str]]],
    max_figures: int,
) -> set[tuple[str, ...]]:
    ranked = sorted(
        (
            (_case_rank(case_rows), case_key)
            for case_key, case_rows in grouped.items()
        ),
        key=lambda item: (-item[0][0], item[0][1], item[1]),
    )
    return {case_key for _, case_key in ranked[:max_figures] if _[0] > 1.0}


def _plot_static(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["total_time"])].append(row)

    top_case_keys = _top_case_keys(grouped, MAX_STATIC_FIGURES)

    for (model_name, total_time), group_rows in grouped.items():
        if (model_name, total_time) not in top_case_keys:
            continue
        plt.figure(figsize=(7, 5))
        rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in group_rows:
            rows_by_method[row["method"]].append(row)
        for method, method_rows in sorted(rows_by_method.items()):
            method_rows.sort(key=lambda item: int(item["gate_proxy"]))
            plt.plot(
                [int(item["gate_proxy"]) for item in method_rows],
                [float(item["spectral_error"]) for item in method_rows],
                marker="o",
                label=method,
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Gate Proxy")
        plt.ylabel("Spectral Error")
        plt.title(f"Selected Cartan-family case: {model_name}, T={total_time}")
        plt.legend()
        plt.tight_layout()
        file_name = f"selected_{_safe_label(model_name)}_T{_fmt(float(total_time))}.png"
        plt.savefig(output_dir / file_name, dpi=200)
        plt.close()


def _plot_driven(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["amplitude"], row["omega"], row["total_time"])].append(row)

    top_case_keys = _top_case_keys(grouped, MAX_DRIVEN_FIGURES)

    for (amplitude, omega, total_time), group_rows in grouped.items():
        if (amplitude, omega, total_time) not in top_case_keys:
            continue
        plt.figure(figsize=(7, 5))
        rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in group_rows:
            rows_by_method[row["method"]].append(row)
        for method, method_rows in sorted(rows_by_method.items()):
            method_rows.sort(key=lambda item: int(item["gate_proxy"]))
            plt.plot(
                [int(item["gate_proxy"]) for item in method_rows],
                [float(item["spectral_error"]) for item in method_rows],
                marker="o",
                label=method,
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Gate Proxy")
        plt.ylabel("Spectral Error")
        plt.title(f"Selected driven TFIM case: A={amplitude}, w={omega}, T={total_time}")
        plt.legend()
        plt.tight_layout()
        file_name = f"selected_driven_tfim_A{_fmt(float(amplitude))}_w{_fmt(float(omega))}_T{_fmt(float(total_time))}.png"
        plt.savefig(output_dir / file_name, dpi=200)
        plt.close()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    results_dir = Path("results")
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for png_file in output_dir.glob("*.png"):
        png_file.unlink()

    static_rows = _load_rows(results_dir / "static_selected_results.csv")
    driven_rows = _load_rows(results_dir / "driven_selected_results.csv")

    if static_rows:
        _plot_static(static_rows, output_dir)
    if driven_rows:
        _plot_driven(driven_rows, output_dir)

    print(f"saved_figures={len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()