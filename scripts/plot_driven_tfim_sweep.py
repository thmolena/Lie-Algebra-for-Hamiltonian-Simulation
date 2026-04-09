from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import warnings

import matplotlib.pyplot as plt


def _fmt(value: float) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    input_path = Path("results") / "driven_tfim_sweep_results.csv"
    output_dir = Path("results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[float, float, float], list[dict[str, str]]] = defaultdict(list)
    with input_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (float(row["amplitude"]), float(row["omega"]), float(row["total_time"]))
            grouped[key].append(row)

    for (amplitude, omega, total_time), rows in grouped.items():
        rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rows_by_method[row["method"]].append(row)

        plt.figure(figsize=(7, 5))
        for method, method_rows in sorted(rows_by_method.items()):
            method_rows.sort(key=lambda item: int(item["gate_proxy"]))
            x_values = [int(item["gate_proxy"]) for item in method_rows]
            y_values = [float(item["spectral_error"]) for item in method_rows]
            plt.plot(x_values, y_values, marker="o", label=method)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Gate Proxy")
        plt.ylabel("Spectral Error")
        plt.title(f"Driven TFIM error vs gate proxy: A={amplitude}, w={omega}, T={total_time}")
        plt.legend()
        plt.tight_layout()
        file_name = f"driven_tfim_A{_fmt(amplitude)}_w{_fmt(omega)}_T{_fmt(total_time)}.png"
        plt.savefig(output_dir / file_name, dpi=200)
        plt.close()


if __name__ == "__main__":
    main()