"""Experiment 1 -- fixed-time oracle benchmark (Table I and the fixed-time figure).

Question: does the *exact* residual R_q = U S_q^dagger cancel the Trotter--Suzuki
error of the corrected step G_q = R_q S_q down to the floating-point floor?

For the TFIM with J = h = 1, total time t = 1 and r = 10 steps, this compares the
baseline error  ||U(t) - S_q(t/r)^r||_2  against the oracle-residual error
||U(t) - [R_q S_q]^r||_2  for n = 4,5,6 qubits and orders q = 1,2,4,6,8.  The oracle
reaches 1e-15..1e-14 (machine precision) while the baseline ranges from ~1e-1 (q=1)
down to ~1e-9 (q=6); at q=8 the baseline is itself near round-off, which is the
expected finite-precision behaviour, not a violation of the exact-cancellation
theorem.

Outputs: tables/error_summary.tex and generated_data/fixed_time_errors.{csv,json}.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from common import ORDERS, TABLE_DIR, global_errors, line_plot_style, save_dataframe, save_figure, save_metadata, scientific, write_latex_table

OUT_FIGURE = "fig1_exact_residual.pdf"


def build_dataset() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for n_qubits in (4, 5, 6):
        for order in ORDERS:
            result = global_errors(n_qubits=n_qubits, order=order, t=1.0, r=10, J=1.0, h=1.0)
            rows.append(
                {
                    "n": n_qubits,
                    "d": 2 ** n_qubits,
                    "order": order,
                    "t": 1.0,
                    "r": 10,
                    "J": 1.0,
                    "h": 1.0,
                    "dt": 0.1,
                    **result,
                }
            )
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    markers = {4: "o", 5: "s", 6: "^"}
    colors = {4: "#2a6f97", 5: "#c8553d", 6: "#6a994e"}
    for n_qubits in (4, 5, 6):
        subset = df[df["n"] == n_qubits].sort_values("order")
        ax.plot(
            subset["order"],
            subset["baseline_error"],
            marker=markers[n_qubits],
            color=colors[n_qubits],
            linestyle="--",
            label=f"baseline n={n_qubits}",
        )
        ax.plot(
            subset["order"],
            subset["oracle_error"],
            marker=markers[n_qubits],
            color=colors[n_qubits],
            linestyle="-",
            label=f"oracle n={n_qubits}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Suzuki order $q$")
    ax.set_ylabel("spectral-norm error")
    ax.set_title("Exact-residual cancellation vs uncorrected Strang--Suzuki")
    ax.legend(ncol=2, fontsize=8)
    line_plot_style(ax)
    save_figure(fig, OUT_FIGURE)


def write_table(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Authentic dense-matrix fixed-time benchmark for the open-boundary \tfim{} with $J=h=1$, $t=1$, and $r=10$ steps. Baseline is $\epsilon_{T,q}=\norm{U(t)-S_q(t/r)^r}_2$; oracle residual is $\epsilon_{R,q}=\norm{U(t)-[R_q(t/r)S_q(t/r)]^r}_2$.}",
        r"\label{tab:error-summary}",
        r"\centering",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$n$ & $d$ & $q$ & $\epsilon_{T,q}$ & $\epsilon_{R,q}$\\",
        r"\midrule",
    ]
    for row in df.sort_values(["n", "order"]).itertuples(index=False):
        lines.append(
            f"{row.n} & {row.d} & {row.order} & ${scientific(row.baseline_error)}$ & ${scientific(row.oracle_error)}$\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    write_latex_table(TABLE_DIR / "error_summary.tex", lines)


def main(force: bool = False) -> pd.DataFrame:
    df = build_dataset()
    save_dataframe(df, "fixed_time_errors.csv", "fixed_time_errors.json")
    save_metadata(
        "fixed_time_errors.meta.json",
        {
            "figure": OUT_FIGURE,
            "parameters": {"J": 1.0, "h": 1.0, "t": 1.0, "r": 10, "n_values": [4, 5, 6], "orders": ORDERS},
            "force": force,
        },
    )
    make_plot(df)
    write_table(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)