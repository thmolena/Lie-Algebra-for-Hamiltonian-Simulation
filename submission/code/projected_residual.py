from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from common import TABLE_DIR, projected_residual_error, save_dataframe, save_figure, save_metadata, scientific, write_latex_table

OUT_FIGURE = "fig2_projected_residual.pdf"


def build_dataset() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for weight in range(6):
        result = projected_residual_error(n_qubits=5, order=2, t=1.0, r=10, max_weight=weight, J=1.0, h=1.0)
        rows.append(
            {
                "w": weight,
                **result,
                "improvement": result["baseline_error"] / result["projected_error"],
            }
        )
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(df["w"], df["projected_error"], marker="o", color="#0b3c5d", label="projected residual")
    ax.axhline(df["baseline_error"].iloc[0], linestyle="--", color="#c8553d", label="baseline S2")
    ax.set_yscale("log")
    ax.set_xlabel("maximum retained Pauli weight")
    ax.set_ylabel("global spectral-norm error")
    ax.set_title("Projected residual compilation, n=5, q=2")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, OUT_FIGURE)


def write_table(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Weight-truncated residual compilation for $n=5$, $q=2$, $J=h=1$, $t=1$, and $r=10$. The column $w$ keeps Pauli strings of weight at most $w$ in the exact residual generator. All values are recomputed from dense matrices.}",
        r"\label{tab:projected-summary}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{cccc}",
        r"$w$ & $\norm{K_q-\Pi_wK_q}_2$ & projected error & improvement\\",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.w} & ${scientific(row.generator_residual_norm)}$ & ${scientific(row.projected_error)}$ & ${row.improvement:.2f}\\times$\\\\"
        )
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"])
    write_latex_table(TABLE_DIR / "projected_summary.tex", lines)


def main(force: bool = False) -> pd.DataFrame:
    df = build_dataset()
    save_dataframe(df, "projected_residual_n5_q2.csv", "projected_residual_n5_q2.json")
    save_metadata(
        "projected_residual_n5_q2.meta.json",
        {
            "figure": OUT_FIGURE,
            "parameters": {"n": 5, "order": 2, "J": 1.0, "h": 1.0, "t": 1.0, "r": 10, "weights": [0, 1, 2, 3, 4, 5]},
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