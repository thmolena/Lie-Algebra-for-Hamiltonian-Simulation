"""Experiment 2 -- weight-truncated (compressed) residual (Table II and figure).

Instead of the full oracle generator K_q, keep only Pauli strings of weight <= w:
K_{q,w} = Pi_w K_q (the unique Frobenius-optimal compression), then correct with
exp(-i K_{q,w}).  This is the first *non-tautological* test -- it measures how much
of the useful correction lives in a small operator family.

For n = 5, q = 2, t = 1, r = 10 the baseline 1.384e-2 drops to 1.545e-4 at w = 3
(an ~89.6x improvement) and to 1.850e-7 at w = 4.  The sharp gain at w = 3 is exactly
the threshold predicted by the leading-Strang locality theorem (the degree-three
TFIM defect has Pauli weight <= 3).

Outputs: tables/projected_summary.tex and
generated_data/projected_residual_n5_q2.{csv,json}.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from common import PALETTE, TABLE_DIR, panel_label, projected_residual_error, save_dataframe, save_figure, save_metadata, scientific, write_latex_table

OUT_FIGURE = "fig2_compressed_residual.pdf"


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
    baseline = float(df["baseline_error"].iloc[0])
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.4, 4.3))

    # (a) Compressed-residual error versus retained Pauli weight (ours vs baseline).
    ax_a.plot(df["w"], df["projected_error"], marker="o", color=PALETTE[2],
              label="compressed residual (ours)")
    ax_a.axhline(baseline, linestyle="--", color=PALETTE[0], label="uncorrected Strang")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("maximum retained Pauli weight $w$")
    ax_a.set_ylabel("global spectral-norm error")
    ax_a.set_title("Compressed residual vs baseline")
    ax_a.grid(alpha=0.25)
    ax_a.legend()
    panel_label(ax_a, "a")

    # (b) Error-reduction factor over the uncorrected baseline, per weight (bar).
    factors = baseline / df["projected_error"].to_numpy()
    ax_b.bar(df["w"].to_numpy(), factors, color=PALETTE[2], width=0.62)
    ax_b.axhline(1.0, linestyle="--", color=PALETTE[0], linewidth=1.0)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("maximum retained Pauli weight $w$")
    ax_b.set_ylabel(r"error-reduction factor $\epsilon_{\mathrm{Strang}}/\epsilon_w$")
    ax_b.set_title("Improvement over baseline")
    ax_b.grid(alpha=0.25, axis="y")
    panel_label(ax_b, "b")

    fig.tight_layout(pad=1.2)
    save_figure(fig, OUT_FIGURE)


def write_table(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Weight-truncated residual compilation for $n=5$, $q=2$, $J=h=1$, $t=1$, and $r=10$. The column $w$ keeps Pauli strings of weight at most $w$ in the exact residual generator. All values are recomputed from dense matrices.}",
        r"\label{tab:projected-summary}",
        r"\centering",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$w$ & $\norm{K_q-\Pi_wK_q}_2$ & projected error & improvement\\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.w} & ${scientific(row.generator_residual_norm)}$ & ${scientific(row.projected_error)}$ & ${row.improvement:.2f}\\times$\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
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