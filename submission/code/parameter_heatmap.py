from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg

from common import exact_step, residual_factor, residual_generator, project_pauli_weight, repeated_step, save_dataframe, save_figure, save_metadata, spectral_error, tfim_terms

OUT_FIGURE = "fig4_parameter_heatmap.pdf"


def build_dataset() -> pd.DataFrame:
    grid = np.arange(0.25, 2.01, 0.25)
    rows: list[dict[str, float]] = []
    for J in grid:
        for h in grid:
            terms = tfim_terms(4, float(J), float(h))
            t = 1.0
            r = 10
            dt = t / r
            U_dt, S_dt, R_dt, _ = residual_factor(terms.A, terms.B, terms.H, dt, 2)
            K_dt = residual_generator(R_dt)
            K_w = project_pauli_weight(K_dt, 4, 3)
            projected_step = scipy.linalg.expm(-1j * K_w) @ S_dt
            exact_total = exact_step(terms.H, t)
            baseline_error = spectral_error(exact_total, repeated_step(S_dt, r))
            projected_error = spectral_error(exact_total, repeated_step(projected_step, r))
            rows.append(
                {
                    "J": float(J),
                    "h": float(h),
                    "baseline_error": baseline_error,
                    "projected_error": projected_error,
                    "improvement_ratio": baseline_error / projected_error,
                    "log10_improvement_ratio": float(np.log10(baseline_error / projected_error)),
                }
            )
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="h", columns="J", values="log10_improvement_ratio").sort_index()
    fig, ax = plt.subplots(figsize=(6.3, 5.1))
    image = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in pivot.index])
    ax.set_xlabel("J")
    ax.set_ylabel("h")
    ax.set_title(r"$\log_{10}(\epsilon_{T,2}/\epsilon_{w\leq 3})$")
    fig.colorbar(image, ax=ax, label=r"$\log_{10}$ improvement")
    save_figure(fig, OUT_FIGURE)


def main(force: bool = False) -> pd.DataFrame:
    df = build_dataset()
    save_dataframe(df, "parameter_heatmap_n4_q2_w3.csv", "parameter_heatmap_n4_q2_w3.json")
    save_metadata(
        "parameter_heatmap_n4_q2_w3.meta.json",
        {
            "figure": OUT_FIGURE,
            "parameters": {"n": 4, "order": 2, "t": 1.0, "r": 10, "cutoff": 3, "grid": [round(x, 2) for x in np.arange(0.25, 2.01, 0.25)]},
            "force": force,
        },
    )
    make_plot(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)