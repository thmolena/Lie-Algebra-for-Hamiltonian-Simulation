from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg

from common import exact_step, line_plot_style, project_pauli_weight, residual_factor, residual_generator, repeated_step, save_dataframe, save_figure, save_metadata, spectral_error, tfim_terms

OUT_FIGURE = "fig3_time_sweep.pdf"


def build_dataset() -> pd.DataFrame:
    terms = tfim_terms(4, 1.0, 1.0)
    rows: list[dict[str, float]] = []
    times = np.linspace(0.1, 4.0, 40)
    for total_time in times:
        r = 10
        dt = float(total_time) / r
        U_dt, S_dt, R_dt, G_dt = residual_factor(terms.A, terms.B, terms.H, dt, 2)
        exact_total = exact_step(terms.H, float(total_time))
        baseline_total = repeated_step(S_dt, r)
        oracle_total = repeated_step(G_dt, r)
        K_dt = residual_generator(R_dt)
        for cutoff in (2, 3):
            K_w = project_pauli_weight(K_dt, 4, cutoff)
            projected_step = scipy.linalg.expm(-1j * K_w) @ S_dt
            projected_total = repeated_step(projected_step, r)
            rows.append(
                {
                    "t": float(total_time),
                    "series": f"projected_w{cutoff}",
                    "error": spectral_error(exact_total, projected_total),
                }
            )
        rows.extend(
            [
                {"t": float(total_time), "series": "baseline", "error": spectral_error(exact_total, baseline_total)},
                {"t": float(total_time), "series": "oracle", "error": spectral_error(exact_total, oracle_total)},
            ]
        )
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    styles = {
        "baseline": {"label": "baseline S2", "color": "#c8553d", "linestyle": "--"},
        "projected_w2": {"label": "projected w≤2", "color": "#6a994e", "linestyle": "-."},
        "projected_w3": {"label": "projected w≤3", "color": "#2a6f97", "linestyle": "-"},
        "oracle": {"label": "oracle", "color": "#0b3c5d", "linestyle": ":"},
    }
    for series, subset in df.groupby("series"):
        subset = subset.sort_values("t")
        ax.plot(subset["t"], subset["error"], **styles[series])
    ax.set_yscale("log")
    ax.set_xlabel("total time")
    ax.set_ylabel("global spectral-norm error")
    ax.set_title("Time-resolved correction hierarchy, n=4, q=2")
    ax.legend()
    line_plot_style(ax)
    save_figure(fig, OUT_FIGURE)


def main(force: bool = False) -> pd.DataFrame:
    df = build_dataset()
    save_dataframe(df, "time_sweep_n4_q2.csv", "time_sweep_n4_q2.json")
    save_metadata(
        "time_sweep_n4_q2.meta.json",
        {
            "figure": OUT_FIGURE,
            "parameters": {"n": 4, "order": 2, "J": 1.0, "h": 1.0, "r": 10, "t_min": 0.1, "t_max": 4.0, "points": 40},
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