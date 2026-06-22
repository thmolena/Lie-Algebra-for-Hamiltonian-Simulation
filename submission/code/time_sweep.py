"""Experiment 3 -- time-resolved correction hierarchy (the time-sweep figure).

Varies the total time t (with n = 4, q = 2, r = 10 fixed) and plots four curves: the
uncorrected baseline S_2, the projected residual at w <= 2 and w <= 3, and the exact
oracle.  It shows that w <= 3 uniformly improves the baseline across the sweep while
w <= 2 is unreliable -- i.e. compression must respect the commutator support of the
leading defect.

Outputs: generated_data/time_sweep_n4_q2.{csv,json} and the figure.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg

from common import COL_ONEHALF, PALETTE, apply_nmi_style, exact_step, line_plot_style, project_pauli_weight, residual_factor, residual_generator, repeated_step, save_dataframe, save_figure, save_metadata, spectral_error, tfim_terms

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
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.2))
    styles = {
        "baseline": {"label": "uncorrected Strang", "color": PALETTE[0], "linestyle": "--"},
        "projected_w2": {"label": r"compressed $w\leq2$", "color": PALETTE[1], "linestyle": "-."},
        "projected_w3": {"label": r"compressed $w\leq3$ (ours)", "color": PALETTE[2], "linestyle": "-"},
        "oracle": {"label": "exact oracle residual", "color": PALETTE[3], "linestyle": ":"},
    }
    for series, subset in df.groupby("series"):
        subset = subset.sort_values("t")
        ax.plot(subset["t"], subset["error"], **styles[series])
    ax.set_yscale("log")
    ax.set_xlabel("total evolution time $t$")
    ax.set_ylabel("global spectral-norm error")
    ax.legend()
    ax.grid(True, axis="both")
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