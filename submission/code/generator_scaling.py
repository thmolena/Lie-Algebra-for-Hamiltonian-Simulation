"""Experiments 5 & 6 -- generator compressibility and order scaling (two figures).

(5) Compressibility: the cumulative squared Pauli-coefficient mass of the residual
    generator K_2 (n = 5, q = 2, dt = 0.1) as a function of Pauli weight, showing
    that most of the mass is captured by weight three.
(6) Order scaling: the spectral norm ||K_q(dt)||_2 versus step size dt for orders
    q = 1,2,4,6, confirming that higher-order formulas have smaller residual
    generators -- consistent with the small-step bound ||K_q|| = O(dt^{q+1}).

Outputs: generated_data/generator_energy_n5_q2.json,
generator_order_scaling.{csv,json}, and the two figures.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA_DIR, line_plot_style, pauli_weight_energy, residual_factor, residual_generator, cumulative_weight_mass, save_dataframe, save_figure, save_metadata, tfim_terms, write_json

OUT_FIGURE_COMPRESS = "fig5_generator_compressibility.pdf"
OUT_FIGURE_SCALING = "fig6_generator_order_scaling.pdf"


def build_energy_dataset() -> list[dict[str, float]]:
    terms = tfim_terms(5, 1.0, 1.0)
    _, _, R_dt, _ = residual_factor(terms.A, terms.B, terms.H, 0.1, 2)
    K_dt = residual_generator(R_dt)
    return cumulative_weight_mass(pauli_weight_energy(K_dt, 5))


def build_scaling_dataset() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    terms = tfim_terms(4, 1.0, 1.0)
    for order in (1, 2, 4, 6):
        for dt in (0.025, 0.05, 0.075, 0.1, 0.15, 0.2):
            _, _, R_dt, _ = residual_factor(terms.A, terms.B, terms.H, dt, order)
            K_dt = residual_generator(R_dt)
            rows.append({"order": order, "dt": dt, "generator_norm": float(np.linalg.norm(K_dt, ord=2))})
    return pd.DataFrame(rows)


def make_compressibility_plot(rows: list[dict[str, float]]) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    ax.step(df["max_weight"], df["cumulative_mass"], where="post", color="#2a6f97")
    ax.scatter(df["max_weight"], df["cumulative_mass"], color="#2a6f97")
    ax.set_xlabel("maximum retained Pauli weight")
    ax.set_ylabel("cumulative squared coefficient mass")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Residual-generator compressibility, n=5, q=2")
    line_plot_style(ax)
    save_figure(fig, OUT_FIGURE_COMPRESS)


def make_scaling_plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = {1: "#c8553d", 2: "#2a6f97", 4: "#6a994e", 6: "#7b2cbf"}
    for order, subset in df.groupby("order"):
        subset = subset.sort_values("dt")
        ax.plot(subset["dt"], subset["generator_norm"], marker="o", color=colors[order], label=f"q={order}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta t$")
    ax.set_ylabel(r"$\|K_q(\delta t)\|_2$")
    ax.set_title("Residual-generator norm scaling")
    ax.legend()
    line_plot_style(ax)
    save_figure(fig, OUT_FIGURE_SCALING)


def main(force: bool = False) -> tuple[list[dict[str, float]], pd.DataFrame]:
    energy_rows = build_energy_dataset()
    scaling_df = build_scaling_dataset()
    write_json((DATA_DIR / "generator_energy_n5_q2.json"), energy_rows)
    save_dataframe(scaling_df, "generator_order_scaling.csv", "generator_order_scaling.json")
    save_metadata(
        "generator_scaling.meta.json",
        {
            "figures": [OUT_FIGURE_COMPRESS, OUT_FIGURE_SCALING],
            "parameters": {
                "compressibility": {"n": 5, "order": 2, "J": 1.0, "h": 1.0, "dt": 0.1},
                "order_scaling": {"n": 4, "J": 1.0, "h": 1.0, "orders": [1, 2, 4, 6], "dt_values": [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]},
            },
            "force": force,
        },
    )
    make_compressibility_plot(energy_rows)
    make_scaling_plot(scaling_df)
    return energy_rows, scaling_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)