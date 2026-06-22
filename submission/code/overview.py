"""Conceptual overview schematic for residual-generator Trotter compilation.

This figure is a *schematic*: it contains no numerical data, only the logical
flow of the framework (product-formula defect -> exact residual -> local
generator -> learned, size-transferable correction) together with the
multi-step stability guarantee.  It is generated deterministically so that the
whole figure set, including the overview, is reproducible from source.
"""
from __future__ import annotations

import argparse

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import PALETTE, plt, save_figure

OUT_FIGURE = "fig0_overview.pdf"

_TITLES = [
    "Product-formula defect",
    "Exact residual",
    "Local generator",
    "Learned, transferable",
]
_BODIES = [
    "$H=A+B$\nstep $S_q(\\delta t)\\neq U(\\delta t)$",
    "$R_q=U\\,S_q^{\\dagger}=e^{-iK_q}$\nunique, norm-optimal",
    "$K_2=\\delta t^3 K_2^{(3)}+\\mathcal{O}(\\delta t^5)$\nweight $\\leq 3$, local",
    "per-site network\ntrain $n{=}4,5\\to n\\leq 10$",
]


def _box(ax, x: float, y: float, w: float, h: float, accent: str, title: str, body: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.8, edgecolor=accent, facecolor="0.985", zorder=2,
        )
    )
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=accent, zorder=4)
    ax.text(x + w / 2, y + h * 0.40, body, ha="center", va="center", fontsize=10, zorder=4)


def _arrow(ax, x0: float, x1: float, y: float) -> None:
    ax.add_patch(
        FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18,
                        linewidth=1.8, color="0.4", zorder=1)
    )


def make_overview() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 3.0))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.0)
    ax.axis("off")
    w, h, y = 2.78, 1.9, 0.6
    xs = [0.12, 3.22, 6.32, 9.42]
    accents = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
    for x, accent, title, body in zip(xs, accents, _TITLES, _BODIES):
        _box(ax, x, y, w, h, accent, title, body)
    for i in range(3):
        _arrow(ax, xs[i] + w + 0.04, xs[i + 1] - 0.04, y + h / 2)
    ax.text(
        6.3, 0.24,
        "stability guarantee:  global error $\\leq r\\,\\eta$   "
        "(per-step residual error $\\eta$, $r$ steps)",
        ha="center", va="center", fontsize=9.5, color="0.3", style="italic",
    )
    save_figure(fig, OUT_FIGURE)


def main(force: bool = False) -> None:
    make_overview()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
