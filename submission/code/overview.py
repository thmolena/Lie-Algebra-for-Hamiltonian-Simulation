"""Method-overview schematic -- the NMI 'Figure 1' convention.

This script renders the single conceptual display item of the manuscript: a
left-to-right diagram of the residual-generator Trotter compilation (RGTC)
pipeline.  It introduces no numerics; it is a programmatic matplotlib drawing
(boxes + arrows) saved as a vector PDF in the Nature Machine Intelligence house
style (Type-42 fonts, sans-serif, colour-blind-safe accents).

Pipeline (read left to right):

  local Hamiltonian H = A + B  ->  product-formula step S_q and exact residual
  R_q = U S_q^dagger  ->  Lie-algebra defect: Hermitian generator K_q = i log R_q
  that is geometrically local (Pauli weight <= 3 for the leading Strang term)
  ->  compression by Pauli weight or a learned per-site network for K-hat
  ->  spectral-norm error vs the exact propagator U(t), with the proven r*eta
  multi-step stability certificate.

Output: figures/fig0_overview.{pdf,png}.
"""
from __future__ import annotations

import argparse

from common import FIGURE_DIR, fig_schematic

OUT_FIGURE = "fig0_overview.pdf"


def main(force: bool = False) -> None:
    fig_schematic(FIGURE_DIR / OUT_FIGURE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
