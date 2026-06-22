"""One-command, deterministic regeneration of every figure, table and data file.

`python make_all.py` runs, in order: the four dense-matrix experiments (fixed-time
exact-residual, compressed projected residual, time-sweep, generator-structure), the
resource-proxy table, the torch-based learned-residual operator-learning experiment,
and the headline comparison figure.  Every step reads its inputs from the TFIM
definition in common.py and a fixed random seed, so reruns reproduce the same
results; the learned experiment and headline figure are wrapped in try/except so the
dense-matrix figures still build in a torch-free setup.

Afterwards run `python validate_submission.py` to check the artifact set.  See
README.txt for environment setup and expected runtime.
"""
from __future__ import annotations

import argparse

from common import ORDERS, TABLE_DIR, ensure_directories, write_latex_table
import fixed_time
import generator_scaling
import overview
import projected_residual
import time_sweep


def write_resource_table() -> None:
    factor_counts = {1: 2, 2: 3}
    for order in (4, 6, 8):
        factor_counts[order] = 5 * factor_counts[order - 2]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Per-step resource proxy. The baseline factor count is the number of elementary $e^{-iA\tau}$ or $e^{-iB\tau}$ exponentials in $S_q$. The oracle residual construction uses one additional dense exponential for $U(\delta t)$ and one dense multiplication by $S_q^\dagger$; it is not a scalable gate-level implementation. All entries are exact integer counts, not measured quantities.}",
        r"\label{tab:resources}",
        r"\centering",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$q$ & Suzuki factors & dense $U(\delta t)$ expm & dense residual multiply\\",
        r"\midrule",
    ]
    for order in ORDERS:
        lines.append(f"{order} & {factor_counts[order]} & 1 & 1\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_latex_table(TABLE_DIR / "resource_proxy.tex", lines)


def main(force: bool = False) -> None:
    ensure_directories()
    overview.main(force=force)
    fixed_time.main(force=force)
    projected_residual.main(force=force)
    time_sweep.main(force=force)
    generator_scaling.main(force=force)
    write_resource_table()
    # Learned-residual operator-learning experiment (requires torch).  Kept
    # optional so the dense-matrix figures still build in a torch-free setup.
    try:
        import learned_residual

        learned_residual.main(force=force)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"skipping learned_residual experiment: {exc}")
    # Headline comparison figure (reads the learned-residual data table).
    try:
        import headline

        headline.main(force=force)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"skipping headline figure: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)