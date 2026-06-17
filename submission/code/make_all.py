from __future__ import annotations

import argparse

from common import ORDERS, TABLE_DIR, ensure_directories, write_latex_table
import fixed_time
import generator_scaling
import parameter_heatmap
import projected_residual
import time_sweep


def write_resource_table() -> None:
    factor_counts = {1: 2, 2: 3}
    for order in (4, 6, 8):
        factor_counts[order] = 5 * factor_counts[order - 2]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Per-step resource proxy. The baseline factor count is the number of elementary $e^{-iA\tau}$ or $e^{-iB\tau}$ exponentials in $S_q$. The oracle residual construction uses one additional dense exponential for $U(\delta t)$ and one dense multiplication by $S_q^\dagger$; it is not a scalable gate-level implementation.}",
        r"\label{tab:resources}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{cccc}",
        r"$q$ & Suzuki factors & dense $U(\delta t)$ expm & dense residual multiply\\",
    ]
    for order in ORDERS:
        lines.append(f"{order} & {factor_counts[order]} & 1 & 1\\\\")
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}"])
    write_latex_table(TABLE_DIR / "resource_proxy.tex", lines)


def main(force: bool = False) -> None:
    ensure_directories()
    fixed_time.main(force=force)
    projected_residual.main(force=force)
    time_sweep.main(force=force)
    parameter_heatmap.main(force=force)
    generator_scaling.main(force=force)
    write_resource_table()
    # Learned-residual operator-learning experiment (requires torch).  Kept
    # optional so the dense-matrix figures still build in a torch-free setup.
    try:
        import learned_residual

        learned_residual.main(force=force)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"skipping learned_residual experiment: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)