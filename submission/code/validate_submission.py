"""Reproducibility gate: verify the generated artifact set is consistent with main.tex.

Cross-checks that (a) every ``\\cite`` key in main.tex resolves to an inline
``\\bibitem`` in the same file (the manuscript uses an inline bibliography, not a
separate refs.bib), (b) every figure referenced by main.tex exists in the figure
directory, and (c) every required dataset, table and figure produced by the
pipeline is present.  Run it after make_all.py; it exits non-zero on any
mismatch.  Membership is checked as a subset (all required artifacts present),
so raster duplicates or auxiliary outputs do not cause spurious failures.
"""
from __future__ import annotations

import re
from pathlib import Path

from common import (
    CODE_DIR,
    DATA_DIR,
    FIGURE_DIR,
    SUBMISSION_DIR,
    TABLE_DIR,
    parse_citation_keys,
    parse_graphics_references,
)

MAIN_TEX = SUBMISSION_DIR / "main.tex"

REQUIRED_FIGURES = {
    "fig0_overview.pdf",
    "fig1_exact_residual.pdf",
    "fig2_compressed_residual.pdf",
    "fig3_time_sweep.pdf",
    "fig4_generator_structure.pdf",
    "fig5_learned_transfer.pdf",
    "fig6_headline_improvement.pdf",
    "fig7_frontier.pdf",
    "fig8_spectral_truncation.pdf",
}

REQUIRED_TABLES = {
    "error_summary.tex",
    "projected_summary.tex",
    "resource_proxy.tex",
    "learned_residual_summary.tex",
    "order_generality.tex",
    "frontier.tex",
    "xxz_generality.tex",
    "oracle_free_q4.tex",
    "spectral_truncation_rate.tex",
    "faithful_compilation.tex",
    "generator_scaling.tex",
    "learned_breakdown.tex",
    "learned_dtsweep.tex",
}

REQUIRED_DATA = {
    "fixed_time_errors.csv",
    "projected_residual_n5_q2.csv",
    "time_sweep_n4_q2.csv",
    "generator_order_scaling.csv",
    "order_generality.csv",
    "frontier_cnot.csv",
    "compiled_faithfulness.csv",
    "xxz_generality.csv",
    "spectral_truncation_rate.csv",
    "faithful_compilation.csv",
    "certificate_selection.csv",
    "oracle_free_q4_frontier.csv",
    "learned_residual_sizes.csv",
}


def parse_bibitem_keys(tex: str) -> set[str]:
    return set(re.findall(r"\\bibitem\{([^}]+)\}", tex))


def present(path: Path) -> set[str]:
    return {child.name for child in path.iterdir() if child.is_file()} if path.exists() else set()


def main() -> None:
    tex = MAIN_TEX.read_text(encoding="utf-8")

    cited = parse_citation_keys(tex)
    defined = parse_bibitem_keys(tex)
    missing = cited - defined
    if missing:
        raise SystemExit(f"\\cite keys without a matching \\bibitem: {sorted(missing)}")

    referenced = parse_graphics_references(tex)
    figs = present(FIGURE_DIR) | present(CODE_DIR / "figures")
    missing_figs = referenced - figs
    if missing_figs:
        raise SystemExit(f"figures referenced by main.tex but absent from the figure set: {sorted(missing_figs)}")

    for label, required, where in (
        ("figures", REQUIRED_FIGURES, figs),
        ("tables", REQUIRED_TABLES, present(TABLE_DIR)),
        ("datasets", REQUIRED_DATA, present(DATA_DIR)),
    ):
        gap = required - where
        if gap:
            raise SystemExit(f"required {label} missing from the regenerated artifact set: {sorted(gap)}")

    print(
        f"submission validation passed: {len(cited)} citations resolve, "
        f"{len(referenced)} referenced figures present, "
        f"{len(REQUIRED_TABLES)} tables and {len(REQUIRED_DATA)} datasets regenerated."
    )


if __name__ == "__main__":
    main()
