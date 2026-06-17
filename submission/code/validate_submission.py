from __future__ import annotations

from pathlib import Path

from common import CODE_DIR, DATA_DIR, FIGURE_DIR, SUBMISSION_DIR, TABLE_DIR, parse_bib_keys, parse_citation_keys, parse_graphics_references

MAIN_TEX = SUBMISSION_DIR / "main.tex"
REFS_BIB = SUBMISSION_DIR / "refs.bib"

EXPECTED_FIGURES = {
    "fig1_fixed_time_errors.pdf",
    "fig2_projected_residual.pdf",
    "fig3_time_sweep.pdf",
    "fig4_parameter_heatmap.pdf",
    "fig5_generator_compressibility.pdf",
    "fig6_generator_order_scaling.pdf",
    "fig7_learned_residual.pdf",
}

EXPECTED_FIGURE_PNGS = {Path(name).with_suffix(".png").name for name in EXPECTED_FIGURES}

EXPECTED_TABLES = {
    "error_summary.tex",
    "projected_summary.tex",
    "resource_proxy.tex",
}

EXPECTED_DATA = {
    "fixed_time_errors.csv",
    "fixed_time_errors.json",
    "fixed_time_errors.meta.json",
    "projected_residual_n5_q2.csv",
    "projected_residual_n5_q2.json",
    "projected_residual_n5_q2.meta.json",
    "time_sweep_n4_q2.csv",
    "time_sweep_n4_q2.json",
    "time_sweep_n4_q2.meta.json",
    "parameter_heatmap_n4_q2_w3.csv",
    "parameter_heatmap_n4_q2_w3.json",
    "parameter_heatmap_n4_q2_w3.meta.json",
    "generator_energy_n5_q2.json",
    "generator_order_scaling.csv",
    "generator_order_scaling.json",
    "generator_scaling.meta.json",
    "learned_residual_sizes.csv",
    "learned_residual_sizes.json",
    "learned_residual_dtsweep.csv",
    "learned_residual_dtsweep.json",
    "learned_residual_steps.csv",
    "learned_residual_steps.json",
    "learned_residual_parity.csv",
    "learned_residual.meta.json",
}

EXPECTED_SCRIPTS = {
    "common.py",
    "fixed_time.py",
    "projected_residual.py",
    "time_sweep.py",
    "parameter_heatmap.py",
    "generator_scaling.py",
    "make_all.py",
    "learned_residual.py",
    "validate_submission.py",
}


def exact_directory_contents(path: Path) -> set[str]:
    return {child.name for child in path.iterdir() if child.is_file()}


def main() -> None:
    tex = MAIN_TEX.read_text(encoding="utf-8")
    bib = REFS_BIB.read_text(encoding="utf-8")

    cited = parse_citation_keys(tex)
    bib_keys = parse_bib_keys(bib)
    missing_citations = cited - bib_keys
    if missing_citations:
        raise SystemExit(f"missing citation keys in refs.bib: {sorted(missing_citations)}")

    referenced_figures = parse_graphics_references(tex)
    if referenced_figures != EXPECTED_FIGURES:
        raise SystemExit(f"main.tex figure references do not match expected artifact: {sorted(referenced_figures)}")

    figure_files = exact_directory_contents(FIGURE_DIR)
    expected_figure_files = EXPECTED_FIGURES | EXPECTED_FIGURE_PNGS
    if figure_files != expected_figure_files:
        raise SystemExit(
            "figure directory contents are out of sync: "
            f"expected {sorted(expected_figure_files)}, found {sorted(figure_files)}"
        )

    table_files = exact_directory_contents(TABLE_DIR)
    if table_files != EXPECTED_TABLES:
        raise SystemExit(
            "table directory contents are out of sync: "
            f"expected {sorted(EXPECTED_TABLES)}, found {sorted(table_files)}"
        )

    data_files = exact_directory_contents(DATA_DIR)
    if data_files != EXPECTED_DATA:
        raise SystemExit(
            "generated_data contents are out of sync: "
            f"expected {sorted(EXPECTED_DATA)}, found {sorted(data_files)}"
        )

    scripts = {path.name for path in CODE_DIR.glob("*.py")}
    if scripts != EXPECTED_SCRIPTS:
        raise SystemExit(
            "code directory scripts are out of sync: "
            f"expected {sorted(EXPECTED_SCRIPTS)}, found {sorted(scripts)}"
        )

    print("submission validation passed")


if __name__ == "__main__":
    main()