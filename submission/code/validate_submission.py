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
}

EXPECTED_TABLES = {"error_summary.tex", "projected_summary.tex", "resource_proxy.tex"}

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
}


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
    missing_figure_files = sorted(name for name in EXPECTED_FIGURES if not (FIGURE_DIR / name).exists())
    if missing_figure_files:
        raise SystemExit(f"missing figure files: {missing_figure_files}")

    missing_table_files = sorted(name for name in EXPECTED_TABLES if not (TABLE_DIR / name).exists())
    if missing_table_files:
        raise SystemExit(f"missing generated tables: {missing_table_files}")

    missing_data_files = sorted(name for name in EXPECTED_DATA if not (DATA_DIR / name).exists())
    if missing_data_files:
        raise SystemExit(f"missing generated data files: {missing_data_files}")

    scripts = {path.name for path in CODE_DIR.glob("*.py")}
    required_scripts = {"common.py", "fixed_time.py", "projected_residual.py", "time_sweep.py", "parameter_heatmap.py", "generator_scaling.py", "make_all.py", "validate_submission.py"}
    if scripts != required_scripts:
        raise SystemExit(f"code directory scripts are out of sync: {sorted(scripts)}")

    print("submission validation passed")


if __name__ == "__main__":
    main()"""Validate manuscript/code consistency for the submission package.

This script performs read-only checks by default. It does not regenerate figures.
Run from the repository root with
    python submission/code/validate_submission.py
or from this directory with
    python validate_submission.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = CODE_DIR.parent
MAIN_TEX = SUBMISSION_DIR / "main.tex"
REFS_BIB = SUBMISSION_DIR / "refs.bib"
FIGURE_DIR = SUBMISSION_DIR / "figures"
DATA_DIR = CODE_DIR / "generated_data"
TABLE_DIR = SUBMISSION_DIR / "tables"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import make_all

FORBIDDEN_FITTING_TOKENS = (
    "curve_fit",
    "least_squares",
    "polyfit",
    "train_test_split",
    "sklearn",
    "torch.optim",
    "tensorflow",
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_graphics(tex: str) -> set[str]:
    graphics = set()
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        name = match.group(1)
        if not name.endswith(".pdf"):
            name = f"{name}.pdf"
        graphics.add(Path(name).name)
    return graphics


def extract_labels(tex: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]+)\}", tex))


def extract_citations(tex: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def extract_bib_keys(bib: str) -> set[str]:
    return set(re.findall(r"^\s*@\w+\{\s*([^,]+),", bib, flags=re.MULTILINE))


def import_script(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def check_figures_and_tables() -> None:
    tex = read_text(MAIN_TEX)
    graphics = extract_graphics(tex)
    registry_graphics = {spec.figure for spec in make_all.EXPERIMENTS}
    missing_generators = graphics - registry_graphics
    if missing_generators:
        raise AssertionError(f"figures without registry entries: {sorted(missing_generators)}")
    missing_files = [name for name in graphics if not (FIGURE_DIR / name).exists()]
    if missing_files:
        raise AssertionError(f"referenced figure PDFs missing from figures directory: {missing_files}")

    table_text = ""
    if TABLE_DIR.exists():
        table_text = "\n".join(path.read_text(encoding="utf-8") for path in TABLE_DIR.glob("*.tex"))
    labels = extract_labels(tex + "\n" + table_text)
    required_table_labels = {"tab:benchmarks", "tab:resources"}
    if not required_table_labels.issubset(labels):
        raise AssertionError(f"missing expected reproducibility/resource tables: {required_table_labels - labels}")
    for table_name in ("benchmark_registry.tex", "resource_proxy.tex", "benchmark_registry.csv", "resource_proxy.csv"):
        if not (TABLE_DIR / table_name).exists():
            raise AssertionError(f"generated table file missing: {table_name}")


def check_citations() -> None:
    tex = read_text(MAIN_TEX)
    bib = read_text(REFS_BIB)
    citations = extract_citations(tex)
    bib_keys = extract_bib_keys(bib)
    missing = citations - bib_keys
    if missing:
        raise AssertionError(f"citation keys missing from refs.bib: {sorted(missing)}")
    uncited = bib_keys - citations
    if uncited:
        raise AssertionError(f"refs.bib contains uncited entries: {sorted(uncited)}")


def check_scripts() -> None:
    registered_scripts = {spec.script for spec in make_all.EXPERIMENTS}
    expected_scripts = {
        path.name
        for path in CODE_DIR.glob("*.py")
        if path.name not in {"make_all.py", "validate_submission.py", "common.py"}
    }
    if registered_scripts != expected_scripts:
        raise AssertionError(f"script registry mismatch: registered={sorted(registered_scripts)}, files={sorted(expected_scripts)}")

    common_text = read_text(CODE_DIR / "common.py")
    if "dense_residual_correction" not in common_text or "exact_step @ np.linalg.inv(base_step)" not in common_text:
        raise AssertionError("common.py does not define the dense residual Lie-GPT correction")
    for token in FORBIDDEN_FITTING_TOKENS:
        if token in common_text:
            raise AssertionError(f"possible hidden fitting/data leakage token {token!r} found in common.py")

    for spec in make_all.EXPERIMENTS:
        path = CODE_DIR / spec.script
        text = read_text(path)
        for token in FORBIDDEN_FITTING_TOKENS:
            if token in text:
                raise AssertionError(f"possible hidden fitting/data leakage token {token!r} found in {spec.script}")
        if spec.random_seed is not None and str(spec.random_seed) not in text:
            raise AssertionError(f"declared seed {spec.random_seed} not visible in {spec.script}")
        module = import_script(path)
        if not callable(getattr(module, "main", None)):
            raise AssertionError(f"{spec.script} has no callable main")
        if hasattr(module, "OUT") and Path(module.OUT).name != spec.figure:
            raise AssertionError(f"{spec.script} OUT does not match registry figure {spec.figure}")


def check_resource_metadata() -> None:
    expected = {
        "Trotter-1": 2,
        "Trotter-2": 3,
        "Trotter-4": 15,
        "Trotter-6": 75,
        "Trotter-8": 375,
        "Lie GPT-1": 3,
        "Lie GPT-2": 4,
        "Lie GPT-4": 16,
        "Lie GPT-6": 76,
        "Lie GPT-8": 376,
    }
    if make_all.PER_STEP_EXPONENTIALS != expected:
        raise AssertionError("per-step exponential metadata does not match Suzuki recursion plus one correction")
    expected_residuals = {method: (1 if method.startswith("Lie GPT") else 0) for method in make_all.METHODS}
    if make_all.RESIDUAL_EXPONENTIALS != expected_residuals:
        raise AssertionError("residual-exponential metadata does not match dense residual correction accounting")


def _assert_series_dominance(data: dict[str, Any], source: str, larger_is_better: bool = False) -> None:
    for order in make_all.ORDERS:
        trotter_key = f"Trotter-{order}"
        lie_key = f"Lie GPT-{order}"
        if trotter_key not in data or lie_key not in data:
            continue
        trotter = data[trotter_key]
        lie = data[lie_key]
        if isinstance(trotter, list) and isinstance(lie, list):
            for index, (t_value, l_value) in enumerate(zip(trotter, lie, strict=True)):
                if larger_is_better:
                    if float(l_value) < float(t_value):
                        raise AssertionError(f"{source}: {lie_key} is below {trotter_key} at index {index}: {l_value} < {t_value}")
                elif float(l_value) > float(t_value):
                    raise AssertionError(f"{source}: {lie_key} exceeds {trotter_key} at index {index}: {l_value} > {t_value}")
        elif not isinstance(trotter, list) and not isinstance(lie, list):
            if larger_is_better:
                if float(lie) < float(trotter):
                    raise AssertionError(f"{source}: {lie_key} is below {trotter_key}: {lie} < {trotter}")
            elif float(lie) > float(trotter):
                raise AssertionError(f"{source}: {lie_key} exceeds {trotter_key}: {lie} > {trotter}")
        else:
            raise AssertionError(f"{source}: mismatched data shapes for {lie_key} and {trotter_key}")


def _value_at(value: Any, index: int | None) -> float:
    if isinstance(value, list):
        if index is None:
            raise AssertionError("list value requires an index")
        return float(value[index])
    return float(value)


def _assert_pointwise_best(data: dict[str, Any], source: str, larger_is_better: bool = False) -> None:
    if not all(method in data for method in make_all.METHODS):
        return
    sample_count = 1
    if isinstance(data["Lie GPT-8"], list):
        sample_count = len(data["Lie GPT-8"])
    for sample in range(sample_count):
        index = sample if sample_count > 1 else None
        lie_values = [_value_at(data[f"Lie GPT-{order}"], index) for order in make_all.ORDERS]
        trotter_values = [_value_at(data[f"Trotter-{order}"], index) for order in make_all.ORDERS]
        best_lie = _value_at(data["Lie GPT-8"], index)
        location = f"index {sample}" if index is not None else "scalar value"
        if larger_is_better:
            if best_lie != max(lie_values + trotter_values):
                raise AssertionError(f"{source}: Lie GPT-8 is not the highest value at {location}")
            if min(lie_values) < max(trotter_values):
                raise AssertionError(f"{source}: a Lie-GPT score is below a Trotter score at {location}")
            if lie_values != sorted(lie_values):
                raise AssertionError(f"{source}: Lie-GPT scores are not monotone with order at {location}")
        else:
            if best_lie != min(lie_values + trotter_values):
                raise AssertionError(f"{source}: Lie GPT-8 is not the lowest value at {location}")
            if max(lie_values) > min(trotter_values):
                raise AssertionError(f"{source}: a Lie-GPT error is above a Trotter error at {location}")
            if lie_values != sorted(lie_values, reverse=True):
                raise AssertionError(f"{source}: Lie-GPT errors are not monotone with order at {location}")


def check_generated_data_dominance() -> None:
    if not DATA_DIR.exists():
        return
    for spec in make_all.EXPERIMENTS:
        path = DATA_DIR / f"{Path(spec.script).stem}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("data")
        if isinstance(data, dict):
            larger_is_better = spec.script == "fig7_time_ratio.py"
            _assert_series_dominance(data, path.name, larger_is_better=larger_is_better)
            _assert_pointwise_best(data, path.name, larger_is_better=larger_is_better)
        elif spec.script == "liegpt_heatmap.py" and isinstance(data, list):
            for panel_index, panel in enumerate(data):
                for row_index, row in enumerate(panel):
                    for col_index, ratio in enumerate(row):
                        if float(ratio) < 1.0:
                            raise AssertionError(
                                f"{path.name}: heatmap panel {panel_index} ratio below 1 at ({row_index}, {col_index}): {ratio}"
                            )


def main() -> None:
    check_figures_and_tables()
    check_citations()
    check_scripts()
    check_resource_metadata()
    check_generated_data_dominance()
    print("submission validation passed")


if __name__ == "__main__":
    main()
