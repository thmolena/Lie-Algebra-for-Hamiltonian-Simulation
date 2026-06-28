"""Experiment -- spectral truncation of the residual generator: convergence rate,
symmetric faithful compilation, and certificate-driven bandwidth selection.

This module supplies the manuscript's central methodological advance: the
Pauli-weight filtration of the residual generator is a genuine *spectral
truncation* of the Lie algebra of Hermitian operators (Definition,
``def:spectral-truncation``), and -- unlike an operator-algebraic spectral
truncation that only preserves a structural property and converges in the limit
-- this one carries (i) a quantitative geometric *convergence rate* in the
truncation level, (ii) an a priori *dynamical* simulation certificate at every
level that lets the truncation bandwidth be chosen from the target accuracy
alone, and (iii) a *symmetric faithful compilation* that repairs the first-order
inner-compilation floor at every base order.  Every number is an exact,
deterministic dense-matrix computation; nothing is sampled or hand-entered.

Outputs (under generated_data/, tables/, figures/):
  spectral_truncation_rate.{csv,json,meta.json}
  faithful_compilation.{csv,json,meta.json}
  certificate_selection.{csv,json}
  tables/spectral_truncation_rate.tex, tables/faithful_compilation.tex
  figures/fig8_spectral_truncation.{pdf}
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.linalg import expm

from common import (
    COL_DOUBLE,
    DATA_DIR,
    PALETTE,
    TABLE_DIR,
    apply_nmi_style,
    exact_step,
    pauli_basis,
    pauli_string,
    panel_label,
    plt,
    product_formula,
    project_pauli_weight,
    residual_factor,
    residual_generator,
    save_dataframe,
    save_figure,
    save_metadata,
    scientific,
    spectral_error,
    tfim_terms,
    write_latex_table,
)
from higher_order_frontier import (
    commuting_layers,
    correction_cnots,
    count_A,
    kept_terms,
)

OUT_FIGURE = "fig8_spectral_truncation.pdf"
TERM_TOL = 1e-10


# --------------------------------------------------------------------------- #
# Compilation of a compressed generator into commuting Pauli-rotation layers.
# --------------------------------------------------------------------------- #
def _layer_matrices(terms, n: int):
    mats = []
    for layer in commuting_layers(terms):
        M = np.zeros((2 ** n, 2 ** n), dtype=complex)
        for word, _weight, coeff in layer:
            M += coeff * pauli_string(word)
        mats.append(M)
    return mats


def compiled_first_order(terms, n: int) -> np.ndarray:
    """exp(-i sum_i L_i) as the ordered product prod_i exp(-i L_i) (Theorem faithful-compilation)."""
    U = np.eye(2 ** n, dtype=complex)
    for M in _layer_matrices(terms, n):
        U = expm(-1j * M) @ U
    return U


def compiled_symmetric(terms, n: int) -> np.ndarray:
    """Symmetric (Strang-ordered) inner product formula over the commuting layers:
    prod_i exp(-i L_i/2) . prod_i exp(-i L_i/2) reversed.  Compilation error O(Lambda^3)."""
    mats = _layer_matrices(terms, n)
    halves = [expm(-0.5j * M) for M in mats]
    U = np.eye(2 ** n, dtype=complex)
    for E in halves:
        U = E @ U
    for E in reversed(halves):
        U = E @ U
    return U


def symmetric_cnots(terms, n: int) -> int:
    """Two-qubit-gate count of the symmetric compile: each layer is applied twice with
    half-angle, except the middle layer whose two half-exponentials merge into one."""
    layers = commuting_layers(terms)
    layer_cost = [sum(2 * (weight - 1) for _word, weight, _coeff in layer) for layer in layers]
    if not layer_cost:
        return 0
    return int(2 * sum(layer_cost) - layer_cost[-1])


# --------------------------------------------------------------------------- #
# (1) Geometric convergence rate of the spectral truncation.
# --------------------------------------------------------------------------- #
def truncation_rate(n: int = 6, t: float = 1.0, r: int = 10, J: float = 1.0, h: float = 1.0,
                    orders=(2, 4, 6)) -> tuple[pd.DataFrame, dict]:
    terms = tfim_terms(n, J, h)
    A, B, H = terms.A, terms.B, terms.H
    dt = t / r
    U_total = exact_step(H, t)
    rows = []
    rates: dict[str, float] = {}
    for q in orders:
        _, S_dt, R_dt, _ = residual_factor(A, B, H, dt, q)
        K = residual_generator(R_dt)
        baseline = spectral_error(U_total, np.linalg.matrix_power(S_dt, r))
        errs = {}
        for w in range(0, n + 1):
            Kw = project_pauli_weight(K, n, w)
            tail = float(np.linalg.norm(K - Kw, ord=2))
            err = spectral_error(U_total, np.linalg.matrix_power(expm(-1j * Kw) @ S_dt, r))
            errs[w] = err
            rows.append({"q": q, "w": w, "baseline_error": baseline,
                         "truncation_tail_2norm": tail, "projected_error": err})
        # geometric-regime rate: log10 slope of error per added weight level, from the
        # leading effective threshold w0 = ceil(q/2)+2 up to just above the round-off floor.
        w0 = int(np.ceil(q / 2)) + 2
        ws = [w for w in range(w0, n) if errs[w] > 1e-13 and errs[w - 1] > 1e-13]
        if len(ws) >= 2:
            slope = float(np.polyfit(np.array(ws, dtype=float),
                                     np.log10([errs[w] for w in ws]), 1)[0])
            rates[f"q{q}_log10_slope_per_level"] = slope
            rates[f"q{q}_factor_per_level"] = float(10.0 ** (-slope))
            rates[f"q{q}_w0"] = w0
    return pd.DataFrame(rows), rates


# --------------------------------------------------------------------------- #
# (2) Symmetric vs first-order faithful compilation.
# --------------------------------------------------------------------------- #
def faithful_compilation(ns=(5, 6), t: float = 1.0, r: int = 10, J: float = 1.0, h: float = 1.0):
    dt = t / r
    rows = []
    for n in ns:
        terms = tfim_terms(n, J, h)
        A, B, H = terms.A, terms.B, terms.H
        U_total = exact_step(H, t)
        for q in (2, 4):
            _, S_dt, R_dt, _ = residual_factor(A, B, H, dt, q)
            K = residual_generator(R_dt)
            base_cnots = count_A(q) * 2 * (n - 1)
            for w in range(q + 1, q + 3):
                kt = kept_terms(K, n, w, tol=TERM_TOL)
                Kw = project_pauli_weight(K, n, w)
                err_oracle = spectral_error(U_total, np.linalg.matrix_power(expm(-1j * Kw) @ S_dt, r))
                err_c1 = spectral_error(U_total, np.linalg.matrix_power(compiled_first_order(kt, n) @ S_dt, r))
                err_c2 = spectral_error(U_total, np.linalg.matrix_power(compiled_symmetric(kt, n) @ S_dt, r))
                c1 = correction_cnots(kt)
                c2 = symmetric_cnots(kt, n)
                rows.append({
                    "n": n, "q": q, "w": w, "n_terms": len(kt),
                    "n_layers": len(commuting_layers(kt)),
                    "oracle_error": err_oracle,
                    "first_order_error": err_c1, "symmetric_error": err_c2,
                    "first_order_cnots": base_cnots + c1, "symmetric_cnots": base_cnots + c2,
                    "first_order_overhead": err_c1 / err_oracle if err_oracle > 0 else float("inf"),
                    "symmetric_overhead": err_c2 / err_oracle if err_oracle > 0 else float("inf"),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (3) Certificate-driven bandwidth selection.
# --------------------------------------------------------------------------- #
def certificate_selection(n: int = 6, q: int = 2, t: float = 1.0, r: int = 10,
                          J: float = 1.0, h: float = 1.0,
                          targets=(1e-3, 1e-4, 1e-6, 1e-8, 1e-10)) -> pd.DataFrame:
    terms = tfim_terms(n, J, h)
    A, B, H = terms.A, terms.B, terms.H
    dt = t / r
    U_total = exact_step(H, t)
    _, S_dt, R_dt, _ = residual_factor(A, B, H, dt, q)
    K = residual_generator(R_dt)
    tail = []
    achieved = []
    for w in range(0, n + 1):
        Kw = project_pauli_weight(K, n, w)
        tail.append(float(np.linalg.norm(K - Kw, ord=2)))
        achieved.append(spectral_error(U_total, np.linalg.matrix_power(expm(-1j * Kw) @ S_dt, r)))
    rows = []
    for eps in targets:
        wstar = next((w for w in range(0, n + 1) if r * tail[w] <= eps), None)
        if wstar is None:
            continue
        rows.append({"target_eps": eps, "w_star": wstar,
                     "certificate_bound": r * tail[wstar],
                     "achieved_error": achieved[wstar],
                     "bound_holds": bool(achieved[wstar] <= eps)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def make_plot(rate: pd.DataFrame, fa: pd.DataFrame, cert: pd.DataFrame,
              rates: dict, n_show: int = 6) -> None:
    apply_nmi_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.5))
    axc, axb, axs = axes

    # (a) geometric convergence rate
    markers = {2: "o", 4: "s", 6: "D"}
    for q in (2, 4, 6):
        d = rate[rate["q"] == q].sort_values("w")
        ax_err = np.clip(d["projected_error"].to_numpy(), 1e-16, None)
        axc.semilogy(d["w"], ax_err, marker=markers[q], ms=3.2,
                     color=PALETTE[{2: 0, 4: 1, 6: 2}[q]], label=f"$q={q}$")
    axc.set_xlabel("truncation level $w$")
    axc.set_ylabel("global error")
    axc.legend(loc="upper right", fontsize=6)
    axc.grid(True, which="both", alpha=0.25)
    axc.set_title("spectral truncation: geometric rate", fontsize=7)
    panel_label(axc, "a")

    # (b) first-order vs symmetric compilation (q=2 repair)
    d = fa[(fa["n"] == n_show) & (fa["q"] == 2)].sort_values("w")
    x = np.arange(len(d))
    width = 0.26
    axb.bar(x - width, d["oracle_error"], width, color=PALETTE[2], label="oracle")
    axb.bar(x, d["first_order_error"], width, color=PALETTE[1], label="first-order")
    axb.bar(x + width, d["symmetric_error"], width, color=PALETTE[0], label="symmetric")
    axb.set_yscale("log")
    axb.set_xticks(x)
    axb.set_xticklabels([f"$w{{=}}{int(w)}$" for w in d["w"]])
    axb.set_ylabel("global error")
    axb.legend(loc="upper right", fontsize=6)
    axb.set_title(f"faithful compilation ($q{{=}}2$, $n{{=}}{n_show}$)", fontsize=7)
    axb.grid(True, which="both", axis="y", alpha=0.25)
    panel_label(axb, "b")

    # (c) certificate vs achieved error
    cc = cert.sort_values("w_star")
    axs.loglog(cc["certificate_bound"], cc["achieved_error"], "o-", ms=3.2, color=PALETTE[3])
    lo = float(min(cc["achieved_error"].min(), cc["certificate_bound"].min())) * 0.3
    hi = float(max(cc["achieved_error"].max(), cc["certificate_bound"].max())) * 3.0
    axs.plot([lo, hi], [lo, hi], "--", color="#888888", lw=0.8, label="$y=x$")
    axs.set_xlabel("a priori certificate $r\\,\\Vert K_q-\\Pi_w K_q\\Vert_2$")
    axs.set_ylabel("achieved error")
    axs.legend(loc="upper left", fontsize=6)
    axs.grid(True, which="both", alpha=0.25)
    axs.set_title("certificate is a valid upper bound", fontsize=7)
    panel_label(axs, "c")

    fig.tight_layout(pad=0.7)
    save_figure(fig, OUT_FIGURE)


# --------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------- #
def write_rate_table(rate: pd.DataFrame, rates: dict) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{\textbf{Geometric convergence rate of the Lie-algebraic spectral truncation "
        r"(open-boundary \tfim{}, $n=6$, $J=h=1$, $t=1$, $r=10$).} "
        r"Global spectral-norm error of the level-$w$ truncated residual compilation $\widehat G_{q,w}$ "
        r"versus truncation level $w$, for base orders $q=2,4,6$. Beyond the effective threshold "
        r"$w_0=\lceil q/2\rceil+2$ the error decays geometrically; the last column reports the measured "
        r"per-level reduction factor (fitted log-slope), far steeper than the provable geometric rate "
        r"($\ge\dt^{-1}$ per level, Theorem~\ref{thm:spectral-rate}) because of symmetric cancellations. "
        r"The $q=6$ truncation reaches the double-precision floor within one level of its threshold at this "
        r"size, so its rate is not resolvable here. Exact, deterministic dense-matrix computations.}",
        r"\label{tab:spectral-rate}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"$q$ & $w_0$ & $w\le2$ & $w\le3$ & $w\le4$ & $w\le5$ & $w\le6$ & factor/level\\",
        r"\midrule",
    ]
    for q in (2, 4, 6):
        d = rate[rate["q"] == q]

        def cell(w):
            v = d[d["w"] == w]
            return scientific(float(v["projected_error"].iloc[0])) if len(v) else "--"

        w0 = int(np.ceil(q / 2)) + 2
        factor = rates.get(f"q{q}_factor_per_level", float("nan"))
        # At n=6 the q=6 truncation reaches the double-precision floor within one level
        # of its threshold, so a multi-point geometric rate is not resolvable at this size.
        factor_cell = f"${factor:.0f}\\times$" if np.isfinite(factor) else "--"
        lines.append(
            f"{q} & {w0} & ${cell(2)}$ & ${cell(3)}$ & ${cell(4)}$ & ${cell(5)}$ & ${cell(6)}$ & "
            f"{factor_cell}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    write_latex_table(TABLE_DIR / "spectral_truncation_rate.tex", lines)


def write_compilation_table(fa: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\caption{\textbf{Symmetric faithful compilation repairs the first-order inner-compilation floor "
        r"(open-boundary \tfim{}, $J=h=1$, $t=1$, $r=10$).} For the compressed weight-$\le w$ residual "
        r"correction of an order-$q$ step, global spectral-norm error of the dense oracle, the first-order "
        r"compiled product (Theorem~\ref{thm:faithful-compilation}), and the symmetric compiled product "
        r"(Theorem~\ref{thm:symmetric-compilation}), with two-qubit-gate (CNOT) counts. The first-order compile "
        r"of the $q=2$ correction stalls far above its oracle; the symmetric compile recovers the oracle "
        r"accuracy at $\approx2\times$ the correction gate count. At $q=4$ the first-order compile is already "
        r"faithful. Exact, deterministic dense-matrix computations.}",
        r"\label{tab:faithful-compilation}",
        r"\centering",
        r"\begin{tabular}{cccccccc}",
        r"\toprule",
        r"$n$ & $q$ & $w$ & oracle & first-order & symmetric & CNOT$_1$ & CNOT$_2$\\",
        r"\midrule",
    ]
    for _, row in fa.sort_values(["n", "q", "w"]).iterrows():
        lines.append(
            f"{int(row['n'])} & {int(row['q'])} & {int(row['w'])} & "
            f"${scientific(float(row['oracle_error']))}$ & ${scientific(float(row['first_order_error']))}$ & "
            f"${scientific(float(row['symmetric_error']))}$ & "
            f"{int(row['first_order_cnots'])} & {int(row['symmetric_cnots'])}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_latex_table(TABLE_DIR / "faithful_compilation.tex", lines)


def main(force: bool = False) -> dict:
    rate, rates = truncation_rate()
    fa = faithful_compilation()
    cert = certificate_selection()

    save_dataframe(rate, "spectral_truncation_rate.csv", "spectral_truncation_rate.json")
    save_dataframe(fa, "faithful_compilation.csv", "faithful_compilation.json")
    save_dataframe(cert, "certificate_selection.csv", "certificate_selection.json")
    save_metadata("spectral_truncation_rate.meta.json",
                  {"parameters": {"n": 6, "orders": [2, 4, 6], "J": 1.0, "h": 1.0, "t": 1.0, "r": 10},
                   "fitted_rates": rates, "force": force})
    save_metadata("faithful_compilation.meta.json",
                  {"parameters": {"ns": [5, 6], "orders": [2, 4], "J": 1.0, "h": 1.0, "t": 1.0, "r": 10,
                                  "inner_formulas": ["first-order ordered product", "symmetric Strang product"]},
                   "force": force})

    write_rate_table(rate, rates)
    write_compilation_table(fa)
    make_plot(rate, fa, cert, rates)

    # main.tex resolves graphics from code/figures/; mirror the figure there.
    import shutil
    from common import CODE_DIR, FIGURE_DIR
    code_fig = CODE_DIR / "figures"
    src = FIGURE_DIR / OUT_FIGURE
    if src.exists() and src.resolve() != (code_fig / OUT_FIGURE).resolve():
        code_fig.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, code_fig / OUT_FIGURE)
    return {"rate": rate, "rates": rates, "faithful": fa, "certificate": cert}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = main(force=args.force)
    print("\n=== fitted geometric rates ===")
    print(out["rates"])
    print("\n=== faithful compilation (oracle / first-order / symmetric) ===")
    print(out["faithful"][["n", "q", "w", "oracle_error", "first_order_error", "symmetric_error",
                           "first_order_cnots", "symmetric_cnots"]].to_string(index=False))
    print("\n=== certificate-driven bandwidth selection ===")
    print(out["certificate"].to_string(index=False))
