===============================================================================
 RESIDUAL-GENERATOR TROTTER COMPILATION (RGTC) -- REPRODUCIBILITY GUIDE
===============================================================================

This folder regenerates, from first principles and with no hand-entered numbers,
every figure, table, and data file used in the manuscript (../main.tex).  All
quantities come from deterministic dense-matrix linear algebra applied to the
transverse-field Ising model (TFIM) defined in common.py; the one learned model
is trained with a fixed random seed.

If you want the mathematics derived from scratch -- the theory of computation
framing, the linear algebra, the Trotter--Suzuki formulas, the residual
generator, the locality argument, and the machine-learning setup -- read the
companion file THEORY.txt.  Every concept there names the exact function in this
folder that implements it.

-------------------------------------------------------------------------------
1.  QUICK START
-------------------------------------------------------------------------------

    # from this folder (submission/code)
    python -m pip install -r requirements.txt      # one-time
    python make_all.py                             # regenerate everything
    python validate_submission.py                  # check the artifact set

`make_all.py` writes:
    ../figures/*.pdf and *.png      (8 figures)
    ../tables/*.tex                 (4 tables)
    generated_data/*.csv, *.json    (raw data + run metadata)

`validate_submission.py` prints "submission validation passed" and exits 0 when
the generated artifact set is exactly what the manuscript references.

To rebuild the PDF afterwards (requires a LaTeX engine such as tectonic):

    cd ..  &&  tectonic main.tex

-------------------------------------------------------------------------------
2.  ENVIRONMENT
-------------------------------------------------------------------------------

Tested with CPython 3.13.12 and the versions pinned in requirements.txt
(numpy 2.4.3, scipy 1.17.1, pandas 3.0.2, matplotlib 3.10.8, torch 2.11.0).

Only learned_residual.py needs PyTorch.  make_all.py imports it inside a
try/except, so in a torch-free environment the six dense-matrix figures and the
overview schematic still build; only the learned-residual figure/table/data are
skipped (with a printed notice).

macOS note: learned_residual.py sets KMP_DUPLICATE_LIB_OK=TRUE and
OMP_NUM_THREADS=1 internally to avoid a conda/OpenMP duplicate-runtime crash and
to keep training deterministic.  No external environment variables are required.

Runtime (laptop CPU): the dense-matrix experiments finish in ~1-2 minutes; the
learned-residual experiment dominates at roughly 12-30 minutes depending on the
machine (it trains two networks and evaluates dense propagators up to n = 10,
a 1024-dimensional Hilbert space).

-------------------------------------------------------------------------------
3.  WHAT EACH SCRIPT PRODUCES
-------------------------------------------------------------------------------

common.py              Shared toolkit imported by everything else: Pauli
                       operators, the TFIM Hamiltonian, Trotter--Suzuki steps,
                       exact propagator, residual factor/generator, Pauli
                       projection, spectral-norm error, deterministic I/O, and
                       publication figure styling.  Produces no artifact itself.

overview.py            Figure 1: conceptual schematic of the RGTC framework
                       (fig0_overview).  Pure diagram, no data.

fixed_time.py          Experiment 1 -> Table I (error_summary.tex) and the
                       fixed-time figure (fig1): exact residual cancels the
                       Trotter error to the floating-point floor.

projected_residual.py  Experiment 2 -> Table II (projected_summary.tex) and the
                       projected figure (fig2): weight-truncated residual; the
                       sharp gain at Pauli weight 3.

time_sweep.py          Experiment 3 -> the time-sweep figure (fig3): correction
                       hierarchy (baseline / w<=2 / w<=3 / oracle) vs total time.

parameter_heatmap.py   Experiment 4 -> the heatmap figure (fig4): improvement
                       ratio over a 64-point (J,h) grid.

generator_scaling.py   Experiments 5 & 6 -> the compressibility figure (fig5)
                       and the order-scaling figure (fig6): Pauli-mass by weight
                       and ||K_q|| vs step size.

learned_residual.py    The operator-learning experiment -> the learned figure
                       (fig7, four panels) and Table III
                       (learned_residual_summary.tex): one translation-
                       equivariant network predicts the local weight-<=3
                       coefficients of K_2 and transfers from n=4,5 to n<=10.

make_all.py            Runs all of the above in order and writes the resource
                       table (Table IV, resource_proxy.tex).

validate_submission.py Checks that the figures, tables, data files, scripts, and
                       citation/figure references are exactly consistent with
                       the manuscript.

-------------------------------------------------------------------------------
4.  DETERMINISM AND EXACT REPRODUCTION
-------------------------------------------------------------------------------

* The dense-matrix experiments use fixed parameter grids and no randomness, so
  they reproduce identically on any machine (up to floating-point-floor digits).

* The learned-residual experiment is seeded once (SEED = 20240517 in
  learned_residual.py) for the disorder sampler, the train/test split, and the
  network initialisation, and runs single-threaded.  Re-running reproduces the
  reported means exactly (for example R^2 = 0.9998 and the 40-45x error
  reduction).

* Honest caveat on "exact": only quantities at the double-precision floor can
  differ in their last digits across BLAS builds -- the oracle residuals
  (~1e-13 to 1e-15 in Table I), the weight-5 row of Table II, and the M_0' mass
  in the appendix.  These are numerically zero / round-off and are explicitly
  flagged as such in the manuscript.  Every physically meaningful number is
  stable.

* Reproducibility is self-checking: each data file is written next to a
  *.meta.json recording the parameters, the library versions, the platform, and
  the git commit, and validate_submission.py fails if any artifact is missing,
  extra, or mismatched.  Figures are always drawn from the saved raw data, never
  from hard-coded arrays.

-------------------------------------------------------------------------------
5.  RELATIONSHIP TO THE PAPER
-------------------------------------------------------------------------------

main.tex (one directory up) consumes only the files this folder generates:
\input{tables/*.tex} for the four tables and \includegraphics of the eight
figures.  The numbers quoted in the manuscript prose are the same numbers in the
CSV/JSON here.  THEORY.txt connects every theorem and experiment in the paper to
the function that realises it, so the code can be read as an executable version
of the manuscript's mathematics.
