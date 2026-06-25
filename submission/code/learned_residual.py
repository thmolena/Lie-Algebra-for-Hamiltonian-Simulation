"""Operator learning of the residual generator.

This experiment instantiates the learned-residual mode discussed in the paper.
Instead of computing the exact residual generator K_q(dt) from a dense matrix
logarithm of the full propagator (the global oracle), a small translation-
equivariant neural network predicts the low-weight Pauli coefficients of K_2
from the *local* couplings of a disordered transverse-field Ising chain.  The
predicted generator is exponentiated and used as a Trotter correction.

Because the leading Strang residual generator is supported on geometrically
local, weight-<=3 Pauli strings (proved in the paper), the same per-site network
applies at every site of a chain of any length, so a model trained on small
chains corrects larger chains it never saw.

This version adds three things beyond the basic demonstration:

  * Sparse Pauli operators, so reconstructions scale to larger systems
    (here up to n = 10 qubits, a 1024-dimensional Hilbert space) without
    materializing 4^n dense Pauli matrices.
  * An *oracle-free* training mode.  Labels are computed from the residual
    generator of a small fixed-size local patch (<= 7 qubits) instead of the
    global propagator, so no dense 2^n oracle is needed to train.  We show the
    oracle-free model matches the oracle-trained model downstream.
  * A non-learned, leading-order BCH baseline (the analytic delta t^3 residual
    generator), and a step-size axis on which leading-order BCH degrades while
    the delta t-conditioned learned model tracks the exact oracle.

All training labels and downstream error metrics use the same first-principles
dense-matrix primitives as the rest of the submission (`common.py`).

Outputs (generated_data/ and figures/):
  learned_residual_sizes.csv/json    error vs chain length (delta t = 0.1)
  learned_residual_dtsweep.csv/json  error vs step size (n = 6)
  learned_residual_steps.csv/json    error vs number of Trotter steps (n = 6)
  learned_residual_parity.csv        predicted vs exact coefficients (transfer)
  learned_residual.meta.json         configuration + environment
  fig5_learned_transfer.{pdf,png}    four-panel summary figure
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product

import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.patches import Patch
from scipy.linalg import expm

from common import (
    COL_DOUBLE,
    PALETTE,
    TABLE_DIR,
    apply_nmi_style,
    exact_step,
    line_plot_style,
    panel_label,
    plt,
    product_formula,
    repeated_step,
    residual_factor,
    residual_generator,
    save_dataframe,
    save_figure,
    save_metadata,
    scientific,
    shaded_band,
    spectral_error,
    write_latex_table,
)

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORDER = 2                       # Strang; the weight<=3 locality theorem is for q=2
MAX_WEIGHT = 3
DT_EVAL = 0.1                   # operating point for the size, stability, parity panels
DT_TRAIN_RANGE = (0.05, 0.30)   # the network is trained across this range of step sizes
DT_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DT0_BCH = 0.02                  # small step used to extract the leading-order generator
PATCH_RADIUS = 2                # margin (sites) around a 3-site support for oracle-free labels

J_RANGE = (0.5, 1.5)
H_RANGE = (0.5, 1.5)

TRAIN_SIZES = (4, 5)
TRANSFER_SIZES = (4, 5, 6, 7, 8, 9, 10)
EVAL_REALIZATIONS = {4: 40, 5: 40, 6: 40, 7: 40, 8: 24, 9: 16, 10: 12}
N_TRAIN_REALIZATIONS = 220
STEP_SWEEP_SIZE = 6
STEP_SWEEP_RS = (1, 2, 4, 6, 8, 10, 14, 18)
DTSWEEP_SIZE = 6
DTSWEEP_REALIZATIONS = 24

HIDDEN = 64
EPOCHS = 4000
LR = 3e-3
WEIGHT_DECAY = 1e-6
SEED = 20240517


# ---------------------------------------------------------------------------
# Sparse Pauli operators
# ---------------------------------------------------------------------------
_SINGLE = {
    "I": sp.identity(2, format="csr", dtype=complex),
    "X": sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
    "Y": sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
    "Z": sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex)),
}


@lru_cache(maxsize=None)
def pauli_sparse(word: str):
    out = _SINGLE[word[0]]
    for ch in word[1:]:
        out = sp.kron(out, _SINGLE[ch], format="csr")
    return out


def pauli_coefficient(word: str, K: np.ndarray, n_qubits: int) -> float:
    """tr(P K) / 2^n with P a Pauli string; O(2^n) via the sparse structure."""
    P = pauli_sparse(word).tocoo()
    val = np.sum(P.data * K[P.col, P.row])
    return float(np.real(val) / (2 ** n_qubits))


def add_pauli(K: np.ndarray, word: str, coeff: float) -> None:
    if coeff == 0.0:
        return
    P = pauli_sparse(word).tocoo()
    K[P.row, P.col] += coeff * P.data


# ---------------------------------------------------------------------------
# Disordered transverse-field Ising model
# ---------------------------------------------------------------------------
def tfim_terms_disordered(js: np.ndarray, hs: np.ndarray):
    """H = sum_i J_i Z_iZ_{i+1} + sum_i h_i X_i on an open chain (dense)."""
    n_qubits = len(hs)
    dimension = 2 ** n_qubits
    A = np.zeros((dimension, dimension), dtype=complex)
    B = np.zeros((dimension, dimension), dtype=complex)
    for site in range(n_qubits - 1):
        word = ["I"] * n_qubits
        word[site] = word[site + 1] = "Z"
        add_pauli(A, "".join(word), float(js[site]))
    for site in range(n_qubits):
        word = ["I"] * n_qubits
        word[site] = "X"
        add_pauli(B, "".join(word), float(hs[site]))
    return A, B, A + B


# ---------------------------------------------------------------------------
# Local Pauli templates (support inside a 3-site window, weight <= 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Template:
    offsets: tuple[int, ...]
    letters: tuple[str, ...]


def build_templates() -> list[Template]:
    letters3 = ("X", "Y", "Z")
    templates: list[Template] = []
    for a in letters3:
        templates.append(Template((0,), (a,)))
    for a, b in product(letters3, repeat=2):
        templates.append(Template((0, 1), (a, b)))
    for a, b in product(letters3, repeat=2):
        templates.append(Template((0, 2), (a, b)))
    for a, b, c in product(letters3, repeat=3):
        templates.append(Template((0, 1, 2), (a, b, c)))
    return templates


TEMPLATES = build_templates()
N_TEMPLATES = len(TEMPLATES)


def template_word(template: Template, anchor: int, n_qubits: int) -> str:
    word = ["I"] * n_qubits
    for off, letter in zip(template.offsets, template.letters):
        word[anchor + off] = letter
    return "".join(word)


def valid_anchor(template: Template, anchor: int, n_qubits: int) -> bool:
    return anchor + max(template.offsets) < n_qubits


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
N_LOCAL_FEATURES = 11
N_FEATURES = N_LOCAL_FEATURES + 1  # + step size


def anchor_features(js: np.ndarray, hs: np.ndarray, anchor: int, dt: float) -> np.ndarray:
    n_qubits = len(hs)

    def jval(idx: int) -> float:
        return float(js[idx]) if 0 <= idx < len(js) else 0.0

    def hval(idx: int) -> float:
        return float(hs[idx]) if 0 <= idx < n_qubits else 0.0

    feats = [
        jval(anchor - 1), jval(anchor), jval(anchor + 1), jval(anchor + 2),
        hval(anchor - 1), hval(anchor), hval(anchor + 1), hval(anchor + 2), hval(anchor + 3),
        1.0 if anchor <= 0 else 0.0,
        1.0 if anchor + 2 >= n_qubits - 1 else 0.0,
        dt,
    ]
    return np.asarray(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Exact residual generators (global oracle and local patch)
# ---------------------------------------------------------------------------
def global_generator(js: np.ndarray, hs: np.ndarray, dt: float) -> np.ndarray:
    A, B, H = tfim_terms_disordered(js, hs)
    _, _, R, _ = residual_factor(A, B, H, dt, ORDER)
    return residual_generator(R)


def coeffs_from_K(K: np.ndarray, n_qubits: int):
    """Exact local-template Pauli coefficients (targets) and validity mask."""
    targs = np.zeros((n_qubits, N_TEMPLATES))
    masks = np.zeros((n_qubits, N_TEMPLATES))
    for anchor in range(n_qubits):
        for t_idx, template in enumerate(TEMPLATES):
            if not valid_anchor(template, anchor, n_qubits):
                continue
            targs[anchor, t_idx] = pauli_coefficient(template_word(template, anchor, n_qubits), K, n_qubits)
            masks[anchor, t_idx] = 1.0
    return targs, masks


def patch_coeffs(js: np.ndarray, hs: np.ndarray, anchor: int, dt: float):
    """Oracle-free template coefficients for one anchor, computed from the
    residual generator of a small local patch (<= 2*PATCH_RADIUS+3 sites)."""
    n_qubits = len(hs)
    lo = max(0, anchor - PATCH_RADIUS)
    hi = min(n_qubits - 1, anchor + 2 + PATCH_RADIUS)
    hs_p = hs[lo:hi + 1]
    js_p = js[lo:hi]                      # bonds fully inside the patch
    K_p = global_generator(js_p, hs_p, dt)
    m = len(hs_p)
    row = np.zeros(N_TEMPLATES)
    mask = np.zeros(N_TEMPLATES)
    for t_idx, template in enumerate(TEMPLATES):
        if not valid_anchor(template, anchor, n_qubits):
            continue
        local_anchor = anchor - lo
        row[t_idx] = pauli_coefficient(template_word(template, local_anchor, m), K_p, m)
        mask[t_idx] = 1.0
    return row, mask


def realization_patch_coeffs(js, hs, dt):
    rows = np.zeros((len(hs), N_TEMPLATES))
    masks = np.zeros((len(hs), N_TEMPLATES))
    for anchor in range(len(hs)):
        rows[anchor], masks[anchor] = patch_coeffs(js, hs, anchor, dt)
    return rows, masks


def bch_leading_coeffs(js, hs, dt):
    """Leading-order analytic generator: coefficients are extracted at a small
    step and rescaled by (dt/dt0)^3, i.e. dt^3 K^{(3)}.  No learning, no global
    oracle (patch-local)."""
    rows, masks = realization_patch_coeffs(js, hs, DT0_BCH)
    return rows * (dt / DT0_BCH) ** 3, masks


# ---------------------------------------------------------------------------
# Generator assembly and oracle projections (sparse)
# ---------------------------------------------------------------------------
def assemble_generator(coeffs_by_anchor: np.ndarray, n_qubits: int) -> np.ndarray:
    dimension = 2 ** n_qubits
    K = np.zeros((dimension, dimension), dtype=complex)
    for anchor in range(n_qubits):
        for t_idx, template in enumerate(TEMPLATES):
            if not valid_anchor(template, anchor, n_qubits):
                continue
            add_pauli(K, template_word(template, anchor, n_qubits), coeffs_by_anchor[anchor, t_idx])
    return 0.5 * (K + K.conj().T)


def low_weight_words(n_qubits: int, max_weight: int):
    for weight in range(1, max_weight + 1):
        for sites in combinations(range(n_qubits), weight):
            for letters in product(("X", "Y", "Z"), repeat=weight):
                word = ["I"] * n_qubits
                for site, letter in zip(sites, letters):
                    word[site] = letter
                yield "".join(word)


def oracle_allweight_generator(K: np.ndarray, n_qubits: int, max_weight: int) -> np.ndarray:
    dimension = 2 ** n_qubits
    K_proj = np.zeros((dimension, dimension), dtype=complex)
    for word in low_weight_words(n_qubits, max_weight):
        add_pauli(K_proj, word, pauli_coefficient(word, K, n_qubits))
    return 0.5 * (K_proj + K_proj.conj().T)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LocalResidualNet(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def sample_couplings(rng, n_qubits):
    return rng.uniform(*J_RANGE, size=n_qubits - 1), rng.uniform(*H_RANGE, size=n_qubits)


def build_training_set(rng, label_kind: str, use_prior: bool = True):
    """label_kind: 'global' (dense oracle) or 'patch' (oracle-free local patches).
    Coefficients are divided by dt^3 to remove the leading scale; the network
    therefore predicts a smooth O(1) object and dt is an input feature.

    When use_prior is True the analytic leading-order (second-order Zassenhaus)
    generator -- the patch-local dt^3 term -- is subtracted from the target, so
    the network learns only the *residual beyond the analytic prior* (delta
    learning).  The prior is added back at prediction time.  Both the prior and
    the patch labels are oracle-free, so the 'patch' mode remains free of any
    dense 2^n propagator."""
    feats, targs, masks = [], [], []
    for n_qubits in TRAIN_SIZES:
        for _ in range(N_TRAIN_REALIZATIONS):
            js, hs = sample_couplings(rng, n_qubits)
            dt = float(rng.uniform(*DT_TRAIN_RANGE))
            if label_kind == "global":
                rows, mk = coeffs_from_K(global_generator(js, hs, dt), n_qubits)
            else:
                rows, mk = realization_patch_coeffs(js, hs, dt)
            if use_prior:
                prior, _ = bch_leading_coeffs(js, hs, dt)
                rows = rows - prior
            f = np.stack([anchor_features(js, hs, a, dt) for a in range(n_qubits)])
            feats.append(f)
            targs.append(rows / dt ** 3)
            masks.append(mk)
    return np.concatenate(feats), np.concatenate(targs), np.concatenate(masks)


def train_model(rng, label_kind: str, use_prior: bool = True):
    feats, targs, masks = build_training_set(rng, label_kind, use_prior)
    f_mean, f_std = feats.mean(0), feats.std(0) + 1e-8
    counts = masks.sum(0) + 1e-8
    t_mean = (targs * masks).sum(0) / counts
    t_std = np.sqrt(((targs - t_mean) ** 2 * masks).sum(0) / counts) + 1e-8

    X = torch.tensor((feats - f_mean) / f_std, dtype=torch.float32)
    Y = torch.tensor((targs - t_mean) / t_std, dtype=torch.float32)
    M = torch.tensor(masks, dtype=torch.float32)

    torch.manual_seed(SEED)
    model = LocalResidualNet(N_FEATURES, N_TEMPLATES)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    final = None
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = (((model(X) - Y) ** 2) * M).sum() / M.sum()
        loss.backward()
        opt.step()
        final = float(loss.item())
    norm = {"f_mean": f_mean, "f_std": f_std, "t_mean": t_mean, "t_std": t_std,
            "use_prior": use_prior}
    return model, norm, final


def predict_coeffs(model, norm, js, hs, dt) -> np.ndarray:
    feats = np.stack([anchor_features(js, hs, a, dt) for a in range(len(hs))])
    Xf = (feats - norm["f_mean"]) / norm["f_std"]
    with torch.no_grad():
        out = model(torch.tensor(Xf, dtype=torch.float32)).numpy()
    coeffs = (out * norm["t_std"] + norm["t_mean"]) * dt ** 3   # undo standardize + dt^3 scale
    if norm.get("use_prior", False):
        prior, _ = bch_leading_coeffs(js, hs, dt)
        coeffs = coeffs + prior   # delta learning: analytic Zassenhaus prior + learned residual
    return coeffs


# ---------------------------------------------------------------------------
# Downstream evaluation
# ---------------------------------------------------------------------------
def downstream_error(K_correction, S, exact_total, r):
    G = expm(-1j * K_correction) @ S
    return spectral_error(exact_total, repeated_step(G, r))


def evaluate_realization(models, norm, js, hs, dt, r, want_allw3=True, want_parity=False):
    """models = {'oracle': (A, normA), 'free': (B, normB)}."""
    n_qubits = len(hs)
    A, B, H = tfim_terms_disordered(js, hs)
    t = r * dt
    S = product_formula(A, B, dt, ORDER)
    exact_total = exact_step(H, t)

    K = global_generator(js, hs, dt)
    targets, masks = coeffs_from_K(K, n_qubits)

    pred_oracle = predict_coeffs(models["oracle"][0], models["oracle"][1], js, hs, dt)
    pred_free = predict_coeffs(models["free"][0], models["free"][1], js, hs, dt)
    bch_rows, _ = bch_leading_coeffs(js, hs, dt)

    metrics = {
        "baseline_error": spectral_error(exact_total, repeated_step(S, r)),
        "oracle_local_error": downstream_error(assemble_generator(targets * masks, n_qubits), S, exact_total, r),
        "learned_oracle_error": downstream_error(assemble_generator(pred_oracle * masks, n_qubits), S, exact_total, r),
        "learned_free_error": downstream_error(assemble_generator(pred_free * masks, n_qubits), S, exact_total, r),
        "bch_leading_error": downstream_error(assemble_generator(bch_rows * masks, n_qubits), S, exact_total, r),
    }
    if "free_noprior" in models:
        pred_free_np = predict_coeffs(models["free_noprior"][0], models["free_noprior"][1], js, hs, dt)
        metrics["learned_free_noprior_error"] = downstream_error(
            assemble_generator(pred_free_np * masks, n_qubits), S, exact_total, r)
    if want_allw3:
        K_allw3 = oracle_allweight_generator(K, n_qubits, MAX_WEIGHT)
        metrics["oracle_allw3_error"] = downstream_error(K_allw3, S, exact_total, r)
    parity = (pred_free, targets, masks) if want_parity else None
    return metrics, parity


def size_sweep(models, norm, rng):
    rows, parity_pred, parity_true = [], [], []
    keys = ["baseline_error", "oracle_local_error", "oracle_allw3_error",
            "learned_oracle_error", "learned_free_error", "learned_free_noprior_error",
            "bch_leading_error"]
    for n_qubits in TRANSFER_SIZES:
        accum = {k: [] for k in keys}
        for _ in range(EVAL_REALIZATIONS[n_qubits]):
            js, hs = sample_couplings(rng, n_qubits)
            m, parity = evaluate_realization(models, norm, js, hs, DT_EVAL, max(1, round(1.0 / DT_EVAL)),
                                             want_allw3=True, want_parity=(n_qubits not in TRAIN_SIZES))
            for k in keys:
                accum[k].append(m[k])
            if parity is not None:
                pred, true, mask = parity
                sel = mask.astype(bool)
                parity_pred.append(pred[sel])
                parity_true.append(true[sel])
        row = {"n_qubits": n_qubits, "trained": n_qubits in TRAIN_SIZES,
               "n_realizations": EVAL_REALIZATIONS[n_qubits]}
        for k in keys:
            samples = np.asarray(accum[k], dtype=float)
            row[f"{k}_mean"] = float(samples.mean())
            row[f"{k}_std"] = float(samples.std(ddof=1)) if samples.size > 1 else 0.0
        # Per-realization error-reduction factor (mean and spread of the ratio).
        ratios = np.asarray(accum["baseline_error"], dtype=float) / np.asarray(accum["learned_free_error"], dtype=float)
        row["reduction_free_mean"] = float(ratios.mean())
        row["reduction_free_std"] = float(ratios.std(ddof=1)) if ratios.size > 1 else 0.0
        rows.append(row)
    parity = (np.concatenate(parity_pred), np.concatenate(parity_true)) if parity_pred else (np.array([]), np.array([]))
    return pd.DataFrame(rows), parity


def dt_sweep(models, norm, rng):
    rows = []
    for dt in DT_SWEEP:
        accum = {k: [] for k in ["baseline_error", "oracle_local_error",
                                 "learned_free_error", "bch_leading_error"]}
        for _ in range(DTSWEEP_REALIZATIONS):
            js, hs = sample_couplings(rng, DTSWEEP_SIZE)
            m, _ = evaluate_realization(models, norm, js, hs, dt, 1, want_allw3=False)
            for k in accum:
                accum[k].append(m[k])
        row = {"dt": dt, "n_realizations": DTSWEEP_REALIZATIONS}
        for k in accum:
            samples = np.asarray(accum[k], dtype=float)
            row[f"{k}_mean"] = float(samples.mean())
            row[f"{k}_std"] = float(samples.std(ddof=1)) if samples.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def step_sweep(models, norm, rng):
    js, hs = sample_couplings(rng, STEP_SWEEP_SIZE)
    rows = []
    for r in STEP_SWEEP_RS:
        m, _ = evaluate_realization(models, norm, js, hs, DT_EVAL, r, want_allw3=False)
        m["r"] = r
        m["t"] = r * DT_EVAL
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
_C = {
    "baseline": PALETTE[0],   # blue
    "bch": PALETTE[1],        # vermillion
    "learn_oracle": PALETTE[4],  # orange
    "learn_free": PALETTE[2],    # bluish green
    "oracle": PALETTE[3],     # reddish purple
}


def make_figure(sizes_df, dt_df, steps_df, parity):
    apply_nmi_style()
    fig, axes = plt.subplots(2, 2, figsize=(COL_DOUBLE, 5.6))

    # (a) Size transfer ---------------------------------------------------
    # No in-plot title -- the panel description (t=1, dt=0.1, held-out shading)
    # lives in the LaTeX caption.
    ax = axes[0, 0]
    x = sizes_df["n_qubits"]
    series_a = [
        ("baseline_error", "uncorrected Strang", "o-", _C["baseline"], True),
        ("bch_leading_error", "leading-order BCH", "P-", _C["bch"], True),
        ("learned_free_noprior_error", "learned (no prior)", "v:", "0.55", False),
        ("learned_oracle_error", "learned (oracle labels)", "s-", _C["learn_oracle"], False),
        ("learned_free_error", "learned (Zassenhaus prior, oracle-free)", "D-", _C["learn_free"], True),
        ("oracle_local_error", r"exact weight-$\leq$3 oracle", "^--", _C["oracle"], True),
    ]
    for key, label, style, color, band in series_a:
        ax.semilogy(x, sizes_df[f"{key}_mean"], style, color=color, label=label)
        if band:
            shaded_band(ax, x, sizes_df[f"{key}_mean"], sizes_df[f"{key}_std"], color)
    ax.axvspan(max(TRAIN_SIZES) + 0.5, max(TRANSFER_SIZES) + 0.3, color="0.93", zorder=0)
    ax.set_xlabel("chain length $n$")
    ax.set_ylabel("global spectral-norm error")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.93", label=r"held out ($n>5$)"))
    ax.legend(handles=handles, fontsize=5.6, loc="best")
    line_plot_style(ax)
    panel_label(ax, "a")

    # (b) Step-size dependence -------------------------------------------
    ax = axes[0, 1]
    x = dt_df["dt"]
    series_b = [
        ("baseline_error", "uncorrected Strang", "o-", _C["baseline"]),
        ("bch_leading_error", "leading-order BCH", "P-", _C["bch"]),
        ("learned_free_error", "learned (oracle-free)", "D-", _C["learn_free"]),
        ("oracle_local_error", r"exact weight-$\leq$3 oracle", "^--", _C["oracle"]),
    ]
    for key, label, style, color in series_b:
        ax.semilogy(x, dt_df[f"{key}_mean"], style, color=color, label=label)
        shaded_band(ax, x, dt_df[f"{key}_mean"], dt_df[f"{key}_std"], color)
    ax.set_xlabel(r"step size $\delta t$")
    ax.set_ylabel("per-step spectral-norm error")
    ax.legend(fontsize=5.6, loc="best")
    line_plot_style(ax)
    panel_label(ax, "b")

    # (c) Stability vs number of steps -----------------------------------
    ax = axes[1, 0]
    ax.loglog(steps_df["r"], steps_df["baseline_error"], "o-", color=_C["baseline"], label="uncorrected Strang")
    ax.loglog(steps_df["r"], steps_df["learned_free_error"], "D-", color=_C["learn_free"], label="learned (oracle-free)")
    ax.loglog(steps_df["r"], steps_df["oracle_local_error"], "^--", color=_C["oracle"], label=r"exact weight-$\leq$3 oracle")
    r_ref = np.asarray(steps_df["r"], dtype=float)
    ax.loglog(r_ref, steps_df["learned_free_error"].iloc[0] * r_ref / r_ref[0], ":", color="0.5", lw=1.3,
              label=r"linear-in-$r$ guide")
    ax.set_xlabel("Trotter steps $r$")
    ax.set_ylabel("total spectral-norm error")
    ax.legend(fontsize=5.6, loc="best")
    line_plot_style(ax)
    panel_label(ax, "c")

    # (d) Predicted vs exact coefficients on held-out transfer sizes (parity).
    #     R^2 is reported in the caption; the y=x line is the perfect-fit guide.
    ax = axes[1, 1]
    pred, true = parity
    if pred.size:
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="0.5", lw=1.0, zorder=1, label=r"$y=x$")
        ax.scatter(true, pred, s=4, color=_C["learn_free"], alpha=0.45, linewidths=0,
                   zorder=2, label="oracle-free, $n=6\\ldots10$")
        ax.set_xlabel("exact coefficient")
        ax.set_ylabel("predicted coefficient")
        ax.legend(fontsize=5.6, loc="best")
    line_plot_style(ax)
    panel_label(ax, "d")

    fig.tight_layout(pad=1.0)
    save_figure(fig, "fig5_learned_transfer.pdf")


def write_summary_table(sizes_df) -> None:
    def pm(mean: float, std: float) -> str:
        """mean +/- std in scientific notation for a LaTeX table cell."""
        return f"${scientific(mean)} \\pm {scientific(std)}$"

    lines = [
        r"\begin{table}[t]",
        (
            r"\caption{Learned residual generator transferred across system size on the "
            r"disordered open-boundary \tfim{} ($J_i,h_i\sim\mathcal{U}[0.5,1.5]$, $t=1$, "
            r"$\delta t=0.1$). Errors are global spectral-norm errors reported as the mean "
            r"$\pm$ one standard deviation over the disordered chains (count $n_{\mathrm c}$ "
            r"in the last column); the same dispersion is shown as bands in "
            r"Fig.~\ref{fig:learned}. The column $\epsilon^{\mathrm{no\,prior}}_{\mathrm{learned}}$ "
            r"is the ablation that learns the full generator directly; $\epsilon_{\mathrm{learned}}$ "
            r"adds the analytic second-order Zassenhaus prior and learns only the residual. "
            r"The reduction factor is "
            r"$\epsilon_{\mathrm{Strang}}/\epsilon_{\mathrm{learned}}$. The oracle-free network is "
            r"trained only on $n=4,5$; sizes $n\geq6$ are absent from training. All values are "
            r"recomputed from dense matrices.}"
        ),
        r"\label{tab:learned-summary}",
        r"\centering",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        (
            r"$n$ & regime & $\epsilon_{\mathrm{Strang}}$ & "
            r"$\epsilon^{\mathrm{no\,prior}}_{\mathrm{learned}}$ & "
            r"$\epsilon_{\mathrm{learned}}$ & reduction & $n_{\mathrm c}$\\"
        ),
        r"\midrule",
    ]
    for row in sizes_df.itertuples(index=False):
        regime = "train" if row.trained else "transfer"
        reduction = row.baseline_error_mean / row.learned_free_error_mean
        lines.append(
            f"{int(row.n_qubits)} & {regime} & "
            f"{pm(row.baseline_error_mean, row.baseline_error_std)} & "
            f"{pm(row.learned_free_noprior_error_mean, row.learned_free_noprior_error_std)} & "
            f"{pm(row.learned_free_error_mean, row.learned_free_error_std)} & "
            f"${reduction:.1f}\\times$ & "
            f"{int(row.n_realizations)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_latex_table(TABLE_DIR / "learned_residual_summary.tex", lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(force: bool = False) -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)

    # Zassenhaus-prior delta learning: the oracle and oracle-free networks learn
    # the residual beyond the analytic leading-order generator; 'free_noprior' is
    # the prior-free ablation (the previous full-target variant) for comparison.
    model_oracle, norm_oracle, loss_oracle = train_model(rng, "global", use_prior=True)
    model_free, norm_free, loss_free = train_model(rng, "patch", use_prior=True)
    model_free_np, norm_free_np, loss_free_np = train_model(rng, "patch", use_prior=False)
    models = {"oracle": (model_oracle, norm_oracle), "free": (model_free, norm_free),
              "free_noprior": (model_free_np, norm_free_np)}

    sizes_df, parity = size_sweep(models, norm_oracle, rng)
    dt_df = dt_sweep(models, norm_oracle, rng)
    steps_df = step_sweep(models, norm_oracle, rng)

    pred, true = parity
    if pred.size:
        rel_l2 = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))
        r2 = float(1.0 - np.sum((pred - true) ** 2) / (np.sum((true - true.mean()) ** 2) + 1e-12))
    else:
        rel_l2 = r2 = float("nan")

    save_dataframe(sizes_df, "learned_residual_sizes.csv", "learned_residual_sizes.json")
    save_dataframe(dt_df, "learned_residual_dtsweep.csv", "learned_residual_dtsweep.json")
    save_dataframe(steps_df, "learned_residual_steps.csv", "learned_residual_steps.json")
    if pred.size:
        save_dataframe(pd.DataFrame({"exact": true, "predicted": pred}), "learned_residual_parity.csv")
    save_metadata("learned_residual.meta.json", {
        "experiment": "learned_residual",
        "order": ORDER, "max_weight": MAX_WEIGHT, "dt_eval": DT_EVAL,
        "dt_train_range": list(DT_TRAIN_RANGE), "dt0_bch": DT0_BCH, "patch_radius": PATCH_RADIUS,
        "train_sizes": list(TRAIN_SIZES), "transfer_sizes": list(TRANSFER_SIZES),
        "eval_realizations": EVAL_REALIZATIONS, "n_train_realizations": N_TRAIN_REALIZATIONS,
        "n_templates": N_TEMPLATES, "n_features": N_FEATURES, "hidden": HIDDEN,
        "epochs": EPOCHS, "lr": LR, "seed": SEED,
        "J_range": list(J_RANGE), "h_range": list(H_RANGE),
        "final_train_loss_oracle": loss_oracle, "final_train_loss_free": loss_free,
        "final_train_loss_free_noprior": loss_free_np, "zassenhaus_prior": True,
        "transfer_coeff_rel_l2": rel_l2, "transfer_coeff_r2": r2,
    })

    make_figure(sizes_df, dt_df, steps_df, parity)
    write_summary_table(sizes_df)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\ncoeff fit (oracle-free, transfer sizes): rel L2 = {rel_l2:.4f}, R^2 = {r2:.4f}")
    print(f"train loss  oracle={loss_oracle:.2e}  oracle-free={loss_free:.2e}  oracle-free(no prior)={loss_free_np:.2e}")
    print("\n=== size sweep (mean error) ===")
    print(sizes_df[["n_qubits", "trained", "baseline_error_mean", "bch_leading_error_mean",
                    "learned_free_noprior_error_mean", "learned_oracle_error_mean",
                    "learned_free_error_mean", "oracle_local_error_mean",
                    "oracle_allw3_error_mean"]].to_string(index=False))
    print("\n=== step-size sweep (n=6) ===")
    print(dt_df.to_string(index=False))
    print("\n=== step-count sweep (n=6) ===")
    print(steps_df[["r", "t", "baseline_error", "learned_free_error", "oracle_local_error"]].to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
