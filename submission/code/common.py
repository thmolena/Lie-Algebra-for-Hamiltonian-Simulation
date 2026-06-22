"""Shared numerical toolkit for residual-generator Trotter compilation (RGTC).

This module builds, from first principles, every primitive the experiments need:

  * Single-qubit Pauli matrices and n-qubit Pauli strings via Kronecker products
    (`single_pauli`, `kron_all`, `pauli_string`, `pauli_basis`).
  * The transverse-field Ising Hamiltonian H = A + B with A = J sum_i Z_i Z_{i+1}
    and B = h sum_i X_i (`tfim_terms`).
  * Trotter--Suzuki product-formula steps S_q of order q in {1,2,4,6,8}, from the
    recursive Suzuki construction (`suzuki_sequence`, `product_formula`).
  * The exact propagator U(dt) = exp(-i H dt) by dense matrix exponential
    (`exact_step`); the residual factor R_q = U S_q^dagger and corrected step
    G_q = R_q S_q (`residual_factor`); and the Hermitian residual generator
    K_q = i log R_q (`residual_generator`).
  * Frobenius-optimal Pauli-weight projection of K_q (`project_pauli_weight`) and
    its squared-coefficient mass by weight (`pauli_weight_energy`).
  * Spectral-norm error from singular values (`spectral_error`) -- the
    operationally meaningful worst-case state-vector error.

It also centralises deterministic I/O (CSV/JSON tagged with environment metadata),
publication-grade figure styling (colour-blind-safe Okabe--Ito palette, Type-42
fonts), and small plotting helpers (`shaded_band`, `panel_label`).  Every other
script imports from here, so all results share one verified, reproducible base.

Run instructions are in README.txt; the mathematics is derived in THEORY.txt.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.dont_write_bytecode = True

# ``code/common.py`` -> parents[1] is the submission directory in the source
# tree.  Every output directory is anchored to a single OUTPUT_ROOT so that the
# package can regenerate the manuscript artifacts in place (default) or into an
# isolated directory chosen by the caller.  The determinism harness and any
# pip-installed user set ``RGTC_OUTPUT_ROOT`` to redirect all outputs without
# touching the numerics.
SOURCE_SUBMISSION_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = SOURCE_SUBMISSION_DIR / "code"
_OUTPUT_ROOT_ENV = os.environ.get("RGTC_OUTPUT_ROOT")
if _OUTPUT_ROOT_ENV:
    OUTPUT_ROOT = Path(_OUTPUT_ROOT_ENV).resolve()
    DATA_DIR = OUTPUT_ROOT / "generated_data"
else:
    OUTPUT_ROOT = SOURCE_SUBMISSION_DIR
    DATA_DIR = CODE_DIR / "generated_data"
ROOT = OUTPUT_ROOT.parent
SUBMISSION_DIR = OUTPUT_ROOT
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
os.environ.setdefault("MPLCONFIGDIR", str(SOURCE_SUBMISSION_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(SOURCE_SUBMISSION_DIR / ".cache"))
# Fixed epoch so the matplotlib PDF backend writes a constant CreationDate; this
# makes every regenerated PDF byte-for-byte identical across runs and machines.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import expm, logm, svdvals

Array = np.ndarray
ORDERS = [1, 2, 4, 6, 8]

# --- Nature Machine Intelligence (NMI) display conventions -------------------
# Design rules (Nature Portfolio artwork & formatting guidance):
#   * Vector PDF output with embedded, editable text (``pdf.fonttype = 42``).
#   * Sans-serif typeface, with in-figure math kept sans-serif
#     (``mathtext.fontset = dejavusans``).
#   * No in-panel titles -- every description lives in the LaTeX caption.
#   * Bold lower-case panel labels (a, b, ...) for multi-panel figures.
#   * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods
#     2011) for categories; the perceptually uniform ``viridis`` map for
#     sequential quantities.
#   * Uncertainty shown wherever a mean/estimate is plotted; the caption states
#     n and what the interval represents (or that values are exact/deterministic).
#   * Top/right spines removed for an uncluttered Nature-style frame.

# Colour-blind-safe qualitative palette (Okabe--Ito) used for categorical series.
NMI_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
]
# Backwards-compatible alias: existing scripts index ``PALETTE``.
PALETTE = NMI_PALETTE

# Column widths in inches (Nature: single column 89 mm, double column 183 mm).
COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20


def apply_nmi_style() -> None:
    """Install NMI-conforming matplotlib defaults (idempotent).

    Called once at import time and again at the top of every plotting routine so
    that the whole figure set shares one consistent, journal-grade visual style
    regardless of import order.  Fonts are embedded as editable Type-42
    (TrueType) and in-figure math is rendered sans-serif.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so text stays selectable/editable
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",  # keep in-figure math sans-serif
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.0,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=NMI_PALETTE),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
        }
    )


# Install the NMI defaults at import time so any script that merely imports
# ``common`` already inherits the house style.
apply_nmi_style()


@dataclass(frozen=True)
class ExperimentMetadata:
    python_version: str
    numpy_version: str
    scipy_version: str
    pandas_version: str
    matplotlib_version: str
    platform: str
    blas_info: str
    git_commit: str | None


@dataclass(frozen=True)
class HamiltonianTerms:
    A: Array
    B: Array
    H: Array


def ensure_directories() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def environment_metadata() -> ExperimentMetadata:
    ensure_directories()
    try:
        from io import StringIO
        import contextlib

        buffer = StringIO()
        with contextlib.redirect_stdout(buffer):
            np.__config__.show()
        blas_info = buffer.getvalue().strip() or "unavailable"
    except Exception:
        blas_info = "unavailable"
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        git_commit = proc.stdout.strip() or None
    except Exception:
        git_commit = None
    return ExperimentMetadata(
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        pandas_version=pd.__version__,
        matplotlib_version=matplotlib.__version__,
        platform=platform.platform(),
        blas_info=str(blas_info),
        git_commit=git_commit,
    )


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def save_dataframe(df: pd.DataFrame, csv_name: str, json_name: str | None = None) -> None:
    ensure_directories()
    csv_path = DATA_DIR / csv_name
    df.to_csv(csv_path, index=False)
    if json_name is not None:
        write_json(DATA_DIR / json_name, df.to_dict(orient="records"))


def save_metadata(meta_name: str, payload: dict[str, Any]) -> None:
    ensure_directories()
    payload = {**payload, "environment": environment_metadata().__dict__}
    write_json(DATA_DIR / meta_name, payload)


def single_pauli(label: str) -> Array:
    matrices = {
        "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    }
    try:
        return matrices[label]
    except KeyError as exc:
        raise ValueError(f"unsupported Pauli label {label!r}") from exc


def kron_all(mats: Sequence[Array]) -> Array:
    if not mats:
        raise ValueError("kron_all requires at least one matrix")
    out = mats[0]
    for mat in mats[1:]:
        out = np.kron(out, mat)
    return out


@lru_cache(maxsize=None)
def op_on_site(label: str, site: int, n_qubits: int) -> Array:
    mats = [single_pauli("I") for _ in range(n_qubits)]
    mats[site] = single_pauli(label)
    return kron_all(mats)


@lru_cache(maxsize=None)
def pauli_string(word: str) -> Array:
    return kron_all([single_pauli(label) for label in word])


def pauli_weight(word: str) -> int:
    return sum(label != "I" for label in word)


@lru_cache(maxsize=None)
def pauli_basis(n_qubits: int) -> tuple[tuple[str, int, Array], ...]:
    basis: list[tuple[str, int, Array]] = []
    for letters in product("IXYZ", repeat=n_qubits):
        word = "".join(letters)
        basis.append((word, pauli_weight(word), pauli_string(word)))
    return tuple(basis)


def tfim_terms(n_qubits: int, J: float, h: float) -> HamiltonianTerms:
    dimension = 2 ** n_qubits
    A = np.zeros((dimension, dimension), dtype=complex)
    B = np.zeros((dimension, dimension), dtype=complex)
    z = single_pauli("Z")
    x = single_pauli("X")
    identity = single_pauli("I")
    for site in range(n_qubits - 1):
        mats = [identity for _ in range(n_qubits)]
        mats[site] = z
        mats[site + 1] = z
        A += J * kron_all(mats)
    for site in range(n_qubits):
        B += h * op_on_site("X", site, n_qubits)
    H = A + B
    if not np.allclose(A, A.conj().T):
        raise ValueError("A is not Hermitian")
    if not np.allclose(B, B.conj().T):
        raise ValueError("B is not Hermitian")
    return HamiltonianTerms(A=A, B=B, H=H)


def suzuki_sequence(order: int, scale: float = 1.0) -> list[tuple[str, float]]:
    if order == 1:
        return [("B", scale), ("A", scale)]
    if order == 2:
        return [("B", scale / 2.0), ("A", scale), ("B", scale / 2.0)]
    if order not in (4, 6, 8):
        raise ValueError(f"unsupported order {order}")
    k = order // 2
    p_k = (4.0 - 4.0 ** (1.0 / (2 * k - 1))) ** (-1)
    lower = order - 2
    seq_p = suzuki_sequence(lower, scale * p_k)
    seq_m = suzuki_sequence(lower, scale * (1.0 - 4.0 * p_k))
    return seq_p + seq_p + seq_m + seq_p + seq_p


def factor_count(order: int) -> int:
    return len(suzuki_sequence(order))


def product_formula(A: Array, B: Array, dt: float, order: int) -> Array:
    dimension = A.shape[0]
    result = np.eye(dimension, dtype=complex)
    cache: dict[tuple[str, float], Array] = {}
    for label, coeff in reversed(suzuki_sequence(order)):
        key = (label, coeff * dt)
        if key not in cache:
            generator = A if label == "A" else B
            cache[key] = expm(-1j * generator * key[1])
        result = result @ cache[key]
    return result


def exact_step(H: Array, dt: float) -> Array:
    return expm(-1j * H * dt)


def spectral_error(U: Array, V: Array) -> float:
    return float(np.max(svdvals(U - V)))


def residual_factor(A: Array, B: Array, H: Array, dt: float, order: int) -> tuple[Array, Array, Array, Array]:
    U = exact_step(H, dt)
    S = product_formula(A, B, dt, order)
    R = U @ S.conj().T
    G = R @ S
    return U, S, R, G


def residual_generator(R: Array) -> Array:
    K = 1j * logm(R)
    return 0.5 * (K + K.conj().T)


def repeated_step(step: Array, repetitions: int) -> Array:
    return np.linalg.matrix_power(step, repetitions)


def global_errors(n_qubits: int, order: int, t: float, r: int, J: float, h: float) -> dict[str, float]:
    terms = tfim_terms(n_qubits, J, h)
    dt = t / r
    U_dt, S_dt, R_dt, G_dt = residual_factor(terms.A, terms.B, terms.H, dt, order)
    exact_total = exact_step(terms.H, t)
    baseline_total = repeated_step(S_dt, r)
    oracle_total = repeated_step(G_dt, r)
    baseline_error = spectral_error(exact_total, baseline_total)
    oracle_error = spectral_error(exact_total, oracle_total)
    if baseline_error < 1.0e-12 and oracle_error < 1.0e-12:
        print(
            f"warning: order {order}, n={n_qubits} is at the floating-point floor "
            f"(baseline={baseline_error:.3e}, oracle={oracle_error:.3e})"
        )
    return {
        "baseline_error": baseline_error,
        "oracle_error": oracle_error,
        "unitarity_R": spectral_error(R_dt.conj().T @ R_dt, np.eye(R_dt.shape[0], dtype=complex)),
        "step_identity": spectral_error(G_dt, U_dt),
        "dimension": float(terms.H.shape[0]),
    }


def project_pauli_weight(K: Array, n_qubits: int, max_weight: int) -> Array:
    projection = np.zeros_like(K, dtype=complex)
    normalizer = float(2 ** n_qubits)
    for _, weight, P in pauli_basis(n_qubits):
        if weight > max_weight:
            continue
        coefficient = np.trace(P @ K) / normalizer
        projection += coefficient * P
    projection = 0.5 * (projection + projection.conj().T)
    return projection


def pauli_weight_energy(K: Array, n_qubits: int) -> dict[int, float]:
    normalizer = float(2 ** n_qubits)
    by_weight: dict[int, float] = {weight: 0.0 for weight in range(n_qubits + 1)}
    for _, weight, P in pauli_basis(n_qubits):
        coefficient = np.trace(P @ K) / normalizer
        by_weight[weight] += float(np.abs(coefficient) ** 2)
    return by_weight


def projected_residual_error(n_qubits: int, order: int, t: float, r: int, max_weight: int, J: float, h: float) -> dict[str, float]:
    terms = tfim_terms(n_qubits, J, h)
    dt = t / r
    U_dt, S_dt, R_dt, _ = residual_factor(terms.A, terms.B, terms.H, dt, order)
    K_dt = residual_generator(R_dt)
    K_w = project_pauli_weight(K_dt, n_qubits, max_weight)
    R_w = expm(-1j * K_w)
    G_w = R_w @ S_dt
    exact_total = exact_step(terms.H, t)
    projected_total = repeated_step(G_w, r)
    baseline_total = repeated_step(S_dt, r)
    return {
        "projected_error": spectral_error(exact_total, projected_total),
        "generator_residual_norm": float(np.linalg.norm(K_dt - K_w, ord=2)),
        "generator_norm": float(np.linalg.norm(K_dt, ord=2)),
        "residual_unitary_step_error": spectral_error(R_dt, R_w),
        "baseline_error": spectral_error(exact_total, baseline_total),
    }


def cumulative_weight_mass(energy_by_weight: dict[int, float]) -> list[dict[str, float]]:
    total = sum(energy_by_weight.values())
    cumulative = 0.0
    rows: list[dict[str, float]] = []
    for weight in sorted(energy_by_weight):
        cumulative += energy_by_weight[weight]
        rows.append(
            {
                "max_weight": float(weight),
                "weight_energy": energy_by_weight[weight],
                "cumulative_energy": cumulative,
                "cumulative_mass": cumulative / total if total else 0.0,
            }
        )
    return rows


def line_plot_style(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def shaded_band(ax: plt.Axes, x, mean, std, color: str, alpha: float = 0.18) -> None:
    """Shade mean +/- one standard deviation, clipped positive for log axes."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    lower = np.clip(mean - std, 1.0e-300, None)
    upper = mean + std
    ax.fill_between(np.asarray(x, dtype=float), lower, upper, color=color, alpha=alpha, linewidth=0.0)


def panel_label(ax: plt.Axes, text: str, dx: float = -0.16, dy: float = 1.04) -> None:
    """Bold lower-case panel label in the upper-left, Nature convention."""
    ax.text(
        dx,
        dy,
        text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def save_figure(fig: plt.Figure, pdf_name: str) -> None:
    ensure_directories()
    pdf_path = FIGURE_DIR / pdf_name
    png_path = pdf_path.with_suffix(".png")
    # Pinned metadata (no embedded creation timestamp) so the rendered PDF/PNG
    # bytes are identical on every run -- the figures are part of the
    # deterministic artifact set verified by ``rgtc-verify``.
    fig.savefig(pdf_path, bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(png_path, dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def _schematic_box(ax, xy, w, h, text, fc, ec="#222222"):
    """Draw a rounded method-schematic box with centred wrapped text."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (xy[0], xy[1]),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.0,
        zorder=5,
    )
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _schematic_arrow(ax, p0, p1):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444", shrinkA=2, shrinkB=2),
    )


def fig_schematic(out: Path) -> Path:
    """Method-overview schematic (the NMI 'Figure 1' convention).

    A left-to-right pipeline for residual-generator Trotter compilation (RGTC):
    a local Hamiltonian H = A + B is split into a Trotter--Suzuki product-formula
    step S_q; the exact residual factor R_q = U S_q^dagger defines a Hermitian
    residual generator K_q = i log R_q; the Lie-algebraic commutator structure of
    the leading defect makes K_q geometrically local, supported on low Pauli
    weight; K_q is compressed by Pauli weight or learned per site by a
    translation-equivariant network from local couplings (no dense propagator);
    the corrected step exp(-i K-hat) S_q is scored by spectral-norm error against
    the exact propagator U(t), with a proven r*eta stability certificate.
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue, green, orange, purple, grey = (
        "#D6E6F2",
        "#D6EFE3",
        "#FBE6D4",
        "#ECDCE9",
        "#ECECEC",
    )
    y = 0.40
    h = 0.36
    boxes = [
        (0.000, 0.158, "local Hamiltonian\n$H=A+B$\n(TFIM chain)", blue),
        (0.198, 0.176, "product-formula\nstep $S_q(\\delta t)$;\nresidual $R_q=US_q^{\\dagger}$", green),
        (0.412, 0.176, "Lie-algebra defect:\ngenerator $K_q=i\\log R_q$\nlocal, weight $\\leq 3$", orange),
        (0.626, 0.176, "compress / learn:\n$\\Pi_w K_q$ or per-site\nnetwork $\\widehat K_q$", purple),
        (0.840, 0.160, "error vs exact\n$U(t)$; $r\\eta$\nstability bound", grey),
    ]
    rights, lefts = [], []
    for x0, w, text, fc in boxes:
        r, l = _schematic_box(ax, (x0, y), w, h, text, fc)
        rights.append(r)
        lefts.append(l)
    for i in range(len(boxes) - 1):
        _schematic_arrow(ax, rights[i], lefts[i + 1])

    # The two structural guarantees that underpin the framework.
    ax.text(
        0.5, 0.075,
        "exact one-step cancellation $R_qS_q=U(\\delta t)$"
        "   $\\bullet$   leading Strang generator $K_2^{(3)}\\in\\mathcal{B}_3$ (weight $\\leq 3$)",
        ha="center", va="center", fontsize=6.4, color="#555555",
    )
    ax.text(
        0.5, 0.95,
        "trained on $n=4,5$ from local patches; transfers to $n=10$ under the proven bound",
        ha="center", va="center", fontsize=6.6, color="#333333",
    )
    save_figure(fig, out.name)
    return out


def write_latex_table(path: Path, lines: Iterable[str]) -> None:
    ensure_directories()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scientific(value: float) -> str:
    return f"{value:.3e}".replace("e-0", "e-").replace("e+0", "e+")


def parse_graphics_references(tex: str) -> set[str]:
    import re

    names = set(re.findall(r"\\(?:includegraphics|paperfigure)(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    normalized: set[str] = set()
    for name in names:
        if "#" in name:
            continue
        normalized.add(name if "." in Path(name).name else f"{name}.pdf")
    return normalized


def parse_citation_keys(tex: str) -> set[str]:
    import re

    keys: set[str] = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        keys.update(part.strip() for part in group.split(",") if part.strip())
    return keys


def parse_bib_keys(bib: str) -> set[str]:
    import re

    return set(re.findall(r"^\s*@\w+\{\s*([^,]+),", bib, flags=re.MULTILINE))