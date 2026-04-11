"""
Generate all clean paper figures for research_paper/main.tex.

Produces six figures, each making exactly ONE comparison:
  fig1_local_error.png   – local error vs δt, Trotter-1 vs GPT-1, n=4
  fig2_n6.png            – same but n=6 (robustness to system size)
  fig3_global.png        – global error vs N, Trotter-1 vs GPT-1, T=0.5
  fig4_random.png        – ratio histogram over 300 random Hamiltonians
  fig5_longtime_gpt1.png – T=0..10, Trotter-1 vs GPT-1 only
  fig6_longtime_gpt2.png – T=0..10, Suzuki-2  vs GPT-2  only
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lc_qaoa.models import tfim_hamiltonian
from lc_qaoa.propagators import (
    commutator, evolution_unitary, exact_propagator,
    first_order_trotter, second_order_suzuki, lc_qaoa_repeated,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "research_paper", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "lines.linewidth": 2.0,
})
C_BASE = "#444444"   # Trotter / Suzuki
C_GPT  = "#1f77b4"   # Lie GPT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spec_err(U, V):
    return float(np.linalg.norm(U - V, ord=2))


def gpt2_step(model, dt):
    """Lie GPT-2 step with corrected α₂=-1/6, α₃=-1/3."""
    A, B = model.A, model.B
    AB = commutator(A, B)
    CA = commutator(A, AB)
    CB = commutator(B, AB)
    core = (
        evolution_unitary(np.zeros_like(A), 0)  # identity placeholder
    )
    # V2 = exp(α3 δt³ CB) exp(α2 δt³ CA) exp(-δt² [A,B]/2) exp(-iδt B) exp(-iδt A)
    step = (
        evolution_unitary(CB, -1.0/3.0 * dt**3)
        @ evolution_unitary(CA, -1.0/6.0 * dt**3)
        @ evolution_unitary(1j * AB, -0.5 * dt**2)   # exp(+δt²/2 · i[A,B])
        @ evolution_unitary(B, dt)
        @ evolution_unitary(A, dt)
    )
    return step


def gpt2_rep(model, T, N):
    dt = T / N
    step = gpt2_step(model, dt)
    R = np.eye(model.dimension, dtype=complex)
    for _ in range(N):
        R = step @ R
    return R


# ---------------------------------------------------------------------------
# TFIM models
# ---------------------------------------------------------------------------
model4 = tfim_hamiltonian(4, 1.0, 0.5)
model6 = tfim_hamiltonian(6, 1.0, 0.5)

dts = np.logspace(-2, 0, 40)


# ===========================================================================
# FIG 1 – local error vs δt, n=4, Trotter-1 vs GPT-1 only
# ===========================================================================
print("Generating fig1_local_error.png ...")

e_t1, e_g1 = [], []
for dt in dts:
    Ue = exact_propagator(model4.hamiltonian, dt)
    e_t1.append(spec_err(first_order_trotter(model4, dt, 1), Ue))
    e_g1.append(spec_err(lc_qaoa_repeated(model4, dt, 1), Ue))
e_t1 = np.array(e_t1)
e_g1 = np.array(e_g1)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.loglog(dts, e_t1, color=C_BASE, ls="--", label="Trotter-1")
ax.loglog(dts, e_g1, color=C_GPT,  ls="-",  label="Lie GPT-1")
# reference slopes anchored to mid-range
mid = len(dts) // 2
ax.loglog(dts, e_t1[mid] * (dts / dts[mid])**2, "k:",  lw=1, alpha=0.55, label=r"$\delta t^2$")
ax.loglog(dts, e_g1[mid] * (dts / dts[mid])**3, "k-.", lw=1, alpha=0.55, label=r"$\delta t^3$")

ax.set_xlabel(r"Step size $\delta t$")
ax.set_ylabel(r"Spectral error $\|e^{-i\delta t H} - U\|_2$")
ax.set_title(r"Local error: Trotter-1 vs.\ Lie GPT-1 ($n{=}4$, TFIM)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_local_error.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("  saved fig1_local_error.png")


# ===========================================================================
# FIG 2 – n=6 system, Trotter-1 vs GPT-1 only
# ===========================================================================
print("Generating fig2_n6.png ...")

e_t1_6, e_g1_6 = [], []
for dt in dts:
    Ue = exact_propagator(model6.hamiltonian, dt)
    e_t1_6.append(spec_err(first_order_trotter(model6, dt, 1), Ue))
    e_g1_6.append(spec_err(lc_qaoa_repeated(model6, dt, 1), Ue))
e_t1_6 = np.array(e_t1_6)
e_g1_6 = np.array(e_g1_6)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.loglog(dts, e_t1_6, color=C_BASE, ls="--", label="Trotter-1")
ax.loglog(dts, e_g1_6, color=C_GPT,  ls="-",  label="Lie GPT-1")
ax.loglog(dts, e_t1_6[mid] * (dts / dts[mid])**2, "k:",  lw=1, alpha=0.55, label=r"$\delta t^2$")
ax.loglog(dts, e_g1_6[mid] * (dts / dts[mid])**3, "k-.", lw=1, alpha=0.55, label=r"$\delta t^3$")

ax.set_xlabel(r"Step size $\delta t$")
ax.set_ylabel(r"Spectral error $\|e^{-i\delta t H} - U\|_2$")
ax.set_title(r"Local error: Trotter-1 vs.\ Lie GPT-1 ($n{=}6$, TFIM)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_n6.png"), dpi=180, bbox_inches="tight")
plt.close(fig)

# print ratio at small dt
ratios_n6 = e_t1_6 / np.maximum(e_g1_6, 1e-16)
print(f"  n=6 ratio at δt=0.06: {np.interp(0.06, dts[::-1], ratios_n6[::-1]):.1f}×")
print("  saved fig2_n6.png")


# ===========================================================================
# FIG 3 – global error vs N, T=0.5, Trotter-1 vs GPT-1 only
# ===========================================================================
print("Generating fig3_global.png ...")

T_glob = 0.5
N_arr = np.array([2, 4, 6, 8, 10, 12, 16, 20, 24, 32])
Ue_T  = exact_propagator(model4.hamiltonian, T_glob)
e1_gl, g1_gl = [], []
for N in N_arr:
    e1_gl.append(spec_err(first_order_trotter(model4, T_glob, N), Ue_T))
    g1_gl.append(spec_err(lc_qaoa_repeated(model4, T_glob, N), Ue_T))
e1_gl = np.array(e1_gl)
g1_gl = np.array(g1_gl)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.loglog(N_arr, e1_gl, color=C_BASE, ls="--", marker="o", ms=5, label="Trotter-1")
ax.loglog(N_arr, g1_gl, color=C_GPT,  ls="-",  marker="s", ms=5, label="Lie GPT-1")
ax.loglog(N_arr, e1_gl[0] * (N_arr / N_arr[0])**-1, "k:",  lw=1, alpha=0.55, label=r"$N^{-1}$")
ax.loglog(N_arr, g1_gl[0] * (N_arr / N_arr[0])**-2, "k-.", lw=1, alpha=0.55, label=r"$N^{-2}$")

ax.set_xlabel(r"Number of Trotter steps $N$")
ax.set_ylabel(r"Spectral error $\|e^{-iT H} - U_N\|_2$")
ax.set_title(r"Global error at $T{=}0.5$: Trotter-1 vs.\ Lie GPT-1 ($n{=}4$)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_global.png"), dpi=180, bbox_inches="tight")
plt.close(fig)

ratios_gl = e1_gl / np.maximum(g1_gl, 1e-16)
for N, r in zip(N_arr, ratios_gl):
    print(f"  N={N:2d}: Trot/GPT1 = {r:.1f}×")
print("  saved fig3_global.png")


# ===========================================================================
# FIG 4 – random Hamiltonians: ratio distribution
# ===========================================================================
print("Generating fig4_random.png ...")

rng   = np.random.RandomState(42)
dim   = 16   # 4 qubits, 2^4 Hilbert space
dt_r  = 0.05
N_rand = 300

ratios_rand = []
for _ in range(N_rand):
    A = rng.randn(dim, dim); A = (A + A.T) / 2
    B = rng.randn(dim, dim); B = (B + B.T) / 2
    # normalise so spectral radius ~1
    A /= max(np.linalg.norm(A, ord=2), 1e-12)
    B /= max(np.linalg.norm(B, ord=2), 1e-12)
    H   = A + B
    Ue  = expm(-1j * dt_r * H)

    # Trotter-1
    U_tr = expm(-1j * dt_r * B) @ expm(-1j * dt_r * A)
    err_tr = np.linalg.norm(Ue - U_tr, ord=2)

    # GPT-1: prepend exp(-δt²[A,B]/2) to cancel the BCH leading error
    AB     = A @ B - B @ A          # [A,B]  (anti-Hermitian)
    U_gpt1 = expm(-0.5 * dt_r**2 * AB) @ expm(-1j * dt_r * B) @ expm(-1j * dt_r * A)
    err_g1 = np.linalg.norm(Ue - U_gpt1, ord=2)

    if err_g1 > 1e-14:
        ratios_rand.append(err_tr / err_g1)

ratios_rand = np.array(ratios_rand)
med = float(np.median(ratios_rand))
frac10 = float((ratios_rand >= 10).mean()) * 100
print(f"  Random: N={len(ratios_rand)}, median={med:.1f}×, ≥10×: {frac10:.0f}%")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
log_r = np.log10(np.maximum(ratios_rand, 1e-6))
ax.hist(log_r, bins=30, color=C_GPT, edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axvline(np.log10(10), color="crimson", lw=2.0, ls="--", label=r"$10\times$ threshold")
ax.axvline(np.log10(med), color="darkorange", lw=2.0, ls="-",
           label=rf"Median $= {med:.1f}\times$")
ax.set_xlabel(r"$\log_{10}(\,\|\text{Trotter-1 error}\| \,/\, \|\text{GPT-1 error}\|\,)$")
ax.set_ylabel("Count")
ax.set_title(rf"Ratio distribution over {N_rand} random Hamiltonians ($\delta t={dt_r}$)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_random.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("  saved fig4_random.png")


# ===========================================================================
# FIG 5 – long-time GPT-1 only: Trotter-1 vs GPT-1, T=0.5..10
# ===========================================================================
print("Generating fig5_longtime_gpt1.png ...")

DT   = 0.01
T_lt = np.concatenate([
    np.arange(0.5, 2.0, 0.5),
    np.arange(2.0, 10.5, 1.0),
])
e1_lt, g1_lt = [], []
for T in T_lt:
    N = max(1, round(T / DT))
    Ue = exact_propagator(model4.hamiltonian, T)
    e1_lt.append(spec_err(first_order_trotter(model4, T, N), Ue))
    g1_lt.append(spec_err(lc_qaoa_repeated(model4, T, N), Ue))
    print(f"  GPT1 T={T:.1f} N={N}: ratio={e1_lt[-1]/max(g1_lt[-1],1e-16):.1f}×")
e1_lt = np.array(e1_lt)
g1_lt = np.array(g1_lt)
r1_lt = e1_lt / np.maximum(g1_lt, 1e-16)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

ax1.semilogy(T_lt, e1_lt, color=C_BASE, ls="--", label="Trotter-1")
ax1.semilogy(T_lt, g1_lt, color=C_GPT,  ls="-",  label="Lie GPT-1")
ax1.set_xlabel(r"Total time $T$")
ax1.set_ylabel(r"Spectral error")
ax1.set_title(r"(a) Error vs.\ $T$ ($\delta t{=}0.01$)")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

ax2.plot(T_lt, r1_lt, color=C_GPT, lw=2.0)
ax2.axhline(10, color="crimson", ls=":", lw=2.0, label=r"$10\times$ threshold")
ax2.fill_between(T_lt, r1_lt, 10, where=(r1_lt >= 10),
                  alpha=0.15, color=C_GPT, label=r"$\geq\!10\times$ region")
ax2.set_xlabel(r"Total time $T$")
ax2.set_ylabel(r"Trotter-1 error $\,/\,$ GPT-1 error")
ax2.set_title(r"(b) Improvement ratio vs.\ $T$")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

fig.suptitle(r"Long-time benchmark: Trotter-1 vs.\ Lie GPT-1 ($n{=}4$, TFIM, $\delta t{=}0.01$)",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_longtime_gpt1.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("  saved fig5_longtime_gpt1.png")


# ===========================================================================
# FIG 6 – long-time GPT-2 only: Suzuki-2 vs GPT-2, T=0.5..10
# ===========================================================================
print("Generating fig6_longtime_gpt2.png ...")

e2_lt, g2_lt = [], []
for T in T_lt:
    N = max(1, round(T / DT))
    Ue = exact_propagator(model4.hamiltonian, T)
    e2_lt.append(spec_err(second_order_suzuki(model4, T, N), Ue))
    g2_lt.append(spec_err(gpt2_rep(model4, T, N), Ue))
    print(f"  GPT2 T={T:.1f} N={N}: ratio={e2_lt[-1]/max(g2_lt[-1],1e-16):.1f}×")
e2_lt = np.array(e2_lt)
g2_lt = np.array(g2_lt)
r2_lt = e2_lt / np.maximum(g2_lt, 1e-16)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

ax1.semilogy(T_lt, e2_lt, color=C_BASE, ls="--", label="Suzuki-2")
ax1.semilogy(T_lt, g2_lt, color=C_GPT,  ls="-",  label="Lie GPT-2")
ax1.set_xlabel(r"Total time $T$")
ax1.set_ylabel(r"Spectral error")
ax1.set_title(r"(a) Error vs.\ $T$ ($\delta t{=}0.01$)")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

ax2.plot(T_lt, r2_lt, color=C_GPT, lw=2.0)
ax2.axhline(10, color="crimson", ls=":", lw=2.0, label=r"$10\times$ threshold")
ax2.fill_between(T_lt, r2_lt, 10, where=(r2_lt >= 10),
                  alpha=0.15, color=C_GPT, label=r"$\geq\!10\times$ region")
ax2.set_xlabel(r"Total time $T$")
ax2.set_ylabel(r"Suzuki-2 error $\,/\,$ GPT-2 error")
ax2.set_title(r"(b) Improvement ratio vs.\ $T$")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

fig.suptitle(r"Long-time benchmark: Suzuki-2 vs.\ Lie GPT-2 ($n{=}4$, TFIM, $\delta t{=}0.01$)",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_longtime_gpt2.png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print("  saved fig6_longtime_gpt2.png")

print("\nAll figures generated successfully.")
