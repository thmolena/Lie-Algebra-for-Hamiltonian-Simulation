"""
Long-time benchmark: Lie GPT vs Trotter/Suzuki, T from 0 to 10.

Generates:
  research_paper/figures/liegpt_longtime.png       (4-panel: two dt regimes)
  research_paper/figures/liegpt_longtime_nsweep.png (N-sweep at T=10)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lc_qaoa.models import tfim_hamiltonian
from lc_qaoa.propagators import (
    commutator, evolution_unitary, exact_propagator,
    first_order_trotter, second_order_suzuki, lc_qaoa_repeated,
)

# ── helpers ─────────────────────────────────────────────────────────────────
def spec_err(U, V):
    return float(np.linalg.norm(U - V, ord=2))


def gpt2_step(model, dt):
    """Lie GPT-2 one step. Corrected coefficients alpha2=-1/6, alpha3=-1/3.

    Sign note:
      commutator(A, B)       = [A,B]         anti-Hermitian for Hermitian A,B
      C  = 1j * commutator   = i[A,B]        Hermitian  (used in GPT-1 layer)
      CA = commutator(A, AB) = [A,[A,B]]     Hermitian  (double commutator)
      CB = commutator(B, AB) = [B,[A,B]]     Hermitian
    evolution_unitary(G, s) = exp(-i*s*G), so:
      evolution_unitary(CA, alpha2*dt^3) = exp(-i*alpha2*dt^3*[A,[A,B]])
      with alpha2=-1/6 this gives exp(+i/6*dt^3*[A,[A,B]])  -- the correct sign.
    """
    A, B   = model.A, model.B
    AB     = commutator(A, B)          # [A,B], anti-Hermitian
    C      = 1j * AB                   # i[A,B], Hermitian
    CA     = commutator(A, AB)         # [A,[A,B]], Hermitian
    CB     = commutator(B, AB)         # [B,[A,B]], Hermitian
    alpha2, alpha3 = -1./6., -1./3.
    # GPT-1 core
    core = (
        evolution_unitary(C,  -0.5 * dt**2)
        @ evolution_unitary(B, dt)
        @ evolution_unitary(A, dt)
    )
    # GPT-2 correction layers
    corr = (
        evolution_unitary(CB, alpha3 * dt**3)
        @ evolution_unitary(CA, alpha2 * dt**3)
    )
    return corr @ core


def gpt2_rep(model, T, N):
    dt   = T / N
    step = gpt2_step(model, dt)
    R    = np.eye(model.dimension, dtype=complex)
    for _ in range(N):
        R = step @ R
    return R


def sweep(model, dt, T_arr):
    """Return (trot1, gpt1, suz2, gpt2) error arrays for fixed dt, varying T."""
    e1, g1, s2, g2 = [], [], [], []
    for T in T_arr:
        N  = max(1, round(T / dt))
        Ue = exact_propagator(model.hamiltonian, T)
        e1.append(spec_err(first_order_trotter(model, T, N), Ue))
        g1.append(spec_err(lc_qaoa_repeated(model, T, N), Ue))
        s2.append(spec_err(second_order_suzuki(model, T, N), Ue))
        g2.append(spec_err(gpt2_rep(model, T, N), Ue))
        print(f"  T={T:5.1f}  N={N:5d}"
              f"  Trot={e1[-1]:.3e} GPT1={g1[-1]:.3e}"
              f"  r1={e1[-1]/max(g1[-1],1e-16):.1f}x"
              f"  Suz2={s2[-1]:.3e} GPT2={g2[-1]:.3e}"
              f"  r2={s2[-1]/max(g2[-1],1e-16):.1f}x")
    return (np.array(e1), np.array(g1), np.array(s2), np.array(g2))


# ── model ────────────────────────────────────────────────────────────────────
model = tfim_hamiltonian(4, 1.0, 0.5)
OUT   = os.path.join(os.path.dirname(__file__), "..", "research_paper", "figures")
os.makedirs(OUT, exist_ok=True)

# sanity: reproduce paper table values
Ue05 = exact_propagator(model.hamiltonian, 0.5)
t1 = spec_err(first_order_trotter(model, 0.5, 10), Ue05)
g1 = spec_err(lc_qaoa_repeated(model, 0.5, 10), Ue05)
s2 = spec_err(second_order_suzuki(model, 0.5, 24), Ue05)
g2 = spec_err(gpt2_rep(model, 0.5, 24), Ue05)
print(f"Sanity T=0.5  Trot1={t1:.3e}(~4.76e-2)  GPT1={g1:.3e}(~3.89e-3)"
      f"  Suz2={s2:.3e}(~1.69e-4)  GPT2={g2:.3e}(~1.58e-5)")

# ── Sweep A: dt=0.04 — ratio ≥10× for T ≲ 2.2 ────────────────────────────
DT_A = 0.04
T_A  = np.concatenate([np.arange(0.1, 1.0, 0.1),
                        np.arange(1.0, 3.1, 0.25)])
print(f"\n── Sweep A: dt={DT_A} ──")
e1A, g1A, s2A, g2A = sweep(model, DT_A, T_A)

# ── Sweep B: dt=0.01 — ratio ≥10× for all T ≤ 10 (N ≤ 1000) ─────────────
DT_B = 0.01
T_B  = np.concatenate([np.arange(0.5, 2.0, 0.5),
                        np.arange(2.0, 10.5, 1.0)])
print(f"\n── Sweep B: dt={DT_B} ──")
e1B, g1B, s2B, g2B = sweep(model, DT_B, T_B)

# ── 4-panel figure ───────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.labelsize": 12, "figure.facecolor": "white"})
C_T = "#555555"; C_G1 = "#1f77b4"; C_S2 = "#aaaaaa"; C_G2 = "#ff7f0e"

r1A = e1A / np.maximum(g1A, 1e-16)
r2A = s2A / np.maximum(g2A, 1e-16)
r1B = e1B / np.maximum(g1B, 1e-16)
r2B = s2B / np.maximum(g2B, 1e-16)

fig, ax4 = plt.subplots(2, 2, figsize=(11, 8.5))

for ax, T_arr, e1, g1, s2, g2, title in [
    (ax4[0,0], T_A, e1A, g1A, s2A, g2A, rf"(a) Error vs $T$, $\delta t={DT_A}$"),
    (ax4[1,0], T_B, e1B, g1B, s2B, g2B,
     rf"(c) Error vs $T$, $\delta t={DT_B}$ ($N\leq1000$ at $T\!=\!10$)"),
]:
    ax.semilogy(T_arr, e1, C_T,  ls="--", lw=1.8, label="Trotter-1")
    ax.semilogy(T_arr, g1, C_G1, ls="-",  lw=2.2, label="Lie GPT-1")
    ax.semilogy(T_arr, s2, C_S2, ls=":",  lw=1.8, label="Suzuki-2")
    ax.semilogy(T_arr, g2, C_G2, ls="-",  lw=2.2, label="Lie GPT-2")
    ax.set_xlabel(r"Total time $T$")
    ax.set_ylabel(r"Spectral error $\|\cdot\|_2$")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

for ax, T_arr, r1, r2, title in [
    (ax4[0,1], T_A, r1A, r2A,
     rf"(b) Ratio vs $T$, $\delta t={DT_A}$ — $\geq\!10\times$ for $T\!\leq\!2.2$"),
    (ax4[1,1], T_B, r1B, r2B,
     rf"(d) Ratio vs $T$, $\delta t={DT_B}$ — $\geq\!10\times$ for all $T\!\leq\!10$"),
]:
    ax.plot(T_arr, r1, C_G1, lw=2.2, label=r"Trotter-1 / GPT-1")
    ax.plot(T_arr, r2, C_G2, lw=2.2, ls="--", label=r"Suzuki-2 / GPT-2")
    ax.axhline(10, color="crimson", lw=1.5, ls=":", label=r"$10\times$ threshold")
    ax.set_xlabel(r"Total time $T$")
    ax.set_ylabel(r"Error ratio (baseline / Lie GPT)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=0)

fig.tight_layout()
p = os.path.join(OUT, "liegpt_longtime.png")
fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
print(f"\nSaved: {p}")

# ── N-sweep at T=10 ──────────────────────────────────────────────────────────
T10 = 10.0
Ue10 = exact_propagator(model.hamiltonian, T10)
N_SWP = [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500, 2000]

rows = []
print(f"\n── N-sweep at T={T10} ──")
print(f"{'N':>6}  {'dt':>7}  {'Trot-1':>12}  {'GPT-1':>12}  {'r1':>7}  "
      f"{'Suz-2':>12}  {'GPT-2':>12}  {'r2':>7}")
for N in N_SWP:
    dt  = T10 / N
    e1  = spec_err(first_order_trotter(model, T10, N), Ue10)
    g1  = spec_err(lc_qaoa_repeated(model, T10, N), Ue10)
    s2  = spec_err(second_order_suzuki(model, T10, N), Ue10)
    g2  = spec_err(gpt2_rep(model, T10, N), Ue10)
    r1  = e1 / max(g1, 1e-16); r2 = s2 / max(g2, 1e-16)
    rows.append((N, dt, e1, g1, r1, s2, g2, r2))
    print(f"{N:6d}  {dt:7.4f}  {e1:12.3e}  {g1:12.3e}  {r1:7.1f}x  "
          f"{s2:12.3e}  {g2:12.3e}  {r2:7.1f}x")

N_a  = np.array([r[0] for r in rows], float)
e1_a = np.array([r[2] for r in rows]); g1_a = np.array([r[3] for r in rows])
s2_a = np.array([r[5] for r in rows]); g2_a = np.array([r[6] for r in rows])
r1_a = np.array([r[4] for r in rows]); r2_a = np.array([r[7] for r in rows])

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes2[0]
ax.loglog(N_a, e1_a, C_T,  ls="--", lw=1.8, label="Trotter-1")
ax.loglog(N_a, g1_a, C_G1, ls="-",  lw=2.2, label="Lie GPT-1")
ax.loglog(N_a, s2_a, C_S2, ls=":",  lw=1.8, label="Suzuki-2")
ax.loglog(N_a, g2_a, C_G2, ls="-",  lw=2.2, label="Lie GPT-2")
idx = 2
ax.loglog(N_a, e1_a[idx]*(N_a/N_a[idx])**-1, "k--", lw=0.7, alpha=0.4, label=r"$N^{-1}$")
ax.loglog(N_a, g1_a[idx]*(N_a/N_a[idx])**-2, "k-",  lw=0.7, alpha=0.4, label=r"$N^{-2}$")
ax.loglog(N_a, g2_a[idx]*(N_a/N_a[idx])**-3, "k:",  lw=0.7, alpha=0.4, label=r"$N^{-3}$")
ax.set_xlabel(r"Number of steps $N$"); ax.set_ylabel(r"Spectral error")
ax.set_title(rf"(a) Error vs $N$, $T={T10}$", fontsize=12)
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)

ax = axes2[1]
ax.semilogx(N_a, r1_a, C_G1, lw=2.2, label=r"Trotter-1 / GPT-1")
ax.semilogx(N_a, r2_a, C_G2, lw=2.2, ls="--", label=r"Suzuki-2 / GPT-2")
ax.axhline(10, color="crimson", lw=1.5, ls=":", label=r"$10\times$ threshold")
ax.axvline(1000, color="gray", lw=1.0, ls=":", alpha=0.6, label="$N=1000$")
ax.set_xlabel(r"Number of steps $N$"); ax.set_ylabel(r"Error ratio")
ax.set_title(rf"(b) Ratio vs $N$, $T={T10}$", fontsize=12)
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3); ax.set_ylim(bottom=0)

fig2.tight_layout()
p2 = os.path.join(OUT, "liegpt_longtime_nsweep.png")
fig2.savefig(p2, dpi=180, bbox_inches="tight"); plt.close(fig2)
print(f"Saved: {p2}")
print("Done.")
