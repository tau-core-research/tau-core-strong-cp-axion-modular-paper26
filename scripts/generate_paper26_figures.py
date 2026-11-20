#!/usr/bin/env python3
"""
Generate 3 Publication-Grade Vector PDF Figures for Tau Core Paper 26 (v1.1.0):
  Fig 1: Z_16 Discrete-Gauge Axion Window (|Delta theta_bar|, SN 1987A, QUMOND, |delta eta_B/eta_B|, beta_iso)
  Fig 2: (a) 2-Field Starobinsky + Conformal Flaton Monotonic ODE Glide, (b) 2-Loop SM + G_PS yb/ytau(mu)
  Fig 3: (a) SL(2,Z) Fundamental Domain F & Solution B Benchmark tau_0, (b) (m_a, |g_{a gamma gamma}|) Experimental Fork
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
FIG_DIR = os.path.join(REPO_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(SCRIPT_DIR, "paper26_verification_results.json")) as f:
    res = json.load(f)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.8,
    "axes.labelsize": 10.2,
    "axes.titlesize": 10.2,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "figure.dpi": 300,
})

M_PL = 2.435323e18
M_INF = 3.1959e13
H_INF = M_INF / 2.0
T_RH = 5.13e9
CHI_TOP = 0.0755**4

# =============================================================================
# FIGURE 1: Z_16 DISCRETE-GAUGE AXION WINDOW & COSMOLOGICAL SAFETY
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

fa_grid = np.geomspace(2.0e8, 2.0e12, 300)
for d_val, col, ls, lbl in [
    (12, "#d95f02", "--", r"$d = 12$ ($\mathbb{Z}_{12}$, $v_{\mathrm{PQ}}=12 f_a$)"),
    (14, "#7570b3", "-.", r"$d = 14$ ($\mathbb{Z}_{14}$, $v_{\mathrm{PQ}}=14 f_a$)"),
    (16, "#1b4f72", "-",  r"$d = 16$ ($\mathbb{Z}_{16}$, Solution A1)"),
]:
    vpq_g = d_val * fa_grid
    th_g = np.minimum(1.0, 2.0 * ((vpq_g / np.sqrt(2.0))**d_val) / ((M_PL**(d_val - 4)) * CHI_TOP))
    ax1.loglog(fa_grid, th_g, color=col, lw=1.9, ls=ls, label=lbl)

ax1.axhline(1.2e-10, color="#c0392b", lw=1.4, ls=":", label=r"nEDM bound $|\bar{\theta}| < 1.2\times 10^{-10}$")
ax1.axvspan(2.0e8, 1.76e9, color="#e74c3c", alpha=0.14, label=r"SN 1987A ($|g_{ap}| > 2.5\times 10^{-10}$)")
fa_qumond_max = 9.0e11 * ((0.003 / 0.12)**(1.0 / 1.165))
ax1.axvspan(fa_qumond_max, 2.0e12, color="#7f8c8d", alpha=0.18, label=r"QUMOND ceiling ($\Omega_a h^2 > 0.003$)")

fa_A1 = res["solution_A1"]["f_a_GeV"]
th_A1 = res["solution_A1"]["abs_delta_theta"]
ax1.plot(fa_A1, th_A1, marker="*", markersize=11, color="#f39c12", markeredgecolor="black", zorder=6,
         label=r"Benchmark A1 ($2.20\times 10^{10}\;\mathrm{GeV}$)")
ax1.set_xlim(2.0e8, 2.0e12)
ax1.set_ylim(1e-42, 1e1)
ax1.set_xlabel(r"Axion decay constant $f_a\;[\mathrm{GeV}]$")
ax1.set_ylabel(r"Planck-induced shift $|\Delta\bar{\theta}_{\mathrm{QCD}}|$")
ax1.set_title(r"(a) Discrete-Gauge $\mathbb{Z}_N$ Quality & Allowed $f_a$ Window")
ax1.grid(True, which="both", alpha=0.22)
ax1.legend(loc="upper left", frameon=True, framealpha=0.93, fontsize=7.7)

# Panel (b): |delta eta_B / eta_B| & beta_iso across f_a in [2e9, 1e11] GeV:
fa_sub = np.geomspace(1.5e9, 1.0e11, 90)
d_eta_pct = []
beta_iso_arr = []
f_eff_CMB = np.sqrt(2.0) * res["solution_A1"]["chi_CMB_GeV"] / 16.0
for fa_v in fa_sub:
    MQ_v = 0.24 * (16.0 * fa_v) / np.sqrt(2.0)
    xv = MQ_v / T_RH
    if xv > 35.0:
        d_eta_pct.append(1e-7)
    else:
        ri, _ = quad(lambda u: (u**2)*np.sqrt(u**2+xv**2)/(np.exp(np.sqrt(u**2+xv**2))+1.0), 0.0, 55.0)
        pi_, _ = quad(lambda u: (u**4)/(np.sqrt(u**2+xv**2)*(np.exp(np.sqrt(u**2+xv**2))+1.0)), 0.0, 55.0)
        r1 = ri / (2.0 * np.pi**2); s1 = r1 + pi_ / (6.0 * np.pi**2)
        dgr = 256.0 * r1 / (np.pi**2 / 30.0); dgs = 256.0 * s1 / (2.0 * np.pi**2 / 45.0)
        eb_rat = 1.0 / (((106.75 + dgs)/106.75) * ((106.75/(106.75 + dgr))**0.75))
        d_eta_pct.append(max(1e-7, abs(eb_rat - 1.0) * 100.0))
    om_v = 0.12 * ((fa_v / 9.0e11)**1.165)
    p_iso = (om_v / 0.12)**2 * (H_INF / (np.pi * f_eff_CMB))**2
    beta_iso_arr.append(p_iso / (2.10e-9 + p_iso))

ax2.loglog(fa_sub, d_eta_pct, color="#1b4f72", lw=2.0, label=r"$|\delta\eta_B/\eta_B|\;[\%]$ ($y_Q = 0.24$)")
ax2.loglog(fa_sub, beta_iso_arr, color="#27ae60", lw=2.0, ls="--", label=r"CMB isocurvature $\beta_{\mathrm{iso}}(\theta_i=1)$")
ax2.axhline(0.038, color="#c0392b", lw=1.3, ls=":", label=r"Planck 2018 $\beta_{\mathrm{iso}} < 0.038$")
ax2.axhline(0.001, color="#2ecc71", lw=1.1, ls="-.", label=r"LiteBIRD target ($\beta_{\mathrm{iso}} \sim 10^{-3}$)")
ax2.axvline(fa_A1, color="#f39c12", lw=1.4, ls="-", alpha=0.85, label=r"Benchmark A1 ($f_a = 2.20\times 10^{10}\;\mathrm{GeV}$)")
ax2.set_xlim(1.5e9, 1.0e11)
ax2.set_ylim(1e-6, 1e2)
ax2.set_xlabel(r"Axion decay constant $f_a\;[\mathrm{GeV}]$")
ax2.set_ylabel(r"Dimensionless ratio / percentage")
ax2.set_title(r"(b) Leptogenesis Boltzmann Safety & CMB Isocurvature")
ax2.grid(True, which="both", alpha=0.22)
ax2.legend(loc="lower left", frameon=True, framealpha=0.92)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_Z16_axion_parameter_space.pdf"), bbox_inches="tight")
plt.close()

# =============================================================================
# FIGURE 2: 2-FIELD ADIABATIC GLIDE ODE & 2-LOOP + G_PS yb/ytau RUNNING
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

t_arr = np.array(res["solution_A1"]["ode_trace_t"])
phi_arr = np.array(res["solution_A1"]["ode_trace_phi"])
r_arr = np.array(res["solution_A1"]["ode_trace_r"]) / 1.0e17

ax1.plot(t_arr, r_arr, color="#1b4f72", lw=2.1, label=r"Flaton radial field $\hat{r}_\chi(t)\;[10^{17}\;\mathrm{GeV}]$")
ax1.plot(t_arr, phi_arr, color="#d95f02", lw=1.1, alpha=0.80, label=r"Starobinsky scalaron $\phi(t)/M_{\mathrm{Pl}}$")
ax1.axhline(0.0, color="black", lw=0.8, ls=":")
ax1.set_xlim(0.0, 300.0)
ax1.set_xlabel(r"Dimensionless time $M_{\mathrm{inf}}\,t$ ($\sim 48$ scalaron oscillations)")
ax1.set_ylabel(r"Field amplitude")
ax1.set_title(r"(a) Monotonic Conformal Flaton Glide ($\min \hat{r}_\chi > 0$)")
ax1.grid(True, alpha=0.22)
ax1.legend(loc="upper right", frameon=True, framealpha=0.92)

# Panel (b): Recompute smooth curves of yb(mu)/ytau(mu) from M_Z to M_R in <1s by loading RGE helpers only:
with open(os.path.join(SCRIPT_DIR, "verify_paper26_strong_cp_closure.py")) as vf:
    v_lines = vf.read().splitlines()
stop_idx = next(i for i, ln in enumerate(v_lines) if "THEOREM 4.1:" in ln)
vp26_ns = {"__file__": os.path.join(SCRIPT_DIR, "verify_paper26_strong_cp_closure.py"), "__name__": "vp26_rge"}
exec("\n".join(v_lines[:stop_idx - 2]), vp26_ns)
class _NS: pass
vp26 = _NS()
for k_n, v_n in vp26_ns.items():
    setattr(vp26, k_n, v_n)

t_des = np.linspace(0.0, vp26.t_MR, 180)
sol_des = solve_ivp(vp26.rge_sm_2loop, [0.0, vp26.t_MR], vp26.y0_MZ, t_eval=t_des, method="DOP853", rtol=1e-10, atol=1e-12)
mu_des = vp26.M_Z * np.exp(t_des)
rat_des = [vp26.extract_sm(sol_des.y[:, i])["yd"][2] / vp26.extract_sm(sol_des.y[:, i])["ye"][2] for i in range(len(t_des))]

# Solution A1 curve: [0, t_MQ0] -> [t_MQ0, t_PS] -> [t_PS, t_MR]
mu_1 = vp26.M_Z * np.exp(vp26.t_grid_1)
rat_1 = [vp26.extract_sm(vp26.sol_1.y[:, i])["yd"][2] / vp26.extract_sm(vp26.sol_1.y[:, i])["ye"][2] for i in range(len(vp26.t_grid_1))]
mu_2 = vp26.M_Z * np.exp(vp26.t_grid_2)
rat_2 = [vp26.extract_sm(vp26.sol_2.y[:, i])["yd"][2] / vp26.extract_sm(vp26.sol_2.y[:, i])["ye"][2] for i in range(len(vp26.t_grid_2))]
t_ps_grid = np.linspace(vp26.t_PS, vp26.t_MR, 50)
mu_3 = vp26.M_Z * np.exp(t_ps_grid)
rat_3 = [vp26.run_ps_bidoublet(vp26.t_PS, tv, (vp26.g3_p, vp26.g2_p, vp26.g2R_p), vp26.o_PS_A1["Yu"],
                               0.25*(3.0*vp26.o_PS_A1["Yd"]+vp26.o_PS_A1["Ye"]),
                               0.25*(vp26.o_PS_A1["Yd"]-vp26.o_PS_A1["Ye"]),
                               vp26.b_PS_full + 32.0/3.0)["yd"][2]
         / vp26.run_ps_bidoublet(vp26.t_PS, tv, (vp26.g3_p, vp26.g2_p, vp26.g2R_p), vp26.o_PS_A1["Yu"],
                                 0.25*(3.0*vp26.o_PS_A1["Yd"]+vp26.o_PS_A1["Ye"]),
                                 0.25*(vp26.o_PS_A1["Yd"]-vp26.o_PS_A1["Ye"]),
                                 vp26.b_PS_full + 32.0/3.0)["ye"][2]
         for tv in [vp26.t_PS + 1e-6] + list(t_ps_grid[1:])]
rat_3[0] = rat_2[-1]

ax2.semilogx(mu_des, rat_des, color="#c0392b", lw=1.7, ls="--", label=r"Pure 2-loop SM desert ($0.6173$, excluded)")
ax2.semilogx(np.concatenate([mu_1, mu_2, mu_3]), np.concatenate([rat_1, rat_2, rat_3]),
             color="#1b4f72", lw=2.2, label=r"Solution A1 ($M_Q \to M_{\mathrm{PS}} \to M_R$: $0.6931$)")
ax2.axhline(2.0/3.0, color="#27ae60", lw=1.4, ls=":", label=r"Geometric $K_4$ target $2/3 = 0.6667$")
ax2.axvline(res["solution_A1"]["M_Q0_GeV"], color="#8e44ad", lw=1.1, ls="-.", label=r"$M_Q(0) = 5.97\times 10^{10}\;\mathrm{GeV}$")
ax2.axvline(vp26.M_PS, color="#7f8c8d", lw=1.1, ls=":", label=r"$M_{\mathrm{PS}} = 3.66\times 10^{11}\;\mathrm{GeV}$")
ax2.set_xlim(vp26.M_Z, vp26.M_R)
ax2.set_ylim(0.55, 1.70)
ax2.set_xlabel(r"Renormalization scale $\mu\;[\mathrm{GeV}]$")
ax2.set_ylabel(r"Yukawa ratio $y_b(\mu) / y_\tau(\mu)$")
ax2.set_title("(b) Preservation of $G_{\\mathrm{PS}}$ $b$\u2013$\\tau$ Stabilization in A1")
ax2.grid(True, which="both", alpha=0.22)
ax2.legend(loc="upper right", frameon=True, framealpha=0.92)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_flaton_adiabatic_glide_and_rge.pdf"), bbox_inches="tight")
plt.close()

# =============================================================================
# FIGURE 3: MODULAR FUNDAMENTAL DOMAIN F & EXPERIMENTAL FALSIFIABILITY FORK
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2))

re_arc = np.linspace(-0.5, 0.5, 300)
im_arc = np.sqrt(1.0 - re_arc**2)
ax1.fill_between(re_arc, im_arc, 2.0, color="#d6eaf8", alpha=0.55, label=r"$\mathrm{SL}(2,\mathbb{Z})$ fundamental domain $\mathcal{F}$")
ax1.plot(re_arc, im_arc, color="#c0392b", lw=2.0, label=r"$CP$-conserving boundary ($\partial\mathcal{F} \cup \{\mathrm{Re}\,\tau=0\}$)")
ax1.plot([-0.5, -0.5], [np.sqrt(0.75), 2.0], color="#c0392b", lw=2.0)
ax1.plot([0.5, 0.5], [np.sqrt(0.75), 2.0], color="#c0392b", lw=2.0)
ax1.plot([0.0, 0.0], [1.0, 2.0], color="#c0392b", lw=1.4, ls="--")

tau_re = res["solution_B"]["tau_re"]
tau_im = res["solution_B"]["tau_im"]
ax1.plot(tau_re, tau_im, marker="o", markersize=9, color="#1b4f72", markeredgecolor="black", zorder=6,
         label=rf"Solution B ($\tau_0 = {tau_re:+.3f} + {tau_im:.3f}i$)")
ax1.annotate(r"13-obs $\mathrm{Spin}(10)\!+\!\mathbf{120}_H$ fit" + "\n" + r"$\arg\det(Y_u Y_d)=0$",
             xy=(tau_re, tau_im), xytext=(-0.44, 1.63),
             arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=1.1), fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1b4f72", alpha=0.92))
ax1.set_xlim(-0.65, 0.65)
ax1.set_ylim(0.70, 1.95)
ax1.set_xlabel(r"$\mathrm{Re}\,\tau$")
ax1.set_ylabel(r"$\mathrm{Im}\,\tau$")
ax1.set_title(r"(a) Cusp-Free Modular Solution B Vacuum $\tau_0 \in \mathcal{F}$")
ax1.grid(True, alpha=0.22)
ax1.legend(loc="lower center", frameon=True, framealpha=0.92)

# Panel (b): Experimental (m_a, |g_{a gamma gamma}|) fork:
ma_eV = np.geomspace(1e-6, 1e-1, 300)
fa_band = 5.691e-3 / ma_eV * 1.0e9
g_KSVZ = (1.0 / (137.036 * 2.0 * np.pi * fa_band)) * abs(0.0 - 1.92)
g_E83  = (1.0 / (137.036 * 2.0 * np.pi * fa_band)) * abs(8.0/3.0 - 1.92)
ax2.loglog(ma_eV * 1e3, g_KSVZ, color="#7f8c8d", lw=1.4, ls="--", label=r"Standard KSVZ ($E/N = 0$)")
ax2.loglog(ma_eV * 1e3, g_E83,  color="#1b4f72", lw=2.0, label=r"$\mathrm{Spin}(10)$ $\mathbf{16}_Q\oplus\overline{\mathbf{16}}_Q$ ($E/N = 8/3$)")
ax2.axhspan(6.6e-11, 1e-9, color="#e74c3c", alpha=0.15, label=r"CAST / HB Stars Excluded")
ax2.axvspan(3.0, 100.0, color="#e67e22", alpha=0.12, label=r"SN 1987A Pion Conv. ($m_a > 3\;\mathrm{meV}$)")
ax2.axvspan(0.04, 0.40, ymin=0.0, ymax=0.55, color="#2ecc71", alpha=0.14, label=r"MADMAX / BREAD Dielectric Band")

ma_A1 = res["solution_A1"]["m_a_meV"]
ga_A1 = res["solution_A1"]["g_agamma_GeVinv"]
beta_A1 = res["solution_A1"]["beta_iso"]
dneff_A1 = res["solution_A1"]["dNeff_tot"]
ax2.plot(ma_A1, ga_A1, marker="*", markersize=12, color="#f39c12", markeredgecolor="black", zorder=6,
         label=rf"Solution A1 ($m_a = {ma_A1:.4f}\;\mathrm{{meV}}$)")
ax2.annotate("Solution A1:\n" + rf"$\beta_{{\mathrm{{iso}}}}={beta_A1:.4f},\;\Delta N_{{\mathrm{{eff}}}}={dneff_A1:.3f}$" + "\nvs. Solution B: Null axion",
             xy=(ma_A1, ga_A1), xytext=(0.0025, 1.5e-12),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.1), fontsize=8.1,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#f39c12", alpha=0.93))
ax2.set_xlim(1e-3, 1e2)
ax2.set_ylim(1e-16, 1e-9)
ax2.set_xlabel(r"Axion mass $m_a\;[\mathrm{meV}]$")
ax2.set_ylabel(r"Photon coupling $|g_{a\gamma\gamma}|\;[\mathrm{GeV}^{-1}]$")
ax2.set_title(r"(b) Experimental Falsifiability: Solution A1 vs. B")
ax2.grid(True, which="both", alpha=0.22)
ax2.legend(loc="lower right", frameon=True, framealpha=0.92)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_modular_spin10_and_experimental_fork.pdf"), bbox_inches="tight")
plt.close()

print("ALL 3 PAPER 26 VECTOR PDF FIGURES GENERATED SUCCESSFULLY.")
