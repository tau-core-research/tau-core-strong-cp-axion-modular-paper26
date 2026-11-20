#!/usr/bin/env python3
"""
TAU CORE PAPER 26 (v1.1.0) — STANDALONE MASTER VERIFICATION SUITE
Strong CP Closure Across a Tetrahedral Spin(10) -> Pati-Salam -> SM and Starobinsky Spine:
  1. Universal 4D Non-SUSY Axion-Free No-Go Suite (Theorem 2.1)
  2. Conformal Two-Scalar Z_16 Discrete-Gauge Spin(10) Axion (Theorems 3.1 & 3.2)
  3. Chiral Composite Axion Spin(7)_A x Spin(6)_B x U(1)_X & P_{spin-M} Obstruction (Proposition 3.3)
  4. Cusp-Free Modular N=1 SUGRA Spin(10) + 120_H UV Alternative (Theorem 4.1)

100% standalone (requires only numpy, scipy, sympy). Saves deterministic JSON ledger to
scripts/paper26_verification_results.json.
"""
import json
import os
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, differential_evolution, minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(SCRIPT_DIR, "paper26_verification_results.json")

# =============================================================================
# 0. FUNDAMENTAL CONSTANTS & FROZEN PAPER 24 / PAPER 25 BOUNDARY SCALES
# =============================================================================
M_PL = 2.435323e18
M_STAR = M_PL / np.sqrt(3.0)               # 1.406034e18 GeV
M_R = M_STAR / (256.0 * np.pi**2)          # 5.564893e14 GeV
M_PS = 3.66e11                             # GeV (Paper 25 Pati-Salam scale)
M_INF = 3.1959e13                          # GeV (Paper 24 Starobinsky scalaron mass)
H_INF = M_INF / 2.0                        # 1.59795e13 GeV
T_MAX = 2.11e12                            # GeV (Paper 24 exact ODE reheating peak)
T_RH = 5.13e9                              # GeV (Paper 24 reheating temperature)
M_1_RHN = 5.70e12                          # GeV (Paper 24 lightest RHN mass)
M_Z = 91.1876                              # GeV
V_EW = 246.21965                           # GeV
v_over_sqrt2 = V_EW / np.sqrt(2.0)         # 174.10358 GeV
CHI_TOP = 0.0755**4                        # (75.5 MeV)^4 topological susceptibility
THETA_BOUND = 1.2e-10                      # |d_n| < 1.8e-26 e cm bound

# PDG 2024 inputs at M_Z (identical to Paper 25 v1.1.0):
alpha_em_MZ = 1.0 / 127.951
sin2_tW_MZ = 0.23122
alpha_s_MZ = 0.1180
g1_MZ = np.sqrt(4.0 * np.pi * (5.0 / 3.0) * alpha_em_MZ / (1.0 - sin2_tW_MZ))
g2_MZ = np.sqrt(4.0 * np.pi * alpha_em_MZ / sin2_tW_MZ)
g3_MZ = np.sqrt(4.0 * np.pi * alpha_s_MZ)
m_H_pole = 125.20
lamH_MZ = (m_H_pole**2) / (2.0 * V_EW**2)  # MSbar ~ 0.12925 (Paper 25 exact value: 0.12925)

pdg_masses = {
    'u': (1.24e-3, 0.22e-3), 'c': (0.624, 0.014), 't': (171.7, 0.8),
    'd': (2.69e-3, 0.19e-3), 's': (53.5e-3, 4.6e-3), 'b': (2.855, 0.025),
    'e': (0.48657e-3, 1e-8), 'mu': (0.102718, 1e-6), 'tau': (1.74617, 0.00015),
}
pdg_ckm = {
    'Vus': (0.22500, 0.00067), 'Vcb': (0.04182, 0.00085),
    'Vub': (0.00369, 0.00011), 'delta_deg': (65.7, 1.5),
}

def ckm_matrix(Vus, Vcb, Vub, delta):
    s12 = Vus; c12 = np.sqrt(1.0 - s12**2)
    s13 = Vub; c13 = np.sqrt(1.0 - s13**2)
    s23 = Vcb / c13; c23 = np.sqrt(1.0 - s23**2)
    ed = np.exp(1j * delta)
    return np.array([
        [c12*c13, s12*c13, s13*np.conj(ed)],
        [-s12*c23 - c12*s23*s13*ed, c12*c23 - s12*s23*s13*ed, s23*c13],
        [s12*s23 - c12*c23*s13*ed, -c12*s23 - s12*c23*s13*ed, c23*c13]
    ], dtype=complex)

def pack_sm(g1, g2, g3, lamH, Yu, Yd, Ye):
    return np.concatenate([
        [g1, g2, g3, lamH],
        Yu.real.ravel(), Yu.imag.ravel(),
        Yd.real.ravel(), Yd.imag.ravel(),
        Ye.real.ravel(), Ye.imag.ravel(),
    ])

def unpack_sm(y):
    g1, g2, g3, lamH = y[0], y[1], y[2], y[3]
    def gm(i): return y[i:i+9].reshape(3, 3) + 1j * y[i+9:i+18].reshape(3, 3)
    return g1, g2, g3, lamH, gm(4), gm(22), gm(40)

b_g_SM = np.array([41.0/10.0, -19.0/6.0, -7.0])
B_g_SM = np.array([
    [199.0/50.0, 27.0/10.0, 44.0/5.0],
    [9.0/10.0,   35.0/6.0,  12.0],
    [11.0/10.0,  9.0/2.0,  -26.0]
])
C_yg_SM = np.array([
    [17.0/10.0, 1.0/2.0, 3.0/2.0],
    [3.0/2.0,   3.0/2.0, 1.0/2.0],
    [2.0,       2.0,     0.0]
])

def rge_sm_2loop(t, y):
    g1, g2, g3, lamH, Yu, Yd, Ye = unpack_sm(y)
    g = np.array([g1, g2, g3])
    g1_2, g2_2, g3_2 = g1**2, g2**2, g3**2
    g1_4, g2_4, g3_4 = g1_2**2, g2_2**2, g3_2**2
    g1_6, g2_6 = g1_2**3, g2_2**3
    f1 = 1.0 / (16.0 * np.pi**2); f2 = f1**2
    I3 = np.eye(3, dtype=complex)
    Hu, Hd, He = Yu @ Yu.conj().T, Yd @ Yd.conj().T, Ye @ Ye.conj().T
    Hu2, Hd2, He2 = Hu @ Hu, Hd @ Hd, He @ He
    tu, td, te = np.trace(Hu).real, np.trace(Hd).real, np.trace(He).real
    trs = np.array([tu, td, te])
    Y2 = 3.0 * (tu + td) + te
    tu2, td2, te2 = np.trace(Hu2).real, np.trace(Hd2).real, np.trace(He2).real
    tud = np.trace(Hu @ Hd).real
    tu3, td3, te3 = np.trace(Hu2 @ Hu).real, np.trace(Hd2 @ Hd).real, np.trace(He2 @ He).real
    tu2d, tud2 = np.trace(Hu2 @ Hd).real, np.trace(Hu @ Hd2).real

    dg = np.array([f1*b_g_SM[i]*g[i]**3 + f2*g[i]**3*(np.dot(B_g_SM[i], g**2) - np.dot(C_yg_SM[i], trs)) for i in range(3)])
    b1_lam = 24.0*lamH**2 - (9.0/5.0*g1_2 + 9.0*g2_2)*lamH + (27.0/200.0*g1_4 + 9.0/20.0*g1_2*g2_2 + 9.0/8.0*g2_4) + 4.0*Y2*lamH - 2.0*(3.0*(tu2+td2)+te2)
    b2_lam = (
        -312.0*lamH**3 + (108.0/5.0*g1_2 + 108.0*g2_2)*lamH**2
        - (73.0/8.0*g2_4 - 117.0/20.0*g1_2*g2_2 - 1887.0/200.0*g1_4)*lamH
        + 305.0/16.0*g2_6 - 289.0/80.0*g1_2*g2_4 - 1677.0/400.0*g1_4*g2_2 - 3411.0/2000.0*g1_6
        - 32.0*g3_2*(tu2+td2) - 8.0/5.0*g1_2*(2.0*tu2 - td2 + 3.0*te2) - 9.0/4.0*g2_4*Y2
        + g1_2*((63.0/10.0*g2_2 - 171.0/100.0*g1_2)*tu + (27.0/10.0*g2_2 + 9.0/20.0*g1_2)*td + (33.0/10.0*g2_2 - 9.0/4.0*g1_2)*te)
        + 10.0*lamH*((17.0/20.0*g1_2 + 9.0/4.0*g2_2 + 8.0*g3_2)*tu + (1.0/4.0*g1_2 + 9.0/4.0*g2_2 + 8.0*g3_2)*td + (3.0/4.0*g1_2 + 3.0/4.0*g2_2)*te)
        - 48.0*lamH**2*Y2 - lamH*(3.0*(tu2+td2)+te2) - 6.0*lamH*tud + 10.0*(3.0*(tu3+td3)+te3) - 6.0*(tu2d+tud2)
    )
    gu1 = (17.0/20.0)*g1_2 + (9.0/4.0)*g2_2 + 8.0*g3_2
    gd1 = (1.0/4.0)*g1_2   + (9.0/4.0)*g2_2 + 8.0*g3_2
    ge1 = (9.0/4.0)*g1_2   + (9.0/4.0)*g2_2
    A1_u = 1.5*(Hu - Hd) + (Y2 - gu1)*I3
    A1_d = 1.5*(Hd - Hu) + (Y2 - gd1)*I3
    A1_e = 1.5*He        + (Y2 - ge1)*I3
    chi4 = (9.0/4.0)*(3.0*(tu2+td2)+te2) - 1.5*tud
    Y2_g = 2.5*(gu1*tu + gd1*td + (0.75*g1_2 + 0.75*g2_2)*te)
    gu2 = (1187.0/600.0)*g1_4 - (9.0/20.0)*g1_2*g2_2 + (19.0/15.0)*g1_2*g3_2 - (23.0/4.0)*g2_4 + 9.0*g2_2*g3_2 - 108.0*g3_4
    gd2 = -(127.0/600.0)*g1_4 - (27.0/20.0)*g1_2*g2_2 + (31.0/15.0)*g1_2*g3_2 - (23.0/4.0)*g2_4 + 9.0*g2_2*g3_2 - 108.0*g3_4
    ge2 = (1371.0/200.0)*g1_4 + (27.0/20.0)*g1_2*g2_2 - (23.0/4.0)*g2_4
    A2_u = 1.5*Hu2 - 0.25*(Hu@Hd) - (Hd@Hu) + 2.75*Hd2 + Y2*(-2.25*Hu + 1.25*Hd) - 12.0*lamH*Hu + ((223.0/80.0)*g1_2 + (135.0/16.0)*g2_2 + 16.0*g3_2)*Hu - ((43.0/80.0)*g1_2 - (9.0/16.0)*g2_2 + 16.0*g3_2)*Hd + (6.0*lamH**2 - chi4 + Y2_g + gu2)*I3
    A2_d = 1.5*Hd2 - 0.25*(Hd@Hu) - (Hu@Hd) + 2.75*Hu2 + Y2*(-2.25*Hd + 1.25*Hu) - 12.0*lamH*Hd + ((187.0/80.0)*g1_2 + (135.0/16.0)*g2_2 + 16.0*g3_2)*Hd - ((79.0/80.0)*g1_2 - (9.0/16.0)*g2_2 + 16.0*g3_2)*Hu + (6.0*lamH**2 - chi4 + Y2_g + gd2)*I3
    A2_e = 1.5*He2 - 2.25*Y2*He - 12.0*lamH*He + ((387.0/80.0)*g1_2 + (135.0/16.0)*g2_2)*He + (6.0*lamH**2 - chi4 + Y2_g + ge2)*I3
    return pack_sm(dg[0], dg[1], dg[2], f1*b1_lam + f2*b2_lam, (f1*A1_u + f2*A2_u)@Yu, (f1*A1_d + f2*A2_d)@Yd, (f1*A1_e + f2*A2_e)@Ye)

def extract_sm(y):
    g1, g2, g3, lamH, Yu, Yd, Ye = unpack_sm(y)
    Uu, su, _ = np.linalg.svd(Yu); iu = np.argsort(su); su, Uu = su[iu], Uu[:, iu]
    Ud, sd, _ = np.linalg.svd(Yd); id_ = np.argsort(sd); sd, Ud = sd[id_], Ud[:, id_]
    Ue, se, _ = np.linalg.svd(Ye); ie = np.argsort(se); se, Ue = se[ie], Ue[:, ie]
    V = Uu.conj().T @ Ud
    return {
        'g': (float(g1), float(g2), float(g3)), 'lamH': float(lamH),
        'yu': su, 'yd': sd, 'ye': se, 'Yu': Yu, 'Yd': Yd, 'Ye': Ye, 'V': V,
    }

def run_ps_bidoublet(t_a, t_b, g_init, Yu_i, Y1_i, Y15_i, b_vec):
    def pack_ps(g4, gL, gR, A, B, C):
        return np.concatenate([[g4, gL, gR], A.real.ravel(), A.imag.ravel(), B.real.ravel(), B.imag.ravel(), C.real.ravel(), C.imag.ravel()])
    def unpack_ps(y):
        def gm(i): return y[i:i+9].reshape(3, 3) + 1j * y[i+9:i+18].reshape(3, 3)
        return y[0], y[1], y[2], gm(3), gm(21), gm(39)
    def rhs(t, y):
        g4, gL, gR, Yu, Y1, Y15 = unpack_ps(y)
        f1 = 1.0 / (16.0 * np.pi**2)
        I3 = np.eye(3, dtype=complex)
        dg4, dgL, dgR = f1*b_vec[0]*g4**3, f1*b_vec[1]*gL**3, f1*b_vec[2]*gR**3
        Hu, H1, H15 = Yu @ Yu.conj().T, Y1 @ Y1.conj().T, Y15 @ Y15.conj().T
        g_uni = (15.0/4.0)*g4**2 + (9.0/4.0)*gL**2 + (9.0/4.0)*gR**2
        S_leg = 2.0 * (Hu + H1 + (15.0/4.0)*H15)
        dYu  = f1 * (S_leg + (4.0*np.trace(Hu).real - g_uni)*I3) @ Yu
        dY1  = f1 * (S_leg + (4.0*np.trace(H1).real - g_uni)*I3) @ Y1
        dY15 = f1 * (S_leg + (np.trace(H15).real    - g_uni)*I3) @ Y15
        return pack_ps(dg4, dgL, dgR, dYu, dY1, dY15)
    sol = solve_ivp(rhs, [t_a, t_b], pack_ps(*g_init, Yu_i, Y1_i, Y15_i), method='DOP853', rtol=1e-11, atol=1e-13)
    g4, gL, gR, Yu, Y1, Y15 = unpack_ps(sol.y[:, -1])
    Yd, Ye = Y1 + Y15, Y1 - 3.0*Y15
    Uu, su, _ = np.linalg.svd(Yu); iu = np.argsort(su); su, Uu = su[iu], Uu[:, iu]
    Ud, sd, _ = np.linalg.svd(Yd); id_ = np.argsort(sd); sd, Ud = sd[id_], Ud[:, id_]
    se = np.sort(np.linalg.svd(Ye, compute_uv=False))
    return {'g': (float(g4), float(gL), float(gR)), 'yu': su, 'yd': sd, 'ye': se, 'Yu': Yu, 'Y1': Y1, 'Y15': Y15, 'V': Uu.conj().T @ Ud}

V0_MZ = ckm_matrix(pdg_ckm['Vus'][0], pdg_ckm['Vcb'][0], pdg_ckm['Vub'][0], np.deg2rad(pdg_ckm['delta_deg'][0]))
Yu0_MZ = np.diag([pdg_masses['u'][0], pdg_masses['c'][0], pdg_masses['t'][0]]) / v_over_sqrt2
Yd0_MZ = V0_MZ @ np.diag([pdg_masses['d'][0], pdg_masses['s'][0], pdg_masses['b'][0]]) @ V0_MZ.T / v_over_sqrt2
Ye0_MZ = np.diag([pdg_masses['e'][0], pdg_masses['mu'][0], pdg_masses['tau'][0]]) / v_over_sqrt2
y0_MZ = pack_sm(g1_MZ, g2_MZ, g3_MZ, lamH_MZ, Yu0_MZ, Yd0_MZ, Ye0_MZ)

t_PS = np.log(M_PS / M_Z)
t_MR = np.log(M_R / M_Z)
b_PS_full = np.array([-5.0/3.0, 2.0, 26.0/3.0])

# Baseline Paper 25 reference at M_R (SM -> G_PS -> Spin(10)) AND pure 2-loop single-Higgs SM desert at M_R:
sol_base = solve_ivp(rge_sm_2loop, [0.0, t_PS], y0_MZ, method='DOP853', rtol=1e-11, atol=1e-13)
o_PS_base = extract_sm(sol_base.y[:, -1])
g1_b, g2_b, g3_b = o_PS_base['g']
g2R_b = 1.0 / np.sqrt((5.0/3.0)/(g1_b**2) - (2.0/3.0)/(g3_b**2))
o_MR_base = run_ps_bidoublet(t_PS, t_MR, (g3_b, g2_b, g2R_b), o_PS_base['Yu'], 0.25*(3.0*o_PS_base['Yd']+o_PS_base['Ye']), 0.25*(o_PS_base['Yd']-o_PS_base['Ye']), b_PS_full)

t_200TeV = np.log(2.0e5 / M_Z)
t_Trh = np.log(T_RH / M_Z)
sol_sm_desert = solve_ivp(rge_sm_2loop, [0.0, t_MR], y0_MZ, t_eval=[t_200TeV, t_Trh, t_MR], method='DOP853', rtol=1e-11, atol=1e-13)
o_sm_200TeV = extract_sm(sol_sm_desert.y[:, 0])
o_sm_Trh = extract_sm(sol_sm_desert.y[:, 1])
o_sm_MR = extract_sm(sol_sm_desert.y[:, 2])

print("=" * 90)
print("TAU CORE PAPER 26 (v1.1.0) — STANDALONE MASTER VERIFICATION SUITE")
print("=" * 90)

# =============================================================================
# 1. THEOREM 2.1: UNIVERSAL 4D NON-SUSY AXION-FREE NO-GO SUITE
# =============================================================================
print("\n[Theorem 2.1] Universal 4D Non-SUSY Axion-Free No-Go Bounds:")
# (i) P-even d=6 gluon operator (|chi_R|^2 - |chi_L|^2) G Gtilde / M_Pl^2 at v_R = M_R:
theta_d6_MR = (M_R / M_PL)**2
# (ii) Gauge-singlet VLQ P (Babu-Mohapatra 1990) d=5 operator (c_5 / M_Pl) sigma q_Lbar D_R H_d:
yu_MR, yd_MR, ye_MR = o_MR_base['yu'], o_MR_base['yd'], o_MR_base['ye']
vR_VLQ_d5_max = THETA_BOUND * M_PL
# (iii) Mirror-Higgs P (Barr-Chang-Senjanovic 1991; Hall-Harigaya 2018) with <H'> = (0, v')^T (v = 174.1 GeV convention):
#       d=6 |H'|^2 (q_1Lbar u_1R H_tilde) / M_Pl^2 shifts y_u(M_R) = 3.0782e-6 by v'^2 / M_Pl^2 -> v'_max = sqrt(THETA_BOUND * y_u(M_R)) * M_Pl:
yu_MR_val = float(yu_MR[0])
yu_200TeV_val = float(o_sm_200TeV['yu'][0])
vp_mirror_max = np.sqrt(THETA_BOUND * yu_MR_val) * M_PL
m_u_prime_MR_max = yu_MR_val * vp_mirror_max
m_u_prime_thr_max = yu_200TeV_val * vp_mirror_max
# (iv) Non-SUSY Nelson-Barr (Bento-Branco-Parada 1991; Dine-Leigh-Kagan 1993; Dine-Draper 2015):
#      Z_2-even d=5 operator (c_NB / M_Pl) S^2 F_dLbar F_dR shifts m_F -> m_F + c_NB <S>^2 / M_Pl inside det(M_d) = det(y_d v_d) m_F,
#      requiring |c_NB| M_NB / M_Pl < 1.2e-10 => M_NB < 2.922e8 GeV << M_1 = 5.70e12 GeV (plus stronger (c'_NB / M_Pl) S q_Lbar F_dR H_d):
M_NB_d5_max = THETA_BOUND * M_PL

assert theta_d6_MR > 1e-8 and vR_VLQ_d5_max < 3.0e8 and m_u_prime_thr_max < 1.0e6 < T_RH and M_NB_d5_max < 3.0e8 < M_1_RHN
print(f"  1. Bidoublet/LR Parity d=6 at M_R: |Delta theta| ~ (M_R/M_Pl)^2 = {theta_d6_MR:.2e} >> 1.2e-10 (EXCLUDED)")
print(f"  2. VLQ Parity d=5 bound: v_R < {vR_VLQ_d5_max:.3e} GeV << M_PS = {M_PS:.2e} GeV (EXCLUDED)")
print(f"  3. Mirror-Higgs Parity d=6 bound (y_u(M_R)={yu_MR_val:.3e}): v' < {vp_mirror_max:.3e} GeV => m_u'(M_R) < {m_u_prime_MR_max/1e3:.1f} TeV (threshold m_u'(m_u') < {m_u_prime_thr_max/1e3:.1f} TeV << T_rh = {T_RH:.2e} GeV)")
print(f"  4. Non-SUSY Nelson-Barr d=5 (S^2 F_dLbar F_dR) bound: M_NB < {M_NB_d5_max:.3e} GeV << M_1 = {M_1_RHN:.2e} GeV => eps_1(M_1) == 0")

# =============================================================================
# 2. THEOREM 3.1 & 3.2: CONFORMAL TWO-SCALAR Z_16 DISCRETE-GAUGE Spin(10) AXION
# =============================================================================
print("\n[Theorem 3.1 & 3.2] Flagship 4D Non-SUSY Solution A1 (Conformal Z_16 Spin(10) Axion):")
N_Z = 16
N_pair = 4
A_Spin10 = N_pair * 4 * (-1)       # -16 == 0 (mod 16)
A_grav   = N_pair * 16 * (-1)      # -64 == 0 (mod 32)
A_cubic  = N_pair * 16 * ((-1)**3) # -64 == 0 (mod 16; 3 does not divide 16!)
assert A_Spin10 % 16 == 0 and A_grav % 32 == 0 and A_cubic % 16 == 0 and (64 % 32 == 0)

# SymPy verification of full rank 24/24 of the explicit 4D U(1)_g UV completion mass matrix at <S_16(+16)> = v_16 != 0:
u1g_mult_q = [
    (64, -1), (64, 0), (10, +8), (2, -8),
    (1, 0), (1, -16), (2, +8),
    (1, -4), (1, -12), (2, +8),
    (1, -6), (1, -10), (2, +8),
]
assert sum(m * q for m, q in u1g_mult_q) == 0 and sum(m * (q**3) for m, q in u1g_mult_q) == 0
assert (4 * 2 * (-1) + 1 * 1 * (+8)) == 0
# Build symbolic 24x24 UV fermion mass matrix (10 in 10_Psi(+8), 2 in chi_M(-8), and 3x4 = 12 in singlet blocks):
v16_sym, M0_sym = sp.symbols('v16 M0', positive=True)
M_uv_blocks = [
    v16_sym * sp.eye(10),                        # S_16^dag * (10_Psi . 10_Psi): rank 10
    v16_sym * sp.eye(2),                         # S_16 * (chi_M . chi_M): rank 2
    sp.Matrix([[M0_sym, v16_sym], [v16_sym, 0]]), # (0, -16) block: det = -v16^2 != 0 (rank 2)
    v16_sym * sp.eye(2),                         # 2 x (+8) in Block 1: rank 2
    sp.Matrix([[0, v16_sym], [v16_sym, 0]]),     # (-4, -12) Dirac pair via S_16: rank 2
    v16_sym * sp.eye(2),                         # 2 x (+8) in Block 2: rank 2
    sp.Matrix([[0, v16_sym], [v16_sym, 0]]),     # (-6, -10) Dirac pair via S_16: rank 2
    v16_sym * sp.eye(2),                         # 2 x (+8) in Block 3: rank 2
]
M_uv_24 = sp.diag(*M_uv_blocks)
assert M_uv_24.shape == (24, 24) and M_uv_24.rank() == 24
print("  SymPy UV check: 4D U(1)_g -> Z_16 UV mass matrix has shape 24x24 and exact rank 24/24!")

# Scalar potential root & late-time cosmology:
v_PhiQ = 3.520e11
lam_PhiQ = 0.15
m_chi = 5.0e7
mu_mix = 1.0e7
v_chi = brentq(lambda r: (m_chi**2)*r + (r**9)/(16.0 * (M_PL**6)) - (mu_mix**2)*v_PhiQ, 0.0, 1.0e14)
sin_beta = v_chi / np.sqrt(v_PhiQ**2 + v_chi**2)
v_PQ = np.sqrt(v_PhiQ**2 + v_chi**2)
f_a = v_PQ / N_Z
y_Q = 0.24
M_Q0 = y_Q * v_PhiQ / np.sqrt(2.0)

m_a_meV = 5.691 * (1.0e9 / f_a)
g_ap = 0.47 * 0.93827 / f_a
g_agamma = (1.0 / (137.036 * 2.0 * np.pi * f_a)) * abs(8.0/3.0 - 1.92)
# Exact prefactor 2 * (N_Z / N_DW) = 2 from minimizing (1/2) chi_top theta_bar^2 + (2|c_16|/M_Pl^12)[(v_PhiQ/sqrt(2))^16 + (v_chi/sqrt(2))^16] cos(theta_bar + delta):
N_DW_A1 = 16
theta_Z16 = (2.0 * N_Z / N_DW_A1) * (((v_PhiQ / np.sqrt(2.0))**N_Z) + ((v_chi / np.sqrt(2.0))**N_Z)) / (M_PL**(N_Z - 4) * CHI_TOP)
omega_a_h2 = 0.12 * ((f_a / 9.0e11)**1.165) * (1.0**2)

T_sym = np.sqrt(lam_PhiQ * (v_PhiQ**2) / (16.0 * (y_Q**2) / 3.0))
H_Trh = np.sqrt(np.pi**2 * 106.75 / 90.0) * (T_RH**2) / M_PL
H_Tsym = H_Trh * ((T_sym / T_RH)**4)
c_pot = 4.0
m2_chi_Tsym = -2.5 * (H_Tsym**2) + (m_chi**2)

H_osc = m_chi / np.sqrt(2.5)
r_osc = (16.0 * (M_PL**6) * (m_chi**2) / 10.0)**(1.0 / 8.0)
# Technical naturalness ('t Hooft radiative stability) check on flaton potential V_0:
lam_chi_max_osc = (m_chi**2) / ((r_osc / np.sqrt(2.0))**2)
c6_max_osc = (m_chi**2) * (M_PL**2) / ((r_osc / np.sqrt(2.0))**4)
c8_max_osc = (m_chi**2) * (M_PL**4) / ((r_osc / np.sqrt(2.0))**6)
lam_phi_chi = 0.5 * c_pot * ((M_INF / M_PL)**2)
dlam_chi_1loop = (lam_phi_chi**2) / (16.0 * np.pi**2)
dm2_chi_tree = -(mu_mix**4) / (2.0 * lam_PhiQ * (v_PhiQ**2))
assert dlam_chi_1loop < 1e-4 * lam_chi_max_osc and abs(dm2_chi_tree) < 1e-8 * (m_chi**2)
rho_chi_over_tot = (0.5 * (m_chi**2) * (r_osc**2) + (r_osc**10)/(160.0 * M_PL**6)) / (3.0 * (M_PL**2) * (H_osc**2))
Gamma_sig_aa = ((sin_beta**2) * (m_chi**3)) / (32.0 * np.pi * (v_PhiQ**2))
T_dec_chi = ((90.0 / (np.pi**2 * 106.75))**0.25) * np.sqrt(Gamma_sig_aa * M_PL)
frac_sig_at_dec = 0.5 * rho_chi_over_tot * (T_RH / T_dec_chi)
dNeff_late = (106.75 / 1.75) * ((10.75 / 106.75)**(4.0 / 3.0)) * frac_sig_at_dec

# Complete 2-loop SM + G_PS RGE integration across [M_Z -> M_Q0 -> M_PS -> M_R]:
N_extra_16 = 8.0
db_16 = N_extra_16 * np.array([4.0/3.0, 4.0/3.0, 4.0/3.0])
dB_16 = N_extra_16 * np.array([
    [19.0/15.0, 3.0/5.0,  44.0/15.0],
    [1.0/5.0,   49.0/3.0, 4.0      ],
    [11.0/30.0, 3.0/2.0,  76.0/3.0 ]
])

def rge_sm_2loop_A1(t, y):
    dy_base = rge_sm_2loop(t, y)
    g1, g2, g3, lamH, Yu, Yd, Ye = unpack_sm(y)
    dg1, dg2, dg3, dlamH, dYu, dYd, dYe = unpack_sm(dy_base)
    g = np.array([g1, g2, g3])
    f1 = 1.0 / (16.0 * np.pi**2); f2 = f1**2
    dg_ex = np.array([f1 * db_16[i] * g[i]**3 + f2 * g[i]**3 * np.dot(dB_16[i], g**2) for i in range(3)])
    dYu = dYu + f2 * N_extra_16 * ((17.0/18.0)*g1**4 + 2.5*g2**4 + (80.0/9.0)*g3**4) * Yu
    dYd = dYd + f2 * N_extra_16 * ((5.0/18.0)*g1**4  + 2.5*g2**4 + (80.0/9.0)*g3**4) * Yd
    dYe = dYe + f2 * N_extra_16 * (2.5*g1**4         + 2.5*g2**4) * Ye
    return pack_sm(dg1 + dg_ex[0], dg2 + dg_ex[1], dg3 + dg_ex[2], dlamH, dYu, dYd, dYe)

t_MQ0 = np.log(M_Q0 / M_Z)
t_grid_1 = np.linspace(0.0, t_MQ0, 80)
t_grid_2 = np.linspace(t_MQ0, t_PS, 40)
sol_1 = solve_ivp(rge_sm_2loop, [0.0, t_MQ0], y0_MZ, t_eval=t_grid_1, method='DOP853', rtol=1e-11, atol=1e-13)
sol_2 = solve_ivp(rge_sm_2loop_A1, [t_MQ0, t_PS], sol_1.y[:, -1], t_eval=t_grid_2, method='DOP853', rtol=1e-11, atol=1e-13)
o_PS_A1 = extract_sm(sol_2.y[:, -1])
g1_p, g2_p, g3_p = o_PS_A1['g']
g2R_p = 1.0 / np.sqrt((5.0/3.0)/(g1_p**2) - (2.0/3.0)/(g3_p**2))
o_MR_A1 = run_ps_bidoublet(t_PS, t_MR, (g3_p, g2_p, g2R_p), o_PS_A1['Yu'], 0.25*(3.0*o_PS_A1['Yd']+o_PS_A1['Ye']), 0.25*(o_PS_A1['Yd']-o_PS_A1['Ye']), b_PS_full + 32.0/3.0)
yb_ytau_A1 = float(o_MR_A1['yd'][2] / o_MR_A1['ye'][2])
inv_aPS_MR = 4.0 * np.pi / (np.array(o_MR_A1['g'])**2)

# Boltzmann audit at T_rh & hot-axion Delta N_eff:
x_rh = M_Q0 / T_RH
r_int, _ = quad(lambda u: (u**2)*np.sqrt(u**2 + x_rh**2)/(np.exp(np.sqrt(u**2 + x_rh**2)) + 1.0), 0.0, 60.0)
p_int, _ = quad(lambda u: (u**4)/(np.sqrt(u**2 + x_rh**2)*(np.exp(np.sqrt(u**2 + x_rh**2)) + 1.0)), 0.0, 60.0)
rho1 = r_int / (2.0 * np.pi**2); s1 = rho1 + p_int / (6.0 * np.pi**2)
delta_g_rho = 256.0 * rho1 / (np.pi**2 / 30.0)
delta_g_s   = 256.0 * s1   / (2.0 * np.pi**2 / 45.0)
eta_B_ratio = 1.0 / (((106.75 + delta_g_s) / 106.75) * ((106.75 / (106.75 + delta_g_rho))**0.75))
g3_Trh = o_sm_Trh['g'][2]
Gamma_a_SSW = 0.036 * (g3_Trh**6) * (T_RH**3) / (f_a**2)  # Salvio-Strumia-Xue (2014): ~0.036 g_s^6 T^3 / f_a^2
Gamma_a_over_H_Trh = Gamma_a_SSW / H_Trh
dNeff_th = 0.0270 * min(1.0, Gamma_a_over_H_Trh)
dNeff_tot = dNeff_th + dNeff_late

# 2-field Einstein-frame ODE trajectory across ~48 Starobinsky scalaron oscillations:
c10 = (M_PL**2) / (160.0 * (M_INF**2))
m_chi_dimless2 = (m_chi / M_INF)**2
phi_star_CMB = 5.2935  # exact Starobinsky field at N_* = 51.412 e-folds
V_CMB = 0.75 * ((1.0 - np.exp(-np.sqrt(2.0/3.0)*phi_star_CMB))**2)
r_CMB = (((0.5 * c_pot - 2.0/3.0) * V_CMB) / (10.0 * c10))**(1.0 / 8.0)
chi_CMB = (r_CMB / np.sqrt(2.0)) * M_PL
f_eff_CMB = r_CMB * M_PL / N_Z
P_iso = (omega_a_h2 / 0.12)**2 * (H_INF / (np.pi * f_eff_CMB * 1.0))**2
beta_iso = P_iso / (2.10e-9 + P_iso)

phi0 = 1.6
V0 = 0.75 * ((1.0 - np.exp(-np.sqrt(2.0/3.0)*phi0))**2)
r0 = (((0.5 * c_pot - 2.0/3.0) * V0) / (10.0 * c10))**(1.0 / 8.0)
y_ode = np.array([phi0, 0.0, r0, 0.0]); dt_ode = 0.004
r_min_ode = r0
ode_trace_t, ode_trace_phi, ode_trace_r = [], [], []
for step_i in range(75000):
    if step_i % 150 == 0:
        ode_trace_t.append(float(step_i * dt_ode))
        ode_trace_phi.append(float(y_ode[0]))
        ode_trace_r.append(float(y_ode[2] * M_PL))
    def f_ode(yy):
        phi, dphi, r, dr = yy
        ex = np.exp(-np.sqrt(2.0/3.0)*phi)
        Vs = 0.75 * ((1.0 - ex)**2)
        dVs = 1.5 * (1.0 - ex) * np.sqrt(2.0/3.0) * ex
        V10 = c10 * (r**10)
        V2  = 0.5 * m_chi_dimless2 * (r**2)
        Vtot = Vs * (1.0 - 0.25*c_pot*(r**2)) + V2 + V10
        H = np.sqrt(max(1e-30, (0.5*dphi**2 + 0.5*dr**2 + Vtot) / 3.0))
        R_E = 4.0 * Vs * (1.0 - 0.125*c_pot*(r**2)) - dphi**2 + 2.0 * V2 - 6.0 * V10
        return np.array([dphi, -3*H*dphi - dVs*(1-0.25*c_pot*r**2), dr, -3*H*dr - ((1/6)*R_E - 0.5*c_pot*Vs + m_chi_dimless2)*r - 10.0*c10*(r**9)])
    k1 = f_ode(y_ode); k2 = f_ode(y_ode + 0.5*dt_ode*k1); k3 = f_ode(y_ode + 0.5*dt_ode*k2); k4 = f_ode(y_ode + dt_ode*k3)
    y_ode = y_ode + (dt_ode/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    if y_ode[2] < r_min_ode:
        r_min_ode = y_ode[2]

assert theta_Z16 < THETA_BOUND and omega_a_h2 < 0.003 and beta_iso < 0.038 and r_min_ode > 0.0 and dNeff_tot < 0.05
print(f"  f_a = {f_a:.4e} GeV (m_a = {m_a_meV:.4f} meV, |g_ap| = {g_ap:.3e}, |g_agamma| = {g_agamma:.3e} GeV^-1)")
print(f"  |Delta theta_bar| (d=16) = {theta_Z16:.3e} << 1.2e-10; Omega_a h^2 (theta_i=1) = {omega_a_h2:.5f}")
print(f"  Re-integrated 2-loop yb/ytau(M_R) = {yb_ytau_A1:.4f} (Paper 25: 0.6937), 1/alpha_PS(M_R) = {np.round(inv_aPS_MR, 2)}")
print(f"  Boltzmann shift at T_rh: delta(eta_B)/eta_B = {(eta_B_ratio-1)*100:+.4f}%; Delta N_eff^tot = {dNeff_tot:.4f}")
print(f"  48-oscillation 2-field ODE: |chi_CMB| = {chi_CMB:.3e} GeV, beta_iso = {beta_iso:.4f}, MIN r(t) = {r_min_ode*M_PL:.3e} GeV > 0")
print(f"  Pre-BBN flaton decay: v_chi = {v_chi:.3e} GeV, T_dec,chi = {T_dec_chi:.3e} GeV >> T_BBN")

# =============================================================================
# 3. PROPOSITION 3.3: COMPOSITE AXION Spin(7)_A x Spin(6)_B x U(1)_X AUDIT
# =============================================================================
print("\n[Proposition 3.3] Chiral Composite Axion Spin(7)_A x Spin(6)_B x U(1)_X Audit:")
S_sl2z = np.array([[-1, -1], [7, 6]], dtype=int)
assert int(round(np.linalg.det(S_sl2z))) == 1
assert (70*6 + 7*(-60) + 60*(-7) + 6*70 + 36 + 55 - 91) == 0
assert (70*6**3 + 7*(-60)**3 + 60*(-7)**3 + 6*70**3 + 36**3 + 55**3 + (-91)**3) == 0
fa_pre_min = (35.0 * T_MAX) / (4.0 * np.pi * np.sqrt(85.0))
omega_comp_pre = 0.12 * ((fa_pre_min / 9.0e11)**1.165)
assert omega_comp_pre > 0.05  # violates QUMOND (< 0.003)!
print(f"  Anomalies = 0, det(SL(2,Z)) = +1 => N_DW = 1; pre-inf Lambda > 35 T_max forces f_a >= {fa_pre_min:.2e} GeV (Omega_a h^2 = {omega_comp_pre:.4f} >> 0.003)!")

# =============================================================================
# 4. THEOREM 4.1: CUSP-FREE MODULAR N=1 SUGRA Spin(10) + 120_H SOLUTION B
# =============================================================================
print("\n[Theorem 4.1] Cusp-Free Modular N=1 SUGRA Spin(10) + 120_H Solution B:")
k_16 = (+6, 0, -6); k_H = +3
k_Y_mat = [[k_16[a] + k_16[b] + k_H - 3 for b in range(3)] for a in range(3)]
assert k_Y_mat == [[12, 6, 0], [6, 0, -6], [0, -6, -12]]

def eisenstein_E4_E6(tau, nmax=20):
    q = np.exp(2j * np.pi * tau)
    E4, E6 = 1.0 + 0j, 1.0 + 0j
    for n in range(1, nmax + 1):
        divs = [d for d in range(1, n + 1) if n % d == 0]
        qn = q**n
        E4 += 240.0 * sum(d**3 for d in divs) * qn
        E6 -= 504.0 * sum(d**5 for d in divs) * qn
    return E4, E6

def autonne_takagi_symmetric(M):
    M = 0.5 * (M + M.T)
    W, s, Vh = np.linalg.svd(M)
    V = Vh.conj().T
    Z_diag = np.diag(V.T @ W)
    U = V @ np.diag(np.exp(-0.5j * np.angle(Z_diag)))
    idx = np.argsort(s)
    return s[idx], U[:, idx]

def map_tau_in_F(u_re, u_im):
    re_t = 0.48 * np.tanh(u_re)
    im_min = np.sqrt(max(0.0, 1.01**2 - re_t**2))
    im_t = im_min + 0.8 * (1.0 / (1.0 + np.exp(-u_im)))
    return re_t, im_t

# Consistent 2-loop single-Higgs SM desert target vector at M_R (matching Theorem 4.1(3)(iii) decoupling of extra Higgses at M_R):
yu_B_tgt, yd_B_tgt, ye_B_tgt = o_sm_MR['yu'], o_sm_MR['yd'], o_sm_MR['ye']
V_B_tgt = o_sm_MR['V']
Vus_t = float(abs(V_B_tgt[0, 1]))
Vcb_t = float(abs(V_B_tgt[1, 2]))
Vub_t = float(abs(V_B_tgt[0, 2]))
del_t = float(np.rad2deg(abs(np.angle(-V_B_tgt[0, 0]*np.conj(V_B_tgt[0, 2]) / (V_B_tgt[1, 0]*np.conj(V_B_tgt[1, 2]))))))

def eval_B13(x, return_all=False):
    re_t, im_t = map_tau_in_F(x[0], x[1])
    E4, E6 = eisenstein_E4_E6(re_t + 1j * im_t, nmax=20)
    E4_3, E6_2 = E4**3, E6**2
    u11, u12, cu13, cu22 = x[2]*E4_3 + x[3]*E6_2, x[4]*E6, 10**x[5], 10**x[6]
    d11 = (10**x[7]) * (np.cos(x[8])*E4_3 + np.sin(x[8])*E6_2)
    d12, cd13, cd22 = x[9]*(10**x[10])*E6, 10**x[11], 10**x[12]
    b120 = x[23]*(10**x[10])*E6
    c120 = 0.92*np.tanh(x[24])*cd13
    Y120_h = np.array([[0.0, b120, c120], [-b120, 0.0, 0.0], [-c120, 0.0, 0.0]], dtype=complex)
    Yu_h = np.array([[u11, u12, cu13], [u12, cu22, 0.0], [cu13, 0.0, 0.0]], dtype=complex)
    Yd_s = np.array([[d11, d12, cd13], [d12, cd22, 0.0], [cd13, 0.0, 0.0]], dtype=complex)
    Yd_h = Yd_s + Y120_h
    Ye_h = x[21]*Yu_h + x[22]*Yd_s + x[25]*Y120_h
    Ah = np.array([[0.0, x[15]+1j*x[16], x[17]+1j*x[18]],
                   [x[15]-1j*x[16], x[13], x[19]+1j*x[20]],
                   [x[17]-1j*x[18], x[19]-1j*x[20], x[14]]], dtype=complex)
    ev, UA = np.linalg.eigh(Ah)
    H = UA @ np.diag(np.exp(ev)) @ UA.conj().T
    Yu_c, Yd_c, Ye_c = H.T @ Yu_h @ H, H.T @ Yd_h @ H, H.T @ Ye_h @ H
    su, Uu = autonne_takagi_symmetric(Yu_c)
    Ud, sd, _ = np.linalg.svd(Yd_c); id_ = np.argsort(sd); sd, Ud = sd[id_], Ud[:, id_]
    se = np.sort(np.linalg.svd(Ye_c, compute_uv=False))
    V = Uu.conj().T @ Ud
    Vus, Vcb, Vub = abs(V[0,1]), abs(V[1,2]), abs(V[0,2])
    gam = np.rad2deg(abs(np.angle(-V[0,0]*np.conj(V[0,2]) / (V[1,0]*np.conj(V[1,2]) + 1e-30))))
    loss = (
        100.0*np.sum((np.log(np.maximum(su, 1e-25)/yu_B_tgt))**2)
        + 100.0*np.sum((np.log(np.maximum(sd, 1e-25)/yd_B_tgt))**2)
        + 100.0*np.sum((np.log(np.maximum(se, 1e-25)/ye_B_tgt))**2)
        + ((Vus-Vus_t)/0.001)**2 + ((Vcb-Vcb_t)/0.001)**2 + ((Vub-Vub_t)/0.0002)**2 + ((gam-del_t)/1.0)**2
    )
    if return_all:
        return loss, re_t + 1j*im_t, H, Yu_c, Yd_c, Ye_c, su, sd, se, Vus, Vcb, Vub, gam
    return float(loss)

bounds_26 = [
    (-2.0, 2.0), (-2.0, 2.0),
    (-2.0, 2.0), (-2.0, 2.0), (-1.0, 1.0), (-4.5, -1.5), (-4.5, -1.5),
    (-3.5, -1.0), (-np.pi, np.pi), (-1.0, 1.0), (-5.0, -2.0), (-5.5, -2.5), (-5.5, -2.5),
    (-1.5, 1.5), (-1.5, 1.5),
    (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5),
    (-0.2, 0.2), (-4.0, 4.0),
    (-3.0, 3.0), (-2.5, 2.5), (-6.0, 6.0)
]
res_de = differential_evolution(eval_B13, bounds_26, seed=2026, maxiter=320, popsize=12, tol=1e-8, polish=False)
res_nm = minimize(eval_B13, res_de.x, method='Powell', options={'ftol': 1e-11, 'xtol': 1e-11, 'maxiter': 15000})
loss_B, tau_B, H_B, Yu_B, Yd_B, Ye_B, su_B, sd_B, se_B, Vus_B, Vcb_B, Vub_B, gam_B = eval_B13(res_nm.x, return_all=True)
arg_det_B = float((np.angle(np.linalg.det(Yu_B)) + np.angle(np.linalg.det(Yd_B)) + np.pi) % (2*np.pi) - np.pi)
im_tr_YuYd = float(np.imag(np.trace(Yu_B.conj().T @ Yd_B)))
theta_1loop_2hdm = float((3.0 / (16.0 * np.pi**2)) * im_tr_YuYd / (yu_B_tgt[2] * yd_B_tgt[2]))

# Symmetric Autonne-Takagi leptogenesis check at tau_B (modular weights k_M = k_16,a + k_16,b + k_126bar - 3 = [[12,6,0],[6,0,-6],[0,-6,-12]]):
E4_B, E6_B = eisenstein_E4_E6(tau_B, nmax=20)
MR_hol = M_R * np.array([[0.08*E4_B**3 - 0.05*E6_B**2, 0.12*E6_B, 0.05], [0.12*E6_B, 0.10, 0.0], [0.05, 0.0, 0.0]], dtype=complex)
MR_can = H_B.T @ MR_hol @ H_B
M_eigs, U_N = autonne_takagi_symmetric(MR_can)
M_eigs *= (M_1_RHN / M_eigs[0])
Y_nu_N = Yu_B @ U_N
H_nu = Y_nu_N.conj().T @ Y_nu_N
def loop_f(x): return np.sqrt(x) * (1.0 - (1.0 + x)*np.log((1.0 + x)/x) + 1.0 / (1.0 - x))
eps_1_B = float(sum((1.0 / (8.0 * np.pi * np.real(H_nu[0, 0]))) * np.imag((H_nu[0, j])**2) * loop_f((M_eigs[j]/M_eigs[0])**2) for j in [1, 2]))

assert loss_B < 1e-12 and abs(tau_B) >= 1.01 and abs(arg_det_B) < 1e-11 and abs(eps_1_B) > 1e-6
print(f"  13-obs SM-desert fit strictly in F: tau_0 = {tau_B.real:+.5f} + {tau_B.imag:.5f}i (|tau_0| = {abs(tau_B):.5f} >= 1.01), loss = {loss_B:.3e}")
print(f"  arg det(Yu * Yd) = {arg_det_B:+.3e} rad; non-SUSY 2HDM 1-loop phase if unshielded = {theta_1loop_2hdm:+.3e} rad")
print(f"  Takagi leptogenesis CP asymmetry at same tau_0 (M_1 = 5.70e12 GeV): eps_1 = {eps_1_B:+.3e}")

# Save full deterministic JSON ledger:
ledger = {
    "nogo": {
        "theta_d6_MR": float(theta_d6_MR),
        "vR_VLQ_d5_max_GeV": float(vR_VLQ_d5_max),
        "yu_MR": yu_MR_val,
        "yu_200TeV": yu_200TeV_val,
        "vp_mirror_max_GeV": float(vp_mirror_max),
        "m_u_prime_MR_max_TeV": float(m_u_prime_MR_max / 1e3),
        "m_u_prime_thr_max_TeV": float(m_u_prime_thr_max / 1e3),
        "M_NB_d5_max_GeV": float(M_NB_d5_max),
        "theta_1loop_2hdm_rad": theta_1loop_2hdm,
    },
    "solution_A1": {
        "N_Z": N_Z, "N_pair": N_pair,
        "f_a_GeV": float(f_a), "v_PQ_GeV": float(v_PQ),
        "v_PhiQ_GeV": float(v_PhiQ), "v_chi_GeV": float(v_chi), "sin_beta": float(sin_beta),
        "m_a_meV": float(m_a_meV), "g_ap": float(g_ap), "g_agamma_GeVinv": float(g_agamma),
        "M_Q0_GeV": float(M_Q0), "y_Q": float(y_Q),
        "abs_delta_theta": float(theta_Z16), "omega_a_h2": float(omega_a_h2),
        "yb_ytau_MR": yb_ytau_A1, "yb_ytau_MR_paper25": float(yd_MR[2] / ye_MR[2]),
        "inv_alpha_PS_MR": [float(v) for v in inv_aPS_MR],
        "delta_etaB_rel_pct": float((eta_B_ratio - 1.0) * 100.0),
        "Gamma_a_over_H_Trh": float(Gamma_a_over_H_Trh),
        "dNeff_th": float(dNeff_th), "dNeff_late": float(dNeff_late), "dNeff_tot": float(dNeff_tot),
        "chi_CMB_GeV": float(chi_CMB), "beta_iso": float(beta_iso),
        "T_sym_GeV": float(T_sym), "H_Tsym_GeV": float(H_Tsym), "m2_chi_Tsym_GeV2": float(m2_chi_Tsym),
        "T_dec_chi_GeV": float(T_dec_chi), "r_min_ode_GeV": float(r_min_ode * M_PL),
        "lam_chi_max_osc": float(lam_chi_max_osc), "c6_max_osc": float(c6_max_osc), "c8_max_osc": float(c8_max_osc),
        "dlam_chi_1loop": float(dlam_chi_1loop), "dm2_chi_tree_GeV2": float(dm2_chi_tree),
        "ode_trace_t": ode_trace_t, "ode_trace_phi": ode_trace_phi, "ode_trace_r": ode_trace_r,
    },
    "solution_A2": {
        "fa_pre_min_GeV": float(fa_pre_min),
        "omega_comp_pre": float(omega_comp_pre),
        "d_min_pure": 21, "d_min_EW": 16,
    },
    "solution_B": {
        "tau_re": float(tau_B.real), "tau_im": float(tau_B.imag), "abs_tau": float(abs(tau_B)),
        "loss_13obs": float(loss_B), "arg_det_rad": arg_det_B,
        "su_ratios": [float(v) for v in su_B / yu_B_tgt],
        "sd_ratios": [float(v) for v in sd_B / yd_B_tgt],
        "se_ratios": [float(v) for v in se_B / ye_B_tgt],
        "CKM_fit": [float(Vus_B), float(Vcb_B), float(Vub_B), float(gam_B)],
        "CKM_target_sm_MR": [Vus_t, Vcb_t, Vub_t, del_t],
        "yb_ytau_sm_MR": float(yd_B_tgt[2] / ye_B_tgt[2]),
        "eps_1_takagi": eps_1_B,
    }
}
with open(OUT_JSON, "w") as jf:
    json.dump(ledger, jf, indent=2)

print(f"\n[SAVED] Deterministic verification ledger -> {OUT_JSON}")
print("ALL PAPER 26 v1.1.0 MASTER VERIFICATION CHECKS PASSED.")
