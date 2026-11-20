# Strong CP Closure Across a Tetrahedral Spin(10) -> Pati-Salam -> SM and Starobinsky Inflationary Spine (Paper 26)

Standalone reproducible verification suite, vector figure generator, and bilingual LaTeX manuscripts (`v1.1.0`):

1. **Four-Dimensional Non-Supersymmetric Axion-Free No-Go Suite (Theorem 2.1):** Excludes generalized Pati-Salam bidoublet parity, gauge-singlet vector-like quark parity, mirror-Higgs parity, non-supersymmetric Nelson-Barr CP, and real-determinant discrete-flavon textures under the joint Starobinsky inflation/leptogenesis, `Spin(10) -> G_PS -> G_SM` RG, and relic-density boundary conditions.
2. **Flagship 4D Non-SUSY Solution A1 (Theorems 3.1-3.2):** Conformal (`xi = +1/6`) two-scalar `Z_16` discrete-gauge `Spin(10)` axion (`Omega_5^Spin(B Z_16) = 0`, explicit rank-24/24 4D `U(1)_g` UV completion, `f_a = 2.2018e10 GeV`, `m_a = 0.2585 meV`, `|Delta theta_bar| = 3.07e-34`, `yb(M_R)/ytau(M_R) = 0.6931`, `delta eta_B/eta_B = -0.014%`, `beta_iso = 0.0128`, `Delta N_eff^tot = 0.0317`).
3. **UV Axion-Free Alternative Solution B (Theorem 4.1):** Cusp-free modular `N=1` supergravity `Spin(10) + 120_H` (`k_16 = (+6,0,-6)`, `k_H = +3`, Green-Schwarz dilaton `S`) reproducing all 13 single-Higgs SM-desert charged-fermion and CKM observables strictly inside `F` (`arg det(Yu Yd) = 0`) under four mandatory UV conditions above `M_R`.

## Quick Verification

```bash
pip install -r requirements.txt
python3 scripts/verify_paper26_strong_cp_closure.py
python3 scripts/generate_paper26_figures.py
```
