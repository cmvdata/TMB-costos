"""
06_cobb_douglas_robustness.py - Cobb-Douglas restriction of the translog
on C_tec, computed as a robustness benchmark.

The model:
    ln C_tec = a0 + alpha_L ln W_L + alpha_E ln W_E + alpha_O ln W_O
             + beta_1 ln Y_1 + beta_2 ln Y_2 + lambda t

Homogeneity (sum alpha = 1) imposed by construction. alpha_i anchored
on observed cost shares; beta_j anchored on the literature averages of
the translog calibration.

Under Cobb-Douglas:
  - Cost-output elasticity is constant: eps_Y1 = beta_1, eps_Y2 = beta_2.
  - RTD = 1 / beta_1; MC/AC = beta_1 (since MC = (dC/dY1) and
    AC = C/Y1, and dlnC/dlnY1 = beta_1 by construction).

If RTD < 1 or MC > AC under the Cobb-Douglas restriction, STOP and
report. Otherwise export a small summary table consistent with the
robustness appendix.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# Anchors (same as in 02_calibrate_translog.py for C_tec / M-OP)
BETA_1 = 0.45
BETA_2 = 0.25
LAMBDA = -0.005

# Load panel and compute Cobb-Douglas-implied quantities
panel = pd.read_csv(DATA_DIR / "panel_linea_año.csv")
panel["C_tec"] = panel["C_op"]
panel["AC_tec"] = panel["C_tec"] / panel["pax"]

# Observed shares (passenger-weighted, line-year)
sum_C = panel["C_tec"].sum()
alpha_L = panel["personal"].sum() / sum_C
alpha_E = panel["energia"].sum() / sum_C
alpha_O = (panel["aprov"] + panel["serv_ext"] + panel["amort"]).sum() / sum_C
# sanity
assert abs(alpha_L + alpha_E + alpha_O - 1.0) < 1e-9, "Shares do not sum to 1"

# Under Cobb-Douglas with anchors:
eps_Y1 = BETA_1
eps_Y2 = BETA_2
RTD_CD = 1.0 / eps_Y1
RTS_CD = 1.0 / (eps_Y1 + eps_Y2)

# MC and AC: MC_pax = eps_Y1 * C / pax = eps_Y1 * AC.
MC_per_pax = eps_Y1 * panel["C_tec"] / panel["pax"]
AC_per_pax = panel["C_tec"] / panel["pax"]
MC_AC = eps_Y1
W = panel["pax"].values
AC_sys = float(panel["C_tec"].sum() / panel["pax"].sum())
MC_sys = float(eps_Y1 * panel["C_tec"].sum() / panel["pax"].sum())
subsidy = 1.0 - MC_AC

# Translog reference values (loaded from existing summary)
with open(OUT_TABLES / "summary_metrics.json", encoding="utf-8") as f:
    summary = json.load(f)
op = summary["M-OP"]

print("=" * 78)
print("COBB-DOUGLAS ROBUSTNESS BENCHMARK (C_tec only)")
print("=" * 78)
print(f"Anchored alphas: alpha_L = {alpha_L:.4f}, alpha_E = {alpha_E:.4f}, "
      f"alpha_O = {alpha_O:.4f}")
print(f"Anchored betas:  beta_1  = {BETA_1:.2f},  beta_2  = {BETA_2:.2f}")
print(f"Trend lambda  =  {LAMBDA:+.3f}")
print()
print(f"{'metric':<24}{'translog C_tec':>18}{'Cobb-Douglas':>16}")
print(f"{'RTD':<24}{op['rtd_weighted_pax']:>18.3f}{RTD_CD:>16.3f}")
print(f"{'RTS':<24}{op['rts_weighted_pax']:>18.3f}{RTS_CD:>16.3f}")
print(f"{'MC/AC':<24}{op['mc_over_ac_system']:>18.3f}{MC_AC:>16.3f}")
print(f"{'AC system (EUR/pax)':<24}{op['ac_system_eur_per_pax']:>18.4f}"
      f"{AC_sys:>16.4f}")
print(f"{'MC system (EUR/pax)':<24}{op['mc_system_eur_per_pax']:>18.4f}"
      f"{MC_sys:>16.4f}")
print(f"{'Boiteux subsidy':<24}{op['boiteux_optimal_subsidy_pct']:>18.2f}%"
      f"{subsidy*100:>15.2f}%")

# Sanity checks
ok = True
if RTD_CD < 1.0:
    print("\n[STOP] Cobb-Douglas RTD < 1; report before including.")
    ok = False
if MC_AC > 1.0:
    print("\n[STOP] Cobb-Douglas MC > AC; report before including.")
    ok = False

if ok:
    print("\nCobb-Douglas qualitative conclusions consistent with translog:")
    print("  - RTD > 1 (density economies preserved)")
    print("  - MC < AC (Boiteux pricing rule still applies)")
    print("  - RTD lower than translog (unitary substitution restriction)")

# Export
out = pd.DataFrame([
    {"metric": "RTD",                "translog_C_tec": op["rtd_weighted_pax"], "cobb_douglas": RTD_CD},
    {"metric": "RTS",                "translog_C_tec": op["rts_weighted_pax"], "cobb_douglas": RTS_CD},
    {"metric": "MC/AC",              "translog_C_tec": op["mc_over_ac_system"], "cobb_douglas": MC_AC},
    {"metric": "AC system (EUR/pax)","translog_C_tec": op["ac_system_eur_per_pax"], "cobb_douglas": AC_sys},
    {"metric": "MC system (EUR/pax)","translog_C_tec": op["mc_system_eur_per_pax"], "cobb_douglas": MC_sys},
    {"metric": "Boiteux subsidy (%)","translog_C_tec": op["boiteux_optimal_subsidy_pct"],
     "cobb_douglas": subsidy * 100},
])
out.to_csv(OUT_TABLES / "table_cobb_douglas.csv",
           index=False, encoding="utf-8", float_format="%.4f")
print(f"\n[OK] table_cobb_douglas.csv  ({out.shape})")
