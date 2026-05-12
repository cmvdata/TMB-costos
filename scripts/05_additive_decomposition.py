"""
05_additive_decomposition.py - Implements the additive decomposition

    C_total(Y, W, institutions) = C_tec(Y, W) + F_inst

Where
    C_tec  = operating model (renamed M-OP). Satisfies Shephard's properties.
    F_inst = Ifercat L9 concession fee + train leasing.
             Quasi-fixed contractual transfers, NOT a translog input.

Outputs (all in English, period decimal separator, UTF-8):
    - outputs/tables/table_h3_decomposition.csv    (H3 closing table)
    - outputs/tables/table_finst_by_line.csv       (F_inst by line-year)
    - outputs/figures/fig_additive_decomposition.pdf
    - outputs/figures/fig_concavity_diagnostic.pdf

The concavity diagnostic re-uses the M-FU calibration ONLY as a negative
proof that forcing F_inst inside the translog breaks Shephard's concavity.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCRIPTS = ROOT / "scripts"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Match figure style with the rest of the paper
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
})

INCLUDED = ["L1", "L2", "L3", "L4", "L5", "L9/L10 Nord", "L9/L10 Sud"]
DISPLAY = {                               # English display labels
    "L1": "L1", "L2": "L2", "L3": "L3", "L4": "L4", "L5": "L5",
    "L9/L10 Nord": "L9/L10 North", "L9/L10 Sud": "L9/L10 South",
}

# ---------------------------------------------------------------------------
# 1. LOAD PANEL
# ---------------------------------------------------------------------------
panel = pd.read_csv(DATA_DIR / "panel_linea_año.csv")
assert len(panel) == 21, "Panel must have 21 line-year observations"

# F_inst per line-year
panel["F_inst"] = panel["canon"] + panel["renting"]
panel["C_tec"] = panel["C_op"]                  # rename for clarity
# Sanity: C_tec + F_inst == C_total
diff = (panel["C_tec"] + panel["F_inst"]) - panel["C_total"]
assert diff.abs().max() < 1.0, "C_tec + F_inst does not reconcile to C_total"

# Per-passenger
panel["AC_tec_pax"] = panel["C_tec"] / panel["pax"]
panel["AC_total_pax"] = panel["C_total"] / panel["pax"]
panel["F_inst_pax"] = panel["F_inst"] / panel["pax"]
panel["F_inst_share"] = panel["F_inst"] / panel["C_total"]

# ---------------------------------------------------------------------------
# 2. F_INST AGGREGATES
# ---------------------------------------------------------------------------
print("=" * 78)
print("ADDITIVE DECOMPOSITION  -  F_inst diagnostics")
print("=" * 78)

print("\nSystem-wide totals by year:")
agg_year = panel.groupby("year").agg(
    C_tec=("C_tec", "sum"),
    F_inst=("F_inst", "sum"),
    C_total=("C_total", "sum"),
    pax=("pax", "sum"),
).assign(
    F_inst_share=lambda d: d["F_inst"] / d["C_total"],
    F_inst_per_pax=lambda d: d["F_inst"] / d["pax"],
    AC_tec=lambda d: d["C_tec"] / d["pax"],
    AC_total=lambda d: d["C_total"] / d["pax"],
)
print(agg_year.to_string(float_format=lambda v: f"{v:,.4f}"))

# Conventional lines vs L9/L10
def _group(ln: str) -> str:
    return "L9/L10" if "L9" in ln else "L1-L5"

panel["group"] = panel["line"].map(_group)
print("\nDistribution of F_inst across line groups (pooled 2023-2025):")
dist = panel.groupby("group").agg(
    F_inst=("F_inst", "sum"),
    C_total=("C_total", "sum"),
    pax=("pax", "sum"),
).assign(
    share_of_F_inst=lambda d: d["F_inst"] / d["F_inst"].sum(),
    F_inst_share_of_total=lambda d: d["F_inst"] / d["C_total"],
    F_inst_per_pax=lambda d: d["F_inst"] / d["pax"],
)
print(dist.to_string(float_format=lambda v: f"{v:,.4f}"))

# ---------------------------------------------------------------------------
# 3. H3 CLOSING TABLE
# ---------------------------------------------------------------------------
# Pull MC and RTD for C_tec from the calibrated M-OP results
with open(OUT_TABLES / "summary_metrics.json", encoding="utf-8") as f:
    summary = json.load(f)

op = summary["M-OP"]
AC_tec_sys = op["ac_system_eur_per_pax"]                        # ~0.91
MC_tec_sys = op["mc_system_eur_per_pax"]                        # ~0.42
RTD_tec = op["rtd_weighted_pax"]                                # ~2.18
F_inst_per_pax_sys = panel["F_inst"].sum() / panel["pax"].sum()
AC_total_sys = panel["C_total"].sum() / panel["pax"].sum()
share_tec = panel["C_tec"].sum() / panel["C_total"].sum()
share_finst = panel["F_inst"].sum() / panel["C_total"].sum()

# MC of F_inst is 0 by construction (fixed contractual transfer)
h3 = pd.DataFrame([
    {"Metric": "Mean AC (EUR/pax)",
     "C_tec (operating model)": AC_tec_sys,
     "F_inst (institutional)": F_inst_per_pax_sys,
     "C_total (full)": AC_total_sys},
    {"Metric": "Mean MC (EUR/pax)",
     "C_tec (operating model)": MC_tec_sys,
     "F_inst (institutional)": 0.0,
     "C_total (full)": MC_tec_sys},
    {"Metric": "Returns to density (RTD)",
     "C_tec (operating model)": f"{RTD_tec:.3f}",
     "F_inst (institutional)": "N/A (fixed)",
     "C_total (full)": "N/A"},
    {"Metric": "Cost share",
     "C_tec (operating model)": share_tec,
     "F_inst (institutional)": share_finst,
     "C_total (full)": 1.0},
    {"Metric": "Satisfies Shephard's properties",
     "C_tec (operating model)": "Yes",
     "F_inst (institutional)": "Not applicable (non-technological)",
     "C_total (full)": "No (concavity fails in 10/21 obs)"},
])
# Format numeric cells using period as decimal separator (default in pandas)
def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return v

h3_fmt = h3.copy()
for c in ["C_tec (operating model)", "F_inst (institutional)", "C_total (full)"]:
    h3_fmt[c] = h3_fmt[c].map(_fmt)

h3_fmt.to_csv(OUT_TABLES / "table_h3_decomposition.csv",
              index=False, encoding="utf-8")
print("\n" + "=" * 78)
print("H3 CLOSING TABLE  ->  outputs/tables/table_h3_decomposition.csv")
print("=" * 78)
print(h3_fmt.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. F_INST APPENDIX TABLE
# ---------------------------------------------------------------------------
appendix = panel[[
    "line", "year", "canon", "renting", "F_inst",
    "F_inst_pax", "F_inst_share",
]].copy()
appendix.columns = [
    "line", "year",
    "concession_fee", "train_leasing",
    "F_inst_total", "F_inst_per_pax", "F_inst_share_of_total",
]
appendix["line"] = appendix["line"].map(DISPLAY)
appendix = appendix.sort_values(["year", "line"]).reset_index(drop=True)
appendix.to_csv(OUT_TABLES / "table_finst_by_line.csv",
                index=False, encoding="utf-8", float_format="%.6f")
print(f"\n[OK] table_finst_by_line.csv  ({appendix.shape})")

# ---------------------------------------------------------------------------
# 5. FIG: ADDITIVE DECOMPOSITION (stacked bars by line, 3-year average)
# ---------------------------------------------------------------------------
mean_by_line = panel.groupby("line").agg(
    AC_tec=("AC_tec_pax", "mean"),
    F_inst_pax=("F_inst_pax", "mean"),
    AC_total=("AC_total_pax", "mean"),
).reindex(INCLUDED)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
x = np.arange(len(INCLUDED))
bot = mean_by_line["AC_tec"].values
top = mean_by_line["F_inst_pax"].values

ax.bar(x, bot, color="#2E5266", edgecolor="white", linewidth=0.5,
       label="Technological AC (C_tec)")
ax.bar(x, top, bottom=bot, color="#8E2D2D", edgecolor="white", linewidth=0.5,
       label="Institutional component (F_inst)")

# Annotate totals
for i, (b, t) in enumerate(zip(bot, top)):
    total = b + t
    ax.text(i, total + 0.15, f"{total:.2f}", ha="center",
            fontweight="bold", fontsize=9)
    if t > 0.05:
        ax.text(i, b + t/2, f"{t:.2f}", ha="center", va="center",
                color="white", fontsize=8)
    ax.text(i, b/2, f"{b:.2f}", ha="center", va="center",
            color="white", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([DISPLAY[ln] for ln in INCLUDED],
                   rotation=15, ha="right")
ax.set_ylabel("Average cost per passenger (EUR/pax)")
ax.set_title("Additive decomposition of average cost by line, 2023-2025 mean"
             "\nC_total = C_tec (technology) + F_inst (institutional transfer)")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(0, max(bot + top) * 1.18)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig_additive_decomposition.pdf", bbox_inches="tight")
plt.close(fig)
print(f"[OK] fig_additive_decomposition.pdf")

# ---------------------------------------------------------------------------
# 6. FIG: CONCAVITY DIAGNOSTIC (max eigenvalue of M-FU substitution matrix)
# ---------------------------------------------------------------------------
# Re-use the M-FU calibration from script 02 (only as a negative diagnostic)
spec = importlib.util.spec_from_file_location(
    "calib", SCRIPTS / "02_calibrate_translog.py")
calib = importlib.util.module_from_spec(spec)
sys.modules["calib"] = calib
spec.loader.exec_module(calib)

panel_for_fu = pd.read_csv(DATA_DIR / "panel_linea_año.csv")
fu = calib.run_model(panel_for_fu, "M-FU")
eig = fu["eig"]                                            # (21, 3)
eig_max = eig.max(axis=1)
df_fu = fu["df"].reset_index(drop=True)
labels_x = [f"{int(df_fu['year'].iloc[i])} {DISPLAY[df_fu['line'].iloc[i]]}"
            for i in range(len(df_fu))]

fig, ax = plt.subplots(figsize=(9.5, 5.5))
colors = ["#8E2D2D" if v > 0 else "#2E5266" for v in eig_max]
xs = np.arange(len(eig_max))
ax.bar(xs, eig_max, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(0.0, color="black", lw=1.0)
ax.set_xticks(xs)
ax.set_xticklabels(labels_x, rotation=70, ha="right", fontsize=8)
ax.set_ylabel("Maximum eigenvalue")
n_viol = int((eig_max > 0).sum())
ax.set_title("Concavity diagnostic for the full specification (M-FU): "
             "maximum eigenvalue of the Allen-Uzawa substitution matrix"
             f"\n{n_viol} of {len(eig_max)} observations violate concavity "
             "(eigenvalue > 0)")
# Light annotation
ax.text(0.99, 0.97,
        f"max = {eig_max.max():+.5f}\nviolations: {n_viol}/{len(eig_max)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec="black", alpha=0.85))
fig.tight_layout()
fig.savefig(OUT_FIG / "fig_concavity_diagnostic.pdf", bbox_inches="tight")
plt.close(fig)
print(f"[OK] fig_concavity_diagnostic.pdf")

# ---------------------------------------------------------------------------
# 7. NUMERIC CHECK AGAINST EXPECTED VALUES
# ---------------------------------------------------------------------------
expected = dict(
    AC_tec=0.91, MC_tec=0.42,
    F_inst_per_pax=0.39, F_inst_share=0.29,
)
got = dict(
    AC_tec=AC_tec_sys,
    MC_tec=MC_tec_sys,
    F_inst_per_pax=F_inst_per_pax_sys,
    F_inst_share=share_finst,
)
print("\n" + "=" * 78)
print("EXPECTED vs COMPUTED")
print("=" * 78)
print(f"{'metric':<22}{'expected':>12}{'computed':>14}{'|delta|':>12}")
TOL_ABS = 0.05
TOL_REL = 0.10
fail = []
for k in expected:
    e, c = expected[k], got[k]
    d = abs(e - c)
    flag = " " if d <= max(TOL_ABS, TOL_REL*abs(e)) else " <-"
    print(f"{k:<22}{e:>12.4f}{c:>14.6f}{d:>12.6f}{flag}")
    if flag.strip():
        fail.append((k, e, c, d))

if fail:
    print("\n[STOP] Computed numbers materially differ from expected values:")
    for k, e, c, d in fail:
        print(f"  {k}: expected ~{e}, got {c:.4f}  (|delta| = {d:.4f})")
else:
    print("\nAll computed values are within tolerance of the expected ranges.")

print("\nDone.")
