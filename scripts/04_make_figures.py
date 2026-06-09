"""
04_make_figures.py - Generates the PDF figures of the paper.

Serif typography, ~200 dpi, vector PDF, transparent background.
All text in English. Period as decimal separator.

Under the Option B narrative the M-FU specification is no longer a
modelled alternative: the operating model is renamed C_tec and the
institutional component F_inst is treated additively. Figures 3, 4
and 7 therefore display C_tec only. Figure 6 shows the AC_tec /
AC_total split with the gap attributed to F_inst.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Style
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
    "savefig.transparent": True,
})

INCLUDED = ["L1", "L2", "L3", "L4", "L5", "L9/L10 Nord", "L9/L10 Sud"]
DISPLAY = {                                # internal → English label
    "L1": "L1", "L2": "L2", "L3": "L3", "L4": "L4", "L5": "L5",
    "L9/L10 Nord": "L9/L10 North", "L9/L10 Sud": "L9/L10 South",
}
INCLUDED_LBL = [DISPLAY[ln] for ln in INCLUDED]

panel = pd.read_csv(DATA_DIR / "panel_linea_año.csv")
t3 = pd.read_csv(OUT_TABLES / "table3_rtd_mc_by_line_year.csv")
torn = pd.read_csv(OUT_TABLES / "table_robustness.csv")
with open(OUT_TABLES / "summary_metrics.json", encoding="utf-8") as f:
    summary = json.load(f)


# ===========================================================================
# FIG 1 — Cost composition of FMB by year (stacked bars)
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.0, 5.0))
agg = panel.groupby("year")[
    ["personal", "energia", "aprov", "serv_ext",
     "amort", "renting", "canon"]
].sum() / 1e6  # M EUR

labels = {
    "personal":  "Operating labour",
    "energia":   "Energy and fuels",
    "aprov":     "Supplies",
    "serv_ext":  "External services",
    "amort":     "Net depreciation",
    "renting":   "Train leasing",
    "canon":     "Ifercat L9 concession fee",
}
colors = ["#2E5266", "#E2B040", "#A1B5C5", "#7A9CC6",
          "#557A95", "#B97D4B", "#8E2D2D"]
bottom = np.zeros(len(agg))
for col, color in zip(agg.columns, colors):
    ax.bar(agg.index.astype(str), agg[col], bottom=bottom,
           label=labels[col], color=color, edgecolor="white", linewidth=0.5)
    bottom += agg[col].values

totales = agg.sum(axis=1)
for i, (y, v) in enumerate(totales.items()):
    ax.text(i, v + 10, f"{v:.0f} M EUR", ha="center", fontweight="bold")

ax.set_ylabel("Total imputed cost (M EUR)")
ax.set_xlabel("Year")
ax.set_title("Cost composition of FMB-Metro by category, 2023-2025"
             "\n(Total imputed to the line-year panel, 7 lines; "
             "2025 annualised x12/9)")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.set_ylim(0, max(totales)*1.10)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig1_cost_composition.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# FIG 2 — RTD by line-year, C_tec
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.0, 5.0))
op = t3[t3["model"] == "M-OP"].copy()
years = sorted(op.year.unique())
x = np.arange(len(INCLUDED))
w = 0.27
year_colors = {"2023": "#2E5266", "2024": "#557A95", "2025": "#7A9CC6"}
for i, y in enumerate(years):
    yvals = [op[(op.year == y) & (op.line == ln)]["RTD"].values[0]
             for ln in INCLUDED]
    ax.bar(x + (i-1)*w, yvals, w, label=str(y),
           color=year_colors[str(y)], edgecolor="white", linewidth=0.4)

ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
ax.text(len(INCLUDED)-0.5, 1.05, "RTD = 1 (no density economies)",
        ha="right", fontsize=8, style="italic")
ax.set_xticks(x)
ax.set_xticklabels(INCLUDED_LBL, rotation=15, ha="right")
ax.set_ylabel("Returns to density (RTD)")
ax.set_title("Returns to density by line and year, technological model (C_tec)")
ax.legend(title="Year", frameon=False)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig2_rtd_by_line.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# FIG 3 — MC vs AC by line, C_tec (averages 2023-2025)
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.5, 5.0))
sub = t3[t3["model"] == "M-OP"]
means = sub.groupby("line")[["AC", "MC"]].mean().reindex(INCLUDED)
xpos = np.arange(len(INCLUDED))
ax.bar(xpos - 0.20, means["AC"], 0.40, label="Average cost (AC)",
       color="#557A95", edgecolor="white", linewidth=0.5)
ax.bar(xpos + 0.20, means["MC"], 0.40, label="Marginal cost (MC)",
       color="#E2B040", edgecolor="white", linewidth=0.5)
for i, ln in enumerate(INCLUDED):
    ax.text(xpos[i] - 0.20, means["AC"].iloc[i] + 0.02,
            f"{means['AC'].iloc[i]:.2f}", ha="center", fontsize=8)
    ax.text(xpos[i] + 0.20, means["MC"].iloc[i] + 0.02,
            f"{means['MC'].iloc[i]:.2f}", ha="center", fontsize=8)
ax.set_xticks(xpos)
ax.set_xticklabels(INCLUDED_LBL, rotation=15, ha="right")
ax.set_ylabel("EUR per passenger")
ax.set_title("Average vs marginal cost by line, technological model (C_tec)"
             "\n(mean of 2023-2025; the MC/AC ratio is the basis of the "
             "Boiteux pricing rule)")
ax.legend(frameon=False)
ax.set_ylim(0, means["AC"].max() * 1.20)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig3_mc_vs_ac.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# FIG 4 — Boiteux optimal subsidy by year, C_tec
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
subs = []
for y in years:
    s = t3[(t3["model"] == "M-OP") & (t3["year"] == y)]
    AC_w = (s["AC"] * s["pax"]).sum() / s["pax"].sum()
    MC_w = (s["MC"] * s["pax"]).sum() / s["pax"].sum()
    subs.append((1 - MC_w/AC_w) * 100)

xpos = np.arange(len(years))
ax.bar(xpos, subs, 0.55, color="#557A95",
       edgecolor="white", linewidth=0.5)
for j, v in enumerate(subs):
    ax.text(j, v + 0.8, f"{v:.1f}%", ha="center", fontsize=10,
            fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels([str(y) for y in years])
ax.set_ylabel("Boiteux optimal subsidy (%)")
ax.set_xlabel("Year")
ax.set_title("Boiteux optimal subsidy (1 - MC/AC) by year, "
             "technological model (C_tec)")
ax.set_ylim(0, max(subs) * 1.20)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig4_subsidy_boiteux.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# FIG 5 — Operating cost shares by year, stacked bars, C_tec
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.0, 4.8))
shares_yr = []
for y in years:
    s = panel[panel.year == y]
    tot_op = (s[["personal", "energia", "aprov", "serv_ext", "amort"]]
              .sum().sum())
    shares_yr.append({
        "personal": s["personal"].sum() / tot_op,
        "energia":  s["energia"].sum() / tot_op,
        "aprov":    s["aprov"].sum() / tot_op,
        "serv_ext": s["serv_ext"].sum() / tot_op,
        "amort":    s["amort"].sum() / tot_op,
    })
sdf = pd.DataFrame(shares_yr, index=[str(y) for y in years])

share_labels = {
    "personal":  "Operating labour",
    "energia":   "Energy",
    "aprov":     "Supplies",
    "serv_ext":  "External services",
    "amort":     "Net depreciation",
}
share_colors = ["#2E5266", "#E2B040", "#A1B5C5", "#7A9CC6", "#557A95"]
bottom = np.zeros(len(sdf))
for col, color in zip(sdf.columns, share_colors):
    ax.bar(sdf.index, sdf[col]*100, bottom=bottom,
           label=share_labels[col], color=color,
           edgecolor="white", linewidth=0.5)
    for i, v in enumerate(sdf[col]):
        if v > 0.04:
            ax.text(i, bottom[i] + v*100/2, f"{v*100:.1f}%",
                    ha="center", va="center",
                    color="white" if col in ("personal", "serv_ext") else "black",
                    fontsize=8)
    bottom += sdf[col].values*100

ax.set_ylabel("Operating cost share (%)")
ax.set_xlabel("Year")
ax.set_title("Operating cost shares (C_tec) by year")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig5_cost_shares_operating.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# FIG 6 — Heterogeneity L9/L10 vs L1-L5: AC_tec vs AC_total
# ===========================================================================
panel2 = panel.copy()
panel2["AC_tec"] = panel2["C_op"] / panel2["pax"]
panel2["AC_total"] = panel2["C_total"] / panel2["pax"]
panel2["F_inst_pax"] = (panel2["canon"] + panel2["renting"]) / panel2["pax"]

groups = {
    "L1-L5 (conventional)": ["L1", "L2", "L3", "L4", "L5"],
    "L9/L10 (driverless)":  ["L9/L10 Nord", "L9/L10 Sud"],
}
# Weight averages by passengers within each group
def _wmean(df, col, w):
    return (df[col] * df[w]).sum() / df[w].sum()

ac_tec = {g: _wmean(panel2[panel2.line.isin(ls)], "AC_tec", "pax")
          for g, ls in groups.items()}
ac_tot = {g: _wmean(panel2[panel2.line.isin(ls)], "AC_total", "pax")
          for g, ls in groups.items()}
f_inst = {g: ac_tot[g] - ac_tec[g] for g in groups}

fig, ax = plt.subplots(figsize=(8.5, 5.0))
xpos = np.arange(len(groups))
# Stacked bars: bottom = AC_tec, top = F_inst
bot = [ac_tec[g] for g in groups]
top = [f_inst[g] for g in groups]

ax.bar(xpos, bot, 0.55, color="#2E5266", edgecolor="white",
       linewidth=0.5, label="Technological AC (C_tec)")
ax.bar(xpos, top, 0.55, bottom=bot, color="#8E2D2D", edgecolor="white",
       linewidth=0.5, label="Institutional component (F_inst)")

for i, g in enumerate(groups):
    total = bot[i] + top[i]
    pct = top[i] / total * 100
    ax.text(i, total + 0.20, f"AC_total = {total:.2f} EUR\n"
            f"F_inst share = {pct:.0f}%",
            ha="center", fontsize=9, fontweight="bold")
    if top[i] > 0.15:
        ax.text(i, bot[i] + top[i]/2, f"{top[i]:.2f}",
                ha="center", va="center", color="white", fontsize=9)
    ax.text(i, bot[i]/2, f"{bot[i]:.2f}",
            ha="center", va="center", color="white", fontsize=9)

ax.set_xticks(xpos)
ax.set_xticklabels(list(groups.keys()))
ax.set_ylabel("Average cost per passenger (EUR/pax)")
ax.set_title("Heterogeneity L9/L10 vs L1-L5: the gap is institutional"
             "\n(passenger-weighted means, 2023-2025)")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(0, max(bot[i] + top[i] for i in range(len(groups))) * 1.30)
fig.tight_layout()
fig.savefig(OUT_FIG / "fig6_heterogeneity_L9.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# FIG 7 — Tornado plot: prior sensitivity of RTD in C_tec
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.0, 4.8))
sub = torn[torn["model"] == "M-OP"].copy()
base = sub[sub.scenario == "baseline"].iloc[0]["RTD"]
sens_keep = ["sigma_low", "sigma_high", "beta_low", "beta_high", "no_trend"]
display_scn = {
    "sigma_low":  "sigma -20%",
    "sigma_high": "sigma +20%",
    "beta_low":   "beta -20%",
    "beta_high":  "beta +20%",
    "no_trend":   "no time trend",
}
sub2 = sub[sub.scenario.isin(sens_keep)].copy()
sub2["delta"] = sub2["RTD"] - base
sub2 = sub2.sort_values("delta")
colors_bar = ["#B97D4B" if d < 0 else "#557A95" for d in sub2["delta"]]
ax.barh([display_scn[s] for s in sub2["scenario"]], sub2["delta"],
        color=colors_bar, edgecolor="white", linewidth=0.5)
for sc, d, rtd in zip(sub2["scenario"], sub2["delta"], sub2["RTD"]):
    ax.text(d + (0.02 if d > 0 else -0.02), display_scn[sc],
            f"RTD={rtd:.2f}",
            va="center", ha="left" if d > 0 else "right", fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set_title(f"Sensitivity of RTD to priors (+/-20%), C_tec  "
             f"(baseline RTD = {base:.3f})")
ax.set_xlabel("Delta RTD vs baseline")
fig.tight_layout()
fig.savefig(OUT_FIG / "fig7_tornado.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# FIG 8 — Conceptual flow diagram (data → calibration → metrics → policy)
#         Updated for Option B: C_total = C_tec + F_inst
# ===========================================================================
fig, ax = plt.subplots(figsize=(11.0, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
ax.grid(False)

boxes = [
    (0.5, 2.5, 2.0, 1.5,
     "DATA\nFMB-TMB 2023-2025\n"
     "Costs (budget)\n"
     "Passengers, car-km,\n"
     "peak-hour trains,\n"
     "MWh, employees",
     "#A1B5C5"),
    (3.0, 2.5, 2.0, 1.5,
     "PANEL\nLine-year (T=21)\n"
     "7 lines x 3 years\n"
     "Mixed-key imputation:\n"
     "share_ckm + share_km",
     "#7A9CC6"),
    (5.5, 2.5, 2.0, 1.5,
     "CALIBRATION\nTranslog C_tec\n"
     "1st order: anchored\n"
     "on shares and beta\n"
     "from literature\n"
     "2nd order: sigma priors",
     "#557A95"),
    (8.0, 2.5, 1.8, 1.5,
     "METRICS\nRTD, RTS\n"
     "MC, AC, MC/AC\n"
     "Boiteux subsidy\n"
     "Regularity tests",
     "#2E5266"),
]
for x, y, w_, h, txt, color in boxes:
    box = FancyBboxPatch((x, y), w_, h, boxstyle="round,pad=0.06",
                         linewidth=1.2, edgecolor="#333333",
                         facecolor=color, alpha=0.85)
    ax.add_patch(box)
    ax.text(x + w_/2, y + h/2, txt, ha="center", va="center",
            fontsize=9, color="white" if color != "#A1B5C5" else "black")

for x1, x2 in [(2.55, 2.95), (5.05, 5.45), (7.55, 7.95)]:
    arr = FancyArrowPatch((x1, 3.25), (x2, 3.25),
                          arrowstyle="->", mutation_scale=18,
                          color="#333333", lw=1.5)
    ax.add_patch(arr)

pol = FancyBboxPatch((0.2, 0.3), 9.6, 1.4,
                     boxstyle="round,pad=0.08",
                     linewidth=1.2, edgecolor="#7A2D2D",
                     facecolor="#E8C3A8", alpha=0.85)
ax.add_patch(pol)
ax.text(5.0, 1.0,
        "POLICY\n"
        "H1 density economies (RTD > 1)  |  "
        "H2 MC << AC -> Boiteux subsidy  |  "
        "H3 additive decomposition C_total = C_tec + F_inst  |  "
        "H4 L9/L10 heterogeneity",
        ha="center", va="center", fontsize=10, fontweight="bold")

arr2 = FancyArrowPatch((8.8, 2.4), (8.8, 1.8),
                       arrowstyle="->", mutation_scale=18,
                       color="#7A2D2D", lw=1.5)
ax.add_patch(arr2)

ax.text(5.0, 5.3,
        "Multiproduct translog model FMB-TMB: analysis flow",
        ha="center", va="center", fontsize=12, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT_FIG / "fig8_conceptual.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
print("\n[OK] 8 figures exported to outputs/figures/")
for i in range(1, 9):
    matches = list(OUT_FIG.glob(f"fig{i}_*.pdf"))
    if matches:
        f = matches[0]
        size_kb = f.stat().st_size / 1024
        print(f"  fig{i}: {f.name}  ({size_kb:.1f} KB)")
