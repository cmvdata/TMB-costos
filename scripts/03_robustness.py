"""
03_robustness.py - Análisis de robustez del modelo translog FMB-TMB.

Escenarios:
(a) Sensibilidad a priors (σ y β): bajo/alto, sin trend
(b) Imputación alternativa por pasajeros (sanity check: RTD ≈ 1 si se re-estima β)
(c) Modelo agregado T=3 (matriz mal condicionada → justifica panel línea-año)
(d) Monoproducto (solo pax) → RTD ≈ 1.43
(e) Excluyendo 2025 → cuotas estables
(f) Validación contable (C_pred vs C_obs)
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_TABLES = ROOT / "outputs" / "tables"
INPUT_DIR = ROOT

# ---------------------------------------------------------------------------
# 0. Cargar funciones de 02_calibrate_translog.py vía importlib (nombre con dígito)
# ---------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    "calib", ROOT / "scripts" / "02_calibrate_translog.py"
)
calib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib)

INCLUDED = ["L1", "L2", "L3", "L4", "L5", "L9/L10 Nord", "L9/L10 Sud"]
panel = pd.read_csv(DATA_DIR / "panel_linea_año.csv")


# ---------------------------------------------------------------------------
# Helper: re-construir panel con clave de imputación alternativa (por pax)
# ---------------------------------------------------------------------------
def build_pax_imputed_panel(panel_orig):
    """Re-imputa todos los costes por cuota de pasajeros (en vez de coches-km).

    Reconstruye personal/energia/aprov/serv_ext/amort/renting/canon por línea
    como FMB_partida_año × share_pax[ln][y].
    """
    df = panel_orig.copy()
    # FMB totales por año (suma sobre las 7 líneas + sobre las 2 excluidas ≈ FMB)
    # Como tenemos el panel ya imputado, recuperamos FMB-año dividiendo por la
    # cobertura (~0.998). Pero más limpio: leer pasajeros TOTAL FMB y FMB partidas.

    # Pasajeros totales FMB por año (incluye L11+Funicular)
    op = pd.read_excel(INPUT_DIR / "operativos_fmb.xlsx",
                       sheet_name="outputs_y_denominadores")
    op = op.rename(columns={op.columns[0]: "year"})
    pax_total_fmb = {int(r["year"]): int(r["pasajeros_total_xlsx"])
                     for _, r in op.iterrows()}

    # FMB partidas: recuperar como sum panel / share_ckm_sumado (cobertura)
    fmb_partidas = {}
    for y in [2023, 2024, 2025]:
        sub = panel_orig[panel_orig.year == y]
        cov = sub["share_ckm"].sum()   # ≈ 0.998
        fmb_partidas[y] = {
            "personal":  sub["personal"].sum() / cov,
            "energia":   sub["energia"].sum() / cov,
            "aprov":     sub["aprov"].sum() / cov,
            "serv_ext":  sub["serv_ext"].sum() / cov,
            "renting":   sub["renting"].sum() / cov,
            # amort se imputaba por share_km — recuperamos con share_km
            "amort":     sub["amort"].sum() / sub["share_km"].sum(),
            # canon iba 100% a L9/L10
            "canon":     sub["canon"].sum(),
        }

    # Re-imputar: share_pax[ln][y] = pax_línea / pax_total_FMB[y]
    df_out = df.copy()
    for i, row in df.iterrows():
        y = int(row["year"])
        share_pax = row["pax"] / pax_total_fmb[y]
        for p in ["personal", "energia", "aprov", "serv_ext",
                  "amort", "renting", "canon"]:
            df_out.at[i, p] = fmb_partidas[y][p] * share_pax
        df_out.at[i, "C_op"] = sum(
            df_out.at[i, k] for k in ["personal", "energia", "aprov",
                                      "serv_ext", "amort"]
        )
        df_out.at[i, "C_total"] = df_out.at[i, "C_op"] + \
            df_out.at[i, "renting"] + df_out.at[i, "canon"]
    return df_out


def build_pre2025_panel(panel_orig):
    return panel_orig[panel_orig.year < 2025].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper: OLS data-driven sobre el panel
# ---------------------------------------------------------------------------
def ols_translog_betas(df, model="M-OP"):
    """OLS de lnC ~ const + lnY1 + lnY2 + yearFE.

    Devuelve β_1 (=coef lnY1), β_2 (=coef lnY2), RTD=1/β_1 implícito.
    """
    if model == "M-OP":
        d = df.copy()
        d["C"] = d["C_op"]
    else:
        d = df.copy()
        d["C"] = d["C_total"]
    d = d[d["C"] > 0].copy()
    d["lnC"] = np.log(d["C"])
    d["lnY1"] = np.log(d["pax"])
    d["lnY2"] = np.log(d["coches_km"])
    years = sorted(d["year"].unique())
    # FE de año
    dummies = [(d["year"] == yy).astype(int) for yy in years[1:]]
    X_cols = [np.ones(len(d)), d["lnY1"].values, d["lnY2"].values]
    X_cols += [arr.values for arr in dummies]
    X = np.column_stack(X_cols)
    y = d["lnC"].values
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    cond = float(np.linalg.cond(X))
    return dict(
        const=float(beta[0]), b1=float(beta[1]), b2=float(beta[2]),
        rtd=float(1/beta[1]) if abs(beta[1]) > 1e-9 else float("inf"),
        cond=cond, n=len(d), rank=int(rank),
    )


# ---------------------------------------------------------------------------
# (a) SENSIBILIDAD A PRIORS
# ---------------------------------------------------------------------------
print("=" * 78)
print("(a) SENSIBILIDAD A PRIORS")
print("=" * 78)
sensitivity_scenarios = {
    "baseline":   {},
    "sigma_low":  dict(sigma_LE=0.20, sigma_LO=0.30, sigma_EO=0.25),
    "sigma_high": dict(sigma_LE=0.70, sigma_LO=0.90, sigma_EO=0.80),
    "beta_low":   dict(beta1_op=0.35, beta2_op=0.20, beta1_fu=0.40, beta2_fu=0.15),
    "beta_high":  dict(beta1_op=0.55, beta2_op=0.30, beta1_fu=0.60, beta2_fu=0.25),
    "no_trend":   dict(lam=0.0),
}

sens_rows = []
for sname, prior_override in sensitivity_scenarios.items():
    for m in ["M-OP", "M-FU"]:
        # Para β_low/high, los priors son distintos por modelo
        po = dict(prior_override)
        if "beta1_op" in po:
            if m == "M-OP":
                po = dict(beta1=po["beta1_op"], beta2=po["beta2_op"])
            else:
                po = dict(beta1=po["beta1_fu"], beta2=po["beta2_fu"])
        r = calib.run_model(panel, m, priors=po)
        sens_rows.append(dict(
            scenario=sname, model=m,
            rtd_w=r["rtd_w"], rts_w=r["rts_w"],
            mc_ac=r["mc_ac_sys"],
            subsidy_pct=r["subsidio"]*100,
            rmse_M=r["rmse"]/1e6,
        ))
        print(f"  {sname:14s} {m:5s}  RTD={r['rtd_w']:.3f}  "
              f"MC/AC={r['mc_ac_sys']:.3f}  RMSE={r['rmse']/1e6:.2f}M")

sens_df = pd.DataFrame(sens_rows)


# ---------------------------------------------------------------------------
# (b) IMPUTACIÓN POR PASAJEROS
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(b) IMPUTACION POR PASAJEROS (sanity check)")
print("=" * 78)
panel_pax = build_pax_imputed_panel(panel)

# Comparación con priors de literatura (β fijo)
print("Calibracion con priors fijos (literatura beta_1=0.45/0.50):")
for m in ["M-OP", "M-FU"]:
    r_ckm = calib.run_model(panel, m)
    r_pax = calib.run_model(panel_pax, m)
    print(f"  {m}: RTD ckm-imp = {r_ckm['rtd_w']:.3f}  |  "
          f"pax-imp = {r_pax['rtd_w']:.3f}  "
          f"(invariante porque beta anclado a literatura)")

# OLS data-driven
print("\nOLS data-driven (lnC ~ lnY1 + lnY2 + yearFE) -- re-estimacion de beta:")
ols_rows = []
for label, df in [("coches-km imp.", panel), ("pax imp.", panel_pax)]:
    for m in ["M-OP", "M-FU"]:
        ols = ols_translog_betas(df, m)
        ols_rows.append(dict(
            imputation=label, model=m,
            beta1_ols=ols["b1"], beta2_ols=ols["b2"],
            RTD_ols=ols["rtd"], cond=ols["cond"],
        ))
        print(f"  {label:18s} {m:5s}  beta1={ols['b1']:+.3f}  "
              f"beta2={ols['b2']:+.3f}  RTD={ols['rtd']:+.3f}  "
              f"cond={ols['cond']:.1f}")

ols_df = pd.DataFrame(ols_rows)


# ---------------------------------------------------------------------------
# (c) MODELO AGREGADO T=3
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(c) MODELO AGREGADO T=3 (diferencias finitas)")
print("=" * 78)
# Agregamos el panel línea-año a sistema-año (sumas por año)
agg = panel.groupby("year").agg(
    C_total=("C_total", "sum"),
    C_op=("C_op", "sum"),
    pax=("pax", "sum"),
    coches_km=("coches_km", "sum"),
).reset_index().sort_values("year")

print("Sistema-anio imputado:")
print(agg.to_string(index=False))

# Diferencias finitas para resolver lnC_t = β1 lnY1_t + β2 lnY2_t + λ
agg_t3_rows = []
for label, ycol in [("M-OP", "C_op"), ("M-FU", "C_total")]:
    dlnC = np.diff(np.log(agg[ycol].values))
    dlnY1 = np.diff(np.log(agg["pax"].values))
    dlnY2 = np.diff(np.log(agg["coches_km"].values))
    X = np.column_stack([dlnY1, dlnY2])    # 2x2
    y = dlnC
    det = X[0, 0]*X[1, 1] - X[0, 1]*X[1, 0]
    cond = float(np.linalg.cond(X))
    # Resolver X · [β1, β2]' = y  (sistema 2x2)
    if abs(det) > 1e-12:
        beta = np.linalg.solve(X, y)
        b1, b2 = float(beta[0]), float(beta[1])
        rtd = 1/b1 if abs(b1) > 1e-9 else float("inf")
        rts = 1/(b1+b2) if abs(b1+b2) > 1e-9 else float("inf")
    else:
        b1 = b2 = rtd = rts = float("nan")
    agg_t3_rows.append(dict(
        model=label, beta1=b1, beta2=b2, RTD=rtd, RTS=rts,
        det_X=float(det), cond_X=cond,
    ))
    print(f"  {label}:  beta1={b1:+.3f}  beta2={b2:+.3f}  "
          f"RTD={rtd:+.3f}  cond(X)={cond:.2f}  det(X)={det:+.5e}")

agg_t3_df = pd.DataFrame(agg_t3_rows)


# ---------------------------------------------------------------------------
# (d) MONOPRODUCTO (solo pax como output)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(d) MONOPRODUCTO (solo pax, beta_1=0.70)")
print("=" * 78)
mono_priors = dict(
    beta1=0.70, beta2=0.001,    # 0.001 ≈ 0 (evita división por 0)
    delta_22=0.0, delta_12=0.0,
    rho_LY2=0.0, rho_EY2=0.0,
)
mono_rows = []
for m in ["M-OP", "M-FU"]:
    r = calib.run_model(panel, m, priors=mono_priors)
    mono_rows.append(dict(
        model=m, RTD=r["rtd_w"], RTS=r["rts_w"],
        MC_AC=r["mc_ac_sys"], subsidy_pct=r["subsidio"]*100,
    ))
    print(f"  {m}: RTD_mono = {r['rtd_w']:.3f}  (vs biproducto = 2.18/1.96)")
mono_df = pd.DataFrame(mono_rows)


# ---------------------------------------------------------------------------
# (e) EXCLUYENDO 2025
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(e) EXCLUYENDO 2025 (panel 2023-2024)")
print("=" * 78)
panel_pre = build_pre2025_panel(panel)
print(f"  Panel: {len(panel_pre)} obs (7 lineas x 2 anios)")
ex25_rows = []
for m in ["M-OP", "M-FU"]:
    r_full = calib.run_model(panel, m)
    r_pre  = calib.run_model(panel_pre, m)
    p_full, p_pre = r_full["params"], r_pre["params"]
    delta_sL = (p_pre["alpha_L"] - p_full["alpha_L"]) * 100
    delta_sE = (p_pre["alpha_E"] - p_full["alpha_E"]) * 100
    delta_rtd = r_pre["rtd_w"] - r_full["rtd_w"]
    ex25_rows.append(dict(
        model=m,
        sL_full=p_full["alpha_L"], sL_pre2025=p_pre["alpha_L"],
        sE_full=p_full["alpha_E"], sE_pre2025=p_pre["alpha_E"],
        RTD_full=r_full["rtd_w"], RTD_pre2025=r_pre["rtd_w"],
        delta_sL_pp=delta_sL, delta_sE_pp=delta_sE,
        delta_RTD=delta_rtd,
    ))
    print(f"  {m}: dsL={delta_sL:+.2f}pp, dsE={delta_sE:+.2f}pp, "
          f"dRTD={delta_rtd:+.4f}")
ex25_df = pd.DataFrame(ex25_rows)


# ---------------------------------------------------------------------------
# (f) VALIDACIÓN CONTABLE (C_pred / C_obs por línea-año)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("(f) VALIDACION CONTABLE (C_pred / C_obs)")
print("=" * 78)
val_rows = []
for m in ["M-OP", "M-FU"]:
    r = calib.run_model(panel, m)
    df_r = r["df"].reset_index(drop=True).copy()
    df_r["C_pred"] = r["C_pred"].values
    df_r["error_pct"] = (df_r["C_pred"] - df_r["C"]) / df_r["C"] * 100
    max_err = df_r["error_pct"].abs().max()
    mean_abs = df_r["error_pct"].abs().mean()
    p95 = np.percentile(df_r["error_pct"].abs(), 95)
    val_rows.append(dict(
        model=m, MAE_pct=mean_abs, max_err_pct=max_err, p95_err_pct=p95,
        pass_5pct=bool(max_err <= 5.0),
    ))
    print(f"  {m}: MAE={mean_abs:.2f}%  max|err|={max_err:.2f}%  "
          f"P95={p95:.2f}%  [<=5%: {max_err <= 5.0}]")
val_df = pd.DataFrame(val_rows)


# ---------------------------------------------------------------------------
# TORNADO TABLE (consolidada)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("TABLA TORNADO (consolidada)")
print("=" * 78)
baseline_op = sens_df[(sens_df.scenario == "baseline") & (sens_df.model == "M-OP")].iloc[0]
baseline_fu = sens_df[(sens_df.scenario == "baseline") & (sens_df.model == "M-FU")].iloc[0]

tornado_rows = []
for _, row in sens_df.iterrows():
    base = baseline_op if row["model"] == "M-OP" else baseline_fu
    tornado_rows.append(dict(
        scenario=row["scenario"], model=row["model"],
        RTD=row["rtd_w"], delta_RTD=row["rtd_w"] - base["rtd_w"],
        MC_AC=row["mc_ac"], delta_MC_AC=row["mc_ac"] - base["mc_ac"],
        subsidy_pct=row["subsidy_pct"],
    ))

# Añadir escenarios (b), (d) y (e)
for _, row in ols_df.iterrows():
    rtd = row["RTD_ols"] if np.isfinite(row["RTD_ols"]) else float("nan")
    tornado_rows.append(dict(
        scenario=f"OLS-{row['imputation']}",
        model=row["model"], RTD=rtd, delta_RTD=float("nan"),
        MC_AC=float("nan"), delta_MC_AC=float("nan"),
        subsidy_pct=float("nan"),
    ))
for _, row in mono_df.iterrows():
    base = baseline_op if row["model"] == "M-OP" else baseline_fu
    tornado_rows.append(dict(
        scenario="monoproducto", model=row["model"],
        RTD=row["RTD"], delta_RTD=row["RTD"] - base["rtd_w"],
        MC_AC=row["MC_AC"], delta_MC_AC=row["MC_AC"] - base["mc_ac"],
        subsidy_pct=row["subsidy_pct"],
    ))
for _, row in ex25_df.iterrows():
    base = baseline_op if row["model"] == "M-OP" else baseline_fu
    tornado_rows.append(dict(
        scenario="excl-2025", model=row["model"],
        RTD=row["RTD_pre2025"], delta_RTD=row["delta_RTD"],
        MC_AC=float("nan"), delta_MC_AC=float("nan"),
        subsidy_pct=float("nan"),
    ))

tornado_df = pd.DataFrame(tornado_rows)
print(tornado_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Exportar
# ---------------------------------------------------------------------------
sens_df.to_csv(OUT_TABLES / "robustness_sensitivity.csv", index=False)
ols_df.to_csv(OUT_TABLES / "robustness_ols_imputation.csv", index=False)
agg_t3_df.to_csv(OUT_TABLES / "robustness_aggregated_T3.csv", index=False)
mono_df.to_csv(OUT_TABLES / "robustness_monoproduct.csv", index=False)
ex25_df.to_csv(OUT_TABLES / "robustness_excl2025.csv", index=False)
val_df.to_csv(OUT_TABLES / "robustness_validation.csv", index=False)
tornado_df.to_csv(OUT_TABLES / "table_robustness.csv", index=False)

# Resumen JSON
out_summary = dict(
    sensitivity={r["scenario"] + "__" + r["model"]: dict(
        RTD=r["rtd_w"], MC_AC=r["mc_ac"], RMSE_M=r["rmse_M"],
    ) for _, r in sens_df.iterrows()},
    ols={r["imputation"] + "__" + r["model"]:
         dict(beta1=r["beta1_ols"], beta2=r["beta2_ols"], RTD=r["RTD_ols"])
         for _, r in ols_df.iterrows()},
    aggregated_T3={r["model"]: dict(
        beta1=r["beta1"], beta2=r["beta2"], RTD=r["RTD"], cond=r["cond_X"],
    ) for _, r in agg_t3_df.iterrows()},
    monoproduct={r["model"]: dict(RTD=r["RTD"], MC_AC=r["MC_AC"])
                 for _, r in mono_df.iterrows()},
    excl2025={r["model"]: dict(
        delta_sL_pp=r["delta_sL_pp"], delta_sE_pp=r["delta_sE_pp"],
        delta_RTD=r["delta_RTD"],
    ) for _, r in ex25_df.iterrows()},
    validation={r["model"]: dict(
        MAE_pct=r["MAE_pct"], max_err_pct=r["max_err_pct"],
        p95_err_pct=r["p95_err_pct"], pass_5pct=r["pass_5pct"],
    ) for _, r in val_df.iterrows()},
)
with open(OUT_TABLES / "robustness_summary.json", "w", encoding="utf-8") as f:
    json.dump(out_summary, f, indent=2, ensure_ascii=False, default=str)

print("\n[OK] Tablas exportadas a outputs/tables/")
print("  - robustness_sensitivity.csv")
print("  - robustness_ols_imputation.csv")
print("  - robustness_aggregated_T3.csv")
print("  - robustness_monoproduct.csv")
print("  - robustness_excl2025.csv")
print("  - robustness_validation.csv")
print("  - table_robustness.csv (tornado)")
print("  - robustness_summary.json")
