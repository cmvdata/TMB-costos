"""
02_calibrate_translog.py - Calibración del modelo translog multiproducto
para FMB-TMB sobre el panel línea-año (T=21), bajo dos especificaciones:
  M-OP (operacional, sin canon ni renting)
  M-FU (pleno, todos los costes recurrentes)

Anclaje:
- Primer orden: cuotas observadas; β tomado de literatura (Savage 1997, Graham 2008).
- Segundo orden: priors σ Allen-Uzawa (Wheat & Smith 2015); δ_kk, ρ_ik priors.
- Homogeneidad lineal en precios impuesta por construcción.

Especificación (en desviaciones de la media geométrica de la muestra):
  ln(C/W_O)_c = α_L·w_L + α_E·w_E + β_1·y_1 + β_2·y_2
              + 0.5·γ_LL·w_L² + 0.5·γ_EE·w_E² + γ_LE·w_L·w_E
              + 0.5·δ_11·y_1² + 0.5·δ_22·y_2² + δ_12·y_1·y_2
              + Σ_{i∈{L,E}}Σ_{k∈{1,2}} ρ_ik · w_i · y_k
              + λ·t
donde:
  w_L = ln(W_L/W_O) − media;  w_E = ln(W_E/W_O) − media
  y_1 = ln(pax) − media;      y_2 = ln(coches_km) − media
  t   = year − 2024
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_TABLES.mkdir(parents=True, exist_ok=True)

INCLUDED = ["L1", "L2", "L3", "L4", "L5", "L9/L10 Nord", "L9/L10 Sud"]

# ---------------------------------------------------------------------------
# 1. PRIORS DE LITERATURA (Wheat & Smith 2015; Savage 1997; Graham 2008)
# ---------------------------------------------------------------------------
SIGMA_LE = 0.40        # Allen-Uzawa labor-energy
SIGMA_LO = 0.60        # labor-other
SIGMA_EO = 0.50        # energy-other

DELTA_11 = 0.05        # output-output
DELTA_22 = 0.05
DELTA_12 = -0.05

RHO_LY1 = -0.04        # precio-output (M-OP y M-FU)
RHO_LY2 = +0.02
RHO_EY1 = +0.01        # ya ajustado de 0.03 a 0.01 (rigidez frecuencia L9/L10)
RHO_EY2 = +0.01

BETA_PRIORS = {
    "M-OP": dict(beta1=0.45, beta2=0.25, lam=-0.005),
    "M-FU": dict(beta1=0.50, beta2=0.20, lam=-0.003),
}


# ---------------------------------------------------------------------------
# 2. FUNCIONES DE CALIBRACIÓN
# ---------------------------------------------------------------------------
def build_design(panel, model="M-OP"):
    """Construye variables centradas y cuotas observadas para un modelo dado."""
    df = panel.copy()
    if model == "M-OP":
        df["C"] = df["C_op"]
        df["s_L_obs"] = df["personal"] / df["C"]
        df["s_E_obs"] = df["energia"] / df["C"]
        df["s_O_obs"] = (df["aprov"] + df["serv_ext"] + df["amort"]) / df["C"]
    else:
        df["C"] = df["C_total"]
        df["s_L_obs"] = df["personal"] / df["C"]
        df["s_E_obs"] = df["energia"] / df["C"]
        df["s_O_obs"] = (df["aprov"] + df["serv_ext"] + df["amort"]
                         + df["renting"] + df["canon"]) / df["C"]
    df["lnC"]  = np.log(df["C"] / df["w_other"])
    df["lnWL"] = np.log(df["w_labor"] / df["w_other"])
    df["lnWE"] = np.log(df["w_energy"] / df["w_other"])
    df["lnY1"] = np.log(df["pax"])
    df["lnY2"] = np.log(df["coches_km"])
    df["t"]    = df["year"] - 2024
    centers = {k: df[k].mean() for k in ["lnC", "lnWL", "lnWE", "lnY1", "lnY2"]}
    df["wL"] = df["lnWL"] - centers["lnWL"]
    df["wE"] = df["lnWE"] - centers["lnWE"]
    df["y1"] = df["lnY1"] - centers["lnY1"]
    df["y2"] = df["lnY2"] - centers["lnY2"]
    df["lnC_c"] = df["lnC"] - centers["lnC"]
    return df, centers


def calibrate(panel, model, priors=None):
    """Calibra el translog. `priors` sobreescribe defaults."""
    df, centers = build_design(panel, model)
    pr = dict(
        sigma_LE=SIGMA_LE, sigma_LO=SIGMA_LO, sigma_EO=SIGMA_EO,
        delta_11=DELTA_11, delta_22=DELTA_22, delta_12=DELTA_12,
        rho_LY1=RHO_LY1, rho_LY2=RHO_LY2,
        rho_EY1=RHO_EY1, rho_EY2=RHO_EY2,
        beta1=BETA_PRIORS[model]["beta1"],
        beta2=BETA_PRIORS[model]["beta2"],
        lam=BETA_PRIORS[model]["lam"],
    )
    if priors:
        pr.update(priors)

    sL = df["personal"].sum() / df["C"].sum()
    sE = df["energia"].sum() / df["C"].sum()
    if model == "M-OP":
        sO = (df["aprov"] + df["serv_ext"] + df["amort"]).sum() / df["C"].sum()
    else:
        sO = ((df["aprov"] + df["serv_ext"] + df["amort"]
               + df["renting"] + df["canon"]).sum() / df["C"].sum())

    gamma_LE = sL * sE * (pr["sigma_LE"] - 1)
    gamma_LO = sL * sO * (pr["sigma_LO"] - 1)
    gamma_EO = sE * sO * (pr["sigma_EO"] - 1)
    gamma_LL = -(gamma_LE + gamma_LO)
    gamma_EE = -(gamma_LE + gamma_EO)
    gamma_OO = -(gamma_LO + gamma_EO)
    rho_OY1 = -(pr["rho_LY1"] + pr["rho_EY1"])
    rho_OY2 = -(pr["rho_LY2"] + pr["rho_EY2"])

    params = dict(
        model=model,
        alpha_L=sL, alpha_E=sE, alpha_O=sO,
        beta1=pr["beta1"], beta2=pr["beta2"],
        gamma_LL=gamma_LL, gamma_EE=gamma_EE, gamma_OO=gamma_OO,
        gamma_LE=gamma_LE, gamma_LO=gamma_LO, gamma_EO=gamma_EO,
        delta_11=pr["delta_11"], delta_22=pr["delta_22"], delta_12=pr["delta_12"],
        rho_LY1=pr["rho_LY1"], rho_LY2=pr["rho_LY2"],
        rho_EY1=pr["rho_EY1"], rho_EY2=pr["rho_EY2"],
        rho_OY1=rho_OY1, rho_OY2=rho_OY2,
        lam=pr["lam"],
        sigma_LE=pr["sigma_LE"], sigma_LO=pr["sigma_LO"], sigma_EO=pr["sigma_EO"],
    )
    return df, centers, params


def predict_lnC_centered(df, p):
    wL, wE, y1, y2, t = df.wL, df.wE, df.y1, df.y2, df.t
    return (
        p["alpha_L"]*wL + p["alpha_E"]*wE
        + p["beta1"]*y1 + p["beta2"]*y2
        + 0.5*p["gamma_LL"]*wL**2 + 0.5*p["gamma_EE"]*wE**2
        + p["gamma_LE"]*wL*wE
        + 0.5*p["delta_11"]*y1**2 + 0.5*p["delta_22"]*y2**2
        + p["delta_12"]*y1*y2
        + p["rho_LY1"]*wL*y1 + p["rho_LY2"]*wL*y2
        + p["rho_EY1"]*wE*y1 + p["rho_EY2"]*wE*y2
        + p["lam"]*t
    )


def compute_shares_pred(df, p):
    wL, wE, y1, y2 = df.wL, df.wE, df.y1, df.y2
    s_L = (p["alpha_L"] + p["gamma_LL"]*wL + p["gamma_LE"]*wE
           + p["rho_LY1"]*y1 + p["rho_LY2"]*y2)
    s_E = (p["alpha_E"] + p["gamma_LE"]*wL + p["gamma_EE"]*wE
           + p["rho_EY1"]*y1 + p["rho_EY2"]*y2)
    return s_L, s_E, 1 - s_L - s_E


def compute_elasticities(df, p):
    wL, wE, y1, y2 = df.wL, df.wE, df.y1, df.y2
    eps_Y1 = (p["beta1"] + p["delta_11"]*y1 + p["delta_12"]*y2
              + p["rho_LY1"]*wL + p["rho_EY1"]*wE)
    eps_Y2 = (p["beta2"] + p["delta_22"]*y2 + p["delta_12"]*y1
              + p["rho_LY2"]*wL + p["rho_EY2"]*wE)
    return eps_Y1, eps_Y2


def check_concavity(s_L, s_E, p):
    H = np.array([[p["gamma_LL"], p["gamma_LE"], p["gamma_LO"]],
                  [p["gamma_LE"], p["gamma_EE"], p["gamma_EO"]],
                  [p["gamma_LO"], p["gamma_EO"], p["gamma_OO"]]])
    out = []
    for i in range(len(s_L)):
        sL = s_L.iloc[i] if hasattr(s_L, "iloc") else s_L[i]
        sE = s_E.iloc[i] if hasattr(s_E, "iloc") else s_E[i]
        sO = 1.0 - sL - sE
        s = np.array([sL, sE, sO])
        M = H + np.outer(s, s) - np.diag(s)
        out.append(np.linalg.eigvalsh(M))
    return np.array(out)


def run_model(panel, model, priors=None):
    """Calibra + predice + diagnostica. Devuelve dict completo."""
    df, centers, p = calibrate(panel, model, priors=priors)
    lnC_pred_c = predict_lnC_centered(df, p)
    C_pred = np.exp(lnC_pred_c + centers["lnC"]) * df["w_other"]
    resid = df["C"] - C_pred

    sL_pred, sE_pred, sO_pred = compute_shares_pred(df, p)
    eps1, eps2 = compute_elasticities(df, p)
    rtd = 1.0 / eps1
    rts = 1.0 / (eps1 + eps2)
    mc_pax = eps1 * df["C"] / df["pax"]
    ac_pax = df["C"] / df["pax"]

    mon_prices = bool(((sL_pred > 0) & (sE_pred > 0) & (sO_pred > 0)).all())
    mon_outputs = bool(((eps1 > 0) & (eps2 > 0)).all())
    eig = check_concavity(sL_pred, sE_pred, p)
    eig_max = eig.max(axis=1)
    n_viol = int((eig_max > 0).sum())
    worst_idx = int(np.argmax(eig_max))
    worst_obs = (int(df["year"].iloc[worst_idx]), df["line"].iloc[worst_idx])
    conc_strict = bool((eig_max <= 0.0).all())
    conc_approx = bool((eig_max <= 1e-3).all())

    W = df["pax"].values
    rtd_w = float(np.sum(rtd.values * W) / W.sum())
    rts_w = float(np.sum(rts.values * W) / W.sum())
    ac_sys = float(df["C"].sum() / df["pax"].sum())
    mc_sys = float((eps1 * df["C"]).sum() / df["pax"].sum())
    mc_ac_sys = float((eps1 * df["C"]).sum() / df["C"].sum())
    rmse = float(np.sqrt((resid ** 2).mean()))
    mape = float((resid.abs() / df["C"]).mean())

    return dict(
        df=df, centers=centers, params=p,
        sL_pred=sL_pred, sE_pred=sE_pred, sO_pred=sO_pred,
        eps1=eps1, eps2=eps2, rtd=rtd, rts=rts,
        mc_pax=mc_pax, ac_pax=ac_pax, mc_ac=mc_pax/ac_pax,
        C_pred=C_pred, resid=resid,
        mon_prices=mon_prices, mon_outputs=mon_outputs,
        concavity_strict=conc_strict, concavity_approx=conc_approx,
        eig=eig, n_violations=n_viol, worst_obs=worst_obs,
        rtd_w=rtd_w, rts_w=rts_w,
        ac_sys=ac_sys, mc_sys=mc_sys, mc_ac_sys=mc_ac_sys,
        subsidio=1-mc_ac_sys, rmse=rmse, mape=mape,
    )


# ---------------------------------------------------------------------------
# 3. MAIN (ejecución como script: imprime y exporta tablas)
# ---------------------------------------------------------------------------
def _check(name, cond, detail=""):
    flag = "OK " if cond else "X  "
    print(f"  [{flag}] {name}  {detail}")


def _rows_by_obs(r, model_label):
    out = []
    for i, row in r["df"].reset_index(drop=True).iterrows():
        out.append(dict(
            model=model_label, year=int(row.year), line=row.line,
            pax=int(row.pax), coches_km=int(round(row.coches_km)),
            C=float(row.C),
            eps_Y1=float(r["eps1"].iloc[i]),
            eps_Y2=float(r["eps2"].iloc[i]),
            RTD=float(r["rtd"].iloc[i]),
            RTS=float(r["rts"].iloc[i]),
            AC=float(r["ac_pax"].iloc[i]),
            MC=float(r["mc_pax"].iloc[i]),
            MC_over_AC=float(r["mc_ac"].iloc[i]),
            s_L_obs=float(row.personal / row.C),
            s_L_pred=float(r["sL_pred"].iloc[i]),
            s_E_pred=float(r["sE_pred"].iloc[i]),
        ))
    return out


def main():
    panel = pd.read_csv(DATA_DIR / "panel_linea_año.csv")
    results = {m: run_model(panel, m) for m in ["M-OP", "M-FU"]}

    # --- Reporte en pantalla ---
    print("=" * 78)
    print("CALIBRACION TRANSLOG  -  RESULTADOS")
    print("=" * 78)
    for m, r in results.items():
        p = r["params"]
        print(f"\n--- {m} ---")
        print(f"  alpha_L = {p['alpha_L']:.4f}   "
              f"alpha_E = {p['alpha_E']:.4f}   "
              f"alpha_O = {p['alpha_O']:.4f}")
        print(f"  beta_1  = {p['beta1']:.3f}    "
              f"beta_2  = {p['beta2']:.3f}    "
              f"lambda  = {p['lam']:+.4f}")
        print(f"  gamma:  LL={p['gamma_LL']:+.5f} "
              f"EE={p['gamma_EE']:+.5f} OO={p['gamma_OO']:+.5f}")
        print(f"          LE={p['gamma_LE']:+.5f} "
              f"LO={p['gamma_LO']:+.5f} EO={p['gamma_EO']:+.5f}")
        print(f"  REGULARIDAD:")
        print(f"    Monotonicidad precios: {r['mon_prices']}  "
              f"(min s_L={r['sL_pred'].min():.4f}, "
              f"min s_E={r['sE_pred'].min():.4f}, "
              f"min s_O={r['sO_pred'].min():.4f})")
        print(f"    Monotonicidad outputs: {r['mon_outputs']}  "
              f"(min eps_Y1={r['eps1'].min():.4f}, "
              f"min eps_Y2={r['eps2'].min():.4f})")
        print(f"    Concavidad estricta:   {r['concavity_strict']}  "
              f"| aproximada (tol=1e-3): {r['concavity_approx']}")
        print(f"    Max eigenvalue M:      {r['eig'].max():+.5f}  "
              f"({r['n_violations']} obs > 0; peor: "
              f"{r['worst_obs'][0]} {r['worst_obs'][1]})")
        print(f"  POLITICA:")
        print(f"    RTD ponderado  = {r['rtd_w']:.3f}")
        print(f"    RTS ponderado  = {r['rts_w']:.3f}")
        print(f"    AC sistema     = {r['ac_sys']:.4f} EUR/pax")
        print(f"    MC sistema     = {r['mc_sys']:.4f} EUR/pax")
        print(f"    MC/AC sistema  = {r['mc_ac_sys']:.4f}  "
              f"(subsidio Boiteux = {r['subsidio']*100:.1f}%)")
        print(f"  AJUSTE:")
        print(f"    RMSE  = {r['rmse']/1e6:.3f} M EUR")
        print(f"    MAPE  = {r['mape']*100:.2f}%")

    op, fu = results["M-OP"], results["M-FU"]
    print("\n" + "=" * 78)
    print("CRITERIOS DE ACEPTACION")
    print("=" * 78)
    _check("RTD M-OP en [2.0, 2.4]",
           2.0 <= op["rtd_w"] <= 2.4, f"= {op['rtd_w']:.3f}")
    _check("RTD M-FU en [1.8, 2.1]",
           1.8 <= fu["rtd_w"] <= 2.1, f"= {fu['rtd_w']:.3f}")
    _check("MC/AC M-OP en [0.40, 0.55]",
           0.40 <= op["mc_ac_sys"] <= 0.55, f"= {op['mc_ac_sys']:.3f}")
    _check("MC/AC M-FU en [0.45, 0.60]",
           0.45 <= fu["mc_ac_sys"] <= 0.60, f"= {fu['mc_ac_sys']:.3f}")
    _check("Monotonicidad precios M-OP", op["mon_prices"])
    _check("Monotonicidad precios M-FU", fu["mon_prices"])
    _check("Monotonicidad outputs M-OP", op["mon_outputs"])
    _check("Monotonicidad outputs M-FU", fu["mon_outputs"])
    _check("Concavidad aproximada M-OP (tol 1e-3)", op["concavity_approx"],
           f"max eig={op['eig'].max():+.5f}, viols={op['n_violations']}/21")
    _check("Concavidad aproximada M-FU (tol 1e-3)", fu["concavity_approx"],
           f"max eig={fu['eig'].max():+.5f}, viols={fu['n_violations']}/21")
    _check("RMSE M-FU > RMSE M-OP (canon distorsiona)",
           fu["rmse"] > op["rmse"],
           f"M-OP={op['rmse']/1e6:.2f}M, M-FU={fu['rmse']/1e6:.2f}M, "
           f"ratio={fu['rmse']/op['rmse']:.2f}")

    # --- Exportar tablas ---
    desc_cols = ["C_total", "C_op", "personal", "energia", "aprov", "serv_ext",
                 "amort", "renting", "canon", "pax", "coches_km",
                 "w_labor", "w_energy"]
    t1 = panel[desc_cols].describe().T[["mean", "std", "min", "max"]]
    t1.to_csv(OUT_TABLES / "table1_descriptives.csv")
    print(f"\n[OK] table1_descriptives.csv  ({t1.shape})")

    param_keys = ["alpha_L", "alpha_E", "alpha_O", "beta1", "beta2", "lam",
                  "gamma_LL", "gamma_EE", "gamma_OO",
                  "gamma_LE", "gamma_LO", "gamma_EO",
                  "delta_11", "delta_22", "delta_12",
                  "rho_LY1", "rho_LY2", "rho_EY1", "rho_EY2",
                  "rho_OY1", "rho_OY2",
                  "sigma_LE", "sigma_LO", "sigma_EO"]
    t2 = pd.DataFrame({
        "param": param_keys,
        "M-OP": [op["params"][k] for k in param_keys],
        "M-FU": [fu["params"][k] for k in param_keys],
    })
    t2.to_csv(OUT_TABLES / "table2_parameters.csv", index=False)
    print(f"[OK] table2_parameters.csv     ({t2.shape})")

    t3 = pd.DataFrame(_rows_by_obs(op, "M-OP") + _rows_by_obs(fu, "M-FU"))
    t3.to_csv(OUT_TABLES / "table3_rtd_mc_by_line_year.csv", index=False)
    print(f"[OK] table3_rtd_mc_by_line_year.csv  ({t3.shape})")

    summary = {}
    for m in ["M-OP", "M-FU"]:
        r = results[m]
        summary[m] = {
            "rtd_weighted_pax":  r["rtd_w"],
            "rts_weighted_pax":  r["rts_w"],
            "ac_system_eur_per_pax":  r["ac_sys"],
            "mc_system_eur_per_pax":  r["mc_sys"],
            "mc_over_ac_system":  r["mc_ac_sys"],
            "boiteux_optimal_subsidy_pct":  r["subsidio"]*100,
            "rmse_eur":  r["rmse"],
            "mape_pct":  r["mape"]*100,
            "monotonicity_prices_pass":  r["mon_prices"],
            "monotonicity_outputs_pass":  r["mon_outputs"],
            "concavity_strict_pass":  r["concavity_strict"],
            "concavity_approx_pass":  r["concavity_approx"],
            "max_eigenvalue":  float(r["eig"].max()),
            "n_concavity_violations":  r["n_violations"],
            "worst_concavity_obs":  list(r["worst_obs"]),
            "params": {k: float(v) for k, v in r["params"].items()
                       if k != "model"},
        }
    summary["panel"] = {
        "n_obs": int(len(panel)), "n_lines": 7, "n_years": 3,
        "lines": INCLUDED, "years": [2023, 2024, 2025],
    }
    with open(OUT_TABLES / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] summary_metrics.json")
    print("\nCalibracion completada.")


if __name__ == "__main__":
    main()
