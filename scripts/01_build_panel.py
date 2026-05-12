"""
01_build_panel.py - Construye el panel línea-año (T=21) para FMB-TMB 2023-2025.

Reglas operativas (confirmadas con el usuario):
1. Costes en € absolutos (input en miles de € → ×1000).
2. Anualización 2025: ×12/9 a todas las partidas EXCEPTO energía.
3. Energía 2025: MWh y €/kWh = valores 2024 (imputación documentada).
4. Imputación por línea-año:
   - Personal, energía, aprov, serv_ext (+ tributos), renting:
     cuota de coches-km = (trenes_hp × km_linea) / suma_lineas_incluidas
   - Amortización neta: cuota de km_linea (normalizada sobre líneas incluidas).
   - Canon Ifercat L9: 100% a L9/L10, split Nord/Sud por pasajeros.
5. Líneas incluidas: L1, L2, L3, L4, L5, L9/L10 Nord, L9/L10 Sud.
6. Excluidas: L11, Funicular (homogeneidad tecnológica).
7. Precios input: w_labor=personal/empleados, w_energy=energia/kWh, w_other=1.

Ajustes adicionales:
- share_ckm varía año a año (L1: 36→34 trenes hp entre 2023 y 2024).
- empleados NO se anualizan (plantilla a fin de año); solo el gasto de personal.
- Personal del presupuesto (no auditado completo) — diferencia ~0.5%.
- Renting imputado por coches-km (limitación: sin asignación pública de material rodante por línea).
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

for d in [DATA_DIR, OUTPUTS_DIR / "tables", OUTPUTS_DIR / "figures",
          ROOT / "paper"]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------
costos = pd.read_excel(INPUT_DIR / "costos_fmb.xlsx",
                       sheet_name="costos_presupuesto_fmb")
ops = pd.read_excel(INPUT_DIR / "operativos_fmb.xlsx",
                    sheet_name="outputs_y_denominadores")
pax_l = pd.read_excel(INPUT_DIR / "operativos_fmb.xlsx",
                      sheet_name="pasajeros_por_linea")
trenes = pd.read_excel(INPUT_DIR / "operativos_fmb.xlsx",
                       sheet_name="trenes_hora_punta")

# Renombrar primera col 'año' → 'year' (encoding cp1252 → mojibake en pandas)
costos = costos.rename(columns={costos.columns[0]: "year"})
ops = ops.rename(columns={ops.columns[0]: "year"})
pax_l = pax_l.rename(columns={pax_l.columns[0]: "year",
                              pax_l.columns[1]: "line"})

YEARS = [2023, 2024, 2025]
INCLUDED = ["L1", "L2", "L3", "L4", "L5", "L9/L10 Nord", "L9/L10 Sud"]
EXCLUDED = ["L11", "Funicular"]

KM_LINEA = {
    "L1": 22.4, "L2": 13.1, "L3": 18.4, "L4": 17.3, "L5": 18.9,
    "L9/L10 Nord": 18.7, "L9/L10 Sud": 17.2,
    "L11": 2.1, "Funicular": 0.7,
}
KM_RED_FMB = 125.4

# ---------------------------------------------------------------------------
# 2. PARSEO TRENES HORA-PUNTA (manejo de strings "6 / 4")
# ---------------------------------------------------------------------------
def parse_trenes(val):
    if pd.isna(val):
        return 0
    if isinstance(val, str) and "/" in val:
        return sum(int(p.strip()) for p in val.split("/"))
    return int(val)

def get_year_value(row, y):
    if y in row.index:
        return row[y]
    if str(y) in row.index:
        return row[str(y)]
    raise KeyError(f"Year {y} not found in {list(row.index)}")

trenes_dict = {}
for _, row in trenes.iterrows():
    name = str(row["linea_bloque"]).strip()
    if "Total" in name:
        continue
    if "Nord" in name:
        key = "L9/L10 Nord"
    elif "Sud" in name:
        key = "L9/L10 Sud"
    else:
        key = name
    trenes_dict[key] = {y: parse_trenes(get_year_value(row, y)) for y in YEARS}

# ---------------------------------------------------------------------------
# 3. COSTES FMB (€ absolutos), anualización 2025, imputación energía 2025
# ---------------------------------------------------------------------------
THOUSAND = 1000.0
costos_by_year = {}
for _, row in costos.iterrows():
    y = int(row["year"])
    factor = 12 / 9 if y == 2025 else 1.0
    aprov = row["aprovisionamientos"] * THOUSAND * factor
    energia = row["energia_y_carburantes"] * THOUSAND * factor
    personal = row["personal_operativo"] * THOUSAND * factor
    serv_ext = row["servicios_exteriores"] * THOUSAND * factor
    tributos = row["tributos_provisiones_otros"] * THOUSAND * factor
    amort = row["amortizacion_neta"] * THOUSAND * factor
    renting = row["renting_trenes"] * THOUSAND * factor
    canon = row["canon_ifercat_l9"] * THOUSAND * factor
    costos_by_year[y] = dict(
        aprov=aprov,
        energia=energia,
        personal=personal,
        serv_ext=serv_ext + tributos,   # regla 6: tributos pequeños → suma a serv_ext
        amort=amort,
        renting=renting,
        canon=canon,
    )

# Energía 2025: imputar con MWh y €/kWh idénticos a 2024
energy_2024_mwh = float(
    ops.set_index("year").loc[2024, "electricidad_metro"]
)
energy_2024_cost = costos_by_year[2024]["energia"]
price_eur_per_kwh_2024 = energy_2024_cost / (energy_2024_mwh * 1000)
costos_by_year[2025]["energia"] = (
    energy_2024_mwh * 1000 * price_eur_per_kwh_2024
)

# ---------------------------------------------------------------------------
# 4. OPERATIVOS SISTEMA-AÑO (empleados, coches-km, kWh)
# ---------------------------------------------------------------------------
ops_by_year = {}
for _, row in ops.iterrows():
    y = int(row["year"])
    coches_km_total = row["coches_km_utiles_metro_miles"] * THOUSAND
    empleados = int(row["empleados_metro"])    # NO anualizar (plantilla EOY)
    mwh = row["electricidad_metro"]
    if pd.isna(mwh) and y == 2025:
        mwh = energy_2024_mwh                  # imputar MWh 2025 = MWh 2024
    ops_by_year[y] = dict(
        coches_km_total=coches_km_total,
        empleados=empleados,
        mwh=float(mwh),
        kwh=float(mwh) * 1000,
    )

# ---------------------------------------------------------------------------
# 5. CLAVES DE IMPUTACIÓN POR LÍNEA-AÑO
# ---------------------------------------------------------------------------
trenkm = {ln: {y: trenes_dict[ln][y] * KM_LINEA[ln] for y in YEARS}
          for ln in INCLUDED + EXCLUDED}

tot_trenkm_incl = {y: sum(trenkm[ln][y] for ln in INCLUDED) for y in YEARS}
tot_trenkm_all = {y: sum(trenkm[ln][y] for ln in INCLUDED + EXCLUDED)
                  for y in YEARS}

# share_ckm normalizado al SISTEMA COMPLETO (suma sobre INCLUIDAS < 1 = cobertura)
share_ckm = {
    ln: {y: trenkm[ln][y] / tot_trenkm_all[y] for y in YEARS}
    for ln in INCLUDED
}
# share_km normalizado a la suma de km de línea de las 9 líneas (=128.7)
sum_km_all = sum(KM_LINEA[ln] for ln in INCLUDED + EXCLUDED)
share_km = {ln: KM_LINEA[ln] / sum_km_all for ln in INCLUDED}

# ---------------------------------------------------------------------------
# 6. PASAJEROS POR LÍNEA-AÑO
# ---------------------------------------------------------------------------
pax_dict = {ln: {} for ln in INCLUDED + EXCLUDED}
for _, row in pax_l.iterrows():
    y = int(row["year"])
    ln = str(row["line"]).strip()
    if ln in pax_dict:
        pax_dict[ln][y] = int(row["pasajeros_xlsx"])

# ---------------------------------------------------------------------------
# 7. PRECIOS DE INPUTS (sistema-año)
# ---------------------------------------------------------------------------
w_labor = {y: costos_by_year[y]["personal"] / ops_by_year[y]["empleados"]
           for y in YEARS}
w_energy = {y: costos_by_year[y]["energia"] / ops_by_year[y]["kwh"]
            for y in YEARS}
w_other = {y: 1.0 for y in YEARS}

# ---------------------------------------------------------------------------
# 8. CONSTRUCCIÓN DEL PANEL LÍNEA-AÑO
# ---------------------------------------------------------------------------
canon_l9_split = {}
for y in YEARS:
    pn = pax_dict["L9/L10 Nord"][y]
    ps = pax_dict["L9/L10 Sud"][y]
    canon_l9_split[y] = {"L9/L10 Nord": pn / (pn + ps),
                         "L9/L10 Sud": ps / (pn + ps)}

rows = []
for y in YEARS:
    C = costos_by_year[y]
    for ln in INCLUDED:
        s_ckm = share_ckm[ln][y]
        s_km = share_km[ln]
        personal = C["personal"] * s_ckm
        energia = C["energia"] * s_ckm
        aprov = C["aprov"] * s_ckm
        serv = C["serv_ext"] * s_ckm
        amort = C["amort"] * s_km
        renting = C["renting"] * s_ckm
        canon = (C["canon"] * canon_l9_split[y][ln]
                 if ln in ("L9/L10 Nord", "L9/L10 Sud") else 0.0)
        C_op = personal + energia + aprov + serv + amort
        C_total = C_op + renting + canon
        coches_km_line = ops_by_year[y]["coches_km_total"] * s_ckm
        rows.append({
            "year": y, "line": ln,
            "C_total": C_total, "C_op": C_op,
            "personal": personal, "energia": energia,
            "aprov": aprov, "serv_ext": serv,
            "amort": amort, "renting": renting, "canon": canon,
            "pax": pax_dict[ln][y], "coches_km": coches_km_line,
            "w_labor": w_labor[y], "w_energy": w_energy[y],
            "w_other": w_other[y],
            "share_ckm": s_ckm, "share_km": s_km,
        })

panel = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 9. VALIDACIONES
# ---------------------------------------------------------------------------
print("=" * 78)
print("PANEL LÍNEA-AÑO  —  VALIDACIONES")
print("=" * 78)
print(f"Dimensiones del panel: {panel.shape}  (esperado: 21 filas × 18 cols)")
assert len(panel) == 21, "El panel debe tener 7 líneas × 3 años = 21 filas"

# --- Cobertura contable por año ---
print("\nCobertura imputado / FMB total, por año:")
print(f"{'Año':<6}{'M-OP imp.':>16}{'M-OP FMB':>16}{'Cob.M-OP':>10}"
      f"{'M-FU imp.':>16}{'M-FU FMB':>16}{'Cob.M-FU':>10}")
TARGET = 0.976
warnings = []
for y in YEARS:
    C_op_fmb = sum(costos_by_year[y][p]
                   for p in ["personal", "energia", "aprov",
                             "serv_ext", "amort"])
    C_fu_fmb = (C_op_fmb + costos_by_year[y]["renting"]
                + costos_by_year[y]["canon"])
    sub = panel[panel.year == y]
    sum_op = sub["C_op"].sum()
    sum_fu = sub["C_total"].sum()
    cov_op = sum_op / C_op_fmb
    cov_fu = sum_fu / C_fu_fmb
    print(f"{y:<6}{sum_op/1e6:>15.1f}M{C_op_fmb/1e6:>15.1f}M"
          f"{cov_op*100:>9.2f}%{sum_fu/1e6:>15.1f}M"
          f"{C_fu_fmb/1e6:>15.1f}M{cov_fu*100:>9.2f}%")
    if abs(cov_fu - TARGET) > 0.02:
        warnings.append(
            f"  {y}: M-FU cobertura {cov_fu*100:.2f}% se desvía "
            f"{(cov_fu-TARGET)*100:+.2f}pp del target {TARGET*100:.1f}%"
        )

if warnings:
    print("\n[WARNINGS de cobertura]")
    for w in warnings:
        print(w)

# --- Matriz share_ckm 7×3 ---
print("\nMatriz share_ckm (% por línea-año):")
m = panel.pivot(index="line", columns="year",
                values="share_ckm").reindex(INCLUDED)
print((m * 100).round(3).to_string())
print(f"\nSumas por año (deben ser 1.0): "
      f"{ {y: round(m[y].sum(), 6) for y in YEARS} }")

# --- Precios de inputs por año ---
print("\nPrecios input por año:")
print(f"{'Año':<6}{'w_labor (€/empl.)':>22}{'w_energy (€/kWh)':>22}"
      f"{'w_other':>10}")
for y in YEARS:
    print(f"{y:<6}{w_labor[y]:>22,.2f}{w_energy[y]:>22.6f}"
          f"{w_other[y]:>10.2f}")

# --- Diagnóstico anomalías ---
print("\nDiagnóstico de valores anómalos:")
print(f"  NaN totales: {int(panel.isna().sum().sum())}")
num_cols = ["C_total", "C_op", "personal", "energia", "aprov",
            "serv_ext", "amort", "renting", "canon", "pax", "coches_km"]
n_neg = (panel[num_cols] < 0).sum().sum()
print(f"  Valores negativos: {int(n_neg)}")

print("  Outliers (|z|>3) por columna:")
any_out = False
for col in num_cols:
    s = panel[col]
    if s.std() > 0:
        z = (s - s.mean()) / s.std()
        n_out = int((z.abs() > 3).sum())
        if n_out > 0:
            any_out = True
            print(f"    {col}: {n_out} obs   "
                  f"max|z|={z.abs().max():.2f}")
if not any_out:
    print("    (ninguno)")

# --- Cobertura por proxy ---
print("\nCobertura proxy trenes_hp × km_linea:")
for y in YEARS:
    cov = tot_trenkm_incl[y] / tot_trenkm_all[y]
    excl = (trenkm["L11"][y] + trenkm["Funicular"][y]) / tot_trenkm_all[y]
    print(f"  {y}: {cov*100:.3f}% incluido (L11+Func excluidos: "
          f"{excl*100:.3f}%)")

# --- Exportación ---
out_csv = DATA_DIR / "panel_linea_año.csv"
panel.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\nPanel exportado a: {out_csv}")
print(f"  Columnas ({len(panel.columns)}): {list(panel.columns)}")
