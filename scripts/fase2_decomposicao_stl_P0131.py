import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from statsmodels.tsa.seasonal import STL

SEED = 42
np.random.seed(SEED)

CAMINHO_REDE = r"data/raw/demanda.csv"
PASTA_FIGURAS = r"reports/figures"
os.makedirs(PASTA_FIGURAS, exist_ok=True)

PRODUTO = "P0131"
PERIODO_SAZONAL = 7  # semanal

df = pd.read_csv(CAMINHO_REDE, sep=",", parse_dates=["data"], dayfirst=False)
serie = (
    df[df["produto"] == PRODUTO]
    .set_index("data")
    .sort_index()["unidades"]
    .asfreq("D")
)

# --- Decomposicao STL ---
stl = STL(
    serie,
    period=PERIODO_SAZONAL,
    seasonal=13,        # impar default; robust=True usa procedimento robusto
    trend=None,
    low_pass=None,
    seasonal_deg=1,
    trend_deg=1,
    low_pass_deg=1,
    robust=True          # Item pede versao robusta
)
res = stl.fit()

observado = res.observed
tendencia = res.trend
sazonal = res.seasonal
residuo = res.resid

print(f"Produto {PRODUTO} — Decomposição STL, período={PERIODO_SAZONAL}, robust=True")
print("=" * 65)
print()

# --- Item 2: Forca da tendencia e forca da sazonalidade (Hyndman) ---
# Formula Hyndman & Athanasopoulos (Forecasting: Principles and Practice):
# F_tend = max(0, 1 - Var(residuo) / Var(tendencia + residuo))
# F_saz  = max(0, 1 - Var(residuo) / Var(sazonal   + residuo))
var_residuo = np.var(residuo, ddof=1)
var_residuo_mais_tend = np.var(tendencia + residuo, ddof=1)
var_residuo_mais_saz = np.var(sazonal + residuo, ddof=1)

forca_tend = max(0.0, 1 - var_residuo / var_residuo_mais_tend)
forca_saz = max(0.0, 1 - var_residuo / var_residuo_mais_saz)

print("ITEM 2: Força da tendência e força da sazonalidade (Hyndman)")
print("-" * 65)
print(f"  Var(resíduo)                         : {var_residuo:>12.3f}")
print(f"  Var(tendência + resíduo)             : {var_residuo_mais_tend:>12.3f}")
print(f"  Var(sazonalidade + resíduo)          : {var_residuo_mais_saz:>12.3f}")
print()
print(f"  Força da TENDÊNCIA    (F_t) = 1 - Var(r)/Var(T+r) = {forca_tend:.6f}  "
      f"({forca_tend*100:.2f}%)")
print(f"  Força da SAZONALIDADE (F_s) = 1 - Var(r)/Var(S+r) = {forca_saz:.6f}  "
      f"({forca_saz*100:.2f}%)")
print()
print("Interpretação rápida (regra Hyndman):")
print(f"  F_t > 0.64 = tendência 'forte'      : {'SIM' if forca_tend > 0.64 else 'NAO'}")
print(f"  F_s > 0.64 = sazonalidade 'forte'   : {'SIM' if forca_saz > 0.64 else 'NAO'}")
print()

# --- Item 1: 4 paineis (observado, tendencia, sazonal, residuo) ---
sns.set_theme(style="whitegrid", font_scale=1.02)

fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

series_painel = [
    ("Observado (y)",  observado, "#2563EB"),
    ("Tendência (T)",  tendencia, "#DC2626"),
    ("Sazonalidade (S, período 7 dias)", sazonal, "#16A34A"),
    ("Resíduo (r = y − T − S)", residuo, "#7C3AED"),
]

for ax, (tit, vals, cor) in zip(axes, series_painel):
    ax.plot(vals.index, vals.values, color=cor, linewidth=1.0 if tit != "Observado (y)" else 0.75)
    ax.axhline(0, color="#94A3B8", linewidth=0.7, linestyle=":", alpha=0.8)
    ax.set_ylabel(tit, fontsize=10)
    ax.set_title(tit, loc="left", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

axes[0].set_title(f"Produto {PRODUTO} — Decomposição STL (período {PERIODO_SAZONAL} dias, robusta)",
                  loc="left", fontsize=12, fontweight="bold", pad=6)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
plt.tight_layout(h_pad=2.2)

caminho_plot1 = os.path.join(PASTA_FIGURAS, f"decomposicao_stl_{PRODUTO}_p{PERIODO_SAZONAL}.png")
plt.savefig(caminho_plot1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"ITEM 1: 4 painéis salvos em: {caminho_plot1}")
print()

# --- Item 3: Descrever o residuo (numeros auxiliares, sem especular) ---
print("ITEM 3: Descrição do painel de resíduo (medidas descritivas)")
print("-" * 65)
print(f"  Média do resíduo          : {residuo.mean():>+.4f}")
print(f"  Mediana do resíduo        : {np.median(residuo):>+.4f}")
print(f"  Desvio padrão (amostral)  : {residuo.std(ddof=1):>+.4f}")
print(f"  Min                       : {residuo.min():>+.4f}")
print(f"  Max                       : {residuo.max():>+.4f}")
q = np.quantile(residuo, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
print(f"  Q01 / Q05 / Q25 / Q75 / Q95 / Q99 : "
      f"{q[0]:>+8.2f} / {q[1]:>+8.2f} / {q[2]:>+8.2f} / {q[3]:>+8.2f} / {q[4]:>+8.2f} / {q[5]:>+8.2f}")
# Amplitude interquartil e faixa de 3x sigma
iqr = q[3] - q[2]
lim_inf = q[2] - 3 * iqr
lim_sup = q[3] + 3 * iqr
n_fora_iqr = ((residuo < lim_inf) | (residuo > lim_sup)).sum()
lim_sigma = 3 * residuo.std(ddof=1)
n_fora_3sigma = (residuo.abs() > lim_sigma).sum()
print(f"  IQR = {iqr:.2f};  Observados fora ±3·IQR : {n_fora_iqr} de {len(residuo)}  ({n_fora_iqr/len(residuo)*100:.3f}%)")
print(f"  σ   = {residuo.std(ddof=1):.2f}; Observados fora ±3·σ   : {n_fora_3sigma} de {len(residuo)}  ({n_fora_3sigma/len(residuo)*100:.3f}%)")

# Top 10 picos (5 positivos + 5 negativos)
idx_top_pos = np.argsort(-residuo.values)[:5]
idx_top_neg = np.argsort(residuo.values)[:5]
print()
print("  5 maiores picos positivos (resíduo >> 0):")
for i in idx_top_pos:
    d = residuo.index[i]
    v = residuo.iloc[i]
    y = observado.iloc[i]
    print(f"    {d.strftime('%d/%m/%Y')}  residuo={v:>+8.1f}   observado={y:>6.0f}")
print("  5 maiores picos negativos (resíduo << 0):")
for i in idx_top_neg:
    d = residuo.index[i]
    v = residuo.iloc[i]
    y = observado.iloc[i]
    print(f"    {d.strftime('%d/%m/%Y')}  residuo={v:>+8.1f}   observado={y:>6.0f}")
