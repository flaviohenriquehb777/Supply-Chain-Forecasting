import pandas as pd
import numpy as np
import os
import sys
import warnings
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import (
    treino_p, teste_p, real, indice_teste, H, PRODUTO_FOCO,
    wmape, vies, placar, previsoes, avaliar, data_corte, data_maxima,
)
import fase4_modelo1_naive_sazonal as m1
import fase4_modelo2_media_movel_28d as m2
import fase4_modelo3_holt_winters as m3
if "Naive Sazonal (últimos 7d)" not in placar:
    avaliar("Naive Sazonal (últimos 7d)", m1.previsao_naive_sazonal)
if "Média Móvel 28d (baseline comprador)" not in placar:
    avaliar("Média Móvel 28d (baseline comprador)", m2.previsao_mm28)

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

# ==============================================================================
# MODELO 4 — SARIMA (2,1,2)(1,1,1,7)
# ==============================================================================
serie_treino = treino_p.sort_index()["unidades"].astype(float)

ORDEM = (2, 1, 2)
ORDEM_SAZONAL = (1, 1, 1, 7)

print("MODELO 4: SARIMA")
print("=" * 70)
print(f"  Série usada no fit      : treino (n = {len(serie_treino)} dias)")
print(f"  Período treino          : {serie_treino.index.min().strftime('%d/%m/%Y')} → "
      f"{serie_treino.index.max().strftime('%d/%m/%Y')}")
print(f"  order                    : {ORDEM}")
print(f"  seasonal_order          : {ORDEM_SAZONAL}")
print(f"  Frequência              : Diária")
print()

t0 = time.time()
modelo_sarima = SARIMAX(
    serie_treino,
    order=ORDEM,
    seasonal_order=ORDEM_SAZONAL,
    enforce_stationarity=False,
    enforce_invertibility=False,
    freq="D",
)
ajuste = modelo_sarima.fit(disp=False)
t_fit = time.time() - t0

print(f"  Tempo de ajuste (fit)  : {t_fit:.2f}s")
print(f"  AIC                     : {ajuste.aic:.2f}")
print(f"  BIC                     : {ajuste.bic:.2f}")
print()

# Coeficientes estimados
print("  Coeficientes estimados:")
for nome, valor in ajuste.params.items():
    print(f"    {nome:<16} = {valor:>12.4f}")
print()

# Previsao de H passos a frente
t1 = time.time()
previsao_sarima = ajuste.forecast(steps=H).values
t_forecast = time.time() - t1

print(f"  Tempo de previsão      : {t_forecast:.3f}s")
print(f"  Mínimo da previsão    : {previsao_sarima.min():.1f}")
print(f"  Máximo da previsão    : {previsao_sarima.max():.1f}")
print()

# Imprime as primeiras 14 previsões
dias_semana_pt = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
print("  Primeiras 14 previsões:")
print(f"  {'data':<12} {'dia_sem':<10} {'prev':>9}   {'real':>9}   {'erro':>9}")
print("  " + "-" * 56)
for i in range(min(14, H)):
    d = indice_teste[i]
    print(f"  {d.strftime('%d/%m/%Y'):<12} {dias_semana_pt[d.weekday()]:<10} "
          f"{previsao_sarima[i]:>9.1f}   {real[i]:>9.0f}   {previsao_sarima[i] - real[i]:>+9.1f}")
print()

# Avaliacao
print("  Aplicando `avaliar()` (clip em zero):")
print("-" * 70)
w_sarima, v_sarima = avaliar("SARIMA (2,1,2)(1,1,1,7)", previsao_sarima)
print("-" * 70)
print()

# Placar completo
print("PLACAR — evolução até SARIMA:")
print("-" * 70)
tabela_placar = pd.DataFrame(placar).T
tabela_placar = tabela_placar.sort_values("WMAPE_%", ascending=True)
print(tabela_placar[["WMAPE_%", "Viés_%", "Demanda_prevista_total", "Demanda_real_total"]].round(3).to_string())
print()

# Diferença vs baseline do comprador (MM28)
w_mm28 = placar["Média Móvel 28d (baseline comprador)"]["WMAPE_%"]
v_mm28 = placar["Média Móvel 28d (baseline comprador)"]["Viés_%"]
print(f"  SARIMA vs. baseline do comprador (MM28):")
print(f"    ΔWMAPE = {w_sarima - w_mm28:+.3f} pp  ({'melhor' if w_sarima < w_mm28 else 'pior'}")
print(f"    ΔViés  = {v_sarima - v_mm28:+.3f} pp")
print()

# Grafico
PASTA_FIGURAS = r"reports/figures"
os.makedirs(PASTA_FIGURAS, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)
fig, ax = plt.subplots(figsize=(16, 7.2))

ini_zoom = serie_treino.index.max() - pd.Timedelta(days=60)
zoom_treino = serie_treino.loc[ini_zoom:]
ax.plot(zoom_treino.index, zoom_treino.values, color="#1D4ED8", linewidth=1.0,
        alpha=0.85, label="Real (treino — últimos 60 dias)", zorder=2)
ax.plot(indice_teste, real, color="#0F172A", marker="o", markersize=4.2,
        linewidth=2.1, alpha=0.95, label="Real (teste — 28 dias)", zorder=7)

ax.plot(indice_teste, previsoes["SARIMA (2,1,2)(1,1,1,7)"].values,
        color="#059669", linewidth=2.2, marker="D", markersize=4, alpha=0.95,
        label=f"SARIMA (WMAPE {w_sarima:.2f}% · Viés {v_sarima:+.2f}%)", zorder=6)
ax.plot(indice_teste, previsoes["Holt-Winters (add/add, período 7)"].values,
        color="#7C3AED", linewidth=1.8, alpha=0.8,
        label=f"Holt-Winters ({placar['Holt-Winters (add/add, período 7)']['WMAPE_%']:.2f}% · "
              f"{placar['Holt-Winters (add/add, período 7)']['Viés_%']:+.2f}%)", zorder=5)
ax.plot(indice_teste, previsoes["Naive Sazonal (últimos 7d)"].values,
        color="#DC2626", linewidth=1.5, alpha=0.7,
        label=f"Naive 7d ({placar['Naive Sazonal (últimos 7d)']['WMAPE_%']:.2f}% · "
              f"{placar['Naive Sazonal (últimos 7d)']['Viés_%']:+.2f}%)", zorder=4)

ax.axvline(x=data_corte, color="#0F172A", linestyle="--", linewidth=1.5, alpha=0.7,
           label=f"Corte ({data_corte.strftime('%d/%m/%Y')})")
ax.axvspan(data_corte + pd.Timedelta(days=1), data_maxima, color="#FEE2E2", alpha=0.3, zorder=1)

ax.set_title(f"{PRODUTO_FOCO} — SARIMA vs. Holt-Winters e Naive (treino 60d + teste 28d)",
             fontsize=13, fontweight="bold", loc="left")
ax.set_ylabel("Unidades / dia", fontsize=11)
ax.set_xlabel("Data", fontsize=11)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=9.4, ncol=1)

caminho_plot = os.path.join(PASTA_FIGURAS, f"{PRODUTO_FOCO}_sarima_vs_holt_vs_naive.png")
plt.tight_layout()
plt.savefig(caminho_plot, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Plot comparativo salvo em: {caminho_plot}")

# CSV atualizado
PASTA_CSV_INTER = r"data/processed"
os.makedirs(PASTA_CSV_INTER, exist_ok=True)
prev_df = pd.DataFrame(previsoes).reset_index().rename(columns={"index": "data"})
prev_df["real"] = real
prev_df.to_csv(os.path.join(PASTA_CSV_INTER, f"previsoes_{PRODUTO_FOCO}_H{H}.csv"), index=False)
