import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import (
    treino_p, teste_p, real, indice_teste, H, PRODUTO_FOCO,
    placar, previsoes, data_corte, data_maxima,
)
import fase4_modelo1_naive_sazonal as m1
import fase4_modelo2_media_movel_28d as m2
import fase4_modelo5_sarimax_promocao as m5
from fase4_infra_avaliacao import avaliar
if "Naive Sazonal (últimos 7d)" not in placar:
    avaliar("Naive Sazonal (últimos 7d)", m1.previsao_naive_sazonal)
if "Média Móvel 28d (baseline comprador)" not in placar:
    avaliar("Média Móvel 28d (baseline comprador)", m2.previsao_mm28)
if "SARIMAX (2,1,2)(1,1,1,7) + promo" not in placar:
    avaliar("SARIMAX (2,1,2)(1,1,1,7) + promo", m5.previsao_sarimax)

SEED = 42
np.random.seed(SEED)

serie_treino = treino_p.sort_index()["unidades"].astype(float)

print("Gráfico consolidado: últimos 45 dias de treino + realizado do teste")
print("  + 3 previsões: MM28, Naive Sazonal 7d, SARIMAX")
print("=" * 70)

PASTA_FIGURAS = r"reports/figures"
os.makedirs(PASTA_FIGURAS, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.08)

N_DIAS_TREINO_VISIVEL = 45

fig, ax = plt.subplots(figsize=(16, 7.5))

ini_zoom = serie_treino.index.max() - pd.Timedelta(days=N_DIAS_TREINO_VISIVEL - 1)
zoom_treino = serie_treino.loc[ini_zoom:]

# Treino (ultimos 45 dias)
ax.plot(zoom_treino.index, zoom_treino.values, color="#1D4ED8", linewidth=1.1,
        alpha=0.85, label=f"Real — treino (últimos {N_DIAS_TREINO_VISIVEL} dias)", zorder=2)

# Teste realizado (destacado)
ax.plot(indice_teste, real, color="#0F172A", marker="o", markersize=6,
        linewidth=2.6, alpha=0.98, label="Real — teste (28 dias)", zorder=7)

# MM28
ax.plot(indice_teste, previsoes["Média Móvel 28d (baseline comprador)"].values,
        color="#D97706", linewidth=2.4, linestyle="--", marker="s", markersize=5,
        alpha=0.95,
        label=f"Média Móvel 28d (WMAPE {placar['Média Móvel 28d (baseline comprador)']['WMAPE_%']:.2f}% · "
              f"Viés {placar['Média Móvel 28d (baseline comprador)']['Viés_%']:+.2f}%)", zorder=4)

# Naive 7d
ax.plot(indice_teste, previsoes["Naive Sazonal (últimos 7d)"].values,
        color="#DC2626", linewidth=2.2, marker="^", markersize=5.5,
        alpha=0.92,
        label=f"Naive Sazonal 7d (WMAPE {placar['Naive Sazonal (últimos 7d)']['WMAPE_%']:.2f}% · "
              f"Viés {placar['Naive Sazonal (últimos 7d)']['Viés_%']:+.2f}%)", zorder=5)

# SARIMAX
ax.plot(indice_teste, previsoes["SARIMAX (2,1,2)(1,1,1,7) + promo"].values,
        color="#059669", linewidth=2.4, marker="D", markersize=5,
        alpha=0.95,
        label=f"SARIMAX + promo (WMAPE {placar['SARIMAX (2,1,2)(1,1,1,7) + promo']['WMAPE_%']:.2f}% · "
              f"Viés {placar['SARIMAX (2,1,2)(1,1,1,7) + promo']['Viés_%']:+.2f}%)", zorder=6)

ax.axvline(x=data_corte, color="#0F172A", linewidth=1.8, linestyle="--", alpha=0.8,
           label=f"Corte temporal ({data_corte.strftime('%d/%m/%Y')})", zorder=8)
ax.axvspan(data_corte + pd.Timedelta(days=1), data_maxima, color="#FEE2E2",
           alpha=0.30, zorder=1)

ax.set_title(f"{PRODUTO_FOCO} — Últimos {N_DIAS_TREINO_VISIVEL} dias de treino + 28 dias de teste"
             " · MM28 × Naive 7d × SARIMAX",
             fontsize=13.5, fontweight="bold", loc="left")
ax.set_ylabel("Unidades vendidas na rede / dia", fontsize=11)
ax.set_xlabel("Data", fontsize=11)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=9.8, ncol=1)

caminho_plot = os.path.join(PASTA_FIGURAS, f"{PRODUTO_FOCO}_consolidado_mm28_naive7_sarimax.png")
plt.tight_layout()
plt.savefig(caminho_plot, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()

print(f"  Plot consolidado salvo em: {caminho_plot}")
