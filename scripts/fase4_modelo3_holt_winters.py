import pandas as pd
import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import (
    treino_p, teste_p, real, indice_teste, H, PRODUTO_FOCO,
    wmape, vies, placar, previsoes, avaliar, data_corte, data_maxima,
)
# Garante os baselines anteriores no placar (mesmos valores da execucao
# dos scripts de modelo 1 e 2, recalculados aqui por serem baratos).
import fase4_modelo1_naive_sazonal as m1
import fase4_modelo2_media_movel_28d as m2
if "Naive Sazonal (últimos 7d)" not in placar:
    avaliar("Naive Sazonal (últimos 7d)", m1.previsao_naive_sazonal)
if "Média Móvel 28d (baseline comprador)" not in placar:
    avaliar("Média Móvel 28d (baseline comprador)", m2.previsao_mm28)

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

# ==============================================================================
# MODELO 3 — HOLT-WINTERS (tendência aditiva, sazonalidade aditiva, período 7)
# ==============================================================================
serie_treino = treino_p.sort_index()["unidades"].astype(float)

print("MODELO 3: HOLT-WINTERS (additivo / additivo, período sazonal 7)")
print("=" * 70)
print(f"  Série usada no fit          : treino (n = {len(serie_treino)} dias)")
print(f"  Período treino              : {serie_treino.index.min().strftime('%d/%m/%Y')} → "
      f"{serie_treino.index.max().strftime('%d/%m/%Y')}")
print(f"  Parâmetros fixados          : trend='add', seasonal='add', seasonal_periods=7")
print(f"  Inicialização               : 'estimated' (estimada pelo otimizador)")
print()

modelo_hw = ExponentialSmoothing(
    serie_treino,
    trend="add",
    seasonal="add",
    seasonal_periods=7,
    initialization_method="estimated",
    freq="D",
)
ajuste = modelo_hw.fit(optimized=True)

alpha = ajuste.params["smoothing_level"]      # alfa (nível)
beta = ajuste.params["smoothing_trend"]       # beta (tendência)
gamma = ajuste.params["smoothing_seasonal"]   # gama (sazonalidade)

s0 = ajuste.params["initial_level"]
b0 = ajuste.params["initial_trend"]

print("  Parâmetros estimados pelo otimizador (MLE):")
print(f"    alfa  (nível, α)            = {alpha:.6f}")
print(f"    beta  (tendência, β)        = {beta:.6f}")
print(f"    gama  (sazonalidade, γ)     = {gamma:.6f}")
print(f"    nível inicial (l₀)          = {s0:.2f}")
print(f"    tendência inicial (b₀)      = {b0:.4f} unidades/dia")
print()

# Previsao de H passos a frente
previsao_hw = ajuste.forecast(H).values

print("  Previsão dos primeiros 14 dias do período de teste:")
print(f"  {'data':<12} {'dia_sem':<10} {'prev':>9}   {'real':>9}   {'erro':>9}")
print("  " + "-" * 60)
dias_semana_pt = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
for i in range(min(14, H)):
    d = indice_teste[i]
    print(f"  {d.strftime('%d/%m/%Y'):<12} {dias_semana_pt[d.weekday()]:<10} "
          f"{previsao_hw[i]:>9.1f}   {real[i]:>9.0f}   {previsao_hw[i] - real[i]:>+9.1f}")
print()

# Avaliacao
print("  Aplicando `avaliar()` (clip em zero):")
print("-" * 70)
w_hw, v_hw = avaliar("Holt-Winters (add/add, período 7)", previsao_hw)
print("-" * 70)
print()

# Placar atualizado
print("PLACAR — evolução até Holt-Winters:")
print("-" * 70)
tabela_placar = pd.DataFrame(placar).T
print(tabela_placar[["WMAPE_%", "Viés_%", "Demanda_prevista_total", "Demanda_real_total"]].round(3).to_string())
print()

# ==============================================================================
# INTERPRETAÇÃO DOS PARÂMETROS ESTIMADOS (valor por valor)
# ==============================================================================
print("INTERPRETAÇÃO DOS PARÂMETROS — o que cada valor diz sobre o P0131")
print("-" * 70)
print(f"  α (nível)       = {alpha:.4f}")
print(f"  β (tendência)   = {beta:.4f}")
print(f"  γ (sazonalidade)= {gamma:.4f}")
print()
print(f"  Leitura rápida:")
print(f"    - α = {alpha:.3f} {'pequeno' if alpha < 0.2 else ('médio' if alpha < 0.6 else 'alto')} → "
      f"nível local responde {'pouco' if alpha < 0.2 else ('moderadamente' if alpha < 0.6 else 'muito')} "
      f"a cada novo dia de venda.")
print(f"    - β = {beta:.4f} {'próximo de zero' if beta < 0.1 else ('médio' if beta < 0.5 else 'alto')} → "
      f"a inclinação da tendência é {'quase fixa no tempo' if beta < 0.1 else ('moderadamente ajustada' if beta < 0.5 else 'muito sensível ao novo dado')}.")
print(f"    - γ = {gamma:.3f} {'pequeno' if gamma < 0.2 else ('médio' if gamma < 0.6 else 'alto')} → "
      f"o padrão de dia da semana é {'muito estável' if gamma < 0.2 else ('moderadamente mutável' if gamma < 0.6 else 'muito mutável')} "
      f"de semana para semana.")
print()

# Grafico: real do treino (ultimos 60d) + previsoes dos 3 modelos no teste
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

# Holt-Winters
ax.plot(indice_teste, previsoes["Holt-Winters (add/add, período 7)"].values,
        color="#7C3AED", linewidth=2.2, marker="D", markersize=4, alpha=0.95,
        label=f"Holt-Winters (WMAPE {w_hw:.2f}% · Viés {v_hw:+.2f}%)", zorder=6)
# Naive 7d
ax.plot(indice_teste, previsoes["Naive Sazonal (últimos 7d)"].values,
        color="#DC2626", linewidth=1.8, alpha=0.8,
        label=f"Naive 7d ({placar['Naive Sazonal (últimos 7d)']['WMAPE_%']:.2f}% · "
              f"{placar['Naive Sazonal (últimos 7d)']['Viés_%']:+.2f}%)", zorder=4)
# MM28
ax.plot(indice_teste, previsoes["Média Móvel 28d (baseline comprador)"].values,
        color="#D97706", linewidth=1.8, alpha=0.8,
        label=f"MM28 ({placar['Média Móvel 28d (baseline comprador)']['WMAPE_%']:.2f}% · "
              f"{placar['Média Móvel 28d (baseline comprador)']['Viés_%']:+.2f}%)", zorder=5)

ax.axvline(x=data_corte, color="#0F172A", linestyle="--", linewidth=1.5, alpha=0.7,
           label=f"Corte ({data_corte.strftime('%d/%m/%Y')})")
ax.axvspan(data_corte + pd.Timedelta(days=1), data_maxima, color="#FEE2E2", alpha=0.3, zorder=1)

ax.set_title(f"{PRODUTO_FOCO} — Holt-Winters vs. baselines (treino 60d + teste 28d)",
             fontsize=13, fontweight="bold", loc="left")
ax.set_ylabel("Unidades / dia", fontsize=11)
ax.set_xlabel("Data", fontsize=11)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=9.4, ncol=1)

caminho_plot = os.path.join(PASTA_FIGURAS, f"{PRODUTO_FOCO}_holtwinters_vs_baselines.png")
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
