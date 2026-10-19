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
    wmape, vies, placar, previsoes, avaliar, data_corte, data_maxima,
)
from fase4_modelo1_naive_sazonal import (
    previsao_naive_sazonal, ultimos_7_dias_treino, n_repeticoes, N_DIAS_REPETICAO,
)
# Garante que a previsao do naive sazonal tambem foi registrada no placar
# (seria feito rodando modelo 1 antes do modelo 2 em um pipeline; aqui garantimos)
if "Naive Sazonal (últimos 7d)" not in placar:
    avaliar("Naive Sazonal (últimos 7d)", previsao_naive_sazonal)

SEED = 42
np.random.seed(SEED)

JANELA_MM = 28  # 4 semanas
DIAS_SEMANA_PT = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]

# ==============================================================================
# MODELO 2 — MÉDIA DAS ÚLTIMAS 4 SEMANAS (baseline do comprador atual)
# ==============================================================================
serie_treino_ord = treino_p.sort_index()["unidades"].astype(float)

media_ultimas_4semanas = serie_treino_ord.tail(JANELA_MM).mean()
mediana_ultimas_4semanas = serie_treino_ord.tail(JANELA_MM).median()
std_ultimas_4semanas = serie_treino_ord.tail(JANELA_MM).std()

dias_inicio_fim_treino = serie_treino_ord.tail(JANELA_MM).index[[0, -1]]
valores_ultimos_28 = serie_treino_ord.tail(JANELA_MM).values

print("MODELO 2: MÉDIA DAS ÚLTIMAS 4 SEMANAS (baseline do comprador atual)")
print("=" * 70)
print(f"  Janela usada no cálculo     : últimos {JANELA_MM} dias do treino (4 semanas)")
print(f"  Período da janela no treino : {dias_inicio_fim_treino[0].strftime('%d/%m/%Y')} → "
      f"{dias_inicio_fim_treino[1].strftime('%d/%m/%Y')}")
print(f"  Média simples da janela     : {media_ultimas_4semanas:,.2f} unidades/dia")
print(f"  Mediana da janela           : {mediana_ultimas_4semanas:,.2f} unidades/dia")
print(f"  Std da janela               : {std_ultimas_4semanas:,.2f} unidades/dia")
print(f"  Mínimo / Máximo da janela   : {valores_ultimos_28.min():.0f} / {valores_ultimos_28.max():.0f}")
print(f"  Período de previsão         : {indice_teste.min().strftime('%d/%m/%Y')} → "
      f"{indice_teste.max().strftime('%d/%m/%Y')} ({H} dias)")
print()

# Previsao: mesmo valor escalar repetido H vezes (sem sazonalidade semanal)
previsao_mm28 = np.repeat(media_ultimas_4semanas, H)

# Avaliacao
print("  Aplicando `avaliar()`:")
print("-" * 70)
w_mm, v_mm = avaliar("Média Móvel 28d (baseline comprador)", previsao_mm28)
print("-" * 70)
print()

# Comparativo com o Naive Sazonal
print("COMPARATIVO: Naive Sazonal 7d vs. Média Móvel 28d")
print("-" * 70)
tabela_placar = pd.DataFrame(placar).T
tabela_placar["ΔWMAPE_vs_NaiveSaz7d"] = tabela_placar["WMAPE_%"] - tabela_placar.loc[
    "Naive Sazonal (últimos 7d)", "WMAPE_%"
]
tabela_placar["ΔViés_vs_NaiveSaz7d"] = tabela_placar["Viés_%"] - tabela_placar.loc[
    "Naive Sazonal (últimos 7d)", "Viés_%"
]
print(tabela_placar.round(4).to_string())
print()

# ==============================================================================
# Diagnóstico do efeito da MM28 no padrão de dia da semana e na tendência
# ==============================================================================

# Parte A: Padrão de dia da semana na janela dos ultimos 28 dias de treino
print("DIAGNÓSTICO 1 — Padrão de dia da semana na janela de 28 dias do treino")
print("             (efeito da média 28d na sazonalidade semanal)")
print("-" * 70)

ultimos_28 = serie_treino_ord.tail(JANELA_MM).copy().to_frame("unidades")
ultimos_28["dia_semana_n"] = ultimos_28.index.weekday
ultimos_28["dia_semana"] = ultimos_28["dia_semana_n"].map(
    dict(zip(range(7), DIAS_SEMANA_PT))
)
media_dia_semana = ultimos_28.groupby(["dia_semana_n", "dia_semana"], observed=True)[
    "unidades"
].agg(["count", "mean", "std"]).reset_index().sort_values("dia_semana_n")

print("Média por dia da semana nas últimas 4 semanas do treino:")
print(media_dia_semana[["dia_semana", "count", "mean", "std"]].round(2).to_string(index=False))
print()

maior = media_dia_semana["mean"].max()
menor = media_dia_semana["mean"].min()
amplitude_relativa = (maior - menor) / menor * 100
media_geral_janela = media_ultimas_4semanas
print(f"  Melhor dia (maior média)    : {media_dia_semana.loc[media_dia_semana['mean'].idxmax(), 'dia_semana']}"
      f" = {maior:.1f} un/dia  ({(maior/media_geral_janela-1)*100:+.1f}% vs média)")
print(f"  Pior dia   (menor média)    : {media_dia_semana.loc[media_dia_semana['mean'].idxmin(), 'dia_semana']}"
      f" = {menor:.1f} un/dia  ({(menor/media_geral_janela-1)*100:+.1f}% vs média)")
print(f"  Amplitude relativa (pior→melhor): {amplitude_relativa:.1f}% "
      f"— diferença de ~{int(round(amplitude_relativa))}% entre o pior e o melhor dia.")
print(f"  Média móvel 28d IGNORA essa estrutura: prevê {media_ultimas_4semanas:.0f} un/dia IGUAL para TODOS os dias,")
print(f"  sem nenhum ajuste de dia da semana.")
print()

# Parte B: Efeito da MM28 na tendencia
print("DIAGNÓSTICO 2 — Efeito da média 28d na TENDÊNCIA")
print("             (MM28 do começo da série vs. MM28 do fim da série)")
print("-" * 70)

mm_expandida = serie_treino_ord.rolling(window=28, min_periods=28).mean()
# Pegue MM28 do meio do treino (há ~1 ano atrás) e compare com MM28 do final
ponto_ref_1ano = serie_treino_ord.index.min() + pd.Timedelta(days=365)
mm_1ano = float(mm_expandida.loc[ponto_ref_1ano]) if ponto_ref_1ano in mm_expandida.index else np.nan
mm_final = float(mm_expandida.iloc[-1])

print(f"  MM28 calculada no final da série (usada na previsão): {mm_final:,.2f} un/dia")
if not np.isnan(mm_1ano):
    print(f"  MM28 calculada ~1 ano após o início da série        : {mm_1ano:,.2f} un/dia")
    crescimento_abs = mm_final - mm_1ano
    crescimento_pct = (mm_final / mm_1ano - 1) * 100
    print(f"  Crescimento em ~1 ano (MM28): {crescimento_abs:+,.2f} un/dia  ({crescimento_pct:+.1f}%)")
    print(f"  ⇒ A MM28 CAPTURA a tendência de crescimento, porque ela olha o nível mais recente da série.")
    print(f"    Porém ela NÃO EXTRAPOLA: se a tendência continuar subindo, a MM28 SUBestima.")
print()

# Grafico do comparativo dos dois modelos
PASTA_FIGURAS = r"reports/figures"
os.makedirs(PASTA_FIGURAS, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)
fig, ax = plt.subplots(figsize=(15, 7.2))

# Zoom: apenas ultimos 90 dias de treino + 28 dias de teste (3 meses)
ini_zoom = serie_treino_ord.index.max() - pd.Timedelta(days=90)
zoom_treino = serie_treino_ord.loc[ini_zoom:]
ax.plot(zoom_treino.index, zoom_treino.values, color="#1D4ED8", linewidth=1.1,
        alpha=0.9, label="Real (treino — últimos 90 dias)", zorder=2)

ax.plot(indice_teste, real, color="#0F172A", marker="o", markersize=4,
        linewidth=2.0, alpha=0.95, label="Real (teste — 28 dias)", zorder=6)

ax.plot(indice_teste, previsoes["Naive Sazonal (últimos 7d)"].values, color="#DC2626",
        linewidth=2.1, marker="s", markersize=3.8, alpha=0.92,
        label=f"Naive Sazonal 7d (WMAPE {placar['Naive Sazonal (últimos 7d)']['WMAPE_%']:.2f}% · "
              f"Viés {placar['Naive Sazonal (últimos 7d)']['Viés_%']:+.2f}%)", zorder=4)
ax.plot(indice_teste, previsoes["Média Móvel 28d (baseline comprador)"].values,
        color="#D97706", linewidth=2.1, marker="^", markersize=4,
        alpha=0.92,
        label=f"MM28 (baseline comprador · WMAPE {w_mm:.2f}% · Viés {v_mm:+.2f}%)", zorder=5)

ax.axvline(x=data_corte, color="#0F172A", linestyle="--", linewidth=1.5, alpha=0.75,
           label=f"Corte ({data_corte.strftime('%d/%m/%Y')})")
ax.axvspan(data_corte + pd.Timedelta(days=1), data_maxima, color="#FEE2E2",
           alpha=0.32, zorder=1)

ax.set_title(f"{PRODUTO_FOCO} — Comparativo Naive Sazonal 7d vs. Média Móvel 28d"
             " (zoom 90 dias de treino + 28 de teste)",
             fontsize=12.6, fontweight="bold", loc="left")
ax.set_ylabel("Unidades / dia", fontsize=11)
ax.set_xlabel("Data", fontsize=11)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=9.4, ncol=1)

caminho_plot = os.path.join(PASTA_FIGURAS, f"{PRODUTO_FOCO}_comparativo_naive7d_vs_mm28d.png")
plt.tight_layout()
plt.savefig(caminho_plot, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# Salva o CSV atualizado com as previsoes dos dois modelos
PASTA_CSV_INTER = r"data/processed"
os.makedirs(PASTA_CSV_INTER, exist_ok=True)
prev_df = pd.DataFrame(previsoes).reset_index()
prev_df.rename(columns={"index": "data"}, inplace=True)
prev_df["real"] = real
caminho_csv = os.path.join(PASTA_CSV_INTER, f"previsoes_{PRODUTO_FOCO}_H{H}.csv")
prev_df.to_csv(caminho_csv, index=False)

print(f"  Plot comparativo salvo em: {caminho_plot}")
print(f"  CSV previsões atualizado em: {caminho_csv}")
