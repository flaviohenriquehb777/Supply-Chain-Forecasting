"""
FASE 5 — Backtest com origens móveis (produto P0131).

12 origens espaçadas de 14 em 14 dias, terminando na última origem possível.
Em cada origem: treina até a origem, prevê 28 dias à frente, compara com realizado.
Modelos: Naive Sazonal, MM28, Holt-Winters, SARIMA, SARIMAX+promo.
Guarda por origem x modelo: WMAPE(28d), erro do total (real - previsto), total previsto.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import (
    serie_p, treino_p, teste_p, real, PRODUTO_FOCO, SEED, H,
)

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# ── Configuração do backtest ──────────────────────────────────────────────────
N_ORIGENS = 12
ESPACAMENTO = 14  # dias
H_FCST = 28       # horizonte

fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")
os.makedirs(fig_dir, exist_ok=True)

# ── Série completa do P0131 ───────────────────────────────────────────────────
serie = serie_p["unidades"].astype(float)
exog = serie_p["pct_lojas_promo"].astype(float)

# ── Definição das origens ─────────────────────────────────────────────────────
ultima_data = serie.index.max()
# última origem possível: treina até (ultima_data - H_FCST)
ultima_origem = ultima_data - pd.Timedelta(days=H_FCST)
origens = pd.date_range(
    end=ultima_origem,
    periods=N_ORIGENS,
    freq=f"{ESPACAMENTO}D",
)

print("=" * 80)
print(f"BACKTEST — Origens móveis  |  Produto: {PRODUTO_FOCO}")
print("=" * 80)
print(f"  Série completa     : {serie.index.min().date()} → {serie.index.max().date()}")
print(f"  Origens            : {N_ORIGENS}, espaçadas de {ESPACAMENTO} dias")
print(f"  Horizonte          : {H_FCST} dias por origem")
print(f"  Primeira origem    : {origens[0].date()}  (teste: {origens[0]+pd.Timedelta(days=1):%d/%m/%Y} → {(origens[0]+pd.Timedelta(days=H_FCST)).date()})")
print(f"  Última origem      : {origens[-1].date()}  (teste: {origens[-1]+pd.Timedelta(days=1):%d/%m/%Y} → {(origens[-1]+pd.Timedelta(days=H_FCST)).date()})")
print()

# ── Modelos ────────────────────────────────────────────────────────────────────
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def prever_naive_sazonal(y_treino, h):
    """Repete os últimos 7 dias do treino h/7 vezes."""
    ultimos_7 = y_treino.iloc[-7:].values
    n_rep = int(np.ceil(h / 7))
    return np.tile(ultimos_7, n_rep)[:h]

def prever_mm28(y_treino, h):
    """Média dos últimos 28 dias repetida h vezes."""
    media = y_treino.iloc[-28:].mean()
    return np.full(h, media)

def prever_holtwinters(y_treino, h):
    modelo = ExponentialSmoothing(
        y_treino,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
        freq="D",
    )
    ajuste = modelo.fit(optimized=True)
    return ajuste.forecast(steps=h).values

def prever_sarima(y_treino, h):
    modelo = SARIMAX(
        y_treino,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),
        trend=None,
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="D",
    )
    ajuste = modelo.fit(disp=False, method="powell", maxiter=200)
    return ajuste.forecast(steps=h).values

def prever_sarimax(y_treino, exog_treino, h, exog_teste):
    modelo = SARIMAX(
        y_treino,
        exog=exog_treino,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),
        trend=None,
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="D",
    )
    ajuste = modelo.fit(disp=False, method="powell", maxiter=200)
    return ajuste.forecast(steps=h, exog=exog_teste).values

# ── Execução do backtest ──────────────────────────────────────────────────────
modelos = ["Naive Sazonal", "MM28", "Holt-Winters", "SARIMA", "SARIMAX+promo"]
colunas = ["origem", "modelo", "WMAPE", "erro_total_real_minus_prev", "total_previsto", "total_real"]
resultados = []

print(f"{'origem':>12} | {'modelo':<18} | {'WMAPE':>7} | {'erro_total':>12} | {'prev':>8} | {'real':>8}")
print("-" * 80)

for origem in origens:
    # treino = tudo até e incluindo a origem
    y_treino = serie.loc[:origem]
    exog_treino = exog.loc[:origem]
    # teste = 28 dias após a origem
    inicio_teste = origem + pd.Timedelta(days=1)
    fim_teste = origem + pd.Timedelta(days=H_FCST)
    y_teste = serie.loc[inicio_teste:fim_teste]
    exog_teste = exog.loc[inicio_teste:fim_teste]

    real = y_teste.values
    total_real = real.sum()

    # Naive Sazonal
    prev = np.clip(prever_naive_sazonal(y_treino, H_FCST), 0, None)
    wmape = np.sum(np.abs(prev - real)) / total_real * 100
    erro = total_real - prev.sum()
    resultados.append([origem, "Naive Sazonal", wmape, erro, prev.sum(), total_real])
    print(f"{origem.date():>12} | {'Naive Sazonal':<18} | {wmape:>6.2f}% | {erro:>+12.1f} | {prev.sum():>8.0f} | {total_real:>8.0f}")

    # MM28
    prev = np.clip(prever_mm28(y_treino, H_FCST), 0, None)
    wmape = np.sum(np.abs(prev - real)) / total_real * 100
    erro = total_real - prev.sum()
    resultados.append([origem, "MM28", wmape, erro, prev.sum(), total_real])
    print(f"{origem.date():>12} | {'MM28':<18} | {wmape:>6.2f}% | {erro:>+12.1f} | {prev.sum():>8.0f} | {total_real:>8.0f}")

    # Holt-Winters
    prev = np.clip(prever_holtwinters(y_treino, H_FCST), 0, None)
    wmape = np.sum(np.abs(prev - real)) / total_real * 100
    erro = total_real - prev.sum()
    resultados.append([origem, "Holt-Winters", wmape, erro, prev.sum(), total_real])
    print(f"{origem.date():>12} | {'Holt-Winters':<18} | {wmape:>6.2f}% | {erro:>+12.1f} | {prev.sum():>8.0f} | {total_real:>8.0f}")

    # SARIMA
    prev = np.clip(prever_sarima(y_treino, H_FCST), 0, None)
    wmape = np.sum(np.abs(prev - real)) / total_real * 100
    erro = total_real - prev.sum()
    resultados.append([origem, "SARIMA", wmape, erro, prev.sum(), total_real])
    print(f"{origem.date():>12} | {'SARIMA':<18} | {wmape:>6.2f}% | {erro:>+12.1f} | {prev.sum():>8.0f} | {total_real:>8.0f}")

    # SARIMAX+promo
    prev = np.clip(prever_sarimax(y_treino, exog_treino, H_FCST, exog_teste), 0, None)
    wmape = np.sum(np.abs(prev - real)) / total_real * 100
    erro = total_real - prev.sum()
    resultados.append([origem, "SARIMAX+promo", wmape, erro, prev.sum(), total_real])
    print(f"{origem.date():>12} | {'SARIMAX+promo':<18} | {wmape:>6.2f}% | {erro:>+12.1f} | {prev.sum():>8.0f} | {total_real:>8.0f}")

    print("-" * 80)

# ── DataFrame de resultados ────────────────────────────────────────────────────
df_res = pd.DataFrame(resultados, columns=colunas)
df_res["origem"] = pd.to_datetime(df_res["origem"])

# ── Tabela agregada por modelo ─────────────────────────────────────────────────
print()
print("=" * 120)
print("RESUMO POR MODELO (média das 12 origens)")
print("=" * 120)

resumo = df_res.groupby("modelo").agg(
    WMAPE_medio=("WMAPE", "mean"),
    WMAPE_mediano=("WMAPE", "median"),
    WMAPE_std=("WMAPE", "std"),
    WMAPE_min=("WMAPE", "min"),
    WMAPE_max=("WMAPE", "max"),
    erro_total_medio=("erro_total_real_minus_prev", "mean"),
    erro_total_mediano=("erro_total_real_minus_prev", "median"),
    total_previsto=("total_previsto", "sum"),
    total_real=("total_real", "sum"),
).reindex(modelos)

# WMAPE médio descartando o pior resultado (1 única origem)
def wmape_sem_pior(grupo):
    pior_idx = grupo["WMAPE"].idxmax()
    return grupo.drop(pior_idx)["WMAPE"].mean()

resumo["WMAPE_medio_sem_pior"] = df_res.groupby("modelo").apply(wmape_sem_pior)

print()
print(f"{'modelo':<18} | {'WMAPE médio':>10} | {'WMAPE sem pior':>14} | {'Pior WMAPE':>11} | {'WMAPE mediana':>13} | {'WMAPE std':>10} | {'Viés global':>11} | {'Erro acum':>10}")
print("-" * 120)
for m in modelos:
    r = resumo.loc[m]
    vies_global = (r["total_previsto"] - r["total_real"]) / r["total_real"] * 100
    erro_acum = r["total_real"] - r["total_previsto"]
    print(f"{m:<18} | {r['WMAPE_medio']:>9.2f}% | {r['WMAPE_medio_sem_pior']:>13.2f}% | {r['WMAPE_max']:>10.2f}% | {r['WMAPE_mediano']:>12.2f}% | {r['WMAPE_std']:>9.2f}% | {vies_global:>+10.2f}% | {erro_acum:>+10.0f}")

# Salva resultados
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(out_dir, exist_ok=True)
df_res.to_csv(os.path.join(out_dir, "fase5_backtest_origens_P0131.csv"), index=False, encoding="utf-8-sig")
print()
print(f"Resultados salvos em: reports/fase5_backtest_origens_P0131.csv")

# ── Gráfico: WMAPE por origem e por modelo ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
cores = {"Naive Sazonal": "#1f77b4", "MM28": "#ff7f0e", "Holt-Winters": "#2ca02c",
         "SARIMA": "#d62728", "SARIMAX+promo": "#9467bd"}

for m in modelos:
    sub = df_res[df_res["modelo"] == m].sort_values("origem")
    ax.plot(sub["origem"], sub["WMAPE"], marker="o", label=m, color=cores[m], linewidth=2, markersize=5)

ax.set_title("WMAPE por origem móvel — Produto P0131 (12 origens, H=28 dias)", fontsize=14, fontweight="bold")
ax.set_xlabel("Origem (data do corte)", fontsize=12)
ax.set_ylabel("WMAPE (%)", fontsize=12)
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "fase5_backtest_WMAPE_por_origem_P0131.png"), dpi=150)
print(f"Gráfico salvo: reports/figures/fase5_backtest_WMAPE_por_origem_P0131.png")
