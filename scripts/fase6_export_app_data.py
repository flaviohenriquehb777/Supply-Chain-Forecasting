"""
FASE 6 — Exportação de dados para a aplicação web (GitHub Pages).

Para CADA um dos 27 produtos ativos, calcula e exporta:
  - histórico recente (últimos 180 dias) da venda diária na rede;
  - previsão SARIMAX para os 28 dias do holdout (trend='ct', exógena pct_lojas_promo);
  - realizado dos 28 dias do holdout;
  - os 12 erros de período do backtest de origens móveis (SARIMAX, trend=None),
    que a aplicação usa para calcular o estoque de segurança em QUALQUER nível de serviço.

A aplicação (front-end) roda a camada de DECISÃO no navegador:
    pedido = previsão + estoque de segurança (quantil empírico) - estoque no CD
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_REDE = os.path.join(BASE, "data", "raw", "demanda.csv")
SAIDA_DIR = os.path.join(BASE, "docs", "data")
SAIDA = os.path.join(SAIDA_DIR, "forecasts.json")

H = 28
N_ORIGENS = 12
ESPACAMENTO = 14
PRODUTOS_REMOVER = ["P0438", "P0051", "P0079"]
PRODUTO_FOCO = "P0131"
DIAS_HISTORICO = 180

os.makedirs(SAIDA_DIR, exist_ok=True)

# ── Carregamento e filtro ─────────────────────────────────────────────────────
df = pd.read_csv(CAMINHO_REDE, sep=",", parse_dates=["data"], dayfirst=False)
df = df[~df["produto"].isin(PRODUTOS_REMOVER)].copy().reset_index(drop=True)

produtos = sorted(df["produto"].unique())
data_maxima = df["data"].max()
data_corte = data_maxima - pd.Timedelta(days=H)


def ajustar_sarimax(y, exog=None, trend=None):
    """Ajusta o SARIMAX (2,1,2)(1,1,1,7) com o otimizador powell."""
    modelo = SARIMAX(
        y, exog=exog, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
        trend=trend, enforce_stationarity=False, enforce_invertibility=False, freq="D",
    )
    return modelo.fit(disp=False, method="powell", maxiter=200)


print("=" * 78)
print("EXPORTACAO DE DADOS PARA A APLICACAO WEB")
print("=" * 78)
print(f"  Produtos ativos : {len(produtos)}")
print(f"  Corte           : {data_corte:%d/%m/%Y}   |  Holdout: {data_corte + pd.Timedelta(days=1):%d/%m/%Y} a {data_maxima:%d/%m/%Y}")
print(f"  Origens backtest: {N_ORIGENS} (espacamento {ESPACAMENTO} dias)")
print()

saida_produtos = {}
t_inicio = time.time()

for i, produto in enumerate(produtos, start=1):
    t0 = time.time()
    serie = (
        df[df["produto"] == produto]
        .sort_values("data").set_index("data").asfreq("D")
    )
    # Higienização do índice diário. Único produto afetado: P0198, que possui
    # 8 dias sem registro no CSV (06 a 13/01/2017) e 9 preços ausentes.
    # Dias sem registro recebem venda e promoção iguais a zero; o preço é
    # propagado do dia mais próximo (ffill/bfill) para não contaminar a média.
    serie["unidades"] = serie["unidades"].fillna(0.0)
    serie["pct_lojas_promo"] = serie["pct_lojas_promo"].fillna(0.0)
    serie["preco_medio"] = serie["preco_medio"].ffill().bfill()
    y = serie["unidades"].astype(float)
    promo = serie["pct_lojas_promo"].astype(float)

    y_treino = y.loc[:data_corte]
    promo_treino = promo.loc[:data_corte]
    y_teste = y.loc[data_corte + pd.Timedelta(days=1):]
    promo_teste = promo.loc[data_corte + pd.Timedelta(days=1):]

    # 1) Previsão no cenário da Fase 4/6 (trend='ct')
    ajuste_final = ajustar_sarimax(y_treino, exog=promo_treino, trend="ct")
    previsao = np.clip(ajuste_final.forecast(steps=H, exog=promo_teste).values, 0, None)

    # 2) Backtest de origens móveis (trend=None) -> erros de período
    ultima_origem = data_maxima - pd.Timedelta(days=H)
    origens = pd.date_range(end=ultima_origem, periods=N_ORIGENS, freq=f"{ESPACAMENTO}D")
    erros = []
    for origem in origens:
        y_tr = y.loc[:origem]
        promo_tr = promo.loc[:origem]
        ini = origem + pd.Timedelta(days=1)
        fim = origem + pd.Timedelta(days=H)
        y_te = y.loc[ini:fim].values
        promo_te = promo.loc[ini:fim]
        prev = np.clip(ajustar_sarimax(y_tr, exog=promo_tr, trend=None)
                       .forecast(steps=H, exog=promo_te).values, 0, None)
        erros.append(float(y_te.sum() - prev.sum()))

    # Histórico termina na data de corte: não se sobrepõe ao holdout, o que
    # evita datas duplicadas no eixo do gráfico da aplicação.
    hist = y.loc[:data_corte].iloc[-DIAS_HISTORICO:]
    desempenho_periodo = {
        "real_total": float(y_teste.sum()),
        "previsto_total": float(previsao.sum()),
    }

    saida_produtos[produto] = {
        "produto": produto,
        "preco_medio": round(float(serie["preco_medio"].mean()), 4),
        "preco_medio_ponderado": round(
            float((serie["preco_medio"] * serie["unidades"]).sum() / serie["unidades"].sum()), 4),
        "historico": {
            "datas": [d.strftime("%Y-%m-%d") for d in hist.index],
            "unidades": [round(float(v), 2) for v in hist.values],
        },
        "previsao": {
            "datas": [d.strftime("%Y-%m-%d") for d in y_teste.index],
            "unidades": [round(float(v), 2) for v in previsao],
        },
        "promo_teste": [round(float(v), 4) for v in promo_teste.values],
        "real_teste": [round(float(v), 2) for v in y_teste.values],
        "erros_backtest": [round(e, 2) for e in erros],
        "totais_teste": desempenho_periodo,
    }
    print(f"  [{i:>2}/{len(produtos)}] {produto}  "
          f"prev={previsao.sum():>8.0f}  real={y_teste.sum():>8.0f}  "
          f"({time.time() - t0:.1f}s)")

# ── Estrutura final ───────────────────────────────────────────────────────────
saida = {
    "meta": {
        "gerado_em": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "produto_foco": PRODUTO_FOCO,
        "horizonte_dias": H,
        "corte": data_corte.strftime("%Y-%m-%d"),
        "inicio_teste": (data_corte + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "fim_teste": data_maxima.strftime("%Y-%m-%d"),
        "n_origens_backtest": N_ORIGENS,
        "custo_faltar_pct": 0.30,
        "custo_sobrar_pct": 0.02,
        "nivel_servico_otimo": 0.9375,
        "produtos_removidos": PRODUTOS_REMOVER,
    },
    "produtos": saida_produtos,
}

with open(SAIDA, "w", encoding="utf-8") as f:
    json.dump(saida, f, ensure_ascii=False, indent=2)

# Espelho em JavaScript: permite abrir o docs/index.html diretamente pelo
# sistema de arquivos (file://), sem servidor local e sem bloqueio de CORS.
SAIDA_JS = os.path.join(SAIDA_DIR, "forecasts.js")
with open(SAIDA_JS, "w", encoding="utf-8") as f:
    f.write("/* Gerado automaticamente por scripts/fase6_export_app_data.py */\n")
    f.write("window.DADOS_FORECAST = ")
    json.dump(saida, f, ensure_ascii=False, separators=(",", ":"))
    f.write(";\n")

print()
print(f"Arquivo gerado: {SAIDA}  ({os.path.getsize(SAIDA) / 1024:.1f} KB)")
print(f"Arquivo gerado: {SAIDA_JS}  ({os.path.getsize(SAIDA_JS) / 1024:.1f} KB)")
print(f"Tempo total: {(time.time() - t_inicio) / 60:.1f} min")
