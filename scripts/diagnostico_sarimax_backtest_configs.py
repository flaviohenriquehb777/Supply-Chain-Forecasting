"""
FASE 5 — Comparativo de configurações do SARIMAX no backtest de 12 origens.
Objetivo: encontrar a configuração que mais se aproxima do resultado do professor.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import serie_p, PRODUTO_FOCO, SEED, H

warnings.filterwarnings("ignore")
np.random.seed(SEED)

N_ORIGENS = 12
ESPACAMENTO = 14
H_FCST = 28

serie = serie_p["unidades"].astype(float)
exog_cont = serie_p["pct_lojas_promo"].astype(float)
exog_bin = (exog_cont > 0.20).astype(float)

ultima_data = serie.index.max()
ultima_origem = ultima_data - pd.Timedelta(days=H_FCST)
origens = pd.date_range(end=ultima_origem, periods=N_ORIGENS, freq=f"{ESPACAMENTO}D")

from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDEM = (2, 1, 2)
ORDEM_SAZONAL = (1, 1, 1, 7)

configs = [
    ("ct+powell+cont",      "ct", "powell", "cont"),
    ("ct+powell+bin",       "ct", "powell", "bin"),
    ("ct+lbfgs+cont",       "ct", "lbfgs",  "cont"),
    ("ct+bfgs+cont",        "ct", "bfgs",   "cont"),
    ("c+powell+cont",       "c",  "powell", "cont"),
    ("none+powell+cont",    None, "powell", "cont"),
    ("ct+nm+cont",          "ct", "nm",     "cont"),
]

print(f"{'config':<22} | {'WMAPE médio':>10} | {'WMAPE mediana':>13} | {'std':>7} | {'viés global':>11} | {'erro acum':>10} | tempo")
print("-" * 100)

for nome, trend, method, exog_tipo in configs:
    t0 = time.time()
    wmapes = []
    erros = []
    reais = []
    prevs = []
    falhou = False
    for origem in origens:
        y_tr = serie.loc[:origem]
        y_te = serie.loc[origem + pd.Timedelta(days=1):origem + pd.Timedelta(days=H_FCST)]
        ex_tr = exog_cont.loc[:origem] if exog_tipo == "cont" else exog_bin.loc[:origem]
        ex_te = exog_cont.loc[origem + pd.Timedelta(days=1):origem + pd.Timedelta(days=H_FCST)] if exog_tipo == "cont" else exog_bin.loc[origem + pd.Timedelta(days=1):origem + pd.Timedelta(days=H_FCST)]
        try:
            kwargs = dict(
                endog=y_tr, exog=ex_tr, order=ORDEM, seasonal_order=ORDEM_SAZONAL,
                enforce_stationarity=False, enforce_invertibility=False, freq="D",
            )
            if trend is not None:
                kwargs["trend"] = trend
            modelo = SARIMAX(**kwargs)
            ajuste = modelo.fit(disp=False, method=method, maxiter=200)
            prev = np.clip(ajuste.forecast(steps=H_FCST, exog=ex_te).values, 0, None)
            real = y_te.values
            wmapes.append(np.sum(np.abs(prev - real)) / real.sum() * 100)
            erros.append(real.sum() - prev.sum())
            reais.append(real.sum())
            prevs.append(prev.sum())
        except Exception as e:
            falhou = True
            break
    if falhou:
        print(f"  {nome:<22} | FALHOU")
        continue
    w_medio = np.mean(wmapes)
    w_med = np.median(wmapes)
    w_std = np.std(wmapes, ddof=1)
    erro_acum = np.sum(erros)
    viés_global = (np.sum(prevs) - np.sum(reais)) / np.sum(reais) * 100
    print(f"  {nome:<22} | {w_medio:>9.2f}% | {w_med:>12.2f}% | {w_std:>6.2f}% | {viés_global:>+10.2f}% | {erro_acum:>+10.0f} | {time.time()-t0:.1f}s")

print()
print("  Referência do professor: SARIMAX WMAPE médio 15.44% | mediana 11.99% | std 13.28% | viés +6.58% | erro acum -7311")
