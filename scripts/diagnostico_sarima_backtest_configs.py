"""
FASE 5 — Teste rápido de configurações do SARIMA (sem exog) no backtest.
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import serie_p, SEED

warnings.filterwarnings("ignore")
np.random.seed(SEED)

N_ORIGENS = 12
ESPACAMENTO = 14
H_FCST = 28

serie = serie_p["unidades"].astype(float)
ultima_data = serie.index.max()
ultima_origem = ultima_data - pd.Timedelta(days=H_FCST)
origens = pd.date_range(end=ultima_origem, periods=N_ORIGENS, freq=f"{ESPACAMENTO}D")

from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDEM = (2, 1, 2)
ORDEM_SAZONAL = (1, 1, 1, 7)

configs = [
    ("none+powell", None, "powell"),
    ("c+powell",    "c",  "powell"),
    ("ct+powell",   "ct", "powell"),
    ("none+lbfgs",  None, "lbfgs"),
]

print(f"{'config':<16} | {'WMAPE médio':>10} | {'mediana':>8} | {'std':>7} | {'viés':>8} | {'erro acum':>10}")
print("-" * 75)

for nome, trend, method in configs:
    t0 = time.time()
    wmapes = []
    erros = []
    reais = []
    prevs = []
    for origem in origens:
        y_tr = serie.loc[:origem]
        y_te = serie.loc[origem + pd.Timedelta(days=1):origem + pd.Timedelta(days=H_FCST)]
        try:
            kwargs = dict(endog=y_tr, order=ORDEM, seasonal_order=ORDEM_SAZONAL,
                          enforce_stationarity=False, enforce_invertibility=False, freq="D")
            if trend is not None:
                kwargs["trend"] = trend
            modelo = SARIMAX(**kwargs)
            ajuste = modelo.fit(disp=False, method=method, maxiter=200)
            prev = np.clip(ajuste.forecast(steps=H_FCST).values, 0, None)
            real = y_te.values
            wmapes.append(np.sum(np.abs(prev - real)) / real.sum() * 100)
            erros.append(real.sum() - prev.sum())
            reais.append(real.sum())
            prevs.append(prev.sum())
        except Exception:
            pass
    w_medio = np.mean(wmapes)
    w_med = np.median(wmapes)
    w_std = np.std(wmapes, ddof=1)
    erro_acum = np.sum(erros)
    viés = (np.sum(prevs) - np.sum(reais)) / np.sum(reais) * 100
    print(f"  {nome:<16} | {w_medio:>9.2f}% | {w_med:>7.2f}% | {w_std:>6.2f}% | {viés:>+7.2f}% | {erro_acum:>+10.0f}")

print()
print("  Referência professor SARIMA: WMAPE médio 18.43% | mediana 13.78% | std 16.32% | viés +8.43% | erro acum -9366")
