import pandas as pd
import numpy as np
import os
import sys
import warnings
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import (
    treino_p, teste_p, real, H, wmape, vies,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

serie_treino = treino_p.sort_index()["unidades"].astype(float)

ORDEM = (2, 1, 2)
ORDEM_SAZONAL = (1, 1, 1, 7)

print("BUSCANDO a config do SARIMA do professor (10.07% / +1.12%)")
print("=" * 80)

for trend in [None, "c", "t", "ct"]:
    for method in ["lbfgs", "bfgs", "nm", "powell", "cg", "ncg", "basinhopping"]:
        t0 = time.time()
        try:
            kwargs = dict(
                endog=serie_treino,
                order=ORDEM,
                seasonal_order=ORDEM_SAZONAL,
                enforce_stationarity=False,
                enforce_invertibility=False,
                freq="D",
            )
            if trend is not None:
                kwargs["trend"] = trend
            modelo = SARIMAX(**kwargs)
            ajuste = modelo.fit(disp=False, method=method, maxiter=200)
            prev = ajuste.forecast(steps=H).values
            prev_clip = np.clip(prev, 0, None)
            w = wmape(real, prev_clip)
            v = vies(real, prev_clip)
            print(f"  trend={str(trend):<4} method={method:<14} | WMAPE {w:>7.3f}% | Viés {v:>+8.3f}% | AIC {ajuste.aic:>9.1f} | {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  trend={str(trend):<4} method={method:<14} | ERRO: {str(e)[:50]}")

print()
print("  Referência: WMAPE 10.07% | Viés +1.12%")
