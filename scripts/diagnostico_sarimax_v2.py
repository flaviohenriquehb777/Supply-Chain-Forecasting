import pandas as pd
import numpy as np
import os
import sys
import warnings
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import (
    treino_p, teste_p, real, indice_teste, H, PRODUTO_FOCO,
    wmape, vies, avaliar,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

serie_treino = treino_p.sort_index()["unidades"].astype(float)
exog_cont_treino = treino_p.sort_index()["pct_lojas_promo"].astype(float)
exog_cont_teste = teste_p.sort_index()["pct_lojas_promo"].astype(float)

# Promo binaria: 1 se > 20% das lojas, 0 caso contrario (regra da Fase 2)
exog_bin_treino = (exog_cont_treino > 0.20).astype(float)
exog_bin_teste = (exog_cont_teste > 0.20).astype(float)

ORDEM = (2, 1, 2)
ORDEM_SAZONAL = (1, 1, 1, 7)

print("EXPERIMENTO 2 — refinando para bater o resultado do professor")
print("=" * 95)
print(f"  Referência do professor:")
print(f"    SARIMA  (2,1,2)(1,1,1,7) : WMAPE 10.07% | Viés +1.12%")
print(f"    SARIMAX + promoção        : WMAPE  8.45% | Viés +1.09% | coef_promo 65.34")
print()

configs = [
    # desc, exog_tipo, trend, method
    ("trend_ct + exog continuo (lbfgs)",    "cont", "ct", "lbfgs"),
    ("trend_ct + exog binario (lbfgs)",     "bin",  "ct", "lbfgs"),
    ("trend_c  + exog continuo (lbfgs)",    "cont", "c",  "lbfgs"),
    ("trend_ct + exog continuo (bfgs)",     "cont", "ct", "bfgs"),
    ("trend_ct + exog continuo (nm)",       "cont", "ct", "nm"),
    ("trend_ct + exog continuo (powell)",   "cont", "ct", "powell"),
    ("trend_ct + exog binario (bfgs)",      "bin",  "ct", "bfgs"),
    ("trend_c  + exog binario (lbfgs)",     "bin",  "c",  "lbfgs"),
    ("trend_none + exog continuo (bfgs)",   "cont", None, "bfgs"),
]

for desc, exog_tipo, trend, method in configs:
    exog_tr = exog_cont_treino if exog_tipo == "cont" else exog_bin_treino
    exog_te = exog_cont_teste if exog_tipo == "cont" else exog_bin_teste
    t0 = time.time()
    try:
        kwargs = dict(
            endog=serie_treino,
            exog=exog_tr,
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
        t_fit = time.time() - t0

        prev = ajuste.forecast(steps=H, exog=exog_te).values
        prev_clip = np.clip(prev, 0, None)
        w = wmape(real, prev_clip)
        v = vies(real, prev_clip)
        total_prev = prev_clip.sum()
        coef_promo = ajuste.params.get("pct_lojas_promo", ajuste.params.get("x1", np.nan))
        if np.isnan(coef_promo):
            for k in ajuste.params.keys():
                if "exog" in k or "x" == k.lower():
                    coef_promo = ajuste.params[k]
                    break

        print(f"  {desc:<40} | WMAPE {w:>7.3f}% | Viés {v:>+8.3f}% | "
              f"prev {total_prev:>6.0f} | coef_promo {coef_promo:>7.2f} | AIC {ajuste.aic:>9.1f} | {t_fit:.1f}s")
    except Exception as e:
        print(f"  {desc:<40} | ERRO: {str(e)[:70]}")

print()
print("-" * 95)
print("Teste do SARIMA (sem exog) com trend='ct' e diferentes métodos:")
print("-" * 95)
for method in ["lbfgs", "bfgs", "nm", "powell"]:
    t0 = time.time()
    try:
        modelo = SARIMAX(
            serie_treino,
            order=ORDEM,
            seasonal_order=ORDEM_SAZONAL,
            trend="ct",
            enforce_stationarity=False,
            enforce_invertibility=False,
            freq="D",
        )
        ajuste = modelo.fit(disp=False, method=method, maxiter=200)
        prev = ajuste.forecast(steps=H).values
        prev_clip = np.clip(prev, 0, None)
        w = wmape(real, prev_clip)
        v = vies(real, prev_clip)
        print(f"  SARIMA trend=ct method={method:<7} | WMAPE {w:>7.3f}% | Viés {v:>+8.3f}% | AIC {ajuste.aic:>9.1f}")
    except Exception as e:
        print(f"  SARIMA trend=ct method={method:<7} | ERRO: {str(e)[:60]}")
