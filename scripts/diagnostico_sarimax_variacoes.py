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
exog_treino = treino_p.sort_index()["pct_lojas_promo"].astype(float)
exog_teste = teste_p.sort_index()["pct_lojas_promo"].astype(float)

ORDEM = (2, 1, 2)
ORDEM_SAZONAL = (1, 1, 1, 7)

print("EXPERIMENTO — variações do SARIMAX (2,1,2)(1,1,1,7)")
print("=" * 90)
print(f"  Produto    : {PRODUTO_FOCO}")
print(f"  Treino n   : {len(serie_treino)} dias")
print(f"  Teste n    : {H} dias  (real total = {real.sum():.0f})")
print()

# Configurações a testar (sem exog = SARIMA, com exog = SARIMAX)
configs = [
    # nome, usar_exog, trend, enforce_stat, enforce_inv, simple_diff
    ("meu_atual_sem_exog",     False, None,   False, False, False),
    ("trend_c_sem_exog",       False, "c",    False, False, False),
    ("trend_t_sem_exog",       False, "t",    False, False, False),
    ("trend_ct_sem_exog",      False, "ct",   False, False, False),
    ("trend_c_enforce_sem_exog", False, "c",  True,  True,  False),
    ("trend_c_simplediff",     False, "c",    False, False, True),
    ("trend_none_simplediff",  False, None,   False, False, True),
    ("trend_ct_simplediff",    False, "ct",   False, False, True),
]

resultados = []

for nome, usa_exog, trend, es, ei, sd in configs:
    t0 = time.time()
    try:
        kwargs = dict(
            endog=serie_treino,
            order=ORDEM,
            seasonal_order=ORDEM_SAZONAL,
            enforce_stationarity=es,
            enforce_invertibility=ei,
            simple_differencing=sd,
            freq="D",
        )
        if usa_exog:
            kwargs["exog"] = exog_treino
        if trend is not None:
            kwargs["trend"] = trend

        modelo = SARIMAX(**kwargs)
        ajuste = modelo.fit(disp=False)
        t_fit = time.time() - t0

        if usa_exog:
            prev = ajuste.forecast(steps=H, exog=exog_teste).values
        else:
            prev = ajuste.forecast(steps=H).values

        prev_clip = np.clip(prev, 0, None)
        w = wmape(real, prev_clip)
        v = vies(real, prev_clip)
        total_prev = prev_clip.sum()

        print(f"  {nome:<32} | WMAPE {w:>7.3f}% | Viés {v:>+8.3f}% | "
              f"prev {total_prev:>7.0f} | fit {t_fit:>5.2f}s | AIC {ajuste.aic:>10.2f}")

        # Imprime coeficientes do modelo com 'c' para vermos o drift
        if trend in ("c", "ct"):
            drift = ajuste.params.get("drift", ajuste.params.get("intercept", np.nan))
            print(f"      {'':30}   drift/intercept = {drift:.4f}")

        resultados.append((nome, usa_exog, trend, w, v, total_prev))
    except Exception as e:
        print(f"  {nome:<32} | ERRO: {str(e)[:60]}")

print()
print("=" * 90)
print("AGORA COM A EXÓGENA (SARIMAX):")
print("-" * 90)

configs_exog = [
    ("meu_atual_com_exog",     "c",    False, False, False),
    ("trend_c_com_exog",       "c",    False, False, False),
    ("trend_none_com_exog",    None,   False, False, False),
    ("trend_t_com_exog",       "t",    False, False, False),
    ("trend_ct_com_exog",      "ct",   False, False, False),
    ("trend_c_enforce_com_exog", "c",  True,  True,  False),
    ("trend_c_simplediff_exog", "c",   False, False, True),
    ("trend_none_simplediff_exog", None, False, False, True),
    ("trend_ct_simplediff_exog", "ct", False, False, True),
]

for nome, trend, es, ei, sd in configs_exog:
    t0 = time.time()
    try:
        kwargs = dict(
            endog=serie_treino,
            exog=exog_treino,
            order=ORDEM,
            seasonal_order=ORDEM_SAZONAL,
            enforce_stationarity=es,
            enforce_invertibility=ei,
            simple_differencing=sd,
            freq="D",
        )
        if trend is not None:
            kwargs["trend"] = trend

        modelo = SARIMAX(**kwargs)
        ajuste = modelo.fit(disp=False)
        t_fit = time.time() - t0

        prev = ajuste.forecast(steps=H, exog=exog_teste).values
        prev_clip = np.clip(prev, 0, None)
        w = wmape(real, prev_clip)
        v = vies(real, prev_clip)
        total_prev = prev_clip.sum()
        coef_promo = ajuste.params.get("pct_lojas_promo", np.nan)

        print(f"  {nome:<32} | WMAPE {w:>7.3f}% | Viés {v:>+8.3f}% | "
              f"prev {total_prev:>7.0f} | coef_promo {coef_promo:>8.2f} | AIC {ajuste.aic:>10.2f}")

        if trend in ("c", "ct"):
            drift = ajuste.params.get("drift", ajuste.params.get("intercept", np.nan))
            print(f"      {'':30}   drift/intercept = {drift:.4f}")

        resultados.append((nome, True, trend, w, v, total_prev))
    except Exception as e:
        print(f"  {nome:<32} | ERRO: {str(e)[:60]}")

print()
print("Resultados do professor (referência):")
print("  SARIMA  (2,1,2)(1,1,1,7)   : WMAPE 10.07%  | Viés +1.12%")
print("  SARIMAX + promoção          : WMAPE  8.45%  | Viés +1.09%  | coef_promo 65.34")
