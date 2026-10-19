"""
FASE 6 — IMPLANTAÇÃO — 6.1 Política de pedido (componentes separados).

Cenário idêntico ao da Fase 4 (holdout de 28 dias):
  - previsão SARIMAX (2,1,2)(1,1,1,7), trend='ct', method='powell'
  - exógena: pct_lojas_promo do período de teste
  - estoque de segurança: quantil 93,75% dos erros do SARIMAX nas 12 origens do backtest

Conta do pedido (política):
  pedido = demanda prevista + estoque de segurança - estoque disponível no CD
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import (
    treino_p, teste_p, real, indice_teste, H, PRODUTO_FOCO,
)

# ── Parâmetros da política ─────────────────────────────────────────────────────
ESTOQUE_CD = 3000.0        # estoque disponível no CD informado pelo negócio
NIVEL_SERVICO = 0.9375     # Cu / (Cu + Co) = 30% / 32%
PRECO_MEDIO = 2.5870
CU = 0.30 * PRECO_MEDIO
CO = 0.02 * PRECO_MEDIO

# ── 1. Previsão SARIMAX no cenário da Fase 4 ───────────────────────────────────
serie_treino = treino_p.sort_index()["unidades"].astype(float)
exog_treino = treino_p.sort_index()["pct_lojas_promo"].astype(float)
exog_teste = teste_p.sort_index()["pct_lojas_promo"].astype(float)

ajuste = SARIMAX(
    serie_treino, exog=exog_treino,
    order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
    trend="ct", enforce_stationarity=False, enforce_invertibility=False, freq="D",
).fit(disp=False, method="powell", maxiter=200)

previsao = np.clip(ajuste.forecast(steps=H, exog=exog_teste).values, 0, None)
demanda_prevista = float(previsao.sum())

# ── 2. Estoque de segurança — quantil empírico dos erros do backtest ────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "reports", "fase5_backtest_origens_P0131.csv")
df_back = pd.read_csv(CSV)
erros_sarimax = df_back.loc[df_back["modelo"] == "SARIMAX+promo",
                            "erro_total_real_minus_prev"].values
estoque_seguranca = float(np.quantile(erros_sarimax, NIVEL_SERVICO))

# ── 3. Conta do pedido ─────────────────────────────────────────────────────────
pedido_bruto = demanda_prevista + estoque_seguranca - ESTOQUE_CD
pedido = max(0.0, pedido_bruto)     # não existe pedido negativo

# ── 4. Comparação com o realizado ──────────────────────────────────────────────
real_vendido = float(real.sum())
disponivel_total = pedido + ESTOQUE_CD      # o que o CD teria para atender o período
sobra = disponivel_total - real_vendido
cobertura_un = pedido + ESTOQUE_CD
cobertura_dias = cobertura_un / (real_vendido / H)

# ── Impressão ──────────────────────────────────────────────────────────────────
print("=" * 78)
print(f"FASE 6.1 — POLÍTICA DE PEDIDO — {PRODUTO_FOCO}")
print("=" * 78)
print(f"Horizonte (H) ................ {H} dias  ({indice_teste.min():%d/%m/%Y} → {indice_teste.max():%d/%m/%Y})")
print(f"Nível de serviço ............. {NIVEL_SERVICO*100:.2f}%")
print(f"Custo de faltar (Cu) ......... R$ {CU:.4f}/un")
print(f"Custo de sobrar  (Co) ........ R$ {CO:.4f}/un")
print()
print("-" * 78)
print("CONTA ABERTA DO PEDIDO (em unidades, horizonte de 28 dias)")
print("-" * 78)
print(f"  (+) Demanda prevista (SARIMAX, 28 dias) ....... {demanda_prevista:>12,.1f}")
print(f"  (+) Estoque de segurança (quantil 93,75%) ..... {estoque_seguranca:>12,.1f}")
print(f"  (-) Estoque disponível no CD ................. {ESTOQUE_CD:>12,.1f}")
print(f"  {'=' * 50:>50}")
print(f"  (=) PEDIDO sugerido .......................... {pedido:>12,.1f} unidades")
print()
print("-" * 78)
print("CONFRONTO COM O REALIZADO")
print("-" * 78)
print(f"  Demanda prevista + segurança (necessidade) ... {demanda_prevista + estoque_seguranca:>12,.1f}")
print(f"  Estoque CD + pedido (disponível no período) .. {disponivel_total:>12,.1f}")
print(f"  Venda real observada (28 dias) ............... {real_vendido:>12,.1f}")
print(f"  Sobra ao fim do período ...................... {sobra:>12,.1f}")
print()
print(f"  Cobertura montada: {cobertura_un:,.1f} un = {cobertura_dias:.1f} dias de venda média")
print(f"  Sem o estoque de segurança, o pedido seria {max(0.0, demanda_prevista - ESTOQUE_CD):>10,.1f} un")
