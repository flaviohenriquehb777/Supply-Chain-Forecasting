"""
FASE 5/6 — Estoque de segurança pela distribuição EMPÍRICA dos erros do backtest.

Em vez da fórmula do livro-texto (que assume erro normal e variância constante),
usamos o quantil empírico dos erros de período (realizado - previsto) medidos
nas 12 origens do backtest. O estoque de segurança é esse quantil.

Convenção do erro: erro = realizado - previsto.
  erro > 0 -> faltou produto (subestimamos)
  erro < 0 -> sobrou produto (superestimamos)
"""

import os
import numpy as np
import pandas as pd

# ── Caminhos ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "reports", "fase5_backtest_origens_P0131.csv")

# ── Parâmetros de custo (Fase 1) ───────────────────────────────────────────────
PRECO_MEDIO = 2.5870           # preço médio do P0131 na base
MARGEM_PERDIDA_PCT = 0.30      # custo de faltar (fração do preço)
CUSTO_CARREGO_PCT = 0.02       # custo de sobrar (fração do preço)
CUSTO_FALTAR_UN = MARGEM_PERDIDA_PCT * PRECO_MEDIO   # Cu = R$ 0,7761
CUSTO_SOBRAR_UN = CUSTO_CARREGO_PCT * PRECO_MEDIO    # Co = R$ 0,0517

NIVEL_SERVICO = 0.9375         # 93,75% = Cu / (Cu + Co)


# ── Item 1: função do estoque de segurança ─────────────────────────────────────
def estoque_seguranca(erros, nivel_servico=NIVEL_SERVICO):
    """
    Recebe a lista de erros de período de um modelo (realizado - previsto) e o
    nível de serviço. Devolve o quantil empírico desses erros nesse nível.

    Interpretação: pedindo previsão + estoque de segurança, o pedido cobre a
    demanda em `nivel_servico` (x100%) dos períodos observados.
    """
    erros = np.asarray(erros, dtype=float)
    return float(np.quantile(erros, nivel_servico))


# ── Carrega os erros já calculados no backtest ─────────────────────────────────
df = pd.read_csv(CSV)
df["origem"] = pd.to_datetime(df["origem"])

modelos = ["Holt-Winters", "SARIMAX+promo"]

print("=" * 78)
print("ESTOQUE DE SEGURANÇA — quantil empírico dos erros do backtest (12 origens)")
print("=" * 78)
print(f"Preço médio P0131 ........ R$ {PRECO_MEDIO:.4f}")
print(f"Custo de faltar (Cu) ..... R$ {CUSTO_FALTAR_UN:.4f}  (30% do preço)")
print(f"Custo de sobrar  (Co) .... R$ {CUSTO_SOBRAR_UN:.4f}  (2% do preço)")
print(f"Nível de serviço ......... {NIVEL_SERVICO*100:.2f}%")
print()

for modelo in modelos:
    erros = df.loc[df["modelo"] == modelo, "erro_total_real_minus_prev"].values

    # Item 2: estoque de segurança (quantil empírico)
    ss = estoque_seguranca(erros, NIVEL_SERVICO)

    # Item 3: percentual de origens que faltaria produto
    pct_falta_sem_ss = np.mean(erros > 0) * 100          # pedindo só a previsão
    pct_falta_com_ss = np.mean(erros > ss) * 100         # previsão + estoque de segurança

    # Item 4: custo do estoque de segurança por período
    custo_periodo = ss * CUSTO_SOBRAR_UN

    print("-" * 78)
    print(f"MODELO: {modelo}")
    print("-" * 78)
    print(f"  Erros de período (realizado - previsto), ordenados:")
    print(f"    {np.round(np.sort(erros), 1).tolist()}")
    print(f"  Estoque de segurança (quantil {NIVEL_SERVICO*100:.2f}%) ... {ss:>10.1f} unidades")
    print()
    print(f"  Faltaria produto pedindo SÓ a previsão ......... {pct_falta_sem_ss:>6.2f}% das origens"
          f"  ({int(round(pct_falta_sem_ss/100*len(erros)))}/{len(erros)})")
    print(f"  Faltaria produto com o estoque de segurança .... {pct_falta_com_ss:>6.2f}% das origens"
          f"  ({int(round(pct_falta_com_ss/100*len(erros)))}/{len(erros)})")
    print()
    print(f"  Custo do estoque de segurança por período ...... R$ {custo_periodo:.2f}")
    print(f"    ({ss:.1f} un x R$ {CUSTO_SOBRAR_UN:.4f})")
    print()

print("=" * 78)
print("Obs.: amostra = 12 erros de período (12 origens). Quantil empírico em 93,75%")
print("      usa interpolação linear entre os 2 maiores erros positivos.")
