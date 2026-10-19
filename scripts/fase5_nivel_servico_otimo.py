"""
FASE 5/6 — Cálculo do nível de serviço ótimo (newsvendor) para o P0131.
Todos os custos são percentuais do preço, então o preço entra como referência
e a razão entre os custos independe dele.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import serie_p, PRODUTO_FOCO

# ── Parâmetros da Fase 1 (contratuais, não são escolha nossa) ──────────────────
MARGEM_PERDIDA_PCT = 0.30   # custo de faltar uma unidade (fração do preço)
CUSTO_CARREGO_PCT = 0.02    # custo de sobrar uma unidade no período (fração do preço)

# ── Item 1: preço médio do produto na base ─────────────────────────────────────
preco_medio = serie_p["preco_medio"].mean()
preco_mediano = serie_p["preco_medio"].median()
# preço ponderado por volume (média dos preços pesada pelas unidades vendidas)
preco_ponderado = (serie_p["preco_medio"] * serie_p["unidades"]).sum() / serie_p["unidades"].sum()

# ── Itens 2 e 3: custo de faltar e de sobrar ───────────────────────────────────
custo_faltar = MARGEM_PERDIDA_PCT * preco_medio      # Cu
custo_sobrar = CUSTO_CARREGO_PCT * preco_medio       # Co

# ── Item 4: razão entre os custos ──────────────────────────────────────────────
razao = custo_faltar / custo_sobrar

# ── Item 5: nível de serviço ótimo (fração de períodos atendidos) ──────────────
nivel_servico = custo_faltar / (custo_faltar + custo_sobrar)

# ── Item 6: percentual de períodos que aceita ficar sem produto ────────────────
prob_ruptura = 1 - nivel_servico

print("=" * 70)
print(f"NÍVEL DE SERVIÇO ÓTIMO — Produto {PRODUTO_FOCO}")
print("=" * 70)
print()
print("1. PREÇO MÉDIO DO PRODUTO NA BASE")
print(f"   preco_medio (média simples):     R$ {preco_medio:>8.4f}")
print(f"   preco_medio (mediano):           R$ {preco_mediano:>8.4f}")
print(f"   preco_medio (ponderado volume):  R$ {preco_ponderado:>8.4f}")
print()
print("2. CUSTO DE FALTAR UMA UNIDADE (Cu) — margem perdida de 30% do preço")
print(f"   Cu = 0,30 x {preco_medio:.4f} = R$ {custo_faltar:.4f} por unidade")
print()
print("3. CUSTO DE SOBRAR UMA UNIDADE (Co) — carrego de 2% do preço no período")
print(f"   Co = 0,02 x {preco_medio:.4f} = R$ {custo_sobrar:.4f} por unidade")
print()
print("4. RAZÃO ENTRE OS CUSTOS (Cu / Co)")
print(f"   razão = {custo_faltar:.4f} / {custo_sobrar:.4f} = {razao:.2f}")
print(f"   (faltar é {razao:.0f}x mais caro que sobrar)")
print()
print("5. NÍVEL DE SERVIÇO ÓTIMO  =  Cu / (Cu + Co)")
print(f"   nível = {custo_faltar:.4f} / ({custo_faltar:.4f} + {custo_sobrar:.4f})")
print(f"   nível = {custo_faltar:.4f} / {custo_faltar + custo_sobrar:.4f}")
print(f"   nível = {nivel_servico:.6f}  =  {nivel_servico*100:.2f}%   <- quantil do pedido")
print()
print("6. PERCENTUAL DE PERÍODOS QUE PODE FICAR SEM PRODUTO")
print(f"   prob_ruptura = 1 - {nivel_servico:.4f} = {prob_ruptura:.6f} = {prob_ruptura*100:.2f}%")
print()
print("   Em 28 dias (1 ciclo):")
print(f"      dias atendidos esperados : {nivel_servico*28:.2f}")
print(f"      dias com ruptura esperada: {prob_ruptura*28:.2f}")
print()
print("=" * 70)
print("IMPLICAÇÃO OPERACIONAL")
print("=" * 70)
print(f"  Pedir exatamente a previsão (a média) = nível de serviço ~50%")
print(f"  Pedir o quantil {nivel_servico*100:.2f}% da distribuição de demanda = nível ótimo")
