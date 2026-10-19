import pandas as pd
import numpy as np
import os
import sys

# Garante que os módulos/objetos criados no script de infra-estrutura estão disponíveis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fase4_infra_avaliacao import (
    treino_p, teste_p, real, indice_teste, H, PRODUTO_FOCO,
    wmape, vies, placar, previsoes, avaliar,
)

SEED = 42
np.random.seed(SEED)

# ==============================================================================
# MODELO 1 — NAIVE SAZONAL (últimos 7 dias do treino, repetidos por 4 semanas)
# ==============================================================================
# Regra: pegar os últimos 7 dias conhecidos do treino (7 dias anteriores ao corte)
# e repetir esse bloco 4 vezes para cobrir os 28 dias (4 semanas) do teste.

N_DIAS_REPETICAO = 7

# Pega a serie do treino, ordenada (esta garantida pelo indice data asfreq)
serie_treino_ord = treino_p.sort_index()["unidades"].astype(float)

ultimos_7_dias_treino = serie_treino_ord.tail(N_DIAS_REPETICAO).values.copy()
datas_ultimos_7 = serie_treino_ord.tail(N_DIAS_REPETICAO).index.strftime("%d/%m/%Y").tolist()

print("MODELO 1: NAIVE SAZONAL (repete os últimos 7 dias do treino 4 vezes)")
print("=" * 70)
print(f"  Janela de cópia do treino  : últimos {N_DIAS_REPETICAO} dias do treino "
      f"(antes do corte)")
print(f"  Datas usadas como template : {datas_ultimos_7[0]} → {datas_ultimos_7[-1]}")
print(f"  Valores template (7 dias)  : {np.round(ultimos_7_dias_treino, 0).astype(int).tolist()}")
print(f"  Período previsto           : {indice_teste.min().strftime('%d/%m/%Y')} "
      f"→ {indice_teste.max().strftime('%d/%m/%Y')} ({H} dias)")
print()

# Repete o bloco de 7 dias quantas vezes precisar (ceil(H/7) = 4) e depois recorta em H
n_repeticoes = int(np.ceil(H / N_DIAS_REPETICAO))
previsao_naive_sazonal = np.tile(ultimos_7_dias_treino, reps=n_repeticoes)[:H]

# Garante o alinhamento com os dias da semana do teste:
#   4/10/2019 (dia 1 do teste)  ← qual dia da semana?
#   a primeira data do template corresponde ao dia 1 do teste
#   (porque copiamos a sequência temporal: terça → quarta → ... → segunda e repetimos)

# Imprime o primeiro ciclo para conferencia:
dias_semana_pt = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
print("  Alinhamento dia da semana — 1ª semana de teste  ↔  template copiado do treino:")
print(f"  {'Data teste':<12} {'dia_sem':<10} {'prev_unid':>10}   ←   "
      f"{'Data treino (template)':<22} {'dia_sem_orig':<10} {'val_orig':>10}")
print("  " + "-" * 98)
for i in range(N_DIAS_REPETICAO):
    d_teste = indice_teste[i]
    d_treino = serie_treino_ord.tail(N_DIAS_REPETICAO).index[i]
    dia_t = dias_semana_pt[d_teste.weekday()]
    dia_o = dias_semana_pt[d_treino.weekday()]
    print(f"  {d_teste.strftime('%d/%m/%Y'):<12} {dia_t:<10} {previsao_naive_sazonal[i]:>10.0f}   ←   "
          f"{d_treino.strftime('%d/%m/%Y'):<22} {dia_o:<10} {ultimos_7_dias_treino[i]:>10.0f}")
print()

# Avalia (clip em zero, placar e dict previsoes sao populados na funcao avaliar)
print("  Aplicando `avaliar()` — clip em zero, salvando no placar:")
print("-" * 70)
w_ns, v_ns = avaliar("Naive Sazonal (últimos 7d)", previsao_naive_sazonal)
print("-" * 70)
print()
print(f"  Placar acumulado após {len(placar)} modelo(s):")
tabela_placar = pd.DataFrame(placar).T
print(tabela_placar.to_string())
print()

# Series do naive sazonal salvos também em CSV auxiliar (para uso em FASE 5)
PASTA_CSV_INTER = r"data/processed"
os.makedirs(PASTA_CSV_INTER, exist_ok=True)
caminho_csv = os.path.join(PASTA_CSV_INTER, f"previsoes_{PRODUTO_FOCO}_H{H}.csv")
prev_df = pd.DataFrame(previsoes).reset_index()
prev_df.rename(columns={"index": "data"}, inplace=True)
prev_df["real"] = real
prev_df.to_csv(caminho_csv, index=False)
print(f"  Previsões do período salvas em: {caminho_csv}")
