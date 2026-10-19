import pandas as pd
import numpy as np

SEED = 42
np.random.seed(SEED)

CAMINHO_LOJA = r"data/raw/demanda_por_loja.csv"

df_loja = pd.read_csv(CAMINHO_LOJA, sep=",", parse_dates=["data"], dayfirst=False)

print("Colunas reais do arquivo:")
print(list(df_loja.columns))
print()
print("=" * 70)
print("CHECAGEM 1: estoque_fim_dia = 0 e unidades vendidas > 0 no mesmo dia")
print("=" * 70)
total_linhas = len(df_loja)
print(f"Total de linhas na base por loja: {total_linhas:,}")
print()

mask_estoque_zero = df_loja["estoque_fim_dia"] == 0
linhas_estoque_zero = df_loja[mask_estoque_zero].copy()
qtde_estoque_zero = len(linhas_estoque_zero)
pct_estoque_zero = (qtde_estoque_zero / total_linhas) * 100

print(f"Quantidade de linhas com estoque_fim_dia = 0: {qtde_estoque_zero:,}")
print(f"Percentual sobre o total da base: {pct_estoque_zero:.4f}%")
print()

mask_venda_positiva_no_zero = linhas_estoque_zero["unidades"] > 0
linhas_estoque_zero_com_venda = linhas_estoque_zero[mask_venda_positiva_no_zero]
qtde_venda_no_zero = len(linhas_estoque_zero_com_venda)
pct_venda_no_zero_sobre_zero = (qtde_venda_no_zero / qtde_estoque_zero) * 100 if qtde_estoque_zero > 0 else 0
pct_venda_no_zero_sobre_total = (qtde_venda_no_zero / total_linhas) * 100

print(f"Quantidade dessas linhas com unidades > 0: {qtde_venda_no_zero:,}")
print(f"Percentual DENTRO das linhas com estoque=0: {pct_venda_no_zero_sobre_zero:.4f}%")
print(f"Percentual sobre o TOTAL da base (130.843 linhas): {pct_venda_no_zero_sobre_total:.6f}%")
print()

print("=" * 70)
print("Distribuicao da quantidade de unidades vendidas quando estoque=0")
print("=" * 70)
print(linhas_estoque_zero_com_venda["unidades"].describe().to_string())
print()
print("Valores unicos de unidades vendidas no estoque=0:")
print(linhas_estoque_zero_com_venda["unidades"].value_counts().sort_index().to_string())
print()

print("=" * 70)
print("Primeiras 20 linhas com estoque_fim_dia = 0 e unidades > 0:")
print("=" * 70)
colunas_mostrar = [c for c in ["produto", "loja", "data", "unidades", "estoque_fim_dia",
                   "preco", "em_promo", "feriado_escolar"] if c in df_loja.columns]
print(linhas_estoque_zero_com_venda[colunas_mostrar].head(20).to_string(index=False))
print()

print("=" * 70)
print("Exemplos de linhas com estoque=0 E unidades=0 (cenario 'normal'):")
print("=" * 70)
linhas_zero_sem_venda = linhas_estoque_zero[linhas_estoque_zero["unidades"] == 0]
print(linhas_zero_sem_venda[colunas_mostrar].head(10).to_string(index=False))
print()

print("=" * 70)
print("Analise adicional: estoque_inicio_dia vs unidades vs estoque_fim_dia")
print("=" * 70)
df_loja = df_loja.sort_values(["produto", "loja", "data"]).reset_index(drop=True)
df_loja["estoque_inicio_dia_calculado"] = (
    df_loja.groupby(["produto", "loja"])["estoque_fim_dia"].shift(1)
)
df_loja["venda_esperada_pela_variacao"] = (
    df_loja["estoque_inicio_dia_calculado"] - df_loja["estoque_fim_dia"]
)
df_loja["diferenca_venda_registrada_vs_variacao"] = (
    df_loja["unidades"] - df_loja["venda_esperada_pela_variacao"]
)

print("Quanto das vezes unidades = estoque_inicio - estoque_fim (ignorando entradas de estoque)?")
df_com_estoque_anterior = df_loja.dropna(subset=["estoque_inicio_dia_calculado"]).copy()
total_comparavel = len(df_com_estoque_anterior)
iguais = (abs(df_com_estoque_anterior["diferenca_venda_registrada_vs_variacao"]) < 1e-9).sum()
entradas = (df_com_estoque_anterior["diferenca_venda_registrada_vs_variacao"] < -1e-9).sum()
saidas_estranhas = (df_com_estoque_anterior["diferenca_venda_registrada_vs_variacao"] > 1e-9).sum()

print(f"Linhas comparaveis (sem o primeiro dia de cada loja/produto): {total_comparavel:,}")
print(f"  - unidades == (inicio - fim):             {iguais:,}  ({iguais/total_comparavel*100:.4f}%)")
print(f"  - (fim - inicio) > unidades (entrou estoque): {entradas:,}  ({entradas/total_comparavel*100:.4f}%)")
print(f"  - unidades > (inicio - fim) (vendeu mais do que tinha?): {saidas_estranhas:,}  ({saidas_estranhas/total_comparavel*100:.6f}%)")
print()

if saidas_estranhas > 0:
    print("Exemplos de 'vendeu mais do que tinha em estoque no inicio do dia':")
    estranhas = df_com_estoque_anterior[df_com_estoque_anterior["diferenca_venda_registrada_vs_variacao"] > 1e-9]
    print(estranhas[["produto","loja","data","estoque_inicio_dia_calculado","unidades","estoque_fim_dia",
                     "venda_esperada_pela_variacao","diferenca_venda_registrada_vs_variacao"]].head(15).to_string(index=False))
