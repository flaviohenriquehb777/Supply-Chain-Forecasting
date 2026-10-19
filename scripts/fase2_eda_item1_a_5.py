import pandas as pd
import numpy as np

SEED = 42
np.random.seed(SEED)

CAMINHO_DADOS = r"data/raw/demanda.csv"

df = pd.read_csv(CAMINHO_DADOS, sep=",", parse_dates=["data"], dayfirst=False)

print("=" * 60)
print("ITEM 1: Características gerais da base")
print("=" * 60)
n_linhas = df.shape[0]
n_produtos = df["produto"].nunique()
data_min = df["data"].min()
data_max = df["data"].max()
n_dias = df["data"].nunique()
print(f"Numero de linhas: {n_linhas:,}")
print(f"Numero de produtos distintos: {n_produtos}")
print(f"Data minima: {data_min.strftime('%d/%m/%Y')}")
print(f"Data maxima: {data_max.strftime('%d/%m/%Y')}")
print(f"Quantidade de dias distintos: {n_dias}")
dias_total = (data_max - data_min).days + 1
print(f"Total de dias no periodo (incluindo extremos): {dias_total}")
print()

print("=" * 60)
print("ITEM 2: Primeiras linhas e tipos de dado")
print("=" * 60)
print("Tipos de dado das colunas:")
print(df.dtypes)
print()
print("Primeiras 10 linhas:")
print(df.head(10).to_string())
print()
print("Ultimas 5 linhas:")
print(df.tail(5).to_string())
print()

print("=" * 60)
print("ITEM 3: Dicionario de dados (interpretacao)")
print("=" * 60)
print("produto        | texto      | Codigo interno do SKU do produto. Chave primaria no nivel de produto.")
print("data           | datetime64 | Data de referencia da venda (dia). Chave primaria no nivel temporal.")
print("unidades       | float64    | Unidades vendidas no dia, somadas em todas as lojas da rede. COLUNA ALVO. Unidade: unidades fisicas.")
print("preco_medio    | float64    | Preco praticado no dia, media ponderada das lojas da rede. Unidade: R$ por unidade.")
print("pct_lojas_promo| float64    | Fracao das lojas ativas com o item em promocao no dia. Unidade: 0 a 1 (0% a 100%).")
print("estoque_total  | float64    | Saldo de estoque do item somado nas lojas da rede ao final do dia. Unidade: unidades fisicas.")
print("lojas_ativas   | int64      | Numero de lojas que comercializaram o item no dia (lojas com venda > 0). Unidade: lojas.")
print()

print("=" * 60)
print("ITEM 4: Valores nulos por coluna")
print("=" * 60)
nulos_qtd = df.isnull().sum()
nulos_pct = (nulos_qtd / n_linhas) * 100
df_nulos = pd.DataFrame({
    "quantidade_nulos": nulos_qtd,
    "percentual_nulos_pct": nulos_pct.round(4)
})
df_nulos = df_nulos[df_nulos["quantidade_nulos"] > 0].sort_values("quantidade_nulos", ascending=False)
if df_nulos.empty:
    print("NAO HA VALORES NULOS EM NENHUMA COLUNA.")
    print()
    print("Distribuicao dos nulos (todas zero):")
    print(pd.DataFrame({
        "quantidade_nulos": nulos_qtd,
        "percentual_nulos_pct": nulos_pct.round(4)
    }).to_string())
else:
    print(df_nulos.to_string())
print()

print("=" * 60)
print("ITEM 5: Continuidade temporal por produto")
print("=" * 60)
dados_continuidade = []
for produto, grupo in df.groupby("produto"):
    datas_grupo = grupo["data"].sort_values()
    data_min_prod = datas_grupo.iloc[0]
    data_max_prod = datas_grupo.iloc[-1]
    dias_esperados = (data_max_prod - data_min_prod).days + 1
    dias_presentes = datas_grupo.nunique()
    dias_faltando = dias_esperados - dias_presentes
    datas_faltantes = pd.date_range(start=data_min_prod, end=data_max_prod, freq="D").difference(datas_grupo)
    dados_continuidade.append({
        "produto": produto,
        "data_primeira_ocorrencia": data_min_prod.strftime("%Y-%m-%d"),
        "data_ultima_ocorrencia": data_max_prod.strftime("%Y-%m-%d"),
        "dias_esperados_no_periodo": dias_esperados,
        "dias_presentes_na_base": dias_presentes,
        "dias_faltando": dias_faltando,
        "lista_datas_faltantes": ", ".join([d.strftime("%Y-%m-%d") for d in datas_faltantes])
    })

df_continuidade = pd.DataFrame(dados_continuidade)
total_dias_faltando = df_continuidade["dias_faltando"].sum()
produtos_com_falta = df_continuidade[df_continuidade["dias_faltando"] > 0]

print(f"Total de lacunas de dias em todos os produtos: {total_dias_faltando} dia(s)")
print(f"Quantidade de produtos com pelo menos 1 dia faltando: {len(produtos_com_falta)} de {n_produtos}")
print()
if len(produtos_com_falta) > 0:
    print("Detalhamento dos produtos com dias faltando:")
    print(produtos_com_falta.to_string(index=False))
else:
    print("TODOS OS PRODUTOS POSSUEM SERIES TEMPORAIS CONTINUAS (sem lacunas entre a primeira e ultima data).")
print()

print("-" * 60)
print("Resumo da continuidade para todos os produtos:")
print("-" * 60)
print(df_continuidade[["produto", "dias_esperados_no_periodo", "dias_presentes_na_base", "dias_faltando"]].to_string(index=False))
