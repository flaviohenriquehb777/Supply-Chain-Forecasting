import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

SEED = 42
np.random.seed(SEED)

CAMINHO_REDE = r"data/raw/demanda.csv"
PASTA_FIGURAS = r"reports/figures"
os.makedirs(PASTA_FIGURAS, exist_ok=True)

H = 28
PRODUTO_FOCO = "P0131"
PRODUTOS_REMOVER = ["P0438", "P0051", "P0079"]

df = pd.read_csv(CAMINHO_REDE, sep=",", parse_dates=["data"], dayfirst=False)

print("FASE 3 — Preparação dos Dados")
print("=" * 70)

# --- Item 1: Remove produtos descontinuados ---
print("ITEM 1: Remoção dos produtos em descontinuação")
print("-" * 70)
n_prod_antes = df["produto"].nunique()
n_linhas_antes = len(df)

df_limpo = df[~df["produto"].isin(PRODUTOS_REMOVER)].copy().reset_index(drop=True)

n_prod_depois = df_limpo["produto"].nunique()
n_linhas_depois = len(df_limpo)

print(f"  Produtos ANTES : {n_prod_antes}")
print(f"  Produtos REMOVIDOS ({len(PRODUTOS_REMOVER)}): {', '.join(PRODUTOS_REMOVER)}")
print(f"  Produtos DEPOIS: {n_prod_depois}")
print(f"  Linhas ANTES   : {n_linhas_antes:,}")
print(f"  Linhas DEPOIS  : {n_linhas_depois:,}")
print(f"  Linhas removidas: {n_linhas_antes - n_linhas_depois:,}")
print()

nulos_alvo = df_limpo["unidades"].isna().sum()
nulos_pct = nulos_alvo / len(df_limpo) * 100
print(f"  Valores nulos na coluna alvo `unidades` após filtro: {nulos_alvo:,} ({nulos_pct:.4f}%)")
if nulos_alvo == 0:
    print("  ✅ Nenhum vazio restante na coluna alvo.")
else:
    print(f"  ⚠️  Ainda existem {nulos_alvo} vazios — investigar antes de seguir.")
print()

# --- Item 2: Define H e ponto de corte temporal ---
print(f"ITEM 2: Definição de H = {H} dias e ponto de corte temporal")
print("-" * 70)

# Painel data × produto (linha por data, coluna por produto)
painel_limpo = (
    df_limpo.pivot_table(index="data", columns="produto", values="unidades", aggfunc="sum")
    .sort_index()
    .asfreq("D")
)
data_maxima = painel_limpo.index.max()
data_corte = data_maxima - pd.Timedelta(days=H)
# Data de INICIO do holdout: dia seguinte ao corte (inclusive)
data_inicio_teste = data_corte + pd.Timedelta(days=1)
dias_series = len(painel_limpo)

print(f"  Data máxima da base  : {data_maxima.strftime('%d/%m/%Y')}")
print(f"  Dias totais de série : {dias_series} dias")
print(f"  H (horizonte)        : {H} dias")
print(f"  Ponto de CORTE (último dia do treino) : {data_corte.strftime('%d/%m/%Y')}")
print(f"  Primeiro dia do teste (holdout)       : {data_inicio_teste.strftime('%d/%m/%Y')}")
print(f"  Último dia do teste (holdout)         : {data_maxima.strftime('%d/%m/%Y')}")
print(f"  Tamanho do treino    : {len(painel_limpo.loc[:data_corte])} dias")
print(f"  Tamanho do teste     : {len(painel_limpo.loc[data_inicio_teste:])} dias")
print()

# --- Item 3: Separa treino/teste da série do produto foco ---
print(f"ITEM 3: Separação treino/teste do produto {PRODUTO_FOCO}")
print("-" * 70)

serie_p = df_limpo[df_limpo["produto"] == PRODUTO_FOCO].copy().sort_values("data")
serie_p = serie_p.set_index("data").asfreq("D")
serie_p = serie_p.sort_index()

treino_p = serie_p.loc[:data_corte]
teste_p = serie_p.loc[data_inicio_teste:]

print(f"  Período treino: {treino_p.index.min().strftime('%d/%m/%Y')} → {treino_p.index.max().strftime('%d/%m/%Y')}")
print(f"  N dias treino : {len(treino_p)}")
print(f"  Período teste : {teste_p.index.min().strftime('%d/%m/%Y')} → {teste_p.index.max().strftime('%d/%m/%Y')}")
print(f"  N dias teste  : {len(teste_p)}")
print(f"  Total checagem: {len(treino_p)} + {len(teste_p)} = {len(treino_p)+len(teste_p)} (esperado = {len(serie_p)})")
print()

# --- Item 4: Plote serie inteira com treino/teste em cores diferentes + linha corte ---
print("ITEM 4: Plot série completa com corte temporal")
print("-" * 70)

sns.set_theme(style="whitegrid", font_scale=1.05)
fig, ax = plt.subplots(figsize=(16, 6.8))

ax.plot(treino_p.index, treino_p["unidades"].values, color="#1D4ED8", linewidth=0.9, alpha=0.92,
        label=f"Treino (n = {len(treino_p)} dias — até {data_corte.strftime('%d/%m/%Y')})",
        zorder=2)
ax.plot(teste_p.index, teste_p["unidades"].values, color="#DC2626", linewidth=1.65, alpha=0.98,
        label=f"Teste  (n = {len(teste_p)} dias — {data_inicio_teste.strftime('%d/%m')} a {data_maxima.strftime('%d/%m/%Y')})",
        zorder=3)
ax.axvline(x=data_corte, color="#0F172A", linewidth=1.65, linestyle="--", alpha=0.85,
           label=f"Corte temporal (último dia do treino = {data_corte.strftime('%d/%m/%Y')})",
           zorder=4)

# Fundo sombreado para a área do teste
ax.axvspan(data_inicio_teste, data_maxima, color="#FEE2E2", alpha=0.35, zorder=1)

ax.set_title(f"Produto {PRODUTO_FOCO} — Separação treino / teste (H = {H} dias de holdout)",
             fontsize=13, fontweight="bold", loc="left")
ax.set_ylabel("Unidades vendidas na rede / dia", fontsize=11)
ax.set_xlabel("Data", fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=9.8, ncol=1)

caminho_plot = os.path.join(PASTA_FIGURAS, f"{PRODUTO_FOCO}_serie_treino_teste_corte_H{H}.png")
plt.tight_layout()
plt.savefig(caminho_plot, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Plot salvo em: {caminho_plot}")

print()
print("Conferência final do pipeline (ordem de operações corretas):")
print(f"  1) Filtro de produtos (foram 30, agora 27) — SEMPRE ANTES do corte")
print(f"  2) Corte temporal = data_max − H ({data_maxima.strftime('%d/%m/%Y')} − {H}d = {data_corte.strftime('%d/%m/%Y')})")
print(f"  3) Primeiro dia do teste = corte + 1 dia = {data_inicio_teste.strftime('%d/%m/%Y')}")
print(f"  4) Regra 2: qualquer lag de venda deve ser ⩾ 28. Nenhuma janela móvel com último dia depois de t−28.")
