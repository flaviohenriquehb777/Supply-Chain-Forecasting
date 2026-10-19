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

df = pd.read_csv(CAMINHO_REDE, sep=",", parse_dates=["data"], dayfirst=False)

PRODUTO = "P0131"

# Filtra e cria serie diaria com indice de data, frequencia explicita D
df_p = df[df["produto"] == PRODUTO].copy()
serie_p = (
    df_p.set_index("data")
    .sort_index()
    .asfreq("D")  # Garante frequencia diaria explicita; lacunas viram NaN
)

print(f"DECISAO REGISTRADA: Grão = produto x dia (base demanda.csv)")
print(f"                    demanda_por_loja.csv fora do escopo do projeto")
print(f"                    Produto foco: {PRODUTO}")
print("=" * 65)
print()

print(f"SERIE DO PRODUTO {PRODUTO}")
print("=" * 65)

# 1. Periodo
total_dias_idx = len(serie_p)
data_min = serie_p.index.min()
data_max = serie_p.index.max()
dias_calendario = (data_max - data_min).days + 1
print(f"ITEM 1:")
print(f"  Total de dias na serie (frequencia diaria explicita): {total_dias_idx}")
print(f"  Data inicial: {data_min.strftime('%d/%m/%Y')}")
print(f"  Data final  : {data_max.strftime('%d/%m/%Y')}")
print(f"  Dias corridos no periodo (inclui ambos os extremos): {dias_calendario}")
print()

# 2. Vazios e zeros
nan_unidades = serie_p["unidades"].isna().sum()
pct_nan = nan_unidades / total_dias_idx * 100
zeros_unidades = (serie_p["unidades"] == 0).sum()
pct_zeros = zeros_unidades / total_dias_idx * 100
nao_zero_nao_nan = ((serie_p["unidades"].notna()) & (serie_p["unidades"] > 0)).sum()
print(f"ITEM 2:")
print(f"  Dias com valor vazio (NaN em unidades): {nan_unidades}  ({pct_nan:.4f}%)")
print(f"  Dias com venda = 0 unidades          : {zeros_unidades}  ({pct_zeros:.4f}%)")
print(f"  Dias com venda > 0 unidades          : {nao_zero_nao_nan}  ({nao_zero_nao_nan/total_dias_idx*100:.4f}%)")
print()

# 3. Media de venda diaria e preco medio
venda_media = serie_p["unidades"].mean()
venda_soma_total = serie_p["unidades"].sum()
preco_medio_geral = serie_p["preco_medio"].mean()
preco_medio_ponderado = (
    (serie_p["preco_medio"] * serie_p["unidades"]).sum() / venda_soma_total
)
print(f"ITEM 3:")
print(f"  Venda media por dia (media simples sobre todos os dias, incluindo NaN=0): {venda_media:.4f} unidades/dia")
print(f"  Soma total de unidades no periodo: {venda_soma_total:,.1f}")
print(f"  Preco medio (media simples diaria): R$ {preco_medio_geral:.4f}")
print(f"  Preco medio ponderado por volume  : R$ {preco_medio_ponderado:.4f}")
print()

# Estatisticas descritivas da venda diaria
print("Distribuicao de unidades por dia:")
q = serie_p["unidades"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
q = q.round(3)
for k, v in q.items():
    print(f"  {k:>8}: {v:>10.3f}")
print()

# 4. Plot venda diaria + media movel centrada de 28 dias
mm_28_centrada = serie_p["unidades"].rolling(window=28, center=True, min_periods=14).mean()

sns.set_theme(style="whitegrid", font_scale=1.05)
fig, ax = plt.subplots(figsize=(16, 6.5))

ax.plot(serie_p.index, serie_p["unidades"].values,
        color="#60A5FA", linewidth=0.6, alpha=0.85,
        label="Venda diária")
ax.plot(mm_28_centrada.index, mm_28_centrada.values,
        color="#DC2626", linewidth=2.2, alpha=0.95,
        label="Média móvel centrada de 28 dias")

ax.set_title(f"Produto {PRODUTO} — Venda diária na rede e tendência suavizada (MM28 centrada)",
             fontsize=13, fontweight="bold", loc="left")
ax.set_xlabel("Data", fontsize=11)
ax.set_ylabel("Unidades vendidas na rede", fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

dias_ferias = ax.get_lines()[1]
ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=10)

# Destaque da faixa promocional / pico principal
max_venda = serie_p["unidades"].max()
data_pico = serie_p["unidades"].idxmax()
ax.annotate(f"Pico: {max_venda:.0f} un\n({data_pico.strftime('%d/%m/%Y')})",
            xy=(data_pico, max_venda),
            xytext=(-140, 30), textcoords="offset points",
            fontsize=9, color="#7F1D1D",
            arrowprops=dict(arrowstyle="->", color="#7F1D1D", lw=1.2))

plt.tight_layout()
caminho_plot = os.path.join(PASTA_FIGURAS, f"venda_diaria_{PRODUTO}_MM28centrada.png")
plt.savefig(caminho_plot, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print(f"ITEM 4:")
print(f"  Plot salvo em: {caminho_plot}")
print(f"  Janela MM28 = {mm_28_centrada.notna().sum()} pontos com MM calculada "
      f"(min_periods=14 nas bordas)")
print(f"  Pico diário da série = {max_venda:.0f} unidades em {data_pico.strftime('%d/%m/%Y')}")
print(f"  Min  diário da série = {serie_p['unidades'].min():.0f} unidades")
