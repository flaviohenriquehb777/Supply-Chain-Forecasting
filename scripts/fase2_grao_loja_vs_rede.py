import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

SEED = 42
np.random.seed(SEED)

CAMINHO_LOJA = r"data/raw/demanda_por_loja.csv"
CAMINHO_REDE = r"data/raw/demanda.csv"
PASTA_FIGURAS = r"reports/figures"
os.makedirs(PASTA_FIGURAS, exist_ok=True)

df_loja = pd.read_csv(CAMINHO_LOJA, sep=",", parse_dates=["data"], dayfirst=False)
df_rede = pd.read_csv(CAMINHO_REDE, sep=",", parse_dates=["data"], dayfirst=False)

PRODUTO = "P0131"

print(f"Analise do produto {PRODUTO} no demanda_por_loja.csv")
print("=" * 65)

df_p = df_loja[df_loja["produto"] == PRODUTO].copy()
total_linhas_p = len(df_p)
n_lojas = df_p["loja"].nunique()
n_dias = df_p["data"].nunique()
print(f"Linhas na base loja para {PRODUTO}: {total_linhas_p:,}")
print(f"Numero de lojas: {n_lojas}")
print(f"Numero de dias: {n_dias}")
print(f"Esperado (lojas x dias): {n_lojas * n_dias:,}")
print()

# 1. Media diaria por loja (como serie: media da distribuicao loja-dia)
media_por_dia_por_loja = df_p["unidades"].mean()
print(f"ITEM 1: Media de unidades vendidas (loja-dia): {media_por_dia_por_loja:.4f}")

# 2. Mediana
mediana_por_dia_por_loja = df_p["unidades"].median()
print(f"ITEM 2: Mediana de unidades vendidas (loja-dia): {mediana_por_dia_por_loja:.1f}")

# 3. Percentual de zeros na serie loja-dia
qtde_zeros = (df_p["unidades"] == 0).sum()
pct_zeros = (qtde_zeros / total_linhas_p) * 100
print(f"ITEM 3: Percentual de dias com 0 unidades na loja: {pct_zeros:.4f}% ({qtde_zeros:,} de {total_linhas_p:,})")
print()

# Estatisticas adicionais para contexto
q25 = df_p["unidades"].quantile(0.25)
q75 = df_p["unidades"].quantile(0.75)
q90 = df_p["unidades"].quantile(0.90)
q99 = df_p["unidades"].quantile(0.99)
max_val = df_p["unidades"].max()
print("Distribuicao completa de unidades (loja-dia):")
print(f"  min:    {df_p['unidades'].min():.0f}")
print(f"  Q25:    {q25:.0f}")
print(f"  mediana:{mediana_por_dia_por_loja:.0f}")
print(f"  Q75:    {q75:.0f}")
print(f"  Q90:    {q90:.0f}")
print(f"  Q99:    {q99:.0f}")
print(f"  max:    {max_val:.0f}")
print(f"  desv_p: {df_p['unidades'].std():.4f}")
print()

# Media por loja (como serie: media da loja sobre todos os dias, depois media das lojas)
loja_media = df_p.groupby("loja")["unidades"].mean()
print(f"Media POR LOJA (media de cada loja em todos os dias, depois media das lojas): {loja_media.mean():.4f}")
print(f"Mediana da media POR LOJA: {loja_media.median():.4f}")
print(f"Desvio padrao entre as medias das lojas: {loja_media.std():.4f}")
print()

# 4. Plot serie diaria loja individual vs serie na rede
df_p_rede = df_rede[df_rede["produto"] == PRODUTO].set_index("data").sort_index()

# Escolhe uma loja: a que tem a media de vendas mais proxima da mediana das medias
media_mediana = loja_media.median()
loja_escolhida = (loja_media - media_mediana).abs().idxmin()
print(f"Loja escolhida para plot: {loja_escolhida} (media diaria = {loja_media[loja_escolhida]:.2f}, prox a mediana das lojas {media_mediana:.2f})")

df_loja_escolhida = df_p[df_p["loja"] == loja_escolhida].set_index("data").sort_index()

serie_loja = df_loja_escolhida["unidades"].reindex(df_p_rede.index, fill_value=0.0)
serie_rede = df_p_rede["unidades"]

# Faz o plot em 2 subplots empilhados (mesmo periodo, escalas diferentes)
sns.set_theme(style="whitegrid", font_scale=1.05)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

# Subplot 1: loja individual
ax1.plot(serie_loja.index, serie_loja.values,
         color="#2563EB", linewidth=0.9, alpha=0.9,
         label=f"Loja {loja_escolhida}")
ax1.set_ylabel("Unidades vendidas", fontsize=11)
ax1.set_title(f"Produto {PRODUTO} — Série diária: Loja individual {loja_escolhida}",
              fontsize=12, fontweight="bold", loc="left")
ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
ax1.legend(loc="upper right", frameon=True, fancybox=True)

# Subplot 2: serie agregada na rede toda
ax2.plot(serie_rede.index, serie_rede.values,
         color="#DC2626", linewidth=0.9, alpha=0.9,
         label="Rede toda (144 lojas, soma)")
ax2.set_ylabel("Unidades vendidas (soma rede)", fontsize=11)
ax2.set_xlabel("Data", fontsize=11)
ax2.set_title(f"Produto {PRODUTO} — Série diária agregada na rede toda",
              fontsize=12, fontweight="bold", loc="left")
ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
ax2.legend(loc="upper right", frameon=True, fancybox=True)

plt.tight_layout(h_pad=3)
caminho_plot = os.path.join(PASTA_FIGURAS, f"serie_comparativa_{PRODUTO}_loja_vs_rede.png")
plt.savefig(caminho_plot, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# Segunda visualizacao: zoom em 90 dias para ver a granularidade
data_fim_zoom = df_p_rede.index.max()
data_inicio_zoom = data_fim_zoom - pd.Timedelta(days=90)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

idx_zoom = (serie_loja.index >= data_inicio_zoom) & (serie_loja.index <= data_fim_zoom)
markerline, stemlines, baseline = ax1.stem(
    serie_loja.index[idx_zoom], serie_loja.values[idx_zoom],
    basefmt=" ", linefmt="#2563EB"
)
plt.setp(markerline, markersize=3, color="#2563EB", marker="o")
ax1.set_ylabel("Unidades vendidas", fontsize=11)
ax1.set_title(f"Produto {PRODUTO} — Últimos 90 dias: Loja {loja_escolhida}",
              fontsize=12, fontweight="bold", loc="left")
ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
ax1.legend([markerline], [f"Loja {loja_escolhida}"], loc="upper right", frameon=True)

idx_zoom2 = (serie_rede.index >= data_inicio_zoom) & (serie_rede.index <= data_fim_zoom)
ax2.plot(serie_rede.index[idx_zoom2], serie_rede.values[idx_zoom2],
         color="#DC2626", linewidth=1.2, marker="o", markersize=3,
         label="Rede toda (soma)")
ax2.set_ylabel("Unidades vendidas (soma rede)", fontsize=11)
ax2.set_xlabel("Data", fontsize=11)
ax2.set_title(f"Produto {PRODUTO} — Últimos 90 dias: Rede toda",
              fontsize=12, fontweight="bold", loc="left")
ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
ax2.legend(loc="upper right", frameon=True)

plt.tight_layout(h_pad=3)
caminho_plot2 = os.path.join(PASTA_FIGURAS, f"serie_comparativa_{PRODUTO}_loja_vs_rede_90dias.png")
plt.savefig(caminho_plot2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print()
print(f"Plots salvos em:")
print(f"  - 1: {caminho_plot}")
print(f"  - 2: {caminho_plot2}")
