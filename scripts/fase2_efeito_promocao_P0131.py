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

PRODUTO = "P0131"
LIMIAR_PROMO = 0.20  # 20% das lojas
LIMIAR_NAO_PROMO = 0.02  # 2% das lojas (quase nenhuma loja em promo)

df = pd.read_csv(CAMINHO_REDE, sep=",", parse_dates=["data"], dayfirst=False)
df_p = df[df["produto"] == PRODUTO].copy().sort_values("data").reset_index(drop=True)

print("DECISÃO REGISTRADA PARA FASE 3")
print("=" * 70)
print("Produtos que saem do painel (razão < 0,6):")
for p in ["P0438", "P0051", "P0079"]:
    print(f"  - {p}")
print("Filtro NÃO aplicado ainda — decisão somente aplicada na FASE 3.")
print()

print(f"PRODUTO {PRODUTO}")
print("=" * 70)

# --- Item 1: Plot venda diaria destacando dias com pct_lojas_promo > LIMIAR_PROMO ---
mask_promo = df_p["pct_lojas_promo"] > LIMIAR_PROMO
dias_promo = df_p.loc[mask_promo, "data"]
venda_promo = df_p.loc[mask_promo, "unidades"]
dias_não_promo = df_p.loc[~mask_promo, "data"]
venda_não_promo = df_p.loc[~mask_promo, "unidades"]

sns.set_theme(style="whitegrid", font_scale=1.05)
fig, ax = plt.subplots(figsize=(16, 6.5))

ax.plot(df_p["data"], df_p["unidades"].values, color="#93C5FD", linewidth=0.55,
        alpha=0.9, label="Venda diária", zorder=1)
ax.scatter(dias_promo, venda_promo.values, s=36, color="#DC2626", edgecolors="#7F1D1D",
           linewidths=0.5, alpha=0.92, zorder=3,
           label=f"Dias com >{int(LIMIAR_PROMO*100)}% das lojas em promoção (n = {mask_promo.sum()})")
ax.scatter(dias_não_promo, venda_não_promo.values, s=14, color="#1E40AF",
           alpha=0.45, zorder=2,
           label=f"Dias sem promo expressiva (n = {(~mask_promo).sum()})")

ax.set_title(f"Produto {PRODUTO} — Venda diária e dias de promoção expressiva"
               f" (>{int(LIMIAR_PROMO*100)}% das lojas em promo)",
               fontsize=13, fontweight="bold", loc="left")
ax.set_ylabel("Unidades vendidas na rede", fontsize=11)
ax.set_xlabel("Data", fontsize=11)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=9.5, ncol=1)

caminho_plot1 = os.path.join(PASTA_FIGURAS, f"venda_{PRODUTO}_destacando_promocao.png")
plt.tight_layout()
plt.savefig(caminho_plot1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ZOOM 90 dias ao redor do grande pico (abril/2019) para melhor visualização
# do efeito promocional.
data_ref_pico = df_p.loc[df_p["unidades"].idxmax(), "data"]
inicio_zoom = data_ref_pico - pd.Timedelta(days=60)
fim_zoom = data_ref_pico + pd.Timedelta(days=30)

mask_zoom = (df_p["data"] >= inicio_zoom) & (df_p["data"] <= fim_zoom)
zoom = df_p[mask_zoom].copy()

fig, ax1 = plt.subplots(figsize=(15, 6.2))
mask_promo_z = zoom["pct_lojas_promo"] > LIMIAR_PROMO
ax1.plot(zoom["data"], zoom["unidades"], color="#2563EB", linewidth=1.1, marker="o", markersize=3,
          label="Venda diária")
ax1.scatter(zoom.loc[mask_promo_z, "data"], zoom.loc[mask_promo_z, "unidades"],
            s=55, color="#DC2626", edgecolors="#7F1D1D", zorder=5,
            label=f">20% das lojas em promo")
ax1.set_ylabel("Unidades vendidas (escala esquerda)", color="#2563EB", fontsize=10.5)
ax1.tick_params(axis="y", labelcolor="#2563EB")
ax1.set_title(f"Produto {PRODUTO} — Zoom de 90 dias em torno do pico máximo (promoção abril/2019)",
               fontsize=12, fontweight="bold", loc="left")
ax2 = ax1.twinx()
ax2.plot(zoom["data"], zoom["pct_lojas_promo"] * 100, color="#16A34A", linewidth=1.4, linestyle="--",
         label="% lojas em promoção (escala direita)")
ax2.set_ylabel("% lojas em promoção (escala direita)", color="#16A34A", fontsize=10.5)
ax2.tick_params(axis="y", labelcolor="#16A34A")
ax2.set_ylim(-3, 105)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

linhas1, rot1 = ax1.get_legend_handles_labels()
linhas2, rot2 = ax2.get_legend_handles_labels()
ax1.legend(linhas1 + linhas2, rot1 + rot2, loc="upper left", frameon=True, fancybox=True, fontsize=9.5)

ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b"))
plt.setp(ax1.get_xticklabels(), rotation=25, ha="right")

caminho_plot2 = os.path.join(PASTA_FIGURAS, f"zoom_{PRODUTO}_picopromocao_abril2019.png")
plt.tight_layout()
plt.savefig(caminho_plot2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print(f"ITEM 1: Plots salvos:")
print(f"  - Série completa destacando promoções: {caminho_plot1}")
print(f"  - Zoom de 90 dias no pico com % lojas em promo (eixo duplo): {caminho_plot2}")
print()

# --- Item 2: Comparativo venda média (promo >20% vs promoção <= 2%
grupo_promo_forte = df_p[df_p["pct_lojas_promo"] > LIMIAR_PROMO]["unidades"]
grupo_sem_promo = df_p[df_p["pct_lojas_promo"] <= LIMIAR_NAO_PROMO]["unidades"]

media_promo_forte = grupo_promo_forte.mean()
media_sem_promo = grupo_sem_promo.mean()
razao_promo = media_promo_forte / media_sem_promo
mediana_promo = grupo_promo_forte.median()
mediana_sem_promo = grupo_sem_promo.median()
n_promo = len(grupo_promo_forte)
n_sem = len(grupo_sem_promo)
total_dias = len(df_p)
dias_intermediarios = total_dias - n_promo - n_sem

print(f"ITEM 2: Comparação de venda média: dias com >{int(LIMIAR_PROMO * 100)}% lojas em promoção")
print("       vs. dias com <={int(LIMIAR_NAO_PROMO * 100)}% lojas em promoção (quase sem promo)")
print("-" * 70)
print(f"  N dias com >{int(LIMIAR_PROMO*100)}% promo     : {n_promo:>4} de {total_dias} ({n_promo/total_dias*100:.2f}% dos dias)")
print(f"  N dias com <={int(LIMIAR_NAO_PROMO*100)}% promo    : {n_sem:>4} de {total_dias} ({n_sem/total_dias*100:.2f}% dos dias)")
print(f"  Dias intermediários (entre {int(LIMIAR_NAO_PROMO*100)}% e {int(LIMIAR_PROMO*100)}%) : {dias_intermediarios:>4} ({dias_intermediarios/total_dias*100:.2f}%)")
print()
print(f"  Média de venda (promo forte)     : {media_promo_forte:>8.2f} un/dia")
print(f"  Média de venda (sem promo)      : {media_sem_promo:>8.2f} un/dia")
print(f"  Razão (promo / sem promo)    : {razao_promo:>8.3f}x")
print(f"  Mediana (promo forte)        : {mediana_promo:>8.1f} un/dia")
print(f"  Mediana (sem promo)         : {mediana_sem_promo:>8.1f} un/dia")
print(f"  Razão mediana (promo/sem)      : {mediana_promo/mediana_sem_promo:>8.3f}x")
print()
print(f"  ⇒ A promoção multiplica a venda média diária em aproximadamente {razao_promo:.2f}x")
print(f"  (em mais de {int(LIMIAR_PROMO*100)}% das lojas) contra dias sem promoção.")
print()

# Distribuição por faixas de % de lojas em promoção (5 faixas)
print("  Bonus: Venda média por faixa de % lojas em promoção")
print("-" * 70)
faixas = [-0.001, 0.02, 0.20, 0.50, 0.80, 1.001]
rotulos_faixas = ["0–2%", "2%–20%", "20%–50%", "50%–80%", "80%–100%"]
df_p["faixa_promo"] = pd.cut(df_p["pct_lojas_promo"], bins=faixas, labels=rotulos_faixas, include_lowest=True)
tabela_faixas = df_p.groupby("faixa_promo", observed=False)["unidades"].agg([
    "count", "mean", "median", "min", "max"
]).round(2)
tabela_faixas.columns = ["n_dias", "media", "mediana", "min", "max"]
print(tabela_faixas.to_string())
