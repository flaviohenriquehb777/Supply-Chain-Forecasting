import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

SEED = 42
np.random.seed(SEED)

CAMINHO_REDE = r"data/raw/demanda.csv"
PASTA_FIGURAS = r"reports/figures"
os.makedirs(PASTA_FIGURAS, exist_ok=True)

df = pd.read_csv(CAMINHO_REDE, sep=",", parse_dates=["data"], dayfirst=False)
PRODUTO = "P0131"

serie_p = (
    df[df["produto"] == PRODUTO]
    .set_index("data")
    .sort_index()
    .asfreq("D")
)

media_geral = serie_p["unidades"].mean()
print(f"Produto {PRODUTO}")
print(f"Media geral de venda diaria: {media_geral:.4f} unidades")
print()

# --- Item 1: Venda media por dia da semana + plot ---
dias_semana_pt = [
    "2a-feira", "3a-feira", "4a-feira", "5a-feira",
    "6a-feira", "Sabado", "Domingo"
]

serie_p["dow"] = serie_p.index.dayofweek  # 0=seg ... 6=dom
serie_p["dia_semana_nome"] = serie_p["dow"].map(
    lambda d: dias_semana_pt[d]
)

venda_por_dow = serie_p.groupby("dow")["unidades"].agg([
    "mean", "median", "std", "count"
]).rename(columns={
    "mean": "media", "median": "mediana", "std": "desv_p", "count": "n"
})
venda_por_dow["desv_p_pct_da_media"] = (
    venda_por_dow["desv_p"] / venda_por_dow["media"] * 100
).round(2)
venda_por_dow["nome"] = dias_semana_pt
venda_por_dow = venda_por_dow[["nome", "n", "media", "mediana", "desv_p", "desv_p_pct_da_media"]]

print("ITEM 1: Venda media por dia da semana")
print("=" * 65)
print(venda_por_dow.round(3).to_string())
print()
print("Diferenca percentual em relacao a media geral ({}):".format(
    round(media_geral, 2)))
for d in range(7):
    m = venda_por_dow.loc[d, "media"]
    pct = (m - media_geral) / media_geral * 100
    print(f"  {dias_semana_pt[d]:>10}: {m:>7.2f}  ({pct:+.2f}%)")
print()

# Plot barras
sns.set_theme(style="whitegrid", font_scale=1.08)
cores = ["#93C5FD"] * 5 + ["#F87171", "#DC2626"]  # util azul claro, fim semana vermelho

fig, ax = plt.subplots(figsize=(12, 6.5))
bars = ax.bar(
    venda_por_dow["nome"], venda_por_dow["media"],
    color=cores, edgecolor="white", linewidth=1.2, width=0.6
)

# Linha horizontal da media geral
ax.axhline(media_geral, color="#0F172A", linestyle="--", linewidth=2, alpha=0.85,
           label=f"Média geral: {media_geral:.1f} un/dia")

# Rotulos acima das barras
for bar, val in zip(bars, venda_por_dow["media"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.0f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#1E293B")

# Diferencas percentual no rodapé da barra
for bar, d in zip(bars, range(7)):
    m = venda_por_dow.loc[d, "media"]
    pct = (m - media_geral) / media_geral * 100
    rotulo = f"{pct:+.0f}%"
    ax.text(bar.get_x() + bar.get_width() / 2, 4, rotulo,
            ha="center", va="bottom", fontsize=9, color="white",
            fontweight="bold")

ax.set_ylabel("Média de unidades vendidas", fontsize=11)
ax.set_title(f"Produto {PRODUTO} — Venda média por dia da semana",
             fontsize=13, fontweight="bold", loc="left")
ax.legend(loc="upper left", frameon=True, fancybox=True)
ax.set_ylim(0, max(venda_por_dow["media"]) * 1.18)

caminho_plot1 = os.path.join(PASTA_FIGURAS, f"venda_media_dia_semana_{PRODUTO}.png")
plt.tight_layout()
plt.savefig(caminho_plot1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Plot salvo em: {caminho_plot1}")
print()

# --- Item 2: Media dias uteis vs sabado vs domingo vs fim de semana ---
print("ITEM 2: Comparação dia útil vs fim de semana")
print("=" * 65)

serie_p["eh_util"] = serie_p["dow"] <= 4  # seg=0 ... sex=4
serie_p["grupo_dia"] = serie_p["dow"].map(lambda d:
    "Dias úteis (seg-sex)" if d <= 4 else
    "Sábado" if d == 5 else "Domingo"
)

grupo_dia = serie_p.groupby("grupo_dia")["unidades"].agg([
    "mean", "median", "std", "count"
]).reindex(["Dias úteis (seg-sex)", "Sábado", "Domingo"])
grupo_dia["razao_vs_uteis"] = grupo_dia["mean"] / grupo_dia.loc["Dias úteis (seg-sex)", "mean"]
grupo_dia = grupo_dia.round(4)
print(grupo_dia.to_string())
print()

media_uteis = grupo_dia.loc["Dias úteis (seg-sex)", "mean"]
media_sab = grupo_dia.loc["Sábado", "mean"]
media_dom = grupo_dia.loc["Domingo", "mean"]
media_fds = serie_p.loc[serie_p["dow"] >= 5, "unidades"].mean()

print(f"  Media dias úteis     : {media_uteis:.2f}")
print(f"  Media sábado         : {media_sab:.2f}  ({media_sab/media_uteis:.2f}x o dia útil)")
print(f"  Media domingo        : {media_dom:.2f}  ({media_dom/media_uteis:.2f}x o dia útil)")
print(f"  Media fim de semana  : {media_fds:.2f}  ({media_fds/media_uteis:.2f}x o dia útil, média simples sab+dom)")
print()

# --- Item 3: Ciclo anual (resposta em texto + indicativo numerico) ---
print("ITEM 3: Indícios de ciclo anual")
print("=" * 65)
n_meses_completos = serie_p.resample("ME")["unidades"].mean()
print(f"Numero de meses no historico: {len(n_meses_completos)} meses  "
      f"({serie_p.index[0].strftime('%b/%y')} a {serie_p.index[-1].strftime('%b/%y')})")
print(f"Anos completos fechados: 2 (2017 e 2018). 2019 vai só até outubro.")
print()

# Media por mes do ano (jan a dez) combinando todos os anos
serie_p["mes"] = serie_p.index.month
mes_nomes = ["jan", "fev", "mar", "abr", "mai", "jun",
             "jul", "ago", "set", "out", "nov", "dez"]
venda_por_mes_ano = serie_p.groupby(["mes", serie_p.index.year])["unidades"].mean().unstack()
venda_por_mes_ano.index = [mes_nomes[i - 1] for i in venda_por_mes_ano.index]

media_mes = serie_p.groupby("mes")["unidades"].mean()
media_mes.index = mes_nomes
pct_mes_vs_media_geral = (media_mes - media_geral) / media_geral * 100

df_mes = pd.DataFrame({
    "media_diaria_por_mes": media_mes.round(2),
    "%_vs_media_geral": pct_mes_vs_media_geral.round(2)
})
print("Media diaria por mês do ano (todos os anos combinados):")
print(df_mes.to_string())
print()

# Serie mensal: media mensal (jan/17 a out/19) para visualizar ciclo
fig, ax = plt.subplots(figsize=(14, 5.5))
n_meses_completos.index = [d.strftime("%b/%y") for d in n_meses_completos.index]
ax.plot(n_meses_completos.index, n_meses_completos.values,
        marker="o", color="#2563EB", linewidth=1.8, markersize=4,
        label="Média mensal de unidades/dia")

# Marca os mesmos meses dos anos diferentes para ver o ciclo
marcadores_anos = {"2017": "o", "2018": "s", "2019": "^"}
ax.set_title(f"Produto {PRODUTO} — Venda média mensal (ciclo anual × tendência)",
             fontsize=13, fontweight="bold", loc="left")
ax.set_ylabel("Média de unidades/dia no mês")
ax.set_xlabel("Mês / Ano")
ax.axhline(media_geral, color="#7F1D1D", linestyle=":", linewidth=1.6,
           label=f"Média geral: {media_geral:.0f} un/dia")
plt.xticks(rotation=45, ha="right")
ax.legend(loc="upper left", frameon=True, fancybox=True)
plt.tight_layout()
caminho_plot2 = os.path.join(PASTA_FIGURAS, f"serie_mensal_{PRODUTO}_ciclo_anual.png")
plt.savefig(caminho_plot2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Plot mensal salvo em: {caminho_plot2}")
