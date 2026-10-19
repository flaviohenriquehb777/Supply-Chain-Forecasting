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

# Monta painel wide: indice data, colunas = produtos, valores = unidades
painel = (
    df.pivot_table(index="data", columns="produto", values="unidades", aggfunc="sum")
    .sort_index()
    .asfreq("D")  # frequencia diaria; lacunas (se houver) viram NaN
)

data_max = painel.index.max()
fim_janela_recente = data_max
inicio_janela_recente = data_max - pd.Timedelta(days=89)  # 90 dias inclusivo: 89 + 1 = 90

fim_janela_anterior = data_max - pd.Timedelta(days=90)
inicio_janela_anterior = data_max - pd.Timedelta(days=454)  # 455 dias inclusivo: 454 + 1 = 455

print(f"Data maxima da base: {data_max.strftime('%d/%m/%Y')}")
print(f"Janela RECENTE (ultimos 90 dias): "
      f"{inicio_janela_recente.strftime('%d/%m/%Y')} a {fim_janela_recente.strftime('%d/%m/%Y')}")
print(f"Janela ANTERIOR (455 a 90 dias atras): "
      f"{inicio_janela_anterior.strftime('%d/%m/%Y')} a {fim_janela_anterior.strftime('%d/%m/%Y')}")
print()

# --- Item 1: Calculo da razao por produto ---
resultados = []
for produto in painel.columns:
    serie = painel[produto]
    serie_recente = serie.loc[inicio_janela_recente:fim_janela_recente]
    serie_anterior = serie.loc[inicio_janela_anterior:fim_janela_anterior]
    media_recente = serie_recente.mean(skipna=True)
    media_anterior = serie_anterior.mean(skipna=True)
    n_recente_valido = serie_recente.notna().sum()
    n_anterior_valido = serie_anterior.notna().sum()
    razao = media_recente / media_anterior if media_anterior > 0 else np.nan
    resultados.append({
        "produto": produto,
        "media_recente_90d": round(media_recente, 4),
        "media_anterior_455_90d": round(media_anterior, 4),
        "razao_recente_sobre_anterior": round(razao, 6) if not np.isnan(razao) else np.nan,
        "n_dias_validos_recente": n_recente_valido,
        "n_dias_validos_anterior": n_anterior_valido,
    })

df_triagem = pd.DataFrame(resultados).sort_values(
    "razao_recente_sobre_anterior", ascending=True, na_position="first"
).reset_index(drop=True)

# Marca a linha do produto foco
foco = "P0131"
df_triagem.insert(1, "eh_foco", df_triagem["produto"] == foco)
formato_flag = lambda x: "← Foco P0131" if x else ""
df_triagem["eh_foco"] = df_triagem["eh_foco"].map(formato_flag)

print("ITEM 1: Tabela completa de razão (média recente 90d) / (média do período 455–90 dias atrás)")
print("         Ordenada por razão CRESCENTE (menores razões no topo)")
print("=" * 90)
print(df_triagem.round(4).to_string(index=False))
print()

n_total = len(df_triagem)
print("6 MENORES razões (risco de produto morto / descontinuado):")
print("-" * 90)
print(df_triagem[["produto", "media_recente_90d", "media_anterior_455_90d",
                   "razao_recente_sobre_anterior", "n_dias_validos_recente",
                   "n_dias_validos_anterior", "eh_foco"]].head(6).to_string(index=False))
print()
print("3 MAIORES razões (produto crescendo):")
print("-" * 90)
print(df_triagem[["produto", "media_recente_90d", "media_anterior_455_90d",
                   "razao_recente_sobre_anterior", "n_dias_validos_recente",
                   "n_dias_validos_anterior", "eh_foco"]].tail(3).iloc[::-1].to_string(index=False))
print()
print(f"Produto foco P0131 — posicao no ranking (1 menor a {n_total} maior): "
      f"{int(df_triagem.reset_index().query(f'produto == \"{foco}\"').index[0]) + 1}")
print()

# --- Item 2: Plote media mensal dos produtos com razao < 0,6 ---
LIMIAR = 0.6
produtos_criticos = df_triagem.loc[
    df_triagem["razao_recente_sobre_anterior"] < LIMIAR, "produto"
].tolist()
print(f"ITEM 2: Produtos com razão < {LIMIAR} ({len(produtos_criticos)} produto(s)):")
for p in produtos_criticos:
    r = df_triagem.loc[df_triagem["produto"] == p, "razao_recente_sobre_anterior"].iloc[0]
    mr = df_triagem.loc[df_triagem["produto"] == p, "media_recente_90d"].iloc[0]
    ma = df_triagem.loc[df_triagem["produto"] == p, "media_anterior_455_90d"].iloc[0]
    print(f"  - {p}: razão = {r:.4f}  |  média recente = {mr:.2f}  |  média anterior = {ma:.2f}")
print()

if len(produtos_criticos) == 0:
    print("Nenhum produto abaixo do limiar. Nada a plotar.")
else:
    # Media mensal para cada produto critico, mantendo como serie temporal
    painel_mensal = (
        painel[produtos_criticos]
        .resample("ME")
        .mean()
    )

    # Formata eixo x como "Mes/Ano"
    eixo_x = [d.strftime("%b/%y") for d in painel_mensal.index]

    sns.set_theme(style="whitegrid", font_scale=1.03)
    fig, ax = plt.subplots(figsize=(15, 6.2))

    paleta = sns.color_palette("tab10", n_colors=len(produtos_criticos))
    for p, cor in zip(produtos_criticos, paleta):
        media_ultimo = painel_mensal[p].iloc[-1]
        media_terico = df_triagem.loc[df_triagem["produto"] == p, "media_anterior_455_90d"].iloc[0]
        ax.plot(eixo_x, painel_mensal[p].values, marker="o", linewidth=1.7,
                markersize=3.2, color=cor,
                label=f"{p}  (razão = {df_triagem.loc[df_triagem['produto']==p,'razao_recente_sobre_anterior'].iloc[0]:.3f})")

    # Linha vertical do fim do periodo "anterior" (90 dias antes do fim)
    data_limite_ref = fim_janela_anterior
    pos_limite = np.searchsorted(painel_mensal.index, data_limite_ref)
    if 0 < pos_limite < len(eixo_x):
        ax.axvline(x=pos_limite - 0.5, color="#0F172A", linestyle="--", linewidth=1.5,
                   alpha=0.75, label=f"Início janela recente (90d antes do fim: {data_limite_ref.strftime('%d/%m/%Y')})")

    ax.set_title(f"Produtos com razão (recente / anterior) < {LIMIAR} — Média mensal de unidades/dia",
                 fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel("Mês / Ano", fontsize=11)
    ax.set_ylabel("Unidades/dia (média mensal)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    # Apenas um rotulo por mes para nao poluir
    ax.xaxis.set_major_locator(mticker := plt.MaxNLocator(nbins=min(15, len(eixo_x))))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", frameon=True, fancybox=True,
              fontsize=9, ncol=min(3, max(1, len(produtos_criticos))))

    plt.tight_layout()
    caminho_plot = os.path.join(
        PASTA_FIGURAS,
        f"triagem_produtos_razao_abaixo_{str(LIMIAR).replace('.', '_')}.png"
    )
    plt.savefig(caminho_plot, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Plot salvo em: {caminho_plot}")

print()
# --- Item 3: Descreve as curvas (apenas numeros, nao especula) ---
print("ITEM 3: Descrição numérica das curvas dos produtos com razão < 0,6")
print("-" * 90)
for p in produtos_criticos:
    # Janelas: primeiro ano completo vs ultimo semestre vs ultimos 30d
    s = painel[p].dropna()
    if len(s) < 90:
        continue
    # Pico historico vs patamar ultimos 90d
    media_90 = s.iloc[-90:].mean()
    media_1ano_ini = s.iloc[:365].mean() if len(s) >= 365 else s.mean()
    max_historico = s.max()
    qtd_dias_zero_90 = (s.iloc[-90:] == 0).sum()
    pct_zero_90 = qtd_dias_zero_90 / 90 * 100
    print(f"  {p}:")
    print(f"    - Max historico de venda/dia          : {max_historico:.0f}")
    print(f"    - Media diária 1º ano do histórico    : {media_1ano_ini:.2f}")
    print(f"    - Média diária últimos 90 dias        : {media_90:.2f}")
    print(f"    - % de dias com ZERO nos últimos 90d  : {pct_zero_90:.1f}% ({qtd_dias_zero_90}/90)")
