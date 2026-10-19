"""
FASE 5/6 — Simulação de CUSTO TOTAL (Critério 3 da Fase 1).

Para cada modelo:
  1. Estoque de segurança = quantil empírico dos erros de período (nível de serviço).
  2. Pedido do período = previsão + estoque de segurança.
  3. Compara com o realizado de cada uma das 12 origens.
  4. Custo por período = falta x Cu + sobra x Co.

Convenção: erro = realizado - previsto (>0 faltou produto).
Pedido = previsto + SS  =>  falta = max(0, erro - SS), sobra = max(0, SS - erro).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Caminhos ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "reports", "fase5_backtest_origens_P0131.csv")
FIG_DIR = os.path.join(BASE, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Parâmetros de custo (Fase 1) ───────────────────────────────────────────────
PRECO_MEDIO = 2.5870
CU = 0.30 * PRECO_MEDIO      # custo de faltar uma unidade  = R$ 0,7761
CO = 0.02 * PRECO_MEDIO      # custo de sobrar uma unidade  = R$ 0,0517
NIVEL_SERVICO = 0.9375
PERIODOS_ANO = 13            # 13 ciclos de 28 dias por ano


# ── Função de simulação ────────────────────────────────────────────────────────
def simular_custo(erros, nivel_servico=NIVEL_SERVICO, cu=CU, co=CO):
    """
    Dada a distribuição de erros de período de um modelo e o nível de serviço:
      - calcula o estoque de segurança (quantil empírico);
      - monta o pedido (previsão + SS) e compara com o realizado de cada origem;
      - devolve (estoque_seguranca, custo_medio_por_periodo, unidades_faltantes_medias).
    """
    erros = np.asarray(erros, dtype=float)

    # 1. estoque de segurança a partir do erro do próprio modelo
    ss = float(np.quantile(erros, nivel_servico))

    # 2. pedido = previsão + SS  ->  falta e sobra por origem
    falta = np.maximum(0.0, erros - ss)      # unidades que faltaram
    sobra = np.maximum(0.0, ss - erros)      # unidades que sobraram

    # 3. custo médio por período e unidades faltantes médias
    custo_medio = float(np.mean(falta * cu + sobra * co))
    falta_media = float(np.mean(falta))

    return ss, custo_medio, falta_media


# ── Carrega erros do backtest ──────────────────────────────────────────────────
df = pd.read_csv(CSV)
df["origem"] = pd.to_datetime(df["origem"])

MODELOS = {
    "Média Móvel 28d": "MM28",
    "Holt-Winters": "Holt-Winters",
    "SARIMAX+promo": "SARIMAX+promo",
}

resultados = []
for nome, chave in MODELOS.items():
    erros = df.loc[df["modelo"] == chave, "erro_total_real_minus_prev"].values
    ss, custo, falta = simular_custo(erros)
    resultados.append({
        "Modelo": nome,
        "chave": chave,
        "Estoque_seguranca_un": ss,
        "Custo_medio_periodo_R$": custo,
        "Falta_media_un": falta,
        "erros": erros,
    })

res = pd.DataFrame(resultados).sort_values("Custo_medio_periodo_R$").reset_index(drop=True)

print("=" * 84)
print(f"SIMULAÇÃO DE CUSTO TOTAL — P0131  (nível de serviço {NIVEL_SERVICO*100:.2f}%, "
      f"Cu=R$ {CU:.4f}, Co=R$ {CO:.4f})")
print("=" * 84)
print()
print(f"{'#':<3} {'Modelo':<18} {'Est.Seg.(un)':>13} {'Custo/período':>14} {'Falta média(un)':>16}")
print("-" * 84)
for i, r in res.iterrows():
    print(f"{i+1:<3} {r['Modelo']:<18} {r['Estoque_seguranca_un']:>13.1f} "
          f"{'R$ '+format(r['Custo_medio_periodo_R$'], '.2f'):>14} "
          f"{r['Falta_media_un']:>16.1f}")
print()

# ── Gráficos lado a lado ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
cores = ["#2ca02c", "#1f77b4", "#d62728"]  # já ordenado por custo
nomes = res["Modelo"].tolist()

bars1 = ax1.bar(nomes, res["Custo_medio_periodo_R$"], color=cores, alpha=0.85, edgecolor="black")
ax1.set_title("Custo total por período (28 dias)\nmenor é melhor", fontsize=13, fontweight="bold")
ax1.set_ylabel("R$ por período", fontsize=12)
ax1.grid(True, axis="y", alpha=0.3)
for b, v in zip(bars1, res["Custo_medio_periodo_R$"]):
    ax1.text(b.get_x() + b.get_width()/2, v + 2, f"R$ {v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax1.set_ylim(0, res["Custo_medio_periodo_R$"].max() * 1.2)

bars2 = ax2.bar(nomes, res["Estoque_seguranca_un"], color=cores, alpha=0.85, edgecolor="black")
ax2.set_title("Estoque de segurança exigido\n(quantil 93,75% do erro do modelo)", fontsize=13, fontweight="bold")
ax2.set_ylabel("unidades por período", fontsize=12)
ax2.grid(True, axis="y", alpha=0.3)
for b, v in zip(bars2, res["Estoque_seguranca_un"]):
    ax2.text(b.get_x() + b.get_width()/2, v + 20, f"{v:.1f} un", ha="center", fontsize=11, fontweight="bold")
ax2.set_ylim(0, res["Estoque_seguranca_un"].max() * 1.2)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "fase5_custo_estoque_seguranca_por_modelo.png")
plt.savefig(fig_path, dpi=150)
print(f"Gráfico salvo: reports/figures/fase5_custo_estoque_seguranca_por_modelo.png")
print()

# ── Comparação direta com o processo atual (baseline MM28) ─────────────────────
base = res[res["Modelo"] == "Média Móvel 28d"].iloc[0]

print("=" * 96)
print("COMPARAÇÃO DIRETA COM O PROCESSO ATUAL (Média Móvel 28d)")
print("=" * 96)
print()
for nome in ["SARIMAX+promo", "Holt-Winters"]:
    r = res[res["Modelo"] == nome].iloc[0]
    econ_periodo = base["Custo_medio_periodo_R$"] - r["Custo_medio_periodo_R$"]
    econ_ano = econ_periodo * PERIODOS_ANO
    var_custo_pct = econ_periodo / base["Custo_medio_periodo_R$"] * 100
    print(f"MODELO: {nome}")
    print(f"  Custo por período ....... antes R$ {base['Custo_medio_periodo_R$']:.2f}  ->  depois R$ {r['Custo_medio_periodo_R$']:.2f}"
          f"   (redução de {var_custo_pct:+.1f}%)")
    print(f"  Estoque de segurança ..... antes {base['Estoque_seguranca_un']:.1f} un  ->  depois {r['Estoque_seguranca_un']:.1f} un"
          f"   ({r['Estoque_seguranca_un'] - base['Estoque_seguranca_un']:+.1f} un)")
    print(f"  Falta média por período .. antes {base['Falta_media_un']:.1f} un  ->  depois {r['Falta_media_un']:.1f} un")
    print(f"  Economia por período ..... R$ {econ_periodo:.2f}")
    print(f"  ECONOMIA ANUAL (13 períodos) .. R$ {econ_ano:,.2f}  neste único produto")
    print()

print("=" * 96)
print(f"Obs.: economia anual = (custo MM28 - custo modelo) x {PERIODOS_ANO} períodos de 28 dias.")
