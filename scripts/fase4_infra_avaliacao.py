import pandas as pd
import numpy as np
import os

SEED = 42
np.random.seed(SEED)

CAMINHO_REDE = r"data/raw/demanda.csv"
H = 28
PRODUTO_FOCO = "P0131"
PRODUTOS_REMOVER = ["P0438", "P0051", "P0079"]

# --- Carregamento e filtro idênticos à FASE 3 (garante estado) ---
df = pd.read_csv(CAMINHO_REDE, sep=",", parse_dates=["data"], dayfirst=False)
df_limpo = df[~df["produto"].isin(PRODUTOS_REMOVER)].copy().reset_index(drop=True)

# Serie do produto foco
serie_p = (
    df_limpo[df_limpo["produto"] == PRODUTO_FOCO]
    .sort_values("data")
    .set_index("data")
    .asfreq("D")
)

data_maxima = serie_p.index.max()
data_corte = data_maxima - pd.Timedelta(days=H)
data_inicio_teste = data_corte + pd.Timedelta(days=1)
treino_p = serie_p.loc[:data_corte]
teste_p = serie_p.loc[data_inicio_teste:]
real = teste_p["unidades"].astype(float).values
indice_teste = teste_p.index

# ==============================================================================
# INFRAESTRUTURA DE AVALIAÇÃO — FUNÇÕES, PLACAR, DICT DE PREVISÕES
# ==============================================================================

def wmape(real: np.ndarray, previsto: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error.
    WMAPE = (Σ |previsto - real|) / (Σ real)  * 100
    Percentual calculado sobre o volume real (denominador = soma do real).
    Retorna percentual entre 0 e 100.
    """
    real = np.asarray(real, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    erro_abs_total = np.sum(np.abs(previsto - real))
    volume_total_real = np.sum(real)
    if volume_total_real <= 0:
        return np.nan
    return 100.0 * erro_abs_total / volume_total_real


def vies(real: np.ndarray, previsto: np.ndarray) -> float:
    """
    Erro médio (viés) percentual ponderado pelo volume.
    Viés = Σ(previsto - real) / Σ(real)  * 100
    - Positivo  ⇒ modelo  SOBREESTIMA  (pediu demais / previu maior que a realidade).
    - Negativo  ⇒ modelo  SUBESTIMA    (pediu de menos / perdeu ruptura).
    - Perto de zero ⇒ modelo não enviesado.
    """
    real = np.asarray(real, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    erro_total = np.sum(previsto - real)
    volume_total_real = np.sum(real)
    if volume_total_real <= 0:
        return np.nan
    return 100.0 * erro_total / volume_total_real


# Dicionários globais de estado
placar = {}
previsoes = {}


def avaliar(nome: str, previsao_raw) -> tuple[float, float]:
    """
    Recebe uma previsão (array ou Series) para o período de teste H=28,
    clipa em zero, calcula WMAPE e Viés, guarda nos dicionários globais
    `placar` e `previsoes` e imprime a linha do modelo.

    Retorna (wmape, vies) para facilitar uso em laços.
    """
    previsao_raw_arr = (
        previsao_raw["unidades"].astype(float).values
        if isinstance(previsao_raw, pd.DataFrame)
        else (
            previsao_raw.astype(float).values
            if isinstance(previsao_raw, pd.Series)
            else np.asarray(previsao_raw, dtype=float)
        )
    )

    # Item 4: Clip em zero ANTES de calcular métricas
    previsao_clipada = np.clip(previsao_raw_arr, a_min=0.0, a_max=None)

    w = wmape(real, previsao_clipada)
    v = vies(real, previsao_clipada)

    placar[nome] = {
        "WMAPE_%": round(w, 4),
        "Viés_%": round(v, 4),
        "Demanda_prevista_total": round(float(np.sum(previsao_clipada)), 2),
        "Demanda_real_total": round(float(np.sum(real)), 2),
        "Dias_avaliados": len(previsao_clipada),
    }

    previsoes[nome] = pd.Series(
        previsao_clipada,
        index=indice_teste,
        name=f"prev_{nome}"
    )

    sinal = "+" if v >= 0 else ""
    print(
        f"  {nome:<32s} |  WMAPE = {w:6.3f}%  |  "
        f"Viés = {sinal}{v:6.3f}%  |  "
        f"Previsto = {placar[nome]['Demanda_prevista_total']:>9,.1f}  |  "
        f"Real = {placar[nome]['Demanda_real_total']:>9,.1f}"
    )
    return w, v


# ==============================================================================
# Execução de inicialização — imprime os parâmetros do período de avaliação
# ==============================================================================

if __name__ == "__main__":
    print("FASE 4 — Modelagem")
    print("=" * 70)
    print("INFRAESTRUTURA DE AVALIAÇÃO INICIALIZADA")
    print("-" * 70)
    print(f"  Horizonte de previsão (H) : {H} dias")
    print(f"  Período de avaliação      : {indice_teste.min().strftime('%d/%m/%Y')}"
          f" → {indice_teste.max().strftime('%d/%m/%Y')}")
    print(f"  Nº de dias avaliados      : {len(indice_teste)} dias")
    print(f"  Produto                   : {PRODUTO_FOCO}")
    print(f"  Demanda real TOTAL do período de teste (holdout) : "
          f"{float(np.sum(real)):,.2f} unidades")
    print(f"  Demanda real MÉDIA por dia no teste             : "
          f"{float(np.mean(real)):,.2f} unidades/dia")
    print(f"  Mediana diária de venda no teste                : "
          f"{float(np.median(real)):,.2f} unidades/dia")
    print()
    print("Métricas de avaliação:")
    print("  WMAPE = Σ |prev - real| / Σ real  × 100  → 0% = perfeito, quanto menor melhor.")
    print("  Viés  = Σ (prev - real) / Σ real × 100  → positivo = SUPERestima; negativo = SUBestima.")
    print("  Regra 4: todas as previsões são clipadas em 0 antes da avaliação.")
    print()
    print("Placar vazio — aguardando primeiro modelo:")
    print(f"  {'Modelo':<32s} | {'WMAPE':>8s} | {'Viés':>8s} | {'Previsto':>10s} | {'Real':>10s}")
    print("-" * 70)
    # Nenhum modelo ainda — placeholder
