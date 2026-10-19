/* ══════════════════════════════════════════════════════════════════════════
   Supply Chain Forecasting — Camada de decisão no navegador
   --------------------------------------------------------------------------
   O modelo (SARIMAX + promoção) é treinado em Python e os resultados são
   exportados para data/forecasts.json. Esta aplicação:
     1) lê o JSON;
     2) exibe histórico, previsão e realizado;
     3) roda a POLÍTICA DE PEDIDO no cliente:
          estoque_seguranca = quantil_empirico(erros_backtest, nivel_servico)
          pedido            = previsao_28d + estoque_seguranca - estoque_CD
   Todas as contas seguem exatamente as definições dos scripts Python.
   ══════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  // ── Formatadores pt-BR ───────────────────────────────────────────────────
  const fmt0 = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
  const fmt1 = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const fmt2 = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtPct = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const fmtPct1 = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

  const CORES = {
    hist: "#64748b",
    real: "#f0b429",
    prev: "#22d3ee",
    grade: "rgba(45, 60, 84, 0.55)",
    texto: "#8ea1bd",
  };

  let DADOS = null;          // conteúdo de forecasts.json
  let PRODUTO_ATUAL = null;  // chave do produto selecionado
  let grafico = null;        // instância do Chart.js

  const el = (id) => document.getElementById(id);

  // ── Utilitários matemáticos (espelham o numpy) ───────────────────────────

  /**
   * Quantil por interpolação linear — equivalente a np.quantile(arr, q)
   * (método padrão 'linear') usado nos scripts Python da Fase 5.
   */
  function quantil(valores, q) {
    const x = valores.slice().sort((a, b) => a - b);
    const n = x.length;
    if (n === 0) return 0;
    if (n === 1) return x[0];
    const pos = (n - 1) * q;
    const base = Math.floor(pos);
    const resto = pos - base;
    if (x[base + 1] !== undefined) {
      return x[base] + resto * (x[base + 1] - x[base]);
    }
    return x[base];
  }

  const soma = (arr) => arr.reduce((a, b) => a + b, 0);

  /** WMAPE(%) = 100 * Σ|prev-real| / Σreal   — mesma fórmula de fase4_infra_avaliacao.py */
  function wmape(real, previsto) {
    const volReal = soma(real);
    if (volReal === 0) return 0;
    let erroAbs = 0;
    for (let i = 0; i < real.length; i++) erroAbs += Math.abs(previsto[i] - real[i]);
    return (100 * erroAbs) / volReal;
  }

  /** Viés / MPE(%) = 100 * Σ(prev-real) / Σreal */
  function vies(real, previsto) {
    const volReal = soma(real);
    if (volReal === 0) return 0;
    let erro = 0;
    for (let i = 0; i < real.length; i++) erro += previsto[i] - real[i];
    return (100 * erro) / volReal;
  }

  // ── Renderização dos KPIs ────────────────────────────────────────────────
  function renderKpis(prod) {
    const real = prod.real_teste;
    const prev = prod.previsao.unidades;
    const totalReal = soma(real);
    const totalPrev = soma(prev);

    el("kpi-previsto").textContent = fmt0.format(totalPrev) + " un";
    el("kpi-previsto-foot").textContent = fmt1.format(totalPrev / prev.length) + " un/dia";

    el("kpi-real").textContent = fmt0.format(totalReal) + " un";
    el("kpi-real-foot").textContent = fmt1.format(totalReal / real.length) + " un/dia";

    const w = wmape(real, prev);
    el("kpi-wmape").textContent = fmtPct.format(w) + "%";

    const v = vies(real, prev);
    const elVies = el("kpi-vies");
    elVies.textContent = (v > 0 ? "+" : "") + fmtPct.format(v) + "%";
    elVies.classList.remove("is-positive", "is-negative", "is-warn");
    // viés positivo = superestimação (sobra, âmbar); negativo = subestimação (falta, vermelho)
    elVies.classList.add(v >= 0 ? "is-warn" : "is-negative");
  }

  // ── Gráfico ──────────────────────────────────────────────────────────────
  function renderGrafico(prod) {
    const datasHist = prod.historico.datas;
    const datasPrev = prod.previsao.datas;
    const rotulos = datasHist.concat(datasPrev);

    const nHist = datasHist.length;
    const nPrev = datasPrev.length;

    const valoresHist = prod.historico.unidades.concat(new Array(nPrev).fill(null));
    const valoresReal = new Array(nHist).fill(null).concat(prod.real_teste);
    const valoresPrev = new Array(nHist).fill(null).concat(prod.previsao.unidades);

    // rótulo curto dd/mm para o eixo X
    const rotulosCurto = rotulos.map((d) => {
      const [, m, dia] = d.split("-");
      return dia + "/" + m;
    });

    const config = {
      type: "line",
      data: {
        labels: rotulosCurto,
        datasets: [
          {
            label: "Histórico",
            data: valoresHist,
            borderColor: CORES.hist,
            backgroundColor: "transparent",
            borderWidth: 1.6,
            pointRadius: 0,
            tension: 0.25,
          },
          {
            label: "Realizado",
            data: valoresReal,
            borderColor: CORES.real,
            backgroundColor: "transparent",
            borderWidth: 2.6,
            pointRadius: 0,
            tension: 0.2,
          },
          {
            label: "Previsão SARIMAX",
            data: valoresPrev,
            borderColor: CORES.prev,
            backgroundColor: "transparent",
            borderWidth: 2.4,
            borderDash: [6, 4],
            pointRadius: 0,
            tension: 0.2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0a1220",
            borderColor: "#223049",
            borderWidth: 1,
            titleColor: "#e8eef8",
            bodyColor: "#cbd8ea",
            padding: 11,
            callbacks: {
              label: (ctx) =>
                ctx.dataset.label + ": " +
                (ctx.parsed.y === null ? "—" : fmt1.format(ctx.parsed.y) + " un"),
            },
          },
        },
        scales: {
          x: {
            grid: { color: CORES.grade, drawTicks: false },
            ticks: { color: CORES.texto, maxTicksLimit: 14, font: { size: 10 } },
            border: { color: "#223049" },
          },
          y: {
            beginAtZero: true,
            grid: { color: CORES.grade, drawTicks: false },
            ticks: { color: CORES.texto, font: { size: 10 } },
            border: { color: "#223049" },
            title: { display: true, text: "Unidades / dia", color: CORES.texto, font: { size: 11 } },
          },
        },
      },
    };

    if (grafico) grafico.destroy();
    grafico = new Chart(el("grafico").getContext("2d"), config);
  }

  // ── Conta aberta + diagnóstico ───────────────────────────────────────────
  // `animar = false` no primeiro desenho do produto, para não piscar sem motivo.
  // Nas interações (estoque do CD, nível de serviço, troca de SKU) anima = true.
  function renderDecisao(animar) {
    const prod = DADOS.produtos[PRODUTO_ATUAL];
    if (!prod) return;

    const nivel = parseFloat(el("slider-nivel").value);
    const estoqueCd = Math.max(0, parseFloat(el("input-estoque-cd").value) || 0);

    const previsaoTotal = soma(prod.previsao.unidades);
    const erros = prod.erros_backtest;

    // Estoque de segurança = quantil empírico do nível de serviço sobre os erros
    const ss = quantil(erros, nivel);
    const pedido = previsaoTotal + ss - estoqueCd;
    const necessidade = previsaoTotal + ss;

    // Conta aberta. O estoque do CD é o próprio campo de entrada, portanto o
    // valor dele não é reescrito aqui — só as parcelas calculadas.
    el("conta-previsao").textContent = fmt0.format(previsaoTotal) + " un";
    el("conta-ss").textContent = fmt0.format(ss) + " un";
    el("conta-ss-sub").textContent =
      "quantil " + fmtPct1.format(nivel * 100) + "% dos " + erros.length + " erros do backtest";

    const elPedido = el("conta-pedido");
    const pedidoTexto = fmt0.format(pedido) + " un";
    const mudou = elPedido.textContent !== pedidoTexto;
    elPedido.textContent = pedidoTexto;
    elPedido.classList.toggle("is-negativo", pedido < 0);
    if (mudou && animar !== false) {
      elPedido.classList.remove("pulsar");
      void elPedido.offsetWidth; // força o reinício da animação
      elPedido.classList.add("pulsar");
    }

    // Diagnóstico
    el("diag-mediana").textContent = fmt0.format(quantil(erros, 0.5)) + " un";
    el("diag-p90").textContent = fmt0.format(quantil(erros, 0.9)) + " un";
    el("diag-max").textContent = fmt0.format(Math.max.apply(null, erros)) + " un";
    el("diag-ratio").textContent =
      (previsaoTotal > 0 ? fmtPct1.format((ss / previsaoTotal) * 100) : "—") + "%";
    el("diag-necessidade").textContent = fmt0.format(necessidade) + " un";
    // Quanto da necessidade total o estoque já disponível no CD cobre, em %.
    el("diag-cobertura-cd").textContent =
      (necessidade > 0 ? fmtPct1.format((estoqueCd / necessidade) * 100) : "—") + "%";

    // Nota interpretativa
    const nota = el("diag-nota");
    if (pedido <= 0) {
      nota.innerHTML =
        "O estoque no CD já cobre a demanda prevista somada ao estoque de segurança. " +
        "<strong>Não é necessário pedir</strong> neste ciclo.";
    } else {
      const nivelTxt = fmtPct1.format(nivel * 100);
      const coberturaCd = necessidade > 0 ? (estoqueCd / necessidade) * 100 : 0;
      const msg =
        "O pedido de <strong>" + fmt0.format(pedido) + " unidades</strong> repõe a necessidade (" +
        fmt0.format(necessidade) + " = previsão " + fmt0.format(previsaoTotal) + " + SS " +
        fmt0.format(ss) + ") descontando o que já existe no CD (" + fmt0.format(estoqueCd) +
        ", " + fmtPct1.format(coberturaCd) + "% da necessidade). " +
        "A previsão e o estoque de segurança <strong>não mudam</strong> com o estoque do CD — " +
        "só o pedido muda, e ele recalculou agora (" + nivelTxt + "% de serviço).";
      nota.innerHTML = msg;
    }
  }

  // ── Orquestração ─────────────────────────────────────────────────────────
  function renderProduto() {
    const prod = DADOS.produtos[PRODUTO_ATUAL];
    if (!prod) return;
    renderKpis(prod);
    renderGrafico(prod);
    renderDecisao(false); // primeiro desenho do produto: sem animação de pulso
  }

  function popularSelect() {
    const sel = el("select-produto");
    sel.innerHTML = "";
    const chaves = Object.keys(DADOS.produtos).sort();
    chaves.forEach((k) => {
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = k;
      sel.appendChild(opt);
    });
    // produto foco como padrão
    const foco = DADOS.meta.produto_foco;
    sel.value = chaves.indexOf(foco) >= 0 ? foco : chaves[0];
    PRODUTO_ATUAL = sel.value;
  }

  function preencherMeta() {
    const m = DADOS.meta;
    const ini = m.inicio_teste.split("-").reverse().join("/");
    const fim = m.fim_teste.split("-").reverse().join("/");
    el("chip-horizonte").textContent = "H = " + m.horizonte_dias + " dias";
    el("chip-corte").textContent = "Corte " + m.corte.split("-").reverse().join("/");
    el("input-estoque-cd").value = 3000;
    el("rodape-gerado").textContent =
      "Modelo exportado em " + m.gerado_em + " · holdout " + ini + " a " + fim;
  }

  function ligarEventos() {
    el("select-produto").addEventListener("change", (e) => {
      PRODUTO_ATUAL = e.target.value;
      renderProduto();
    });

    const slider = el("slider-nivel");
    slider.addEventListener("input", () => {
      el("out-nivel").textContent = fmtPct.format(parseFloat(slider.value) * 100) + "%";
      renderDecisao();
    });

    el("input-estoque-cd").addEventListener("input", renderDecisao);
  }

  function mostrarErro(msg) {
    el("loading").classList.add("hidden");
    const box = document.createElement("div");
    box.className = "erro-box";
    box.innerHTML =
      "<h2>Não foi possível carregar os dados</h2>" +
      "<p>" + msg + "</p>" +
      "<p>Verifique se o arquivo <span class='mono'>docs/data/forecasts.js</span> foi gerado " +
      "executando <span class='mono'>uv run python scripts/fase6_export_app_data.py</span>.</p>";
    document.body.insertBefore(box, document.querySelector(".app-shell"));
  }

  async function iniciar() {
    try {
      // Os dados chegam embutidos por data/forecasts.js (window.DADOS_FORECAST).
      // Assim a aplicação funciona tanto no GitHub Pages quanto abrindo o
      // arquivo localmente (file://), sem servidor e sem bloqueio de CORS.
      if (typeof window.DADOS_FORECAST === "undefined") {
        throw new Error("objeto window.DADOS_FORECAST não encontrado");
      }
      DADOS = window.DADOS_FORECAST;

      popularSelect();
      preencherMeta();
      ligarEventos();
      renderProduto();

      el("loading").classList.add("hidden");
    } catch (err) {
      mostrarErro(err.message);
    }
  }

  document.addEventListener("DOMContentLoaded", iniciar);
})();
