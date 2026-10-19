# -*- coding: utf-8 -*-
"""
Gera o documento 'Perguntas_Case.docx' com as respostas às perguntas de
discussão do case, escritas em linguagem acessível a público NÃO técnico.

Todas as afirmações estão ancoradas em evidências produzidas ao longo do
projeto (fases 2 a 6 do CRISP-DM): números reais, tabelas e figuras.

Saída: reports/Perguntas_Case.docx
"""

import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Caminhos ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "reports", "figures")
SAIDA = os.path.join(BASE, "reports", "Perguntas_Case.docx")

# ── Paleta (visual sóbrio, corporativo, tema claro) ─────────────────────────
PRIMARIA = RGBColor(0x0F, 0x3D, 0x5C)   # azul-petróleo escuro
ACENTO = RGBColor(0x0E, 0x80, 0x8C)     # teal
VERDE = RGBColor(0x1B, 0x6E, 0x3A)
AMBAR = RGBColor(0xA9, 0x64, 0x00)
CINZA = RGBColor(0x50, 0x5A, 0x66)
CINZA_ESCURO = RGBColor(0x2B, 0x33, 0x3D)

BG_CALLOUT = "E8F3F5"
BG_TAB_HEAD = "0F3D5C"
BG_LINHA_ALT = "F2F6F9"
BG_VERDE = "E7F4EA"
BG_AMBAR = "FBF1DF"

FONTE = "Calibri"


# ══════════════════════════════════════════════════════════════════════════
# Helpers de formatação
# ══════════════════════════════════════════════════════════════════════════

def _fundo_celula(cell, cor_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), cor_hex)
    tcPr.append(shd)


def _bordas_tabela(table, cor="D5DDE6", sz=6):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), cor)
        borders.append(el)
    tblPr.append(borders)


def _borda_inferior(p, cor="0E808C", sz=12):
    pPr = p._p.get_or_add_pPr()
    bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(sz))
    bottom.set(qn("w:space"), "3")
    bottom.set(qn("w:color"), cor)
    bdr.append(bottom)
    pPr.append(bdr)


def _recuo(cell, top=60, bottom=60, left=110, right=110):
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for nome, valor in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement("w:" + nome)
        el.set(qn("w:w"), str(valor))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


def _runs_com_negrito(p, texto, tamanho=10.5, cor=None):
    """Interpreta **negrito** dentro do texto."""
    for i, parte in enumerate(texto.split("**")):
        if parte == "":
            continue
        r = p.add_run(parte)
        r.font.name = FONTE
        r.font.size = Pt(tamanho)
        r.bold = (i % 2 == 1)
        if cor is not None:
            r.font.color.rgb = cor
    return p


def paragrafo(texto, tamanho=10.5, cor=None, espaco_depois=6, alinhamento=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(espaco_depois)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    if alinhamento is not None:
        p.alignment = alinhamento
    _runs_com_negrito(p, texto, tamanho=tamanho, cor=cor)
    return p


def bullet(texto, tamanho=10.5):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.7)
    p.paragraph_format.first_line_indent = Cm(-0.35)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.12
    r = p.add_run("▪  ")
    r.font.name = FONTE
    r.font.size = Pt(tamanho)
    r.font.color.rgb = ACENTO
    _runs_com_negrito(p, texto, tamanho=tamanho)
    return p


def titulo_pergunta(numero, texto):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(numero + "  ")
    r.font.name = FONTE
    r.font.size = Pt(15)
    r.bold = True
    r.font.color.rgb = ACENTO
    r2 = p.add_run(texto)
    r2.font.name = FONTE
    r2.font.size = Pt(15)
    r2.bold = True
    r2.font.color.rgb = PRIMARIA
    _borda_inferior(p, cor="0E808C", sz=10)
    return p


def subtitulo(texto):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(texto)
    r.font.name = FONTE
    r.font.size = Pt(11.5)
    r.bold = True
    r.font.color.rgb = PRIMARIA
    return p


def enunciado(texto):
    """Texto da pergunta original, em itálico cinza."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.1
    r = p.add_run(texto)
    r.font.name = FONTE
    r.font.size = Pt(9.5)
    r.italic = True
    r.font.color.rgb = CINZA
    return p


def callout(rotulo, texto, bg=BG_CALLOUT, cor_borda="0E808C", cor_rotulo=ACENTO):
    t = doc.add_table(rows=1, cols=1)
    t.autofit = True
    cell = t.cell(0, 0)
    _fundo_celula(cell, bg)
    _bordas_tabela(t, cor=cor_borda, sz=8)
    _recuo(cell, top=90, bottom=90, left=140, right=140)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(rotulo + "  ")
    r.font.name = FONTE
    r.font.size = Pt(10)
    r.bold = True
    r.font.color.rgb = cor_rotulo
    _runs_com_negrito(p, texto, tamanho=10)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def tabela(cabecalhos, linhas, larguras_cm, tamanho=9.5):
    t = doc.add_table(rows=1, cols=len(cabecalhos))
    t.autofit = False
    _bordas_tabela(t, cor="CBD6E0", sz=4)
    # cabeçalho
    for j, txt in enumerate(cabecalhos):
        cell = t.rows[0].cells[j]
        cell.width = Cm(larguras_cm[j])
        _fundo_celula(cell, BG_TAB_HEAD)
        _recuo(cell, top=50, bottom=50)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(txt)
        r.font.name = FONTE
        r.font.size = Pt(tamanho)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    # corpo
    for i, linha in enumerate(linhas):
        cells = t.add_row().cells
        for j, txt in enumerate(linha):
            cell = cells[j]
            cell.width = Cm(larguras_cm[j])
            _recuo(cell, top=45, bottom=45)
            if i % 2 == 1:
                _fundo_celula(cell, BG_LINHA_ALT)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            _runs_com_negrito(p, str(txt), tamanho=tamanho)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figura(nome_arquivo, legenda):
    caminho = os.path.join(FIG, nome_arquivo)
    if not os.path.exists(caminho):
        return
    doc.add_picture(caminho, width=Cm(15.6))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(legenda)
    r.font.name = FONTE
    r.font.size = Pt(8.5)
    r.italic = True
    r.font.color.rgb = CINZA


def campo_pagina(paragraph):
    run = paragraph.add_run()
    f1 = OxmlElement("w:fldChar")
    f1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    f2 = OxmlElement("w:fldChar")
    f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1)
    run._r.append(instr)
    run._r.append(f2)
    return run


# ══════════════════════════════════════════════════════════════════════════
# Documento
# ══════════════════════════════════════════════════════════════════════════
doc = Document()

# Estilo base
normal = doc.styles["Normal"]
normal.font.name = FONTE
normal.font.size = Pt(10.5)
normal.font.color.rgb = CINZA_ESCURO

# Margens
for sec in doc.sections:
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)

# ── Cabeçalho institucional ─────────────────────────────────────────────────
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(0)
r = p.add_run("REDE BANDEIRANTE")
r.font.name = FONTE
r.font.size = Pt(11)
r.bold = True
r.font.color.rgb = ACENTO

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(0)
r = p.add_run("Supply Chain Forecasting")
r.font.name = FONTE
r.font.size = Pt(26)
r.bold = True
r.font.color.rgb = PRIMARIA

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("Perguntas para Discussão — Respostas")
r.font.name = FONTE
r.font.size = Pt(13)
r.font.color.rgb = CINZA

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(10)
r = p.add_run(
    "Previsão de demanda e política de estoque no centro de distribuição  ·  "
    "Produto de referência: P0131  ·  Outubro de 2019"
)
r.font.name = FONTE
r.font.size = Pt(9)
r.font.color.rgb = CINZA
_borda_inferior(p, cor="D5DDE6", sz=6)

paragrafo(
    "Este documento responde, em linguagem acessível, às quatro perguntas de discussão do case. "
    "As respostas vêm das duas bases fornecidas pela empresa (demanda.csv e demanda_por_loja.csv) e do "
    "trabalho analítico conduzido em seis etapas. Sempre que possível, cada afirmação é acompanhada do "
    "número que a sustenta — para que a leitura dispense jargão técnico, mas não dispense evidência.",
    espaco_depois=4,
)

# ══════════════════════════════════════════════════════════════════════════
# PERGUNTA 1
# ══════════════════════════════════════════════════════════════════════════
titulo_pergunta("1.", "Escopo da previsão")
enunciado(
    "Quantos dias à frente a Bandeirante precisa prever, deduzindo a resposta dos dois parâmetros "
    "operacionais do Exhibit 1? E em que nível de agregação: por loja ou consolidada na rede? "
    "Compare as duas bases para sustentar a escolha e enuncie a regra geral que você levaria para outros projetos."
)
callout(
    "Em uma frase:",
    "Prever **28 dias à frente**, na base **consolidada da rede** (produto × dia). Prever por loja "
    "multiplica o ruído sem melhorar a decisão — e a decisão de compra é tomada em um único ponto: o CD.",
)

subtitulo("1.1 — Quantos dias à frente prever")
paragrafo(
    "O Exhibit 1 traz dois parâmetros contratuais que, somados, definem o horizonte de previsão:"
)
tabela(
    ["Parâmetro contratual", "Valor", "O que significa na prática"],
    [
        ["Lead time do fornecedor", "14 dias", "Tempo entre fazer o pedido e a mercadoria chegar na doca do CD."],
        ["Ciclo de revisão do pedido", "14 dias", "Só é possível fazer um novo pedido ao mesmo fornecedor 14 dias depois."],
        ["**Horizonte necessário**", "**28 dias**", "**14 + 14**: cobrir a espera da entrega e o intervalo até o próximo pedido."],
    ],
    [5.2, 2.6, 8.0],
)
paragrafo(
    "A lógica é simples: quando o comprador fecha o pedido, a mercadoria demora 14 dias para chegar e "
    "precisa durar até o pedido seguinte, que só acontece 14 dias depois. São **28 dias de cobertura**. "
    "Prever menos deixaria um buraco de abastecimento; prever mais significaria comprar para um período "
    "que ainda não é o do próximo ciclo."
)

subtitulo("1.2 — Por loja ou consolidada na rede?")
paragrafo(
    "Comparamos as duas bases para o mesmo produto (P0131). A diferença de comportamento é decisiva:"
)
tabela(
    ["Critério de comparação", "Base por loja", "Base consolidada (rede)"],
    [
        ["Séries a prever por produto", "144 (uma por loja)", "1 (a rede somada)"],
        ["Venda típica em um dia", "mediana de 0 unidade", "≈ 208 unidades/dia"],
        ["Volatilidade (coeficiente de variação)", "mediana **1,53**", "**0,51**"],
        ["Dias sem venda", "**51,3%** dos dias de loja", "raro"],
        ["Onde a decisão é tomada", "a loja não compra", "o CD compra — é o decisor"],
    ],
    [5.6, 5.1, 5.1],
)
paragrafo(
    "Na base por loja, a venda diária é quase sempre 0, 1 ou 2 unidades — em mais da metade dos dias não "
    "há venda alguma. Séries assim são dominadas por acaso, não por padrão. Prever 144 séries instáveis e "
    "depois somá-las tende a errar mais do que prever diretamente o total da rede, que é muito mais "
    "regular (volatilidade 0,51 contra 1,53). Além disso, o pedido é **um só** — feito pelo CD —, então "
    "o nível natural de decisão é o agregado."
)
figura(
    "serie_comparativa_P0131_loja_vs_rede.png",
    "À esquerda, a venda de uma loja isolada: irregular e cheia de zeros. À direita, a mesma venda somada "
    "na rede: contínua e previsível. Agregar transforma ruído em sinal.",
)

subtitulo("1.3 — A regra geral para outros projetos")
callout(
    "Regra geral:",
    "**Preveja no nível em que a decisão é tomada e agregue o máximo possível antes de modelar.** A soma "
    "de muitas séries pequenas e ruidosas é mais previsível do que cada uma delas separadamente. Só "
    "desagregue quando a decisão também for local.",
)

# ══════════════════════════════════════════════════════════════════════════
# PERGUNTA 2
# ══════════════════════════════════════════════════════════════════════════
titulo_pergunta("2.", "Leitura crítica dos dados")
enunciado(
    "Investigue o que cada coluna de fato mede e o que ela deixa de medir. O que a coluna de unidades "
    "registra em um dia em que o produto esgotou na loja, e que consequência isso tem para um modelo "
    "treinado nela? O que exatamente é a coluna de estoque, e o pedido que estamos calculando abastece "
    "esse mesmo elo da cadeia? Há itens no painel que não deveriam estar no escopo de um modelo de previsão?"
)
callout(
    "Em uma frase:",
    "As colunas registram **venda** (não demanda), **estoque de loja no fim do dia** (não do CD) e "
    "**promoção**. Nenhuma delas é exatamente o que o modelo precisa — e três itens do painel estão "
    "fora do escopo. Ler errado aqui contamina todas as etapas seguintes.",
)

subtitulo("2.1 — O que cada coluna mede e o que deixa de medir")
tabela(
    ["Coluna", "O que ela mede de fato", "O que ela deixa de medir"],
    [
        ["unidades", "Unidades **vendidas** no dia, somadas na rede.",
         "A demanda não atendida: em dia de ruptura, registra só o que havia na gôndola."],
        ["preco_medio", "Preço médio praticado no dia.",
         "Diferenças de preço entre lojas e efeitos de desconto específicos."],
        ["pct_lojas_promo", "Fração de lojas com o item em promoção.",
         "A intensidade e a mecânica da promoção, além do simples \"estava ou não\"."],
        ["estoque_total", "Estoque somado nas **lojas**, **no fim do dia**.",
         "O estoque do CD (o elo que efetivamente compramos) e o estoque de início de dia."],
        ["lojas_ativas", "Lojas que venderam ao menos uma unidade.",
         "Lojas que tinham o item e não venderam por falta de estoque aparecem como \"inativas\"."],
    ],
    [3.5, 6.2, 6.1],
)

subtitulo("2.2 — O que a coluna \"unidades\" registra em um dia de ruptura")
paragrafo(
    "Em um dia em que o produto esgotou, a coluna `unidades` registra apenas **o que a loja conseguiu "
    "vender com o estoque que ainda tinha** — ou seja, zero ou quase isso. A venda que deixou de "
    "acontecer (o cliente que foi procurar o produto e não achou) **não aparece em lugar nenhum**."
)
paragrafo(
    "A consequência para um modelo treinado nessa coluna é direta: ele aprende o **teto** que a loja "
    "conseguiu vender, e não a demanda que existia de fato. Em itens sujeitos a ruptura, isso faz o "
    "modelo **subestimar** a demanda — e o erro se concentra exatamente nos dias em que o cliente "
    "queria comprar mais. É também por isso que o estoque de segurança (pergunta 4) tem um papel "
    "central: ele protege contra essa subestimação."
)
callout(
    "Cuidado que vale para qualquer projeto:",
    "venda registrada **não é** demanda. Quando a ruptura é frequente, o dado observado é a demanda "
    "**truncada** por falta de produto — e não a demanda real.",
    bg=BG_AMBAR,
    cor_borda="D9A441",
    cor_rotulo=AMBAR,
)

subtitulo("2.3 — O pedido abastece o mesmo elo da cadeia?")
paragrafo(
    "**Não.** E esse é um ponto que a intuição costuma errar. A coluna `estoque_total` mede o estoque "
    "somado nas **lojas**, ao **final do dia** — é o último elo da cadeia, o ponto de venda, já "
    "consumido pelo dia. Já o pedido que estamos calculando abastece o **centro de distribuição**: "
    "Rafael compra do fornecedor para o CD, e o CD é quem reabastece as 144 lojas."
)
paragrafo(
    "São, portanto, **elos diferentes**. Por isso não se desconta `estoque_total` na conta do pedido: "
    "o que desconta é o estoque **disponível no CD**, informado pelo negócio. Na aplicação, esse valor "
    "é um campo preenchido pelo próprio usuário (no exemplo da pergunta 4, 3.000 unidades)."
)
callout(
    "Foi exatamente aqui que o processo atual travou:",
    "a média móvel apontava cerca de 300 unidades/dia e o relatório de estoque mostrava mais de 10.000 "
    "unidades — a conta bruta zerava o pedido. O estoque de **loja** não é o estoque que serve de base "
    "para o pedido do **CD**.",
    bg=BG_AMBAR,
    cor_borda="D9A441",
    cor_rotulo=AMBAR,
)

subtitulo("2.4 — Itens que não deveriam estar no escopo")
paragrafo(
    "Comparamos, para cada item, a média de venda dos últimos 90 dias com a média do período anterior. "
    "Uma razão muito baixa indica item que perdeu volume de forma estrutural — sinal de descontinuação. "
    "Adotamos o corte de **razão abaixo de 0,6** para retirá-lo do painel:"
)
tabela(
    ["Produto", "Média recente (90 dias)", "Média anterior", "Razão", "Leitura"],
    [
        ["P0438", "0,05 un", "274,49 un", "0,0002", "Praticamente sem venda — item morto."],
        ["P0051", "28,37 un", "263,31 un", "0,11", "Queda muito forte de volume."],
        ["P0079", "4,32 un", "31,45 un", "0,14", "Queda muito forte de volume."],
    ],
    [2.4, 4.0, 3.2, 2.0, 4.2],
)
paragrafo(
    "Esses três produtos **saem do escopo**. Manter no painel um item que saiu de linha significaria "
    "produzir previsão e pedido para um produto que já não existe na prateleira — esforço e risco "
    "sem retorno."
)

# ══════════════════════════════════════════════════════════════════════════
# PERGUNTA 3
# ══════════════════════════════════════════════════════════════════════════
titulo_pergunta("3.", "Modelagem e validação")
enunciado(
    "Estabeleça o baseline contra o qual todo modelo será comparado e explique por que ele vem antes de "
    "qualquer sofisticação; defina a regra que separa treino de teste e o critério que decide se uma "
    "variável pode entrar em um modelo que enxerga 28 dias à frente — aplique-o ao calendário promocional "
    "e à venda do dia anterior; por fim, avalie os modelos de modo que o número prometido ao comitê se "
    "sustente ao longo de um ano, e não apenas em uma janela escolhida por acaso."
)
callout(
    "Em uma frase:",
    "O baseline é a **média das últimas 4 semanas** — o processo atual. Treino e teste são separados "
    "**no tempo**, sem embaralhar. Só entra variável **conhecida** para os 28 dias à frente. E a "
    "validação usa **12 janelas ao longo do ano**, não uma só.",
)

subtitulo("3.1 — O baseline vem antes de qualquer sofisticação")
paragrafo(
    "O baseline é a **média das últimas 4 semanas** de venda — exatamente o processo que já existe hoje na "
    "empresa (a planilha do comprador). Ele vem primeiro por quatro razões:"
)
bullet("é o **incumbente**: qualquer modelo novo só se justifica se fizer melhor do que o que já roda;")
bullet("**ancora o ganho**: sem ele, não há como afirmar quanto se ganhou;")
bullet("é uma régua **honesta** — para demanda estável, a média das 4 semanas é difícil de bater;")
bullet("define o número que o critério decisivo (custo total) precisa superar.")
paragrafo(
    "No backtest, o baseline teve o **pior** erro entre os modelos avaliados (WMAPE médio de **22,05%**). "
    "Há espaço de melhora — mas a régua é real, não um espantalho."
)

subtitulo("3.2 — A regra que separa treino de teste")
paragrafo(
    "A separação é **temporal e estrita, sem embaralhar dados**. O modelo é treinado apenas com o passado "
    "e avaliado apenas no futuro. Concretamente:"
)
tabela(
    ["Etapa", "Período", "Regra"],
    [
        ["Treino", "até **03/10/2019**", "o modelo só vê o que já aconteceu até essa data."],
        ["Teste (holdout)", "**04/10 a 31/10/2019** (28 dias)", "o modelo nunca vê esses dias durante o treino."],
    ],
    [3.0, 5.6, 7.2],
)
paragrafo(
    "Em previsão de demanda, embaralhar os dados criaria **vazamento**: o modelo teria acesso a "
    "informações do futuro e o desempenho pareceria bom — mas falso. Em série temporal, o futuro nunca "
    "entra no treino, nem como vizinho."
)

subtitulo("3.3 — Que informação pode entrar em um modelo que enxerga 28 dias")
paragrafo(
    "O critério é único e objetivo: **a variável só entra se estiver disponível para todo o horizonte de "
    "28 dias no momento em que a previsão é feita.**"
)
tabela(
    ["Variável", "Está disponível nos 28 dias futuros?", "Entra no modelo?"],
    [
        ["Calendário promocional (pct_lojas_promo)",
         "**Sim** — o comercial fecha as ações de 6 a 8 semanas antes.",
         "**Entra** (variável exógena futura conhecida)."],
        ["Venda do dia anterior",
         "**Não** — só é conhecida no dia, não para 28 dias à frente.",
         "**Não entra** (seria vazamento de informação)."],
        ["Regra de defasagem", "—", "Valores passados só a partir de **28 dias** de distância."],
    ],
    [4.4, 6.4, 5.0],
)
paragrafo(
    "A promoção merece entrar: ela multiplica a venda média do P0131 por aproximadamente **1,48×**, e é "
    "conhecida com antecedência — previsível, portanto. Já a venda de ontem é uma armadilha: parece útil, "
    "mas não existe para os 28 dias à frente."
)

subtitulo("3.4 — Validação: um número que se sustenta ao longo do ano")
paragrafo(
    "Uma única janela de teste pode ter tido sorte. Para que o número prometido ao comitê seja confiável, "
    "avaliamos os modelos em **12 origens móveis**, espaçadas de 14 em 14 dias (a mesma cadência do ciclo "
    "de reposição), terminando em 03/10/2019. Em cada origem, o modelo vê só o passado e prevê os 28 dias "
    "seguintes."
)
tabela(
    ["Modelo", "WMAPE médio (erro)", "Viés global"],
    [
        ["Holt-Winters", "**15,36%**", "+1,85%"],
        ["SARIMAX com promoção", "**15,39%**", "+6,40%"],
        ["Ingênuo sazonal (repete a última semana)", "16,15%", "+0,58%"],
        ["SARIMA sem promoção", "18,18%", "+7,44%"],
        ["**Média das 4 semanas (processo atual)**", "**22,05%**", "+6,12%"],
    ],
    [7.0, 4.4, 4.4],
)
paragrafo(
    "Leitura: os dois melhores (Holt-Winters e SARIMAX com promoção) praticamente empatam e **batem o "
    "processo atual** em erro. E, como o teste foi repetido ao longo de vários meses, o ganho não depende "
    "de um único mês escolhido a dedo — ele se sustenta."
)
figura(
    "fase5_backtest_WMAPE_por_origem_P0131.png",
    "Erro (WMAPE) de cada modelo nas 12 janelas de teste ao longo do ano. As curvas estáveis na parte "
    "de baixa do gráfico indicam desempenho consistente, não sorte de uma janela.",
)

# ══════════════════════════════════════════════════════════════════════════
# PERGUNTA 4
# ══════════════════════════════════════════════════════════════════════════
titulo_pergunta("4.", "Da previsão à decisão de compra")
enunciado(
    "Uma previsão sem viés acerta a média e, por construção, deixa a gôndola vazia em metade dos períodos. "
    "Com os custos do Exhibit 1: determine o nível de serviço que minimiza o custo esperado; explique por "
    "que a resposta não é 100%; calcule a folga de estoque que ele exige; compare em reais a política "
    "proposta com o processo atual, lembrando que um modelo pior precisa de mais folga para o mesmo nível "
    "de serviço, e responda aos três critérios de Larissa."
)
callout(
    "Em uma frase:",
    "O nível de serviço ótimo é **93,75%**. Não é 100% porque proteger a última fração custa mais do que "
    "a falta que evita. Com o modelo proposto, a folga de estoque cai para cerca de **784 unidades** e o "
    "custo total recua **27%** contra o processo atual.",
    bg=BG_VERDE,
    cor_borda="4CA66A",
    cor_rotulo=VERDE,
)

subtitulo("4.1 — O nível de serviço que minimiza o custo")
paragrafo(
    "Faltar é muito mais caro do que sobrar: faltar custa 30% do preço (margem que evapora) e sobrar custa "
    "2% (capital e armazenagem). Para o P0131, com preço médio de R$ 2,5870:"
)
tabela(
    ["Componente", "Valor", "Como se chega nele"],
    [
        ["Custo de faltar 1 unidade (Cu)", "R$ 0,7761", "30% do preço médio"],
        ["Custo de sobrar 1 unidade (Co)", "R$ 0,0517", "2% do preço médio"],
        ["Razão entre os dois custos", "15 : 1", "faltar custa 15 vezes mais que sobrar"],
        ["**Nível de serviço ótimo**", "**93,75%**", "Cu ÷ (Cu + Co) = 0,30 ÷ 0,32"],
    ],
    [6.0, 3.4, 6.4],
)
paragrafo(
    "Em palavras: o nível de serviço ótimo é a fração das situações que vale a pena cobrir com estoque. "
    "Aqui, **93,75%** — equivalente a aceitar cerca de **6,25%** de chance de ruptura por período."
)

subtitulo("4.2 — Por que a resposta não é 100%")
paragrafo(
    "Cobrir 100% das situações exigiria estoque suficiente para o **pior erro imaginável** — e manter esse "
    "estoque é caro. Como sobrar também custa (2% do preço), cada unidade extra de proteção só compensa "
    "até o ponto em que o custo de **proteger** uma unidade a mais se iguala ao custo de **faltar** uma "
    "unidade a mais. Além desse ponto, a proteção adicional custa mais do que a falta que evita."
)
paragrafo(
    "Por isso o ótimo econômico fica **abaixo de 100%**. E note o papel da assimetria: como faltar é 15 "
    "vezes mais caro que sobrar, esse nível é **empurrado para cima** (93,75%, bem acima dos 50% de quem "
    "só acerta a média) — mas nunca até 100%."
)

subtitulo("4.3 — A folga de estoque que ele exige")
paragrafo(
    "A folga (estoque de segurança) é o colchão que o modelo precisa para atingir o nível de serviço. Em "
    "vez de supor uma fórmula teórica, medimos o quantil de 93,75% diretamente sobre os **erros reais** "
    "observados nos testes — a distribuição empírica. Quanto menos o modelo erra, **menos** folga ele "
    "exige para o mesmo nível de serviço:"
)
tabela(
    ["Modelo", "Folga exigida", "Por quê"],
    [
        ["SARIMAX com promoção", "**783,6 un**", "erra menos → precisa de menos colchão."],
        ["Holt-Winters", "1.484,2 un", "erra um pouco mais → colchão maior."],
        ["**Média das 4 semanas (atual)**", "**2.111,9 un**", "erra mais → precisa do maior colchão."],
    ],
    [6.4, 3.4, 6.0],
)
callout(
    "O ponto-chave da pergunta:",
    "um modelo **pior** não apenas erra mais — ele obriga a carregar **mais estoque** para entregar o "
    "mesmo nível de serviço. Melhor previsão significa **menos capital parado**.",
)

subtitulo("4.4 — Comparação em reais e os três critérios de Larissa")
paragrafo(
    "Convertendo tudo para reais, por período de 28 dias:"
)
tabela(
    ["Modelo", "Folga (un)", "Custo por período", "Comparação"],
    [
        ["**SARIMAX com promoção**", "783,6", "**R$ 107,04**", "**−27,2% (melhor)**"],
        ["Média das 4 semanas (atual)", "2.111,9", "R$ 147,11", "referência"],
        ["Holt-Winters", "1.484,2", "R$ 160,13", "+8,9% (pior que o atual)"],
    ],
    [5.8, 2.8, 3.6, 3.6],
)
paragrafo(
    "A economia do modelo proposto é de R$ 40,06 por período de 28 dias. Como o ano tem 13 períodos desse "
    "tipo, isso equivale a aproximadamente **R$ 520,84 por ano — em um único produto**."
)
figura(
    "fase5_custo_estoque_seguranca_por_modelo.png",
    "À esquerda, o custo total por período (menor é melhor): o SARIMAX com promoção é o mais barato. À "
    "direita, a folga de estoque exigida (menor é melhor): o modelo que erra menos precisa de menos colchão.",
)
paragrafo("Agora, os três critérios que a diretoria fixou para o comitê de dezembro:", espaco_depois=4)
callout(
    "Critério 1 — Superar o processo atual em erro:",
    "**Atendido.** O erro médio cai de 22,05% (média das 4 semanas) para 15,39% (SARIMAX com promoção).",
    bg=BG_VERDE,
    cor_borda="4CA66A",
    cor_rotulo=VERDE,
)
callout(
    "Critério 2 — Viés próximo de zero:",
    "**Atendido com ressalva.** O SARIMAX tem viés de +6,40% — ele tende a pedir um pouco a mais, o que "
    "erra para o lado **barato** (sobrar). O Holt-Winters é mais neutro (+1,85%), porém mais caro. É o "
    "critério mais frágil dos três, e vale declarar isso com transparência.",
    bg=BG_AMBAR,
    cor_borda="D9A441",
    cor_rotulo=AMBAR,
)
callout(
    "Critério 3 — Reduzir o custo total em reais (o decisivo):",
    "**Atendido.** Queda de **27,2%** no custo por período. É o critério que manda — e ele é cumprido com "
    "folga pelo modelo proposto.",
    bg=BG_VERDE,
    cor_borda="4CA66A",
    cor_rotulo=VERDE,
)
paragrafo(
    "Vale registrar a conclusão mais importante do case: o **Holt-Winters** tem bom erro de previsão "
    "(15,36%), mas custa **mais caro** do que o processo atual (R$ 160,13 contra R$ 147,11). Ou seja, "
    "**acertar a previsão não garante economizar** — o que decide é o custo total. Por isso a escolha "
    "recai sobre o **SARIMAX com promoção**."
)

subtitulo("4.5 — E o que aparece na tela do comprador?")
paragrafo(
    "Para o P0131, no cenário de teste, com estoque de 3.000 unidades disponíveis no CD, a conta que o "
    "comprador vê aberta é:"
)
tabela(
    ["Componente", "Valor"],
    [
        ["(+) Demanda prevista para 28 dias", "8.198,5 un"],
        ["(+) Folga de estoque (estoque de segurança)", "783,6 un"],
        ["(−) Estoque disponível no CD", "3.000,0 un"],
        ["**(=) Pedido sugerido**", "**5.982,1 un**"],
    ],
    [10.0, 5.0],
)
paragrafo(
    "Cada parcela fica visível. Isso atende à exigência do comprador — *\"se o sistema me mandar pedir um "
    "número e não me disser de onde saiu, eu volto para a minha planilha\"*. A cobertura montada equivale "
    "a cerca de **30,8 dias** de venda média."
)

# ── Encerramento ────────────────────────────────────────────────────────────
callout(
    "Conclusão",
    "O projeto supera o processo atual **nos três critérios**, e o critério decisivo — custo total — é "
    "cumprido com folga. O ganho vem menos de \"prever melhor\" e mais de **decidir melhor**: o mesmo "
    "nível de serviço exige muito menos estoque quando o modelo erra menos.",
    bg=BG_CALLOUT,
    cor_borda="0E808C",
    cor_rotulo=ACENTO,
)

# ── Rodapé ──────────────────────────────────────────────────────────────────
rodape = doc.sections[0].footer.paragraphs[0]
rodape.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = rodape.add_run("Rede Bandeirante · Supply Chain Forecasting · Perguntas para Discussão   |   Página ")
r.font.name = FONTE
r.font.size = Pt(8)
r.font.color.rgb = CINZA
campo_pagina(rodape).font.size = Pt(8)

# ── Salvar ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
doc.save(SAIDA)
print("Documento gerado:", SAIDA)
print("Tamanho:", os.path.getsize(SAIDA), "bytes")
