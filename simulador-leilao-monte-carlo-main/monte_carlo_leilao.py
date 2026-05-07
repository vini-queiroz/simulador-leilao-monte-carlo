"""
============================================================
  SIMULADOR DE MONTE CARLO — OPERAÇÕES DE LEILÃO
  Analisa viabilidade de compra/revenda de materiais
  adquiridos em leilão com estimativa de risco e retorno.
============================================================
Dependências: numpy, matplotlib, scipy, tabulate
  pip install numpy matplotlib scipy tabulate
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from dataclasses import dataclass, field
from typing import Literal, Optional
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  1. ESTRUTURAS DE DADOS
# ─────────────────────────────────────────────

DistType = Literal["triangular", "normal", "uniforme"]


@dataclass
class ParametroDistribuicao:
    """
    Define a distribuição de probabilidade de uma variável.

    Para distribuição TRIANGULAR  → informar: minimo, media (moda), maximo
    Para distribuição NORMAL      → informar: media, desvio_padrao
    Para distribuição UNIFORME    → informar: minimo, maximo
    Teremos também opção de valor FIXO → informar: valor fixo
    """
    tipo: DistType = "triangular"

    # Triangular / Uniforme
    minimo: Optional[float] = None
    maximo: Optional[float] = None

    # Triangular / Normal
    media: Optional[float] = None

    # Normal
    desvio_padrao: Optional[float] = None

    # Fixo
    valor_fixo: Optional[float] = None

    def validar(self, nome: str) -> None:
        if self.tipo == "triangular":
            assert all(v is not None for v in [self.minimo, self.media, self.maximo]), \
                f"[{nome}] Triangular exige: minimo, media (moda), maximo."
            assert self.minimo <= self.media <= self.maximo, \
                f"[{nome}] Ordem inválida: minimo ≤ media ≤ maximo."
        elif self.tipo == "normal":
            assert all(v is not None for v in [self.media, self.desvio_padrao]), \
                f"[{nome}] Normal exige: media e desvio_padrao."
            assert self.desvio_padrao > 0, \
                f"[{nome}] desvio_padrao deve ser positivo."
        elif self.tipo == "uniforme":
            assert all(v is not None for v in [self.minimo, self.maximo]), \
                f"[{nome}] Uniforme exige: minimo e maximo."
            assert self.minimo < self.maximo, \
                f"[{nome}] minimo deve ser menor que maximo."
        elif self.tipo == "fixo":
            assert self.valor_fixo is not None, \
                f"[{nome}] Distribuição fixa exige: valor_fixo."
        else:
            raise ValueError(f"[{nome}] Tipo de distribuição desconhecido: {self.tipo}")


@dataclass
class CustoAdicional:
    """
    Representa um custo extra da operação.
    Pode ser fixo (valor determinístico) ou variável (distribuição).
    """
    nome: str
    valor_fixo: Optional[float] = None                    # custo determinístico
    distribuicao: Optional[ParametroDistribuicao] = None  # custo variável

    def __post_init__(self):
        assert (self.valor_fixo is not None) ^ (self.distribuicao is not None), \
            f"[{self.nome}] Informe valor_fixo OU distribuicao, não ambos."


@dataclass
class ConfiguracaoSimulacao:
    """
    Configuração completa de uma simulação Monte Carlo.
    """
    nome_operacao: str
    custo_aquisicao: ParametroDistribuicao
    preco_venda: ParametroDistribuicao
    custos_adicionais: list[CustoAdicional] = field(default_factory=list)
    n_simulacoes: int = 10_000
    confianca_var: float = 0.95   # Nível de confiança para VaR (ex: 95%)
    semente: Optional[int] = None  # Reprodutibilidade


# ─────────────────────────────────────────────
#  2. GERAÇÃO DE AMOSTRAS
# ─────────────────────────────────────────────

def amostrar(dist: ParametroDistribuicao, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Gera `n` amostras aleatórias a partir de uma ParametroDistribuicao.
    Retorna array numpy com os valores gerados.
    """
    if dist.tipo == "triangular":
        # scipy.stats.triang usa parâmetro `c` = (moda - min) / (max - min)
        loc  = dist.minimo
        escala = dist.maximo - dist.minimo
        c    = (dist.media - dist.minimo) / escala
        return stats.triang.rvs(c=c, loc=loc, scale=escala, size=n,
                                random_state=rng.integers(0, 2**31))

    elif dist.tipo == "normal":
        amostras = rng.normal(dist.media, dist.desvio_padrao, n)
        return np.clip(amostras, 0, None)  # valores negativos não fazem sentido para custos/preços

    elif dist.tipo == "uniforme":
        return rng.uniform(dist.minimo, dist.maximo, n)
    
    elif dist.tipo == "fixo":
        return np.full(n, dist.valor_fixo)

    raise ValueError(f"Tipo de distribuição inválido: {dist.tipo}")


# ─────────────────────────────────────────────
#  3. MOTOR DE SIMULAÇÃO
# ─────────────────────────────────────────────

def executar_simulacao(config: ConfiguracaoSimulacao) -> dict:
    """
    Executa a simulação de Monte Carlo e retorna os resultados brutos.

    Retorna um dict com:
      - lucros         : array com o lucro de cada simulação
      - custos_totais  : array com custo total de cada simulação
      - receitas       : array com receita de cada simulação
      - breakdowns     : dict {nome_custo: array_amostras}
      - Taxa de Administração : regra de negocio, para valores até 999 = 200 reais, pra valores > 1000 = 300 reais
      - Comissão do leiloeiro: regra de negocio: valor dinamico, 5% do valor do lance.
    """
    # Valida todas as distribuições
    config.custo_aquisicao.validar("custo_aquisicao")
    config.preco_venda.validar("preco_venda")
    for ca in config.custos_adicionais:
        if ca.distribuicao:
            ca.distribuicao.validar(ca.nome)

    n   = config.n_simulacoes
    rng = np.random.default_rng(config.semente)

    # Amostra custo de aquisição
    custo_aquisicao = amostrar(config.custo_aquisicao, n, rng)

    # Breakdown inicial
    breakdowns = {"Aquisição": custo_aquisicao}

    # =========================
    # Taxa administrativa dinâmica
    # =========================

    taxa_adm = np.where(custo_aquisicao <  1000, 200, 300)

    breakdowns["Taxa de Administração"] = taxa_adm

    # =========================
    # Comissão dinâmica (5%)
    # =========================

    comissao = custo_aquisicao * 0.05

    breakdowns["Comissão Leilão"] = comissao

    # =========================
    # Custos adicionais
    # =========================

    custos_extras = np.zeros(n)

    for ca in config.custos_adicionais:

        if ca.valor_fixo is not None:
            amostra = np.full(n, ca.valor_fixo)

        else:
            amostra = amostrar(ca.distribuicao, n, rng)

        breakdowns[ca.nome] = amostra
        custos_extras += amostra

    # =========================
    # Custo total final
    # =========================

    custos_totais = (
        custo_aquisicao
        + taxa_adm
        + comissao
        + custos_extras
    )

    # Receita de venda
    receitas = amostrar(config.preco_venda, n, rng)
    breakdowns["Receita de Venda"] = receitas

    # Lucro = Receita − Custo Total
    lucros = receitas - custos_totais

    return {
        "lucros":        lucros,
        "custos_totais": custos_totais,
        "receitas":      receitas,
        "breakdowns":    breakdowns,
    }


# ─────────────────────────────────────────────
#  4. ESTATÍSTICAS
# ─────────────────────────────────────────────

def calcular_estatisticas(resultado: dict, config: ConfiguracaoSimulacao) -> dict:
    """
    Calcula e retorna um dicionário com todas as métricas da simulação.
    """
    lucros = resultado["lucros"]
    n      = config.n_simulacoes

    prob_lucro   = np.mean(lucros > 0) * 100
    prob_prejuizo = np.mean(lucros < 0) * 100
    lucro_medio  = np.mean(lucros)
    lucro_mediano = np.median(lucros)
    lucro_min    = np.min(lucros)
    lucro_max    = np.max(lucros)
    desvio_pad   = np.std(lucros)
    custo_medio = np.mean(resultado["custos_totais"])

    roi = (
        lucro_medio / custo_medio * 100
    ) if custo_medio > 0 else 0

    # Value at Risk: perda máxima esperada com (1-confiança)% de probabilidade
    # VaR(95%) = percentil 5% da distribuição de lucros
    nivel_var = 1.0 - config.confianca_var
    var       = np.percentile(lucros, nivel_var * 100)

    # CVaR (Conditional VaR / Expected Shortfall): média dos piores casos além do VaR
    cvar = np.mean(lucros[lucros <= var])

    # Índice de Sharpe simplificado (lucro médio / desvio padrão)
    sharpe = lucro_medio / desvio_pad if desvio_pad > 0 else np.nan

    # Percentis úteis
    p10 = np.percentile(lucros, 10)
    p25 = np.percentile(lucros, 25)
    p75 = np.percentile(lucros, 75)
    p90 = np.percentile(lucros, 90)

    return {
        "n_simulacoes":   n,
        "prob_lucro":     prob_lucro,
        "prob_prejuizo":  prob_prejuizo,
        "lucro_medio":    lucro_medio,
        "lucro_mediano":  lucro_mediano,
        "lucro_min":      lucro_min,
        "lucro_max":      lucro_max,
        "desvio_padrao":  desvio_pad,
        "var":            var,
        "cvar":           cvar,
        "sharpe":         sharpe,
        "p10": p10, "p25": p25, "p75": p75, "p90": p90,
        "receita_media":  np.mean(resultado["receitas"]),
        "custo_medio":    np.mean(resultado["custos_totais"]),
        "roi": roi,
    }


# ─────────────────────────────────────────────
#  5. IMPRESSÃO DO RELATÓRIO
# ─────────────────────────────────────────────

def formatar_brl(valor: float) -> str:
    """Formata número como Real Brasileiro."""
    sinal = "-" if valor < 0 else ""
    return f"{sinal}R$ {abs(valor):>12,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def imprimir_relatorio(stats_dict: dict, config: ConfiguracaoSimulacao) -> None:
    """Imprime relatório formatado no terminal."""
    separador = "═" * 56

    print(f"\n{separador}")
    print(f"  📊  RELATÓRIO — {config.nome_operacao.upper()}")
    print(separador)

    tabela_geral = [
        ["Simulações executadas",  f"{stats_dict['n_simulacoes']:,}"],
        ["Receita média esperada",  formatar_brl(stats_dict['receita_media'])],
        ["Custo total médio",       formatar_brl(stats_dict['custo_medio'])],
        ["ROI Médio", f"{stats_dict['roi']:.2f}%"],
    ]
    print("\n▸ RESUMO GERAL")
    print(tabulate(tabela_geral, tablefmt="simple"))

    tabela_lucro = [
        ["Lucro médio",    formatar_brl(stats_dict['lucro_medio'])],
        ["Lucro mediano",  formatar_brl(stats_dict['lucro_mediano'])],
        ["Lucro mínimo",   formatar_brl(stats_dict['lucro_min'])],
        ["Lucro máximo",   formatar_brl(stats_dict['lucro_max'])],
        ["Desvio Padrão",  formatar_brl(stats_dict['desvio_padrao'])],
    ]
    print("\n▸ DISTRIBUIÇÃO DO LUCRO")
    print(tabulate(tabela_lucro, tablefmt="simple"))

    tabela_percentis = [
        ["P10", formatar_brl(stats_dict['p10'])],
        ["P25", formatar_brl(stats_dict['p25'])],
        ["P75", formatar_brl(stats_dict['p75'])],
        ["P90", formatar_brl(stats_dict['p90'])],
    ]
    print("\n▸ PERCENTIS")
    print(tabulate(tabela_percentis, tablefmt="simple"))

    nivel_pct = int(config.confianca_var * 100)
    tabela_risco = [
        ["Prob. de Lucro",  f"{stats_dict['prob_lucro']:.1f}%"],
        ["Prob. de Prejuízo", f"{stats_dict['prob_prejuizo']:.1f}%"],
        [f"VaR ({nivel_pct}%)",  formatar_brl(stats_dict['var'])],
        [f"CVaR ({nivel_pct}%)", formatar_brl(stats_dict['cvar'])],
        ["Sharpe simplificado",  f"{stats_dict['sharpe']:.3f}" if not np.isnan(stats_dict['sharpe']) else "N/A"],
    ]
    print("\n▸ RISCO")
    print(tabulate(tabela_risco, tablefmt="simple"))

    # Semáforo visual
    p = stats_dict["prob_lucro"]
    if p >= 75:
        icone, cor_texto = "🟢", "FAVORÁVEL"
    elif p >= 50:
        icone, cor_texto = "🟡", "MODERADO"
    else:
        icone, cor_texto = "🔴", "DESFAVORÁVEL"

    print(f"\n{separador}")
    print(f"  {icone}  PARECER: {cor_texto}  ({p:.1f}% de probabilidade de lucro)")
    print(separador + "\n")

def imprimir_breakdown(resultado: dict) -> None:
    """
    Exibe breakdown detalhado dos custos da operação.
    """

    breakdowns = resultado["breakdowns"]

    # Remove receita (não é custo)
    custos = {
        nome: valores
        for nome, valores in breakdowns.items()
        if nome != "Receita de Venda"
    }

    custo_total_medio = sum(np.mean(v) for v in custos.values())

    tabela = []

    for nome, valores in custos.items():

        media = np.mean(valores)
        minimo = np.min(valores)
        maximo = np.max(valores)

        participacao = (
            media / custo_total_medio * 100
        ) if custo_total_medio > 0 else 0

        tabela.append([
            nome,
            formatar_brl(media),
            formatar_brl(minimo),
            formatar_brl(maximo),
            f"{participacao:.1f}%"
        ])

    print("\n▸ BREAKDOWN DOS CUSTOS")

    print(tabulate(
        tabela,
        headers=[
            "Componente",
            "Média",
            "Mínimo",
            "Máximo",
            "% Participação"
        ],
        tablefmt="rounded_outline"
    ))    

# ─────────────────────────────────────────────
#  6. VISUALIZAÇÕES
# ─────────────────────────────────────────────

def gerar_graficos(
    resultado: dict,
    stats_dict: dict,
    config: ConfiguracaoSimulacao,
    salvar_em: Optional[str] = None,
    mostrar: bool = True,
) -> None:
    """
    Gera 4 gráficos:
      1. Histograma dos lucros com VaR marcado
      2. Curva de distribuição acumulada (CDF)
      3. Boxplot comparativo custo × receita
      4. Convergência do lucro médio ao longo das simulações
    """
    lucros  = resultado["lucros"]
    receitas = resultado["receitas"]
    custos  = resultado["custos_totais"]
    var     = stats_dict["var"]
    nivel_pct = int(config.confianca_var * 100)

    # Paleta de cores
    COR_LUCRO    = "#2ECC71"
    COR_PREJUIZO = "#E74C3C"
    COR_VAR      = "#E67E22"
    COR_MEDIA    = "#3498DB"
    COR_FUNDO    = "#F8F9FA"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(COR_FUNDO)
    fig.suptitle(
        f"Simulação de Monte Carlo — {config.nome_operacao}\n"
        f"({config.n_simulacoes:,} iterações)",
        fontsize=14, fontweight="bold", y=0.98
    )

    # ── 1. HISTOGRAMA DOS LUCROS ──────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor(COR_FUNDO)

    bins = min(80, max(30, config.n_simulacoes // 200))
    contagens, limites, patches = ax1.hist(lucros, bins=bins, edgecolor="white", linewidth=0.4)

    # Colorir barras: verde = lucro, vermelho = prejuízo
    for patch, lim_esq in zip(patches, limites[:-1]):
        patch.set_facecolor(COR_LUCRO if lim_esq >= 0 else COR_PREJUIZO)

    ax1.axvline(0,                  color="black",    linestyle="--", linewidth=1.2, label="Break-even")
    ax1.axvline(stats_dict["lucro_medio"], color=COR_MEDIA, linestyle="-",  linewidth=2,   label=f"Média: {formatar_brl(stats_dict['lucro_medio'])}")
    ax1.axvline(var,                color=COR_VAR,    linestyle=":",  linewidth=2,   label=f"VaR {nivel_pct}%: {formatar_brl(var)}")

    ax1.set_title("Distribuição dos Lucros", fontweight="bold")
    ax1.set_xlabel("Lucro (R$)")
    ax1.set_ylabel("Frequência")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Anotação de probabilidade
    ax1.text(
        0.97, 0.95,
        f"P(Lucro) = {stats_dict['prob_lucro']:.1f}%",
        transform=ax1.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

    # ── 2. CURVA CDF ─────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor(COR_FUNDO)

    lucros_sorted = np.sort(lucros)
    cdf           = np.arange(1, len(lucros_sorted) + 1) / len(lucros_sorted)

    ax2.plot(lucros_sorted, cdf * 100, color="#8E44AD", linewidth=2)
    ax2.axvline(0,   color="black",  linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.axvline(var, color=COR_VAR,  linestyle=":",  linewidth=2,
                label=f"VaR {nivel_pct}%: {formatar_brl(var)}")

    # Marcador no break-even
    prob_be = np.interp(0, lucros_sorted, cdf) * 100
    ax2.axhline(prob_be, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax2.scatter([0], [prob_be], color="black", zorder=5)
    ax2.annotate(
        f"Break-even\n{prob_be:.1f}% abaixo",
        xy=(0, prob_be), xytext=(20, -15), textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=1)
    )

    ax2.set_title("Distribuição Acumulada (CDF)", fontweight="bold")
    ax2.set_xlabel("Lucro (R$)")
    ax2.set_ylabel("Probabilidade Acumulada (%)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── 3. BOXPLOT CUSTO vs RECEITA ───────────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor(COR_FUNDO)

    dados_box  = [custos, receitas, lucros]
    labels_box = ["Custo Total", "Receita", "Lucro"]
    cores_box  = ["#E74C3C", "#2ECC71", "#3498DB"]

    bplot = ax3.boxplot(
        dados_box,
        labels=labels_box,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        widths=0.5,
    )
    for patch, cor in zip(bplot["boxes"], cores_box):
        patch.set_facecolor(cor)
        patch.set_alpha(0.7)

    ax3.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_title("Custo × Receita × Lucro", fontweight="bold")
    ax3.set_ylabel("Valor (R$)")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. CONVERGÊNCIA DA MÉDIA ──────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor(COR_FUNDO)

    # Calcula média acumulada para mostrar a convergência
    pontos_plot = np.linspace(10, config.n_simulacoes, num=min(500, config.n_simulacoes), dtype=int)
    medias_acum = [np.mean(lucros[:k]) for k in pontos_plot]

    ax4.plot(pontos_plot, medias_acum, color=COR_MEDIA, linewidth=2, label="Média acumulada")
    ax4.axhline(stats_dict["lucro_medio"], color="black", linestyle="--",
                linewidth=1.2, label=f"Convergência: {formatar_brl(stats_dict['lucro_medio'])}")
    ax4.axhline(0, color=COR_PREJUIZO, linestyle=":", linewidth=1.2, alpha=0.6, label="Break-even")

    ax4.set_title("Convergência do Lucro Médio", fontweight="bold")
    ax4.set_xlabel("Nº de Simulações")
    ax4.set_ylabel("Lucro Médio (R$)")
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if salvar_em:
        plt.savefig(salvar_em, dpi=150, bbox_inches="tight")
        print(f"  📁 Gráfico salvo em: {salvar_em}")

    if mostrar:
        plt.show()

    plt.close()


# ─────────────────────────────────────────────
#  7. FUNÇÃO PRINCIPAL (PONTO DE ENTRADA)
# ─────────────────────────────────────────────

def analisar_operacao(
    config: ConfiguracaoSimulacao,
    salvar_grafico: Optional[str] = None,
    mostrar_grafico: bool = True,
) -> dict:
    """
    Executa toda a pipeline:
      simulação → estatísticas → relatório → gráficos

    Retorna o dicionário de estatísticas para uso programático.
    """
    resultado   = executar_simulacao(config)
    stats_dict  = calcular_estatisticas(resultado, config)
    imprimir_relatorio(stats_dict, config)
    imprimir_breakdown(resultado)
    gerar_graficos(resultado, stats_dict, config,
                   salvar_em=salvar_grafico, mostrar=mostrar_grafico)
    return stats_dict

# ─────────────────────────────────────────────
#  9. OTIMIZAÇÃO DE LANCES
# ─────────────────────────────────────────────
def otimizar_lance(
    config_base: ConfiguracaoSimulacao,
    lance_min: int,
    lance_max: int,
    passo: int = 50,
):
    resultados = []
    for lance in range(lance_min, lance_max + 1, passo):
        config_teste = copy.deepcopy(config_base)

        # Define lance fixo
        config_teste.custo_aquisicao = ParametroDistribuicao(
            tipo="fixo",
            valor_fixo=lance
        )

        # Executa simulação
        resultado = executar_simulacao(config_teste)

        # Calcula estatísticas
        stats = calcular_estatisticas(resultado, config_teste)

        # Guarda métricas
        resultados.append({
            "lance": lance,
            "lucro_medio": stats["lucro_medio"],
            "prob_lucro": stats["prob_lucro"],
            "var_95": stats["var"],
            "sharpe": stats["sharpe"],
            "desvio_padrao": stats["desvio_padrao"],
            "roi": stats["roi"],
        })

    return resultados

def plotar_fronteira_risco_retorno(resultados):

    riscos = [r["desvio_padrao"] for r in resultados]
    retornos = [r["lucro_medio"] for r in resultados]
    lances = [r["lance"] for r in resultados]

    plt.figure(figsize=(10, 6))

    plt.scatter(riscos, retornos)

    # Adiciona rótulo do lance em cada ponto
    for i, lance in enumerate(lances):
        plt.annotate(
            f"R${lance}",
            (riscos[i], retornos[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.title("Fronteira Risco × Retorno")
    plt.xlabel("Risco (Desvio Padrão)")
    plt.ylabel("Retorno Esperado (Lucro Médio)")

    plt.grid(True)

    plt.show()

# ─────────────────────────────────────────────
#  8. EXEMPLOS DE USO
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ──────────────────────────────────────────
    #  CENÁRIO 1 — Custo de operação elevado
    #  Cenário simples com distribuições triangulares
    # ──────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  EXEMPLO 1 — Lote de Ferramentas (distribuição triangular)")
    print("=" * 56)

    config_ferramentas = ConfiguracaoSimulacao(
        nome_operacao="Lote de Eletronicos",

        custo_aquisicao=ParametroDistribuicao(
            tipo="triangular",
            minimo=700,    # Melhor caso: lote saiu barato
            media=900,    # Caso mais provável (moda)
            maximo=1100,   # Pior caso: disputa acirrada
        #    tipo="fixo",
        #    valor_fixo=850
        ),

        preco_venda=ParametroDistribuicao(

            #Distribuição normal de venda

            tipo="normal",
            media=2200,
            desvio_padrao=200,

            #Distribuição triangular de venda

            #tipo="triangular",
            #minimo=1800,   # Mercado desfavorável
            #media=2400,    # Preço de mercado esperado
            #maximo=3000,   # Mercado aquecido

            #Distribuição uniforme de venda

            #tipo="uniforme",
            #minimo=2000,
            #maximo=2800,
        ),

        custos_adicionais=[
            #CustoAdicional(nome="Taxa de Administração",  valor_fixo=250),  #Parametrização feita na função executar_simulacao(config) - regra de negocio
            #CustoAdicional(nome="Armazenamento", valor_fixo=100),
            #CustoAdicional(nome="Comissão leilão", valor_fixo=40),     #Parametrização feita na função executar_simulacao(config) - regra de negocio
            CustoAdicional(
                nome="Reparos / Limpeza",
                distribuicao=ParametroDistribuicao(
                    tipo="triangular",
                    minimo=450, media=700, maximo=850
                )
            ),
        ],

        n_simulacoes=10_000,
        confianca_var=0.95,
        semente=42,
    )

    stats1 = analisar_operacao(
        config_ferramentas,
        salvar_grafico="simulacao_N1.png",
        mostrar_grafico=True,  # False para ambientes sem display
    )
    resultado_otimizacao = otimizar_lance(
        config_base=config_ferramentas,
        lance_min=700,
        lance_max=1200,
        passo=50
    )

    #plotar_fronteira_risco_retorno(resultado_otimizacao)

    print("\n" + "=" * 56)
    print("  OTIMIZAÇÃO DE LANCES")
    print("=" * 56)

    for r in resultado_otimizacao:

        print(
            f"Lance: R$ {r['lance']:>4} | "
            f"Lucro Médio: R$ {r['lucro_medio']:>7.2f} | "
            f"ROI: {r['roi']:>6.2f}% | "
            f"P(Lucro): {r['prob_lucro']:.1f}% | "
            f"VaR95: R$ {r['var_95']:>7.2f} | "
            f"Sharpe: {r['sharpe']:.2f}"

        )

    # ──────────────────────────────────────────
    #  Cenário 2 — EQUIPAMENTOS ELETRONICOS (NORMAL + UNIFORME)
    #  Cenário com distribuições mistas
    # ──────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  EXEMPLO 2 — Eletrônicos (distribuição normal + uniforme)")
    print("=" * 56)

    config_eletronicos = ConfiguracaoSimulacao(
        nome_operacao="Lote Eletrônicos",

        # Custo normalmente distribuído (leilões com histórico conhecido)
        custo_aquisicao=ParametroDistribuicao(
            tipo="normal",
            media=1300,
            desvio_padrao=100,

        ),

        # Preço de venda com variação uniforme (mercado imprevisível)
        preco_venda=ParametroDistribuicao(
            tipo="uniforme",
            minimo=2000,
            maximo=2800,
        ),

        custos_adicionais=[
            #CustoAdicional(nome="Taxa de Administração",  valor_fixo=200),      #Parametrização feita na função executar_simulacao(config) - regra de negocio
            #CustoAdicional(nome="Seguro de Transporte", valor_fixo=0),
            #CustoAdicional(nome="Comissão Leilão 5%",  valor_fixo=30),              #Parametrização feita na função executar_simulacao(config) - regra de negocio
            CustoAdicional(
                nome="Teste e Certificação",
                distribuicao=ParametroDistribuicao(
                    tipo="uniforme",
                    minimo=600,
                    maximo=1000,
                )
            ),
        ],

        n_simulacoes=10_000,
        confianca_var=0.95,
        semente=99,
    )

    stats2 = analisar_operacao(
        config_eletronicos,
        salvar_grafico="simulacao_eletronicos.png",
        mostrar_grafico=False,
    )

    # ──────────────────────────────────────────
    #  EXEMPLO 3 — COMPARATIVO ENTRE CENÁRIOS
    #  Mostra como usar o retorno para comparar duas operações
    # ──────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  COMPARATIVO ENTRE OPERAÇÕES")
    print("═" * 56)

    comparativo = [
        ["Métrica",              "Ferramentas",                        "Eletrônicos"],
        ["Lucro Médio",          formatar_brl(stats1["lucro_medio"]),  formatar_brl(stats2["lucro_medio"])],
        ["Prob. Lucro",          f"{stats1['prob_lucro']:.1f}%",       f"{stats2['prob_lucro']:.1f}%"],
        ["VaR 95%",              formatar_brl(stats1["var"]),          formatar_brl(stats2["var"])],
        ["CVaR 95%",             formatar_brl(stats1["cvar"]),         formatar_brl(stats2["cvar"])],
        ["Desvio Padrão",        formatar_brl(stats1["desvio_padrao"]),formatar_brl(stats2["desvio_padrao"])],
        ["Sharpe Simplificado",  f"{stats1['sharpe']:.3f}",            f"{stats2['sharpe']:.3f}"],
    ]
    print(tabulate(comparativo, headers="firstrow", tablefmt="rounded_outline"))
    print()
