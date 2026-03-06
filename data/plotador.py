import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PARTE 1: O PLOTADOR GENÉRICO (ROBUSTO E REUTILIZÁVEL)
# =============================================================================

def plot_convergencia(series_dados, titulo="Histórico de Convergência", xlabel="Iterações / Avaliações",
                      ylabel="Fitness", escala_log=True, salvar_como=None):
    """
    Plota múltiplas curvas de convergência com área de desvio padrão.

    :param series_dados: Lista de dicionários. Ex:
                         [{'label': 'GA', 'x': [1,2,3], 'y_media': [10,5,2], 'y_desvio': [1, 0.5, 0.1], 'cor': 'blue'}, ...]
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for serie in series_dados:
        x = np.array(serie['x'])
        y_media = np.array(serie['y_media'])
        y_desvio = np.array(serie.get('y_desvio', np.zeros_like(y_media)))
        cor = serie.get('cor', None)  # Se None, o matplotlib escolhe automaticamente
        label = serie.get('label', 'Série')

        # Plota a linha principal da média
        linha, = ax.plot(x, y_media, label=label, color=cor, linewidth=2)

        # Plota a área de sombra (Desvio Padrão)
        if np.any(y_desvio > 0):
            ax.fill_between(x,
                            y_media - y_desvio,
                            y_media + y_desvio,
                            color=linha.get_color(),
                            alpha=0.2,  # Transparência da sombra
                            edgecolor="none")

    if escala_log:
        ax.set_yscale('log')

    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()

    if salvar_como:
        fig.savefig(salvar_como, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {salvar_como}")

    plt.show()


def plot_boxplot_parametros(nome_parametro, valor_esperado, grupos_dados, titulo=None, salvar_como=None):
    """
    Plota boxplots lado a lado para comparar a distribuição final de um parâmetro.

    :param valor_esperado: Float. O valor real do gabarito para traçar a linha horizontal.
    :param grupos_dados: Lista de dicionários. Ex:
                         [{'label': 'PSO Conj 1', 'valores': [25e9, 24e9, 26e9, 25.5e9]}, ...]
    """
    labels = [g['label'] for g in grupos_dados]
    valores = [g['valores'] for g in grupos_dados]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Criando o boxplot (showmeans=True plota um triângulo verde indicando a média)
    bplot = ax.boxplot(valores, labels=labels, patch_artist=True,
                       showmeans=True, meanline=False,
                       boxprops=dict(facecolor='lightblue', color='black', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow'))

    # Traçando a linha do valor real/esperado
    linha_alvo = ax.axhline(valor_esperado, color='green', linestyle='--', linewidth=2,
                            label=f'Gabarito: {valor_esperado:.2e}')

    if titulo is None:
        titulo = f"Distribuição Final: {nome_parametro}"

    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_ylabel(f"Valor de {nome_parametro}", fontsize=12)

    # Customizando a legenda para explicar o que é a caixa
    ax.plot([], [], color='red', linewidth=2, label='Mediana')
    ax.plot([], [], marker='D', color='w', markerfacecolor='yellow', markeredgecolor='black', label='Média')
    ax.legend(loc='best', fontsize=10)

    ax.grid(True, axis='y', ls="--", alpha=0.7)
    fig.tight_layout()

    if salvar_como:
        fig.savefig(salvar_como, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {salvar_como}")

    plt.show()


# =============================================================================
# PARTE 2: O CONVERSOR/EXTRATOR DE DADOS DOS SEUS ARQUIVOS .CSV
# =============================================================================

def carregar_dados_das_pastas(log_dir_global, cores_personalizadas:dict=None):
    """
    Vasculha as pastas do log global, encontra os arquivos 'Convergencia_*.csv' e
    os converte no formato exato exigido pelos plotadores genéricos acima.
    """
    dados_convergencia = []
    dados_parametros_finais = {}  # Dicionário: {'modulo_concreto': [{'label': 'PSO', 'valores': [...]}]}

    if cores_personalizadas is None:
        cores_personalizadas = {"PSO": "blue", "GA": "green", "BO": "red"}

    # Procura recursivamente por todos os arquivos de Convergência nas subpastas
    caminho_busca = os.path.join(log_dir_global, "**", "Convergencia_*.csv")
    arquivos_csv = glob.glob(caminho_busca, recursive=True)

    if not arquivos_csv:
        print("Nenhum arquivo de convergência encontrado nas pastas.")
        return [], {}

    for arquivo in arquivos_csv:
        # Extrai os nomes das pastas para usar de Rótulo (Label)
        partes_caminho = arquivo.split(os.sep)
        nome_conjunto = partes_caminho[-2]  # Ex: "Conjunto 1"
        nome_algo = partes_caminho[-3]  # Ex: "GA"
        label_curva = f"{nome_algo} ({nome_conjunto})"

        try:
            df = pd.read_csv(arquivo, sep=';', decimal=',')

            # 1. Extraindo dados para a Curva de Convergência
            if 'Media_Fitness' in df.columns:
                dados_convergencia.append({
                    'label': label_curva,
                    'x': df['Iteracao'].tolist(),
                    'y_media': df['Media_Fitness'].tolist(),
                    'y_desvio': df['Desvio_Fitness'].tolist() if 'Desvio_Fitness' in df.columns else None,
                    'cor': cores_personalizadas.get(nome_algo, None)  # Usa a cor do algoritmo
                })

            # 2. Extraindo dados da ÚLTIMA LINHA para os Boxplots de Parâmetros
            # Pega apenas as colunas que começam com "Run" e NÃO são Fitness
            colunas_runs = [c for c in df.columns if c.startswith("Run") and "Fitness" not in c]

            # Itera sobre os nomes reais dos parâmetros (ex: "modulo_concreto")
            # Descobrimos o nome do parâmetro removendo o "RunX_" do início da coluna
            parametros_unicos = set(["_".join(c.split("_")[1:]) for c in colunas_runs])

            ultima_linha = df.iloc[-1]  # A última linha contém o valor final convergido

            for param in parametros_unicos:
                # Pega as colunas exatas das 4 runs para ESTE parâmetro (Ex: Run1_modulo_concreto, Run2_...)
                cols_deste_param = [c for c in colunas_runs if c.endswith(f"_{param}")]
                valores_finais_runs = ultima_linha[cols_deste_param].dropna().astype(float).tolist()

                if param not in dados_parametros_finais:
                    dados_parametros_finais[param] = []

                dados_parametros_finais[param].append({
                    'label': label_curva,
                    'valores': valores_finais_runs
                })

        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")

    return dados_convergencia, dados_parametros_finais


# =============================================================================
# EXEMPLO DE USO GERAL (COMO CHAMAR O CÓDIGO)
# =============================================================================
if __name__ == '__main__':

    from tests.indexador_2026 import indexar_problema, indexar_device

    # 1. Defina o caminho onde a sua rodada foi salva
    pasta_da_rodada = r"C:\Users\thiag\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 4\log\rodada_20260305_135559"

    # dados específicos
    problema = 4
    dadosprob = indexar_problema(problema)

    # (Opcional) Dicionário do seu Gabarito para o Boxplot puxar a linha certa
    # gabarito_real = {
    #     'modulo_concreto': 32.209e9,
    #     'poisson_concreto': 0.20,
    #     'h_concreto': 0.06,
    #     'kv': 1.1e8,
    #     # ... adicione os outros
    # }
    gabarito_real = dadosprob.expected_values

    print("Extraindo dados...")
    # As cores vão agrupar visualmente o PSO em azul, GA verde, BO vermelho (ficará lindo)
    dados_conv, dados_box = carregar_dados_das_pastas(pasta_da_rodada)

    if dados_conv:
        print("Plotando Convergência Geral...")
        plot_convergencia(dados_conv,
                          titulo="Convergência Global dos Algoritmos",
                          escala_log=True)

        print("Plotando Boxplots Físicos...")
        for param_nome, lista_grupos in dados_box.items():
            if param_nome in gabarito_real:
                valor_esperado = gabarito_real[param_nome]
                plot_boxplot_parametros(param_nome, valor_esperado, lista_grupos)