import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PARTE 1: FUNÇÕES DE PLOTAGEM (TOTALMENTE DESACOPLADAS)
# =============================================================================

def plot_convergencia(series_dados, titulo="Histórico de Convergência", xlabel="Iterações / Avaliações",
                      ylabel="Fitness", escala_log=True, limite_y=None, salvar_como=None):
    """
    Plota múltiplas curvas de convergência com área de desvio padrão.

    :param series_dados: Lista de dicionários. Ex:
                         [{'label': 'GA', 'x': [1,2,3], 'y_media': [10,5,2], 'y_desvio': [1, 0.5, 0.1], 'cor': 'blue'}, ...]
    :param limite_y: Tupla (ymin, ymax). Força a escala do eixo Y para ser idêntica em diferentes gráficos.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    max_x = 0
    for serie in series_dados:
        x = np.array(serie['x'])

        # Converte para float e transforma qualquer NaN (vazio) em zero para não quebrar a matemática
        y_media = np.nan_to_num(np.array(serie['y_media'], dtype=float))

        y_desvio_bruto = serie.get('y_desvio')
        if y_desvio_bruto is None:
            y_desvio = np.zeros_like(y_media)
        else:
            y_desvio = np.nan_to_num(np.array(y_desvio_bruto, dtype=float))

        cor = serie.get('cor', None)  # Se None, o matplotlib escolhe automaticamente
        label = serie.get('label', 'Série')

        max_x = max(max_x, len(x))

        # Plota a linha principal (média)
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

    if limite_y is not None:
        ax.set_ylim(limite_y)  # Essencial para comparar gráficos diferentes lado a lado

    ax.set_xlim(left=0, right=max_x)  # Ajusta o eixo X exatamente para o tamanho das iterações

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
    Plota boxplots lado a lado para comparar a distribuição final de UM parâmetro entre vários algoritmos/conjuntos.

    :param valor_esperado: Float. O valor real do gabarito para traçar a linha horizontal.
    :param grupos_dados: Lista de dicionários. Ex:
                         [{'label': 'PSO Conj 1', 'valores': [25e9, 24e9, 26e9, 25.5e9]}, ...]
    """
    labels = [g['label'] for g in grupos_dados]
    valores = [g['valores'] for g in grupos_dados]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Criando o boxplot (showmeans=True plota um triângulo verde indicando a média)
    ax.boxplot(valores, labels=labels, patch_artist=True,
               showmeans=True, meanline=False,
               boxprops=dict(facecolor='lightblue', color='black', alpha=0.7),
               medianprops=dict(color='red', linewidth=2),
               meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow'))

    # Traçando a linha do valor real/esperado
    ax.axhline(valor_esperado, color='green', linestyle='--', linewidth=2, label=f'Gabarito: {valor_esperado:.2e}')

    titulo = titulo or f"Distribuição Final: {nome_parametro}"
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


def plot_boxplot_erros_conjunto(nome_conjunto, dados_parametros, gabarito_real, titulo=None, salvar_como=None):
    """
    NOVO: Plota o Erro Relativo (%) de TODOS os parâmetros para um ÚNICO conjunto/algoritmo.
    """
    labels = []
    erros_percentuais = []

    for param, valores in dados_parametros.items():
        if param in gabarito_real:
            val_esperado = gabarito_real[param]
            # Calcula o erro relativo de cada rodada: ((Val - Esp) / Esp) * 100
            erros = [abs((v - val_esperado) / val_esperado) * 100 for v in valores]
            labels.append(param)
            erros_percentuais.append(erros)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(erros_percentuais, labels=labels, patch_artist=True,
               showmeans=True,
               boxprops=dict(facecolor='lightcoral', color='black', alpha=0.7),
               medianprops=dict(color='black', linewidth=2),
               meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow'))

    ax.axhline(0, color='green', linestyle='--', linewidth=2, label='Erro Zero (Gabarito)')

    titulo = titulo or f"Erro Relativo (%) dos Parâmetros - {nome_conjunto}"
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_ylabel("Erro Relativo (%)", fontsize=12)
    ax.set_yscale('symlog', linthresh=1.0)  # Usa escala log, mas permite chegar ao zero

    # Rotaciona os nomes dos parâmetros se forem muito grandes
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, axis='y', ls="--", alpha=0.7)

    fig.tight_layout()
    if salvar_como:
        fig.savefig(salvar_como, dpi=300, bbox_inches='tight')
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
    dados_box_por_param = {}  # Para comparar 1 param entre vários algoritmos
    dados_box_por_conjunto = {}  # Para comparar todos os params de 1 algoritmo

    # Procura recursivamente por todos os arquivos de Convergência nas subpastas
    cores_personalizadas = cores_personalizadas or {"PSO": "blue", "GA": "green", "BO": "red"}
    arquivos_csv = glob.glob(os.path.join(log_dir_global, "**", "Convergencia_*.csv"), recursive=True)

    if not arquivos_csv:
        print("Nenhum arquivo de convergência encontrado nas pastas.")
        return [], {}

    for arquivo in arquivos_csv:
        # Extrai os nomes das pastas para usar de Rótulo (Label)
        partes = arquivo.split(os.sep)
        nome_conjunto, nome_algo = partes[-2], partes[-3]
        label_curva = f"{nome_algo} - {nome_conjunto}"

        try:
            df = pd.read_csv(arquivo, sep=';', decimal=',')

            # Garante que as colunas sejam convertidas para floats matemáticos
            for col in df.columns:
                if df[col].dtype == 'object':  # Se o Pandas interpretou como texto
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

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

            if label_curva not in dados_box_por_conjunto:
                dados_box_por_conjunto[label_curva] = {}

            for param in parametros_unicos:
                # Pega as colunas exatas das 4 runs para ESTE parâmetro (Ex: Run1_modulo_concreto, Run2_...)
                cols_deste_param = [c for c in colunas_runs if c.endswith(f"_{param}")]
                valores = ultima_linha[cols_deste_param].dropna().astype(float).tolist()

                # Alimenta o dicionário focado no Parâmetro
                if param not in dados_box_por_param:
                    dados_box_por_param[param] = []
                dados_box_por_param[param].append({'label': label_curva, 'valores': valores})

                # Alimenta o dicionário focado no Conjunto
                dados_box_por_conjunto[label_curva][param] = valores

        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")

    return dados_convergencia, dados_box_por_param, dados_box_por_conjunto


# =============================================================================
# EXEMPLO DE USO GERAL (COMO CHAMAR O CÓDIGO)
# =============================================================================
if __name__ == '__main__':

    from tests.indexador_2026 import indexar_problema, indexar_device

    # 1. Defina o caminho onde a sua rodada foi salva
    pasta_da_rodada = r"C:\Users\thiag\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 4\log\rodada_20260305_135559"
    # Caminho de salvamento
    pasta_graficos = os.path.join(pasta_da_rodada, "Graficos")
    os.makedirs(pasta_graficos, exist_ok=True)

    # dados específicos
    problema = 4
    dadosprob = indexar_problema(problema)

    gabarito_real = dadosprob.expected_values # dict

    print("Extraindo dados...")
    dados_conv, box_param, box_conjunto = carregar_dados_das_pastas(pasta_da_rodada)

    # 1. CURVA DE CONVERGÊNCIA (Todos juntos, com escala Y travada para comparação justa)
    if dados_conv:
        print("Plotando Convergência Geral...")
        plot_convergencia(dados_conv,
                          titulo="Convergência Global",
                          limite_y=(1e-4, 1e2),  # <-- AJUSTE AQUI SUA TRAVA DE ESCALA!
                          salvar_como=os.path.join(pasta_graficos, "Convergencia_Todos.png"))

    # 2. BOXPLOT: 1 Parâmetro vs Todos os Algoritmos
    print("\nPlotando parâmetros individuais...")
    for param_nome, lista_grupos in box_param.items():
        if param_nome in gabarito_real:
            plot_boxplot_parametros(param_nome, gabarito_real[param_nome], lista_grupos,
                                    salvar_como=os.path.join(pasta_graficos, f"Boxplot_{param_nome}.png"))

    # 3. BOXPLOT: Todos os Parâmetros (Erro %) vs 1 Algoritmo
    print("\nPlotando visão global de erros por conjunto...")
    for conjunto_nome, dict_parametros in box_conjunto.items():
        plot_boxplot_erros_conjunto(conjunto_nome, dict_parametros, gabarito_real,
                                    salvar_como=os.path.join(pasta_graficos,
                                                             f"Erro_Global_{conjunto_nome.replace(' ', '_')}.png"))