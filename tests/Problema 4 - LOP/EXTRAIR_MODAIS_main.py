from utils.special_functions import SpecialFun
from external.ansys.parser import Ansys

import numpy as np
import pandas as pd
import time
import os
import glob
from datetime import datetime
import shutil


def avaliar_rodada(parameters, base_dir, csv_convergencia, local_dir=None, base_script_filename=None, noise=False):
    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros

    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"

    # Camadas e pastas de trabalho do Ansys
    ansys_working_dir = os.path.join(local_dir if local_dir else base_dir, 'ANSYS')
    os.makedirs(ansys_working_dir, exist_ok=True)
    input_dir = os.path.join(base_dir, 'input')

    # Diretório temporário e isolado para o Ansys rodar sem conflito de sobrescrita
    unique_ansys_dir = os.path.join(ansys_working_dir, f"worker_avaliacao_{os.getpid()}")
    os.makedirs(unique_ansys_dir, exist_ok=True)

    try:
        shutil.copy(os.path.join(base_dir, "ModBase.db"), unique_ansys_dir)
    except FileNotFoundError:
        print("\nAviso: não existe ModBase.db na pasta base. Nenhuma cópia foi feita.")

    base_script_filename = base_script_filename or "script.mac"
    base_freq_filename = "target_freq.txt"
    base_modes_filename = "target_modes.txt"

    out_freq_filename = "out_freq.txt"
    out_modes_filename = "out_modes.txt"

    ansys = Ansys(ansys_exe_path, unique_ansys_dir, input_dir, base_script_filename, base_freq_filename,
                  base_modes_filename, os.path.join(unique_ansys_dir, 'output'))
    ansys.set_output_filenames(out_freq_filename, out_modes_filename)
    ansys.max_attempts = 6

    def fitness_function(param):
        input_file = ansys.create_input_file(param, keys)
        ansys.run_ansys(input_file, True, True)

        comp_freq = ansys.read_frequencies()
        comp_modes = ansys.read_modes()

        paired_comp_freq, paired_comp_modes, mac_error_sum, macs = SpecialFun.pair_modes_mac(
            comp_freq, comp_modes, ansys.base_modes
        )
        freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, paired_comp_freq)

        peso_freq = 1
        peso_mac = 1
        fitness = peso_freq * freq_error_sum + peso_mac * mac_error_sum

        return fitness, {"Freq.": paired_comp_freq, "Mode": paired_comp_modes, "Mac": macs}

    # =========================================================================
    # LÓGICA DE EXTRAÇÃO E SALVAMENTO NA PASTA DE ORIGEM DO CSV
    # =========================================================================
    pasta_destino_csv = os.path.dirname(csv_convergencia)

    # Lê o CSV. Decimal = '.' porque o arquivo gerado pelo seu log usa notação científica/ponto nos números
    df = pd.read_csv(csv_convergencia, sep=';', decimal='.')

    # Identifica dinamicamente todas as colunas de Fitness de cada Run (ex: 'Run1_Fitness', 'Run2_Fitness')
    run_cols = [c for c in df.columns if c.startswith("Run") and c.endswith("Fitness")]
    runs = [c.split('_')[0] for c in run_cols]

    resultados_finais = []

    for run in runs:
        # Encontra a iteração (linha) que obteve o menor fitness para esta run em específico
        min_idx = df[f'{run}_Fitness'].idxmin()

        # Coleta o melhor valor de cada parâmetro do modelo cruzando a chave 'key' com o prefixo 'RunX_'
        run_params = []
        for key in keys:
            col_name = f'{run}_{key}'
            run_params.append(df.loc[min_idx, col_name])

        print(f"\n--- Avaliando {run} (Menor Fitness Encontrado na Iteração {df.loc[min_idx, 'Iteracao']}) ---")

        # Executa a simulação com os melhores parâmetros
        fitness, datas = fitness_function(run_params)

        freqs = datas["Freq."]
        macs = datas["Mac"]
        base_freqs = ansys.base_freq

        # Cálculo de erro relativo das frequências (em %) frente a frequência de referência
        erro_freqs = [abs(f - bf) / bf for f, bf in zip(freqs, base_freqs)]

        # Monta o dicionário com resultados que será uma linha no CSV final
        res_dict = {"Run": run}

        for i, (f, ef, mac) in enumerate(zip(freqs, erro_freqs, macs)):
            res_dict[f"Freq. {i + 1}"] = f
            res_dict[f"Erro Freq. {i + 1} (%)"] = ef
            res_dict[f"MAC {i + 1}"] = mac

        resultados_finais.append(res_dict)

    # Transforma a lista de dicionários num DataFrame
    res_df = pd.DataFrame(resultados_finais)

    # Calcula as métricas estatísticas de fechamento (Média e Desvio) descartando a coluna texto "Run"
    mean_row = res_df.mean(numeric_only=True).to_dict()
    mean_row["Run"] = "Media"

    std_row = res_df.std(numeric_only=True).to_dict()
    std_row["Run"] = "Desvio"

    # Concatena as estatísticas ao DataFrame final
    res_df = pd.concat([res_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # Salva na mesma pasta do arquivo original com prefixo
    out_csv = os.path.join(pasta_destino_csv, f"Modais_compiled.csv")
    res_df.to_csv(out_csv, index=False, sep=';', decimal='.')

    print(f"\nResultados extraídos e salvos com sucesso em: {out_csv}")

    return res_df


def processar_todos_csvs(diretorio_raiz, parameters, base_dir, local_dir=None, base_script_filename=None):
    """
    Varre o diretorio_raiz recursivamente buscando 'Convergencia_*.csv' e aciona a avaliação do Ansys.
    """
    padrao = os.path.join(diretorio_raiz, "**", "Convergencia_*.csv")
    arquivos_csv = glob.glob(padrao, recursive=True)

    if not arquivos_csv:
        print(f"Nenhum arquivo encontrado com o padrão 'Convergencia_*.csv' em {diretorio_raiz}")
        return

    print(f"Encontrados {len(arquivos_csv)} arquivos para processar.\n")

    for csv_path in arquivos_csv:
        print(f"\n{'-' * 60}\nProcessando: {csv_path}\n{'-' * 60}")
        avaliar_rodada(
            parameters=parameters,
            base_dir=base_dir,
            csv_convergencia=csv_path,
            local_dir=local_dir,
            base_script_filename=base_script_filename
        )
        print(f"Resultados de {os.path.basename(csv_path)} extraídos com sucesso.")


# =============================================================================
# EXCECUÇÃO PRINCIPAL (Altere os caminhos e parâmetros conforme sua estrutura)
# =============================================================================
if __name__ == "__main__":

    from tests.indexador_2026 import *

    computador = "DESKTOP"
    pc = indexar_device(computador)

    for problema in [1]:
        dadosp = indexar_problema(problema)

        base_dir = os.path.join(pc.base_path, f"Problema {problema}")
        results_dir = os.path.join(base_dir, "Resultados")
        ansys_dir = os.path.join(pc.local_path, f"Problema {problema}")

        parameters = dadosp.parameters

        processar_todos_csvs(results_dir, parameters, base_dir, ansys_dir, base_script_filename=dadosp.script_filename)