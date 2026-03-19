from utils.special_functions import SpecialFun
from external.ansys.parser import Ansys

import numpy as np
import time
import os
from datetime import datetime
import shutil

def avaliar_rodada(parameters, target_params, base_dir, local_dir=None, base_script_filename=None, noise=False):
    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
    # caminhos
    ansys_working_dir = os.path.join(local_dir if local_dir else base_dir, 'ANSYS')
    os.makedirs(ansys_working_dir, exist_ok=True)
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(local_dir if local_dir else os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

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
                  base_modes_filename, output_dir)
    ansys.set_output_filenames(out_freq_filename, out_modes_filename)
    ansys.max_attempts = 6

    # --- APLICAÇÃO DE RUÍDO NOS DADOS DE REFERÊNCIA (PROBLEMAS SINTÉTICOS) ---
    if noise:
        # Simula incerteza experimental diferente para cada rodada
        # Nível de Ruído (Sigma): 1% (0.01) ou 3% (0.03) são valores comuns
        NOISE_LEVEL = noise

        # Semente aleatória única para este processo (garante variação entre workers)
        np.random.seed(int(time.time()))

        # 1. Ruído nas Frequências (Multiplicativo)
        # freq_new = freq_old * (1 + N(0, sigma))
        freq_noise = np.random.normal(0, NOISE_LEVEL, ansys.base_freq.shape)
        ansys.base_freq = ansys.base_freq * (1 + freq_noise)

        # 2. Ruído nos Modos (Multiplicativo)
        # Afeta a amplitude de cada ponto do modo
        mode_noise = np.random.normal(0, NOISE_LEVEL, ansys.base_modes.shape)
        ansys.base_modes = ansys.base_modes * (1 + mode_noise)


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

        return fitness, {"freq error": freq_error_sum, "mac error": mac_error_sum, "Freq.": paired_comp_freq, "Freq. Error": [abs((bf-nf)/bf) for bf, nf in zip(ansys.base_freq, paired_comp_freq)], "Mode": paired_comp_modes, "MAC": macs}


    # agora basta passar os parâmetros corretos para a função e averiguar se o fitness zera para validar o modelo e script

    # params = [32.206e9, 0.2, 15e9, 0.2, 210e9, 0.3, 210e9, 0.3, 1.1e8, 9.7e7, 0.9e8] # v2: E uniforme pra laje e add do poison e dens.

    fitness, datas = fitness_function(target_params)
    print(f"\nFitness: {fitness} | Frequência: {datas["freq error"]} | Modos: {datas["mac error"]}")
    print(f"\nFrequências esperadas: {ansys.base_freq}"
          f"\nFrequências encontradas: {datas['Freq.']}"
          f"\nErros nas frequências: {datas['Freq. Error']}")
    print(f"\nModos esperados: {ansys.base_modes}"
          f"\nModos encontrados: {datas['Mode']}"
          f"\nMACs: {datas['MAC']}")

    ansys.mapdl.exit()

    return

# # avaliação 8
# parameters = [
#     Continuous(20e9, 35e9, 'modulo_concreto'),
#     Continuous(0.1, 0.49, 'poisson_concreto'),
#     # Continuous(0.25, 0.6, 'h_concreto'),
#
#     Continuous(10e9, 20e9, 'modulo_madeira'),
#     Continuous(0.1, 0.49, 'poisson_madeira'),
#
#     Continuous(150e9, 250e9, 'modulo_perfis'),
#     Continuous(0.1, 0.49, 'poisson_perfis'),
#
#     Continuous(150e9, 250e9, 'modulo_cordoalhas'),
#     Continuous(0.1, 0.49, 'poisson_cordoalhas'),
#
#     Continuous(1e7, 1e9, 'kv'),
#     Continuous(1e7, 1e9, 'kh'),
#     Continuous(1e7, 1e9, 'kt')
# ]
# params = [32.206e9, 0.2, 15e9, 0.2, 210e9, 0.3, 210e9, 0.3, 1.1e8, 9.7e7, 0.9e8] # v2: E uniforme pra laje e add do poison e dens.
# base_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 8"

# # avaliação 9
# parameters = [
#             Continuous(1e6,1e9,'GXY'),
#             Continuous(1e6, 1e9, 'GYZ'),
#             Continuous(1e6, 1e9, 'GXZ'),
#             Continuous(0.02, 0.06, 'h_concreto')
# ]
# params = [0.6e9, 0.4e9, 0.5e9, 0.04]
# base_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 9"

# # avaliação 10
# parameters = [# 10
#             Continuous(20e9, 35e9, 'modulo_concreto'),
#             Continuous(0.1, 0.49, 'poisson_concreto'),
#             Continuous(0.02, 0.06, 'h_concreto'),
#
#             Continuous(10e9, 20e9, 'modulo_madeira'),
#
#             Continuous(150e9, 250e9, 'modulo_cordoalhas'),
#
#             Continuous(1e7, 1e9, 'kv'),
#             Continuous(1e7, 1e9, 'kh'),
#
#             Continuous(1e6, 1e9, 'GXY'),
#             Continuous(1e6, 1e9, 'GYZ'),
#             Continuous(1e6, 1e9, 'GXZ')
#         ]
# params = [32.209e9, 0.2, 0.06, 15e9, 210e9, 1.1e8, 9.7e7, 1.84e8, 2.07e8, 4.06e7]
# base_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 10"
#
# avaliar_rodada(parameters, params, base_dir)
