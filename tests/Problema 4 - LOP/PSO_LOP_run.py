from optimization.pso_optimizer.pso_optimizer import PSO
from external.ansys.parser import Ansys
from utils.special_functions import SpecialFun

import numpy as np
import time
import os
from datetime import datetime
import shutil


def PSO_run(irun, parameters, base_dir, local_dir=None, log_dir=None, base_script_filename=None, noise=False, population_size=None, iterations=None, w=0.6, w_rate=0.99, c1=2.05, c2=2.05, init_vel_ratio=0.2):

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

    keys = [p.key for p in parameters]

    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"

    # caminhos
    ansys_working_dir = os.path.join(local_dir if local_dir else base_dir, 'ANSYS')
    os.makedirs(ansys_working_dir, exist_ok=True)
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(local_dir if local_dir else os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    unique_ansys_dir = os.path.join(ansys_working_dir, f"worker_PSO_{irun}_{os.getpid()}")
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
        np.random.seed(int(time.time()) + irun)

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

    population_size = population_size or len(keys)*10 # indivíduos avaliados por geração (recomendado ao menos 10x o número de variáveis)
    iterations = iterations or round(8*len(keys)) # quantidade de iterações (suficientemente grande para a convergência do algoritmo)

    # declaração do otimizador:
    rodada = PSO(fitness_function, parameters, population_size, w, w_rate, c1, c2, init_vel_ratio) # objeto otimizador
    rodada.set_tolerance(fit_rel = 1e-3, patience = round(0.20*iterations)) # critério de parada
    rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

    # ajuste do registro:
    log = "full" # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
    log_dir = log_dir or os.path.join(base_dir, 'log', 'runs', 'PSO')
    log_title = f"PSO_pop({population_size})_iter({iterations})_{irun}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"  # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)

    # log_dir = None # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
    rodada.set_log(log_title, log_dir, False)

    # chamada:
    try:
        best = rodada.run(iterations, log=log)
    finally:
        try:
            ansys.mapdl.exit(force=True)
        except:
            pass

    executed_iterations = len(rodada.populations)

    # garantia de encerramento
    # Ansys.kill_ansys_process()
    time.sleep(1)

    return best, executed_iterations