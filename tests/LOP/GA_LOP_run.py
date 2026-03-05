from optimization.ga_optimizer import GA
from external.parser import Ansys
from external.special_functions import SpecialFun

import numpy as np
import time
import os
from datetime import datetime
import shutil


def GA_run(irun, parameters, base_dir, local_dir=None, log_dir=None, base_script_filename=None, noise=False, population_size=None, generations=None, elitism_rate=0.10, crossover_rate=0.60, mutation_strength=0.10, selection_method='tournament'):

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

    keys = [p.key for p in parameters]

    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"

    # caminhos
    ansys_working_dir = os.path.join(local_dir if local_dir else base_dir, 'ANSYS')
    os.makedirs(ansys_working_dir, exist_ok=True)
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(local_dir if local_dir else base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    unique_ansys_dir = os.path.join(ansys_working_dir, f"worker_GA_{irun}_{os.getpid()}")
    os.makedirs(unique_ansys_dir, exist_ok=True)
    shutil.copy(os.path.join(base_dir, "ModBase.db"), unique_ansys_dir)

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

        paired_comp_freq, paired_comp_modes, mac_error_sum = SpecialFun.pair_modes_mac(
            comp_freq, comp_modes, ansys.base_modes
        )
        freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, paired_comp_freq)

        peso_freq = 1
        peso_mac = 1
        fitness = peso_freq * freq_error_sum + peso_mac * mac_error_sum

        return fitness, {"freq error": freq_error_sum, "mac error": mac_error_sum, "Freq.": paired_comp_freq, "Mode": paired_comp_modes}

    population_size = population_size or len(keys)*10 # indivíduos avaliados por geração (recomendado ao menos 10x o número de variáveis)
    generations = generations or round(8*len(keys)) # quantidade de iterações (suficientemente grande para a convergência do algoritmo)

    # declaração do otimizador:
    rodada = GA(fitness_function, parameters, population_size, elitism_rate, crossover_rate, mutation_strength) # objeto otimizador
    rodada.set_selection_parents(selection_method)
    rodada.set_tolerance(fit_rel = 1e-3, patience = round(0.25*generations)) # critério de parada
    rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

    # ajuste do registro:
    log = "full" # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
    log_dir = log_dir or os.path.join(base_dir, 'log', 'runs', 'GA')
    log_title = f"GA_pop({population_size})_gen({generations})_{irun}_{datetime.now().strftime("%Y%m%d_%H%M%S")}" # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)

    # log_dir = None # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
    rodada.set_log(log_title, log_dir, False)

    # chamada:
    try:
        best = rodada.run(generations, log=log)
    finally:
        try:
            ansys.mapdl.exit(force=True)
        except:
            pass

    # garantia de encerramento
    # Ansys.kill_ansys_process()
    time.sleep(1)

    return best