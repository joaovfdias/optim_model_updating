from optimization.ga_optimizer import GA
from external.parser import Ansys
from external.special_functions import SpecialFun

import numpy as np
import time
import os
from datetime import datetime
import shutil


def GA_run(irun, base_dir, parameters, population_size, generations, elitism_rate, crossover_rate, mutation_strength, local_dir=None, log_data=None):

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

    keys = [p.key for p in parameters]

    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
    # caminhos
    # base_dir = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"
    ansys_working_dir = os.path.join(local_dir if local_dir else base_dir, 'ANSYS')
    os.makedirs(ansys_working_dir, exist_ok=True)
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(local_dir if local_dir else base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    unique_ansys_dir = os.path.join(ansys_working_dir, f"worker_{irun}_{os.getpid()}")
    os.makedirs(unique_ansys_dir, exist_ok=True)

    base_script_filename = "script problema 2.mac"
    base_freq_filename = "target_freq.txt"
    base_modes_filename = "target_modes.txt"

    out_freq_filename = "out_freq.txt"
    out_modes_filename = "out_modes.txt"

    ansys = Ansys(ansys_exe_path, unique_ansys_dir, input_dir, base_script_filename, base_freq_filename,
                  base_modes_filename, output_dir)
    ansys.set_output_filenames(out_freq_filename, out_modes_filename)
    ansys.max_attempts = 6

    # --- APLICAÇÃO DE RUÍDO NOS DADOS DE REFERÊNCIA ---
    # Simula incerteza experimental diferente para cada rodada
    # Nível de Ruído (Sigma): 1% (0.01) ou 3% (0.03) são valores comuns
    NOISE_LEVEL = 0.03

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

    # parâmetros do algoritmo: serão agora passados na chamada do meta-otimizador
    # elitism_rate = 0.10 # proporção dos melhores da população que serão preservados
    # crossover_rate = 0.60 # chance de ocorrência de cruzamento entre indivíduos selecionados
    # mutation_strength = 0.10 # taxa máxima de mutação de cada gene daqueles indivíduos não originados de crossover
    #
    # population_size = len(keys)*10 # indivíduos avaliados por geração (recomendado ao menos 10x o número de variáveis)
    # generations = 30 # quantidade de iterações (suficientemente grande para a convergência do algoritmo)

    # declaração do otimizador:
    rodada = GA(fitness_function, parameters, population_size, elitism_rate, crossover_rate, mutation_strength) # objeto otimizador
    rodada.set_tolerance(fit_rel = 10e-3, patience = 15) # critério de parada
    rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

    # ajuste do registro:
    log = "full" # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
    if log_data:
        log_dir = log_data["dir"]
        log_title = log_data["title"]
    else:
        log_dir = os.path.join(base_dir, 'meta-opt', 'results', 'GA')
        log_title = f"GA_pop({population_size})_gen({generations})_elit({elitism_rate:.4f})_cross({crossover_rate:.4f})_mut({mutation_strength:.4f})_{irun}_{datetime.now().strftime("%Y%m%d_%H%M%S")}" # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)

    # log_dir = None # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
    rodada.set_log(log_title, log_dir, False)

    # caso de retornar log csv:
    if log_data and log_data["resume"]: rodada.resume_from_log(os.path.join(log_data["dir"], log_data["resume"]))

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

    # # Limpeza (Cleanup) - Executa sempre, dando erro ou sucesso
    # try:
    #     # Pequena pausa para garantir que o ANSYS liberou os arquivos .lock
    #     time.sleep(1)
    #
    #     if os.path.exists(unique_ansys_dir):
    #         shutil.rmtree(unique_ansys_dir)  # Apaga a pasta e tudo dentro
    #         # print(f"Limpeza concluída: {unique_subdir_name}") # Descomente para debug
    # except Exception as clean_error:
    #     # Não queremos parar a otimização se falhar a limpeza, apenas avise
    #     print(f"[AVISO] Não foi possível limpar a pasta {unique_subdir_name}: {clean_error}")

    return best