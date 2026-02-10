from optimization.pso_optimizer.pso_optimizer import PSO
from external.parser import Ansys
from external.special_functions import SpecialFun

import os
from datetime import datetime


def PSO_run(irun, base_dir, parameters, population_size, iterations, w, w_rate, c1, c2, init_vel_ratio, log_data=None):

    keys = [p.key for p in parameters]

    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
    # caminhos
    # base_dir = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"
    ansys_working_dir = os.path.join(base_dir, 'ANSYS')
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(os.getcwd(), 'output')

    base_script_filename = "script problema 2.mac"
    base_freq_filename = "target_freq.txt"
    base_modes_filename = "target_modes.txt"

    out_freq_filename = "out_freq.txt"
    out_modes_filename = "out_modes.txt"

    ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename,
                  base_modes_filename, output_dir)
    ansys.set_output_filenames(out_freq_filename, out_modes_filename)
    ansys.max_attempts = 6

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

    # parâmetros do algoritmo:
    # w = 0.6 # proporção da velocidade atual que participa da próxima
    # w_rate = 0.99 # taxa de decaimento de inércia por iteração
    # c1 = 2.05 # influencia a exploração individual
    # c2 = 2.05 # influencia a convergência para o mínimo do grupo
    # init_vel_ratio = 0.2 # proporção do espaço de busca que pode ser empregado para velocidade inicial
    #
    # population_size = len(keys)*10 # indivíduos avaliados por geração (recomendado ao menos 10x o número de variáveis)
    # iteracoes = 30 # quantidade de iterações (suficientemente grande para a convergência do algoritmo)

    # declaração do otimizador:
    rodada = PSO(fitness_function, parameters, population_size, w, w_rate, c1, c2, init_vel_ratio) # objeto otimizador
    # rodada.set_tolerance(fit_abs = 2e-2, patience = 10) # critério de parada
    rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

    # ajuste do registro:
    log = "full"  # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
    if log_data:
        log_dir = log_data["dir"]
        log_title = log_data["title"]
    else:
        log_dir = os.path.join(base_dir, 'meta-opt', 'results', 'PSO')
        log_title = f"PSO_pop({population_size})_iter({iterations})_w({w})_wrate({w_rate})_c1({c1})_c2({c2})_initvel({init_vel_ratio})_{irun}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"  # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)

    # log_dir = None # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
    rodada.set_log(log_title, log_dir)

    # caso de retornar log csv:
    if log_data["resume"]: rodada.resume_from_log(log_data["resume"])

    # chamada:
    return rodada.run(iterations, log=log)

    # ansys.mapdl.exit()
