from optimization.parameter import *
from optimization.ga_optimizer import GA
from external.parser import Ansys
from external.special_functions import SpecialFun

import os


def GA_run(parameters, irun):

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

    keys = [p.key for p in parameters]

    ansys_exe_path = r"D:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
    ansys_working_dir = None
    input_dir = os.path.join(os.getcwd(), 'input')
    base_script_filename = "script.mac"
    base_freq_filename = "out_freq.txt"
    base_modes_filename = ["out_modos_x.txt", "out_modos_y.txt", "out_modos_z.txt"]
    output_dir = os.path.join(os.getcwd(), 'output')

    out_freq_filename = "out_freq.txt"
    out_modes_filename = ["out_modos_x.txt", "out_modos_y.txt", "out_modos_z.txt"]

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

        return fitness, {"Freq.": paired_comp_freq, "Mode": paired_comp_modes}

    # parâmetros do algoritmo:
    elitism_rate = 0.10 # proporção dos melhores da população que serão preservados
    crossover_rate = 0.60 # chance de ocorrência de cruzamento entre indivíduos selecionados
    mutation_strength = 0.10 # taxa máxima de mutação de cada gene daqueles indivíduos não originados de crossover

    population_size = len(keys)*10 # indivíduos avaliados por geração (recomendado ao menos 10x o número de variáveis)
    generations = 30 # quantidade de iterações (suficientemente grande para a convergência do algoritmo)

    # declaração do otimizador:
    rodada = GA(fitness_function, parameters, population_size, elitism_rate, crossover_rate, mutation_strength) # objeto otimizador
    # rodada.set_tolerance(fit_abs = 2e-2, patience = 10) # critério de parada
    rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

    # ajuste do registro:
    log = "full" # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
    log_title = f"GA_trel_{irun}" # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)
    log_dir = None # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
    rodada.set_log(log_title, log_dir)

    # caso queira retomar a rodada de algum log cvs:
    # rodada.resume_from_log(r"C:\Users\Thiago Artur\Documents\.Mestrado (Local)\PyGit\tests\test log recovery\teste_GA_log_completo.csv")

    # chamada:
    best = rodada.run(generations, log=log)