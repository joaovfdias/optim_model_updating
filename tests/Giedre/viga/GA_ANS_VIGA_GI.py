from optimization.parameter import *
from optimization.ga_optimizer import GA
from external.parser import Ansys
from external.special_functions import SpecialFun

import os

"""
    - são necessários arquivo de script do modelo genérico e arquivos de saída do modelo base
    - diretórios precisam ser declarados usando a formatação r"{diretório}"
    - consultar documentação das classes e métodos para entender entrada e formatos
"""

# parâmetros do modelo:
def run_viga(pop, gen, noise, MAC=None):
    parameters =    [
                    Continuous(20e9, 30e9, 'modulo'),
                    Continuous(0.1, 0.5, 'poisson'),
                    Continuous (2400, 2600, 'dens'),
                    Continuous(10e5, 10e7, 'rigidez1'),
                    Continuous(10e5, 10e7, 'rigidez2'),
                    ]

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

    # parâmetros de entrada da classe Ansys:
    # entrada obrigatória:
    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v251\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
    # entradas opcionais (caso vazias, será utilizado default: diretório \\ANSYS, arquivos "script.txt", "out_base_freq.txt" e "out_base_modes.txt"):
    ansys_working_dir = None
    input_dir = os.path.join(os.getcwd(), r'viga\input')
    base_script_filename = "script_viga.txt"
    base_freq_filename = f"out_base_freq_viga{noise}.txt" if (noise != 0) else "out_base_freq_viga.txt"
    base_modes_filename = f"out_base_modes_viga{noise}.txt" if noise else "out_base_modes_viga.txt"
    output_dir = os.path.join(os.getcwd(), 'output')
    # nome do arquivo de saída conforme configurado no script Ansys (precisa ser configurado usando Ansys.set_output_filenames):
    out_freq_filename = "out_freq_viga.txt"
    out_modes_filename = "out_modes_viga.txt"

    # objeto da classe Ansys declarado antes de fitness_function:
    ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
    ansys.set_output_filenames(out_freq_filename, out_modes_filename) # ajusta o nome dos arquivos de saída de freq e modos do Ansys, que serão gerados em ansys_working_dir

    # função objetivo (usa 'Ansys' para rodar e obter os parâmetros modais necessários e 'SpecialFun' para fazer os cálculos de erro e MAC):
    def fitness_function(param):

        input_file = ansys.create_input_file(param, keys) # gera o arquivo de input para o ansys com base na lista de parâmetros (valores) e keys (nomes)
        ansys.run_ansys(input_file) # executa esse arquivo

        comp_freq = ansys.read_frequencies() # armazena as frequências exportadas atuais
        comp_modes = ansys.read_modes() # armazena os modos exportados atuais

        paired_comp_freq, paired_comp_modes, mac_error_sum = SpecialFun.pair_modes_mac(comp_freq, comp_modes, ansys.base_modes)  # adicionada etapa de pareamento, já retorna a soma dos MAC
        freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq,
                                                     paired_comp_freq)  # parcela correspondente ao erro nas frequências

        peso_freq = 1  # ponderação
        peso_mac = 1

        fitness = peso_freq * freq_error_sum + peso_mac * mac_error_sum if MAC else peso_freq * freq_error_sum

        return fitness, {"freq error": freq_error_sum,
                         "mac error": mac_error_sum,
                         "Freq.": paired_comp_freq,
                         "Mode": paired_comp_modes}  # caso haja dados adicionais para registrar, o 2º retorno da função deve ser um dicionário com {"Identificador": Valor (escalar, vetor, matriz)}. Caso não haja, retornar apenas fitness.

    # parâmetros do algoritmo:
    elitism_rate = 0.10
    crossover_rate = 0.60
    mutation_strength = 0.10

    population_size = pop
    generations = gen

    # declaração do otimizador:
    rodada = GA(fitness_function, parameters, population_size, elitism_rate, crossover_rate, mutation_strength)
    # rodada.set_tolerance(fit_tol = 1e-2, patience = 4) # critério de parada
    rodada.sync_time(ansys.anstime) # sincroniza o log label do algoritmo e a subpasta no output do ansys para facilitar controle

    # ajuste do registro:
    log = "full"  # tipo de registro (True: simplificado, "full": todos os indivíduos)
    log_title = f"viga_GA_freq_MAC_N{noise}" if MAC else f"viga_GA_freq_N{noise}"  # alterar nome do arquivo gerado, se quiser
    log_dir = os.path.join(os.getcwd(), 'log') # alterar diretório do registro, por padrão \log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
    rodada.set_log(log_title, log_dir)

    # chamada:
    best = rodada.run(generations, log=log)
    return 0