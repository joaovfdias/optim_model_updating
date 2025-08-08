from optimization.parameter import *
from optimization.bo_optimizer.bayesian import BO
from external.parser import Ansys
from external.special_functions import SpecialFun

import os

"""
    TEMPLATE DE CHAMADA DO ALGORITMO BO PARA CALIBRAÇÃO USANDO ANSYS
    - são necessários arquivo de script do modelo genérico e arquivos de saída do modelo base
    - diretórios precisam ser declarados usando a formatação r"{diretório}"
    - consultar documentação das classes e métodos em caso de dúvidas com entrada e formatos
"""

def BO_run_laje(input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir, fitness_metric, initial_points, evaluations, acq_func="PI", xi=None, run_idx=1):
    # parâmetros do modelo:
    parameters =    [
                    Continuous(20e9, 30e9, 'modulo'),
                    Continuous(0.1, 0.49, 'poisson'),
                    Continuous (2400, 2600, 'dens'),
                    Continuous(10e5, 10e7, 'rigidez1'),
                    Continuous(10e5, 10e7, 'rigidez2'),
                    Continuous(10e5, 10e7, 'rigidez3'),
                    Continuous(10e5, 10e7, 'rigidez4')
                    ]

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

    # parâmetros de entrada da classe Ansys:
    # entrada obrigatória:
    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v251\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
    # entradas opcionais (caso vazias, será utilizado default: {diretório atual}\ANSYS, arquivos "script.txt", "out_base_freq.txt" e "out_base_modes.txt"):
    ansys_working_dir = None
    #     input_dir = os.path.join(os.getcwd(), 'input') # {diretório atual}\input (localização do script e dados de referência)
    #     base_script_filename = "script_laje.txt"
    #     base_freq_filename = "out_base_freq_laje.txt"
    #     base_modes_filename = "out_base_modes_laje.txt"
    #     output_dir = os.path.join(os.getcwd(), 'output_BO') # {diretório atual}\output (onde serão armazenados os scripts executáveis do Ansys)
    # nome do arquivo de saída conforme configurado no script Ansys (alterar usando Ansys.set_output_filenames):
    out_freq_filename = "out_freq_laje.txt"
    out_modes_filename = "out_modes_laje.txt"

    # objeto da classe Ansys declarado antes de fitness_function:
    ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
    ansys.set_output_filenames(out_freq_filename, out_modes_filename) # ajusta o nome dos arquivos de saída de freq. e modos do Ansys, que serão gerados em ansys_working_dir
    ansys.max_attempts = 6 # define quantas tentativas de rodada o Ansys executa em caso de erro

    # função objetivo com pareamento (usa 'Ansys' para rodar e obter os parâmetros modais necessários e 'SpecialFun' para fazer os cálculos de erro e MAC):
    def fitness_function_freq_mac(param):

        input_file = ansys.create_input_file(param, keys) # gera o arquivo de input para o ansys com base na lista de parâmetros (valores) e keys (nomes)
        ansys.run_ansys(input_file, True, True) # executa esse arquivo (sinalizar quais dados ele espera que o Ansys retorne)

        comp_freq = ansys.read_frequencies() # armazena as frequências exportadas atuais
        comp_modes = ansys.read_modes() # armazena os modos exportados atuais

        paired_comp_freq, paired_comp_modes, mac_error_sum = SpecialFun.pair_modes_mac(comp_freq, comp_modes, ansys.base_modes) # adicionada etapa de pareamento, já retorna a somatória de (1-mac)
        freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, paired_comp_freq) # parcela correspondente ao erro nas frequências

        # ponderação:
        peso_freq = 1
        peso_mac = 1

        fitness = peso_freq * freq_error_sum + peso_mac * mac_error_sum

        return fitness, {"freq error": freq_error_sum, "mac error": mac_error_sum, "Freq.": paired_comp_freq, "Mode": paired_comp_modes} # caso haja dados adicionais para registrar, o 2º retorno da função deve ser um dicionário com {"Identificador": Valor (escalar, vetor, matriz)}. Caso não haja, retornar apenas fitness.

    def fitness_function_freq(param):

        input_file = ansys.create_input_file(param, keys) # gera o arquivo de input para o ansys com base na lista de parâmetros (valores) e keys (nomes)
        ansys.run_ansys(input_file, True, False) # executa esse arquivo (sinalizar quais dados ele espera que o Ansys retorne)

        comp_freq = ansys.read_frequencies() # armazena as frequências exportadas atuais

        freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, comp_freq) # parcela correspondente ao erro nas frequências

        # ponderação:
        peso_freq = 1

        fitness = peso_freq * freq_error_sum

        return fitness, {"Freq.": comp_freq} # caso haja dados adicionais para registrar, o 2º retorno da função deve ser um dicionário com {"Identificador": Valor (escalar, vetor, matriz)}. Caso não haja, retornar apenas fitness.

    fit_fun = {"freq": fitness_function_freq, "freq+mac": fitness_function_freq_mac}  # auxiliar do inicializador de população
    fitness_function = fit_fun[fitness_metric]

    sampling_method = 'lhs'
    kappa = xi

    print(f"\n\nRodada {run_idx}: initial_points = {initial_points}, evaluations = {evaluations}, aqc_func = {acq_func}, xi/kappa = {xi}"
          f"\nFunction: {fitness_metric}")

    rodada = BO(fitness_function, parameters, initial_points)
    rodada.set_sampling_method(sampling_method)
    rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

    # ajuste do registro:
    log_title = None #  f"BO_laje_{fitness_metric}_{noise_level}noise_{acq_func}acq_{xi}xi" # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)
    log_dir = r"D:\Users\Thiago\OneDrive\Documentos\2025.1\Cilamce\Rodadas\logs" # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
    rodada.set_log(log_title, log_dir)

    result = rodada.run(evaluations, acq_func=acq_func, xi=xi, kappa=kappa, status=True)