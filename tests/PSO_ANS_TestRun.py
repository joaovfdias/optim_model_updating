from optimization.parameter import *
from optimization.pso_optimizer.pso_optimizer import PSO
from external.parser import Ansys
from external.additional_functions import SpecialFun

import os

"""
    TEMPLATE DE CHAMADA DO ALGORITMO PSO PARA CALIBRAÇÃO USANDO ANSYS
    - são necessários arquivo de script do modelo genérico e arquivos de saída do modelo base
    - diretórios precisam ser declarados usando a formatação r"{diretório}"
    - consultar documentação das classes e métodos em caso de dúvidas com entrada e formatos
"""

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
input_dir = os.path.join(os.getcwd(), 'input') # {diretório atual}\input (localização do script e dados de referência)
base_script_filename = "script_laje.txt"
base_freq_filename = "out_base_freq_laje.txt"
base_modes_filename = "out_base_modes_laje.txt"
output_dir = os.path.join(os.getcwd(), 'output') # {diretório atual}\output (onde serão armazenados os scripts executáveis do Ansys)
# nome do arquivo de saída conforme configurado no script Ansys (alterar usando Ansys.set_output_filenames):
out_freq_filename = "out_freq_laje.txt"
out_modes_filename = "out_modes_laje.txt"

# objeto da classe Ansys declarado antes de fitness_function:
ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
ansys.set_output_filenames(out_freq_filename, out_modes_filename) # ajusta o nome dos arquivos de saída de freq. e modos do Ansys, que serão gerados em ansys_working_dir
ansys.max_attempts = 6 # define quantas tentativas de rodada o Ansys executa em caso de erro

# função objetivo com pareamento (usa 'Ansys' para rodar e obter os parâmetros modais necessários e 'SpecialFun' para fazer os cálculos de erro e MAC):
def fitness_function(param):

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

    return fitness, {"Freq.": paired_comp_freq, "Mode": paired_comp_modes} # caso haja dados adicionais para registrar, o 2º retorno da função deve ser um dicionário com {"Identificador": Valor (escalar, vetor, matriz)}. Caso não haja, retornar apenas fitness.

# parâmetros do algoritmo:
w = 0.6 # proporção da velocidade atual que participa da próxima
w_rate = 0.99 # taxa de decaimento de inércia por iteração
c1 = 2.05 # influencia a exploração individual
c2 = 2.05 # influencia a convergência para o mínimo do grupo
init_vel_ratio = 0.2 # proporção do espaço de busca que pode ser empregado para velocidade inicial

population_size = 2 # indivíduos avaliados por geração (recomendado ao menos 10x o número de variáveis)
iteracoes = 5 # quantidade de iterações (suficientemente grande para a convergência do algoritmo)

# declaração do otimizador:
rodada = PSO(fitness_function, parameters, population_size, w, w_rate, c1, c2, init_vel_ratio) # objeto otimizador
rodada.set_tolerance(fit_abs = 2e-2, patience = 10) # critério de parada
rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

# ajuste do registro:
log = "full" # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
log_title = "teste_PSO_laje" # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)
log_dir = None # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
rodada.set_log(log_title, log_dir)

# chamada:
best = rodada.run(iteracoes, log=log)