from optimization.parameter import *
from optimization.ga_optimizer import GA
from external.parser import Ansys
from external.special_functions import SpecialFun

import os

"""
        TEMPLATE DE CHAMADA DO GA PARA CALIBRAÇÃO USANDO ANSYS
    - são necessários arquivo de script do modelo genérico e arquivos de referência
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
ansys_working_dir = r"D:\Users\Thiago\Desktop\TesteParser2\Ansys"
input_dir = r"D:\Users\Thiago\Desktop\TesteParser2" # {diretório atual}\input (localização do script e dados de referência)
base_script_filename = "script.mac"
base_freq_filename = "out_freq_base.txt"
base_modes_filename = "out_modes_base.txt"
output_dir = os.path.join(input_dir, 'output') # {diretório atual}\output (onde serão armazenados os scripts executáveis do Ansys)
# nome do arquivo de saída conforme configurado no script Ansys (alterar usando Ansys.set_output_filenames):
out_freq_filename = "out_freq.txt"
out_modes_filename = "out_modos.txt"

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
elitism_rate = 0.10 # proporção dos melhores da população que serão preservados
crossover_rate = 0.60 # chance de ocorrência de cruzamento entre indivíduos selecionados
mutation_strength = 0.10 # taxa máxima de mutação de cada gene daqueles indivíduos não originados de crossover

population_size = 50 # indivíduos avaliados por geração (recomendado ao menos 10x o número de variáveis)
generations = 30 # quantidade de iterações (suficientemente grande para a convergência do algoritmo)

# declaração do otimizador:
rodada = GA(fitness_function, parameters, population_size, elitism_rate, crossover_rate, mutation_strength) # objeto otimizador
rodada.set_tolerance(fit_abs = 2e-2, patience = 10) # critério de parada
rodada.sync_time(ansys.anstime) # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

# ajuste do registro:
log = "full" # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
log_title = "teste_GA_viga" # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)
log_dir = None # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
rodada.set_log(log_title, log_dir)

# caso queira retomar a rodada de algum log cvs:
# rodada.resume_from_log(r"C:\Users\Thiago Artur\Documents\.Mestrado (Local)\PyGit\tests\test log recovery\teste_GA_log_completo.csv")

# chamada:
best = rodada.run(generations, log=log)

ansys.mapdl.exit()