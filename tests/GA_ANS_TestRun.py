from ..optimization.parameter import *
from ..optimization.ga_optimizer import GA
from ..external.parser import Ansys
from ..external.special_functions import SpecialFun


# declaração dos parâmetros do algoritmo
parameters =    [
                Continuous(20e9, 30e9, 'modulo'),
                Continuous(0.1, 0.5, 'poisson'),
                Continuous (2400, 2600, 'dens'),
                Continuous(10e5, 10e7, 'rigidez1'),
                Continuous(10e5, 10e7, 'rigidez2')
                ]

keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

# parâmetros de entrada da classe Ansys
ansys_exe_path = r"D:\Program Files\ANSYS Inc\ANSYS Student\v242\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
ansys_working_dir = r"C:\Users\Thiago Artur\Documents\.Mestrado (Local)\Python\Problema Teste\Ansys"
input_dir = r"C:\Users\Thiago Artur\Documents\.Mestrado (Local)\Python\Problema Teste\Input"
base_script_filename = "script_ulele.txt"
base_freq_filename = "out_base_freq_ulele.txt"
base_modes_filename = "out_base_modos_ulele.txt"
output_dir = None #r"D:\Users\Thiago\Documents\.Mestrado (Local)\Python\OtimizadorGit\Problema Teste\Output"

out_freq_filename = "out_freq_ulele.txt"
out_modes_filename = "out_modes_ulele.txt"

# objeto da classe Ansys declarado fora de fitness_function
ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
ansys.set_output_filenames(out_freq_filename, out_modes_filename) # ajusta o nome dos arquivos de saída de freq e modos do Ansys, que serão gerados em ansys_working_dir

# função objetivo (usa 'Ansys' para rodar e obter os parâmetros modais necessários e de 'SpecialFun' para fazer os cálculos de erro e MAC):
def fitness_function(param):

    input_file = ansys.create_input_file(param, keys) # gera o arquivo de input para o ansys com base na lista de parâmetros (valores) e keys (nomes)
    ansys.run_ansys(input_file) # executa esse arquivo

    comp_freq = ansys.read_frequencies() # armazena as frequências exportadas atuais
    comp_modes = ansys.read_modes() # armazena os modos exportados atuais

    freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, comp_freq) # parcela correspondente ao erro nas frequências
    mac_error_sum = SpecialFun.mac_error(ansys.base_modes, comp_modes) # parcela correspondente ao erro nos modos

    peso_freq = 1 # ponderação
    peso_mac = 1

    fitness = peso_freq * freq_error_sum + peso_mac * mac_error_sum

    return fitness, [comp_freq, comp_modes] # retorna o valor do fitness do invíduo conforme seu conjunto de parâmetros e os dados modais associados ao modelo (se houver)

# parâmetros do algoritmo
elitism_rate = 0.10
crossover_rate = 0.60
mutation_strength = 0.10 # no tipo de mutação "gaussian" para variáveis contínuas, alterar essa taxa diretamente (define a faixa percentual em que os paramêtros podem variar)

population_size = 5
generations = 5

# chamada
rodada = GA(fitness_function, parameters, population_size, elitism_rate, crossover_rate, mutation_strength)

# arquivo log
log = "full"
log_name = f"VIGA_freq+mac_semruido_ulele"
log_dir = r"C:\Users\Thiago Artur\Documents\.Mestrado (Local)\Python\Problema Teste\log"

# rodada
#rodada.set_log(log_name, log_dir)
best = rodada.run(generations, log=log)