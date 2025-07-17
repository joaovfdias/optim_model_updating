from optimization.parameter import Continuous
from optimization.pso_optimizer.pso_optimizer import PSO
from external.parser import Ansys
from external.special_functions import SpecialFun

"""
    TEMPLATE PARA CHAMADA DO ALGORITMO PSO PARA CALIBRAÇÃO USANDO ANSYS
    - são necessários arquivo de script do modelo genérico e arquivos de saída do modelo base
    - diretórios precisam ser declarados usando a formatação r"{diretório}"
    - consultar documentação das classes e métodos para entender entrada e formatos
"""

# parâmetros do modelo:
parameters =    [
                Continuous(20e9, 30e9, 'modulo'),
                Continuous(0.1, 0.5, 'poisson'),
                Continuous (2400, 2600, 'dens'),
                Continuous(10e5, 10e7, 'rigidez1'),
                Continuous(10e5, 10e7, 'rigidez2')
                ]

keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

# parâmetros de entrada da classe Ansys:
# entrada obrigatória:
ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v251\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
# entradas opcionais (caso vazias, será utilizado default: diretório \\ANSYS, arquivos "script.txt", "out_base_freq.txt" e "out_base_modes.txt"):
ansys_working_dir = None
input_dir = None #r"D:\Users\Thiago\Documents\.Mestrado (Local)\Python\OtimizadorGit\Problema Teste\Input"
base_script_filename = None #"script_ulele.txt"
base_freq_filename = None #"out_base_freq_ulele.txt"
base_modes_filename = None #"out_base_modos_ulele.txt"
output_dir = None #r"D:\Users\Thiago\Documents\.Mestrado (Local)\Python\OtimizadorGit\Problema Teste\Output"
# nome do arquivo de saída conforme configurado no script Ansys (precisa ser configurado usando Ansys.set_output_filenames):
out_freq_filename = None #"out_freq_ulele.txt"
out_modes_filename = None #"out_modes_ulele.txt"

# objeto da classe Ansys declarado antes de fitness_function:
ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
ansys.set_output_filenames(out_freq_filename, out_modes_filename) # ajusta o nome dos arquivos de saída de freq e modos do Ansys, que serão gerados em ansys_working_dir

# função objetivo (usa 'Ansys' para rodar e obter os parâmetros modais necessários e 'SpecialFun' para fazer os cálculos de erro e MAC):
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

    return fitness, [comp_freq, comp_modes] # retorna o valor do fitness do indivíduo e os dados modais associados ao modelo (se houver apenas frequências, retornar [comp_freq])

# parâmetros do algoritmo:
w = 0.6 # inércia
w_rate = 0.99 # taxa de decaimento de inércia
c1 = 2.05 # governa a exploração da população
c2 = 2.05 # governa a convergência
init_vel_ratio = 0.2 # proporção do espaço de busca que pode ser empregado para velocidade inicial

population_size = 5
iteracoes = 5

# declarção do otimizador:
rodada = PSO(fitness_function, parameters, population_size, w, w_rate, c1, c2, init_vel_ratio)

rodada.sync_time(ansys.anstime) # sincroniza o log label do algoritmo e a subpasta no output do ansys para facilitar controle

# ajuste do registro:
log = "full" # tipo de registro (True: simplificado, "full": todos os indivíduos)
log_title = "teste" # alterar nome do arquivo gerado, se quiser
log_dir = None # alterar diretório do registro, por padrão \log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
rodada.set_log(log_title, log_dir)

# chamada:
best = rodada.run(iteracoes, log=log)