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
ansys_exe_path = r"D:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
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

ansys.kill_ansys_process()