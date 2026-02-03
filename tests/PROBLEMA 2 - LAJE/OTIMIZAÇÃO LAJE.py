import os

from optimization.parameter import *
from external.special_functions import SpecialFun
from external.parser import Ansys

parameters = [
    Continuous(20e9,35e9, 'modulo_viga_1'),
    Continuous(20e9, 35e9, 'modulo_viga_2'),
    Continuous(20e9, 35e9, 'modulo_centro'),
    # Continuous(20e9, 35e9, 'modulo_borda_1'),
    # Continuous(20e9, 35e9, 'modulo_borda_2'),

    Continuous(0.1, 0.40, 'poisson'),
    Continuous (2400, 2600, 'dens'),

    Continuous(50e6, 50e8, 'rigidez1'),
    Continuous(50e6, 50e8, 'rigidez2'),
    Continuous(50e6, 50e8, 'rigidez3'),
    Continuous(50e6, 50e8, 'rigidez4')
]

keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
# caminhos
base_dir = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"
ansys_working_dir = os.path.join(base_dir, 'ANSYS')
input_dir = os.path.join(base_dir, 'input')
output_dir = os.path.join(os.getcwd(), 'output')

base_script_filename = "script problema 2.mac"
base_freq_filename = "target_freq.txt"
base_modes_filename = "target_modes.txt"

out_freq_filename = "out_freq.txt"
out_modes_filename = "out_modes.txt"

ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
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


# agora basta passar os parâmetros corretos para a função e averiguar se o fitness zera para validar o modelo e script

params = [32e9,28e9,30e9,0.2,2500,50e7,40e7,55e7,60e7] # v2: E uniforme pra laje e add do poison e dens.

fitness, datas = fitness_function(params)
print(fitness)
print(datas)