from optimization.parameter import *
from external.parser import Ansys
from external.special_functions import SpecialFun
import os

from optimization.bo_optimizer import *

parameters = [
    Continuous(20e9, 30e9, 'modulo'),
    Continuous(0.1, 0.49, 'poisson'),
    Continuous(2400, 2600, 'dens'),
    Continuous(1e5, 1e7, 'rigidez1'),
    Continuous(1e5, 1e7, 'rigidez2'),
    Continuous(1e5, 1e7, 'rigidez3'),
    Continuous(1e5, 1e7, 'rigidez4')
]
keys = [p.key for p in parameters]

ansys_exe_path = r"D:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
ansys_working_dir = None
input_dir = os.path.join(os.getcwd(), 'input')
base_script_filename = "script_laje.txt"
base_freq_filename   = "out_base_freq_laje.txt"
base_modes_filename  = "out_base_modes_laje.txt"
output_dir = os.path.join(os.getcwd(), 'output')

out_freq_filename  = "out_freq_laje.txt"
out_modes_filename = "out_modes_laje.txt"

ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
ansys.set_output_filenames(out_freq_filename, out_modes_filename)
ansys.max_attempts = 6

def fitness_function(param):
    input_file = ansys.create_input_file(param, keys)
    ansys.run_ansys(input_file, True, True)

    comp_freq  = ansys.read_frequencies()
    comp_modes = ansys.read_modes()

    paired_comp_freq, paired_comp_modes, mac_error_sum = SpecialFun.pair_modes_mac(
        comp_freq, comp_modes, ansys.base_modes
    )
    freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, paired_comp_freq)

    peso_freq = 1
    peso_mac  = 1
    fitness = peso_freq * freq_error_sum + peso_mac * mac_error_sum

    return fitness, {"Freq.": paired_comp_freq, "Mode": paired_comp_modes}

n_dims = len(parameters)

# Preset “default” já calcula init_points = max(8, ceil(3 * n_dims))
cfg = BOConfig.from_preset("default", n_dims=n_dims, overwrite=False)

# Ajustes
cfg.max_iter = 100 # iterações do BO -> total de evals ~ init_points + max_iter*batch_size

rodada = BO(fitness_function, parameters, config=cfg)

best = rodada.run()  # imprime progresso e retorna o melhor Individual
