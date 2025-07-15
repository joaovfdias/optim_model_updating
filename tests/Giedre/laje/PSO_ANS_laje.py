from optimization.parameter import *
from optimization.pso_optimizer.pso_optimizer import PSO
from external.parser import Ansys
from external.special_functions import SpecialFun

import os


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

keys = [parameter.key for parameter in parameters]

# parâmetros de entrada da classe Ansys:
ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v251\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
ansys_working_dir = None
input_dir = os.path.join(os.getcwd(), 'input')
base_script_filename = "script_laje.txt"
base_freq_filename = "out_base_freq_laje.txt"
base_modes_filename = "out_base_modes_laje.txt"
output_dir = os.path.join(os.getcwd(), 'output')

out_freq_filename = "out_freq_laje.txt"
out_modes_filename = "out_modes_laje.txt"

# objeto da classe Ansys declarado antes de fitness_function:
ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
ansys.set_output_filenames(out_freq_filename, out_modes_filename)

# função objetivo:
def fitness_function(param):

    input_file = ansys.create_input_file(param, keys)
    ansys.run_ansys(input_file)

    comp_freq = ansys.read_frequencies()
    comp_modes = ansys.read_modes()

    paired_comp_freq, paired_comp_modes, mac_error_sum = SpecialFun.pair_modes_mac(comp_freq, comp_modes, ansys.base_modes)
    freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, paired_comp_freq)

    peso_freq = 1
    peso_mac = 1

    fitness = peso_freq * freq_error_sum + peso_mac * mac_error_sum

    return fitness, {"Freq.": paired_comp_freq, "Mode": paired_comp_modes}

# parâmetros do algoritmo:
w = 0.6
w_rate = 0.99
c1 = 2.05
c2 = 2.05
init_vel_ratio = 0.2

population_size = 70
iteracoes = 50

# declarção do otimizador:
rodada = PSO(fitness_function, parameters, population_size, w, w_rate, c1, c2, init_vel_ratio)
rodada.set_tolerance(fit_tol = 1e-4, patience = 10)
rodada.sync_time(ansys.anstime)

# ajuste do registro:
log = "full"
log_title = "laje_PSO_semruido"
log_dir = os.path.join(os.getcwd(), 'log')
rodada.set_log(log_title, log_dir)

# chamada:
best = rodada.run(iteracoes, log=log)