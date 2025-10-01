from optimization.sensitivity import SensitivityAnalyzer, ParamSpec, Sampler
import pandas as pd
import numpy as np
from optimization.parameter import *
from external.parser import Ansys
from external.special_functions import SpecialFun

import os

parameters = [
    Continuous(150e9, 250e9, 'modulo_banz'),
    Continuous(0.1, 0.49, 'poisson_banz'),
    Continuous(7500, 8200, 'dens_banz'),

    Continuous(150e9, 250e9, 'modulo_diag'),
    Continuous(0.1, 0.49, 'poisson_diag'),
    Continuous(7500, 8200, 'dens_diag'),

    Continuous(150e9, 250e9, 'modulo_contrav'),
    Continuous(0.1, 0.49, 'poisson_contrav'),
    Continuous(7500, 8200, 'dens_contrav'),

    Continuous(1e5, 1e7, 'rigidez1'),
    Continuous(1e5, 1e7, 'rigidez2'),
    Continuous(1e5, 1e7, 'rigidez3'),
    Continuous(1e5, 1e7, 'rigidez4'),

    Continuous(400, 800, 'massa')
]

keys = [p.key for p in parameters]

ansys_exe_path = r"D:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
ansys_working_dir = None
input_dir = os.path.join(os.getcwd(), 'input')
base_script_filename = "script.mac"
base_freq_filename   = "out_freq.txt"
base_modes_filename  = ["out_modos_x.txt", "out_modos_y.txt", "out_modos_z.txt"]
output_dir = os.path.join(os.getcwd(), 'output')

out_freq_filename  = "out_freq.txt"
out_modes_filename = ["out_modos_x.txt", "out_modos_y.txt", "out_modos_z.txt"]

ansys = Ansys(ansys_exe_path, ansys_working_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
ansys.set_output_filenames(out_freq_filename, out_modes_filename)
ansys.max_attempts = 6

def evaluate(row):
    param = row

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

    ddata = {}
    for i, f in enumerate(paired_comp_freq, start=1):
        ddata[f"freq #{i}"] = f
    for i, m in enumerate(paired_comp_modes, start=1):
        mac = SpecialFun.modal_assurance_criterion(comp_modes[i-1], m)
        ddata[f"MAC #{i}"] = mac

    return ddata

# Gera DoE sem precisar do Optimizer
rng = np.random.default_rng(42)
specs = [ParamSpec(p.key, p.lower_bound, p.upper_bound) for p in parameters]

X = Sampler.lhs(200, specs, rng)
Y = X.apply(evaluate, axis=1, result_type="expand")

df_eval = pd.concat([X, Y], axis=1)

# Rodar análise de sensibilidade
sa = SensitivityAnalyzer(minimize=True)
selected = sa.workflow(df_eval, fitness_col="Fitness", interactive=True)
print("Parâmetros escolhidos:", selected)
