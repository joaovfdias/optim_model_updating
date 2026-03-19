from sensitivity.sensitivity import SensitivityAnalyzer, ParamSpec, Sampler
import pandas as pd
import numpy as np
from optimization.parameter import *
from external.ansys.parser import Ansys
from utils.special_functions import SpecialFun

import os

parameters = [
            # Continuous(20e9, 35e9, 'modulo_concreto'),
            # Continuous(10e9, 20e9, 'modulo_madeira'),
            # Continuous(150e9, 250e9, 'modulo_aco_a36'),
            Continuous(150e9, 250e9, 'modulo_aco_cabos'),
            Continuous(0, 1000e3, 'protensao'),

            Continuous(0.2, 0.8, 'h_concreto'),

            Continuous(1e7, 1e9, 'kv'),
            Continuous(1e7, 1e9, 'kh'),
            Continuous(1e7, 1e9, 'kt')
        ]

keys = [p.key for p in parameters]

ansys_exe_path = r"D:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
ansys_working_dir = None
input_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Análise 2"
base_script_filename = "scriptLOP.mac"
base_freq_filename = "out_freq.txt"
base_modes_filename = "out_modos_y.txt"
output_dir = os.path.join(os.getcwd(), 'output')

out_freq_filename = "out_freq.txt"
out_modes_filename = "out_modos_y.txt"

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
