from optimization.sensitivity import SensitivityAnalyzer, ParamSpec, Sampler
import pandas as pd
import numpy as np
from optimization.parameter import *
from external.parser import Ansys
from external.special_functions import SpecialFun

import os

parameters = [
    Continuous(0.1, 0.49, 'poisson'),
    Continuous (2400, 2600, 'dens'),

    Continuous(20e9, 35e9, 'modulo_viga_1'),
    Continuous(20e9, 35e9, 'modulo_viga_2'),
    Continuous(20e9, 35e9, 'modulo_centro'),
    Continuous(20e9, 35e9, 'modulo_borda_1'),
    Continuous(20e9, 35e9, 'modulo_borda_2'),

    Continuous(50e6, 50e8, 'rigidez1'),
    Continuous(50e6, 50e8, 'rigidez2'),
    Continuous(50e6, 50e8, 'rigidez3'),
    Continuous(50e6, 50e8, 'rigidez4')
]

keys = [p.key for p in parameters]

ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"

base_dir = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"
ansys_working_dir = os.path.join(base_dir, 'ANSYS')
input_dir = os.path.join(base_dir, 'input', 'sensib')
output_dir = os.path.join(base_dir, 'output')

base_script_filename = "script problema 2 sensib.mac"
base_freq_filename = "target_freq.txt"
base_modes_filename = "target_modes.txt"

out_freq_filename = "out_freq.txt"
out_modes_filename = "out_modes.txt"

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

X = Sampler.lhs(550, specs, rng)
Y = X.apply(evaluate, axis=1, result_type="expand")

df_eval = pd.concat([X, Y], axis=1)

# Rodar análise de sensibilidade
sa = SensitivityAnalyzer(minimize=True)
selected = sa.workflow(df_eval, fitness_col="Fitness", interactive=True)
print("Parâmetros escolhidos:", selected)
