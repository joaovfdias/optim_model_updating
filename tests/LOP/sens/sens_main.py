from sensitivity.sensitivity import SensitivityAnalyzer, ParamSpec, Sampler
import pandas as pd
import numpy as np
from external.ansys.parser import Ansys
from utils.special_functions import SpecialFun

import time
import os
import shutil


def sensitivity_analysis(parameters, base_dir, ansys_exe_path=None, local_dir=None, scriptfilename=None):
    keys = [p.key for p in parameters]

    ansys_exe_path = ansys_exe_path or r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"

    # caminhos
    ansys_working_dir = os.path.join(local_dir if local_dir else base_dir, 'ANSYS')
    os.makedirs(ansys_working_dir, exist_ok=True)
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(local_dir if local_dir else os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    unique_ansys_dir = os.path.join(ansys_working_dir, f"worker_sensitivity")
    os.makedirs(unique_ansys_dir, exist_ok=True)
    try:
        shutil.copy(os.path.join(base_dir, "ModBase.db"), unique_ansys_dir)
    except FileNotFoundError:
        print("\nAviso: não existe ModBase.db na pasta base. Nenhuma cópia foi feita.")

    base_script_filename = scriptfilename or "scriptLOP.mac"
    base_freq_filename = "target_freq.txt"
    base_modes_filename = "target_modes.txt"

    out_freq_filename = "out_freq.txt"
    out_modes_filename = "out_modes.txt"

    ansys = Ansys(ansys_exe_path, unique_ansys_dir, input_dir, base_script_filename, base_freq_filename, base_modes_filename, output_dir)
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

    start_time = time.time()

    # Gera DoE sem precisar do Optimizer
    rng = np.random.default_rng()
    specs = [ParamSpec(p.key, p.lower_bound, p.upper_bound) for p in parameters]

    amostragem = 100*len(parameters)

    X = Sampler.lhs(amostragem, specs, rng)
    Y = X.apply(evaluate, axis=1, result_type="expand")

    df_eval = pd.concat([X, Y], axis=1)

    # Rodar análise de sensibilidade
    sa = SensitivityAnalyzer(minimize=True)
    sa.logpath = os.path.join(base_dir, "sensitivity")
    selected = sa.workflow(df_eval, fitness_col="Fitness", interactive=False)
    print("Parâmetros escolhidos:", selected)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\nTempo para amostragem de {amostragem} indivíduos ({amostragem/len(parameters)}x o número de parâmetros: "
          f"\n{elapsed:.4f} s")

    ansys.mapdl.exit()

    return