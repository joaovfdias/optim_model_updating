from optimization.turbo_optimizer.turbo import TuRBO
from external.parser import Ansys
from external.special_functions import SpecialFun

from datetime import datetime
import numpy as np
import time
import os
import shutil


def TuRBO_run(
        irun,
        parameters,
        base_dir,
        local_dir=None,
        log_dir=None,
        base_script_filename=None,
        noise=False,
        initial_points=None,
        evaluations=None,
        batch_size=4,
        acqf="ts"
):

    keys = [parameter.key for parameter in parameters]  # identificadores dos parâmetros (equivalente ao script: %key%)

    keys = [p.key for p in parameters]

    ansys_exe_path = r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"

    # caminhos
    ansys_working_dir = os.path.join(local_dir if local_dir else base_dir, 'ANSYS')
    os.makedirs(ansys_working_dir, exist_ok=True)
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(local_dir if local_dir else os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)

    unique_ansys_dir = os.path.join(ansys_working_dir, f"worker_BO_{irun}_{os.getpid()}")
    os.makedirs(unique_ansys_dir, exist_ok=True)
    try:
        shutil.copy(os.path.join(base_dir, "ModBase.db"), unique_ansys_dir)
    except FileNotFoundError:
        print("\nAviso: não existe ModBase.db na pasta base. Nenhuma cópia foi feita.")

    base_script_filename = base_script_filename or "script.mac"
    base_freq_filename = "target_freq.txt"
    base_modes_filename = "target_modes.txt"

    out_freq_filename = "out_freq.txt"
    out_modes_filename = "out_modes.txt"

    ansys = Ansys(ansys_exe_path, unique_ansys_dir, input_dir, base_script_filename, base_freq_filename,
                  base_modes_filename, output_dir)
    ansys.set_output_filenames(out_freq_filename, out_modes_filename)
    ansys.max_attempts = 6

    # --- APLICAÇÃO DE RUÍDO NOS DADOS DE REFERÊNCIA (PROBLEMAS SINTÉTICOS) ---
    if noise:
        # Simula incerteza experimental diferente para cada rodada
        # Nível de Ruído (Sigma): 1% (0.01) ou 3% (0.03) são valores comuns
        NOISE_LEVEL = noise

        # Semente aleatória única para este processo (garante variação entre workers)
        np.random.seed(int(time.time()) + irun)

        # 1. Ruído nas Frequências (Multiplicativo)
        # freq_new = freq_old * (1 + N(0, sigma))
        freq_noise = np.random.normal(0, NOISE_LEVEL, ansys.base_freq.shape)
        ansys.base_freq = ansys.base_freq * (1 + freq_noise)

        # 2. Ruído nos Modos (Multiplicativo)
        # Afeta a amplitude de cada ponto do modo
        mode_noise = np.random.normal(0, NOISE_LEVEL, ansys.base_modes.shape)
        ansys.base_modes = ansys.base_modes * (1 + mode_noise)

    def fitness_function(param):
        input_file = ansys.create_input_file(param, keys)
        ansys.run_ansys(input_file, True, True)

        comp_freq = ansys.read_frequencies()
        comp_modes = ansys.read_modes()

        paired_comp_freq, paired_comp_modes, mac_error_sum, macs = SpecialFun.pair_modes_mac(
            comp_freq, comp_modes, ansys.base_modes
        )
        freq_error_sum = SpecialFun.norm_freq_errors(ansys.base_freq, paired_comp_freq)

        peso_freq = 1
        peso_mac = 1
        fitness = peso_freq * freq_error_sum + peso_mac * mac_error_sum

        return fitness, {"freq error": freq_error_sum, "mac error": mac_error_sum, "Freq.": paired_comp_freq,
                         "Freq. Error": [abs((bf - nf) / bf) for bf, nf in zip(ansys.base_freq, paired_comp_freq)],
                         "Mode": paired_comp_modes, "MAC": macs}


    initial_points = initial_points or 2 * len(parameters)
    evaluations = evaluations or 40 * len(parameters)

    rodada = TuRBO(
        fitness_function,
        parameters,
        initial_points
    )

    log_dir = log_dir or os.path.join(base_dir, 'log', 'runs', 'TuRBO')

    log_title = (
        f"TuRBO_ini({initial_points})"
        f"_eval({evaluations})"
        f"_acq({acqf})"
        f"_q({batch_size})"
        f"_{irun}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    rodada.set_log(log_title, log_dir, False)

    try:

        result = rodada.run(
            evaluations=evaluations,
            acqf=acqf,
            batch_size=batch_size,
            status=False
        )

    finally:
        try:
            ansys.mapdl.exit(force=True)
        except:
            pass
