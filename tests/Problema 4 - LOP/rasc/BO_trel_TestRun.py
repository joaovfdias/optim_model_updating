from external.ansys.parser import Ansys
from utils.special_functions import SpecialFun
import os

from optimization.bo_optimizer import *


def BO_run(parameters, irun, input_dir, log_dir, tuning):

    keys = [p.key for p in parameters]

    ansys_exe_path = r"D:\Program Files\ANSYS Inc\ANSYS Student\v252\commonfiles\launcherQT\src\..\..\..\ansys\bin\winx64\MAPDL.EXE"
    ansys_working_dir = None
    # input_dir = input_dir
    base_script_filename = "script.mac"
    base_freq_filename = "out_freq.txt"
    base_modes_filename = "out_modos_y.txt"
    output_dir = os.path.join(os.getcwd(), 'output')

    out_freq_filename = "out_freq.txt"
    out_modes_filename = "out_modos_y.txt"

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

        return fitness, {"freq error": freq_error_sum, "mac error": mac_error_sum, "Freq.": paired_comp_freq, "Mode": paired_comp_modes}

    # default: já calcula init_points = max(8, ceil(3 * n_dims))
    cfg = BOConfig()
    cfg.acquisition = "POI"
    if tuning: cfg.acq_hyper_tuning = 3

    iterations = 400

    rodada = BO(fitness_function, parameters, config=cfg)
    rodada.sync_time(ansys.anstime)  # sincroniza timestamp de optimizer e ansys para facilitar controle dos registros

    log = "full"  # tipo de registro (True: simplificado - melhor de cada iteração, "full": todos os indivíduos)
    log_title = f"BO_trel_tuning.{tuning}_{irun}"  # alterar nome do arquivo gerado, se quiser (todos recebem "_timestamp" no final)
    # log_dir = None  # alterar diretório do registro, por padrão {diretório atual}\log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
    rodada.set_log(log_title, log_dir)

    best = rodada.run(iterations)

    ansys.mapdl.exit()
