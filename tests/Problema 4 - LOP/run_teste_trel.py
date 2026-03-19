import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Process, Queue

from BO_LOP_run import BO_run
from GA_LOP_run import GA_run
from PSO_LOP_run import PSO_run

from optimization.parameter import Continuous
from data.compile import compile_convergence_history

irun = 1

base_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Pesquisa\Rodadas\Problema 3"
local_dir = r"C:\Users\Thiago Artur\Documents\Rodadas\Problema 3"

script_name = "scriptTREL.mac"
noise = 0.03

parameters = [
    Continuous(180e9, 220e9, 'modulo_banz'),

    Continuous(180e9, 220e9, 'modulo_diag'),

    Continuous(180e9, 220e9, 'modulo_contrav'),

    Continuous(1e7, 1e8, 'rigidez1'),
    Continuous(1e7, 1e8, 'rigidez2'),
    Continuous(1e7, 1e8, 'rigidez3'),
    Continuous(1e7, 1e8, 'rigidez4'),

    Continuous(400, 800, 'massa')
]

target_params = [205e9, 215e9, 195e9, 8e7, 6.8e7, 7.6e7, 7.2e7, 600]

iter = None

PSO_run = PSO_run(irun, parameters, base_dir, base_script_filename=script_name, iterations=iter)