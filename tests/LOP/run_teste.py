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

irun = 12

parameters = [ # analise 12
    Continuous(28e9, 34e9, 'modulo_concreto'),
    Continuous(0.04, 0.06, 'h_concreto'),

    Continuous(12e9, 18e9, 'modulo_madeira'),

    Continuous(200e9, 220e9, 'modulo_cordoalhas'),

    Continuous(1e8, 5e8, 'GXY'),
    Continuous(1e7, 5e7, 'GXZ')
]

target_params = [32.209e9, 0.06, 15e9, 210e9, 1.84e8, 4.06e7]

base_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 13"

script_name = "scriptLOP.mac"

iter = None

PSO_run = PSO_run(irun, parameters, base_dir, base_script_filename=script_name, iterations=iter)