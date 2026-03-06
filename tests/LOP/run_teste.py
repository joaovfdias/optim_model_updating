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

irun = 18

if irun == 14:
    parameters = [ # analise 14
        Continuous(0.04, 0.06, 'h_concreto'),

        Continuous(1e7, 1e9, 'GXY'), # 600-900 MPa  6e8
        Continuous(1e6, 1e8, 'GYZ') # 50-150 MPa  5e7
    ]

elif irun == 15:
    parameters = [  # analise 15
        Continuous(0.04, 0.06, 'h_concreto'),
        Continuous(1e6, 1e9, 'ey_wood')  # esperado 10 a 500 MPa
    ]

elif irun == 16: # inutil
    parameters = [  # analise 15
        Continuous(0.04, 0.06, 'h_concreto'),
        Continuous(10e3, 150e3, 'protensao'),
        Continuous(1e6, 1e9, 'ey_wood')  # esperado 10 a 500 MPa
    ]

elif irun == 17:
    parameters = [  # analise
        Continuous(0.04, 0.06, 'h_concreto'),
        Continuous(2000,3000,'dens_concreto'),
        Continuous(1e6, 1e9, 'ey_wood')  # esperado 10 a 500 MPa
    ]

elif irun == 18:
    parameters = [  # analise 14
        Continuous(0.04, 0.06, 'h_concreto'),
        Continuous(1e6, 1e9, 'ey_wood'),  # esperado 10 a 500 MPa
        Continuous(1e7, 1e9, 'GXY'),  # 600-900 MPa  6e8
        Continuous(1e6, 1e8, 'GYZ')  # 50-150 MPa  5e7
    ]

else:
    raise(ValueError(f"Problem {irun} not defined."))

base_dir = os.path.join(r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input", f"Analise {irun}")

script_name = "scriptLOP.mac"

iter = 50

PSO_run = PSO_run(irun, parameters, base_dir, base_script_filename=script_name, iterations=iter)