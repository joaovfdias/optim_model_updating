from tests.LOP.BO_trel_TestRun import BO_run
from tests.LOP.BO_skopt_trel_TestRun import BO_skopt_run
from tests.LOP.GA_trel_TestRun import GA_run
from tests.LOP.PSO_trel_TestRun import PSO_run
from optimization.parameter import *
import os

from multiprocessing import Process


def run_BO(parameters, irun, input_dir, log_dir, tuning):
    BO_run(parameters, irun, input_dir, log_dir, tuning)

def run_BO_skopt(parameters, irun, input_dir, log_dir):
    BO_skopt_run(parameters, irun, input_dir, log_dir)

def run_GA(parameters, irun, input_dir, log_dir):
    GA_run(parameters, irun, input_dir, log_dir)

def run_PSO(parameters, irun, input_dir, log_dir):
    PSO_run(parameters, irun, input_dir, log_dir)

if __name__ == '__main__':

    base_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input"

    parameters = [
        [ #1
        Continuous(20e9, 35e9, 'modulo_concreto'),
        Continuous(10e9, 20e9, 'modulo_madeira'),
        Continuous(150e9, 250e9, 'modulo_aco_a36'),
        Continuous(150e9, 250e9, 'modulo_aco_cabos'),
        Continuous(0, 1000e3, 'protensao'),

        Continuous(1e7, 1e9, 'kv'),
        Continuous(1e7, 1e9, 'kh'),
        Continuous(1e7, 1e9, 'kt')
        ],
        [ #2
            # Continuous(20e9, 35e9, 'modulo_concreto'),
            # Continuous(10e9, 20e9, 'modulo_madeira'),
            # Continuous(150e9, 250e9, 'modulo_aco_a36'),
            Continuous(150e9, 250e9, 'modulo_aco_cabos'),
            Continuous(0, 1000e3, 'protensao'),

            Continuous(0.2, 0.8, 'h_concreto'),

            Continuous(1e7, 1e9, 'kv'),
            Continuous(1e7, 1e9, 'kh'),
            Continuous(1e7, 1e9, 'kt')
        ],
        [ #3
            Continuous(1e6,1e9,'GXY'),
            Continuous(1e6, 1e9, 'GYZ'),
            Continuous(1e6, 1e9, 'GXZ')
        ],
        [ #4
            Continuous(20e9, 35e9, 'modulo_concreto'),
            Continuous(10e9, 20e9, 'modulo_madeira'),
            Continuous(1e6, 1e9, 'GXY'),
            Continuous(1e6, 1e9, 'GYZ'),
            Continuous(1e6, 1e9, 'GXZ'),
            Continuous(150e9, 250e9, 'modulo_aco_a36'),

            Continuous(0.25, 0.6, 'h_concreto'),

            Continuous(1e7, 1e9, 'kv'),
            Continuous(1e7, 1e9, 'kh'),
            Continuous(1e7, 1e9, 'kt')
        ],
        [ #5
            Continuous(20e9, 35e9, 'modulo_concreto'),
            Continuous(10e9, 20e9, 'modulo_madeira'),
            Continuous(1e6, 1e9, 'GXY'),
            # Continuous(1e6, 1e9, 'GYZ'),
            Continuous(1e6, 1e9, 'GXZ'),
            # Continuous(150e9, 250e9, 'modulo_aco_a36'),

            # Continuous(0.25, 0.6, 'h_concreto'),

            Continuous(1e7, 1e9, 'kv'),
            Continuous(1e7, 1e9, 'kh'),
            Continuous(1e7, 1e9, 'kt')
        ],
        [ #6
            Continuous(20e9, 35e9, 'modulo_concreto'),
            Continuous(10e9, 20e9, 'modulo_madeira'),
            Continuous(1e6, 1e9, 'GXY'),
            Continuous(1e6, 1e9, 'GYZ'),
            Continuous(1e6, 1e9, 'GXZ'),
            # Continuous(150e9, 250e9, 'modulo_aco_a36'),

            # Continuous(0.25, 0.6, 'h_concreto'),

            Continuous(1e7, 1e9, 'kv'),
            Continuous(1e7, 1e9, 'kh'),
            Continuous(1e7, 1e9, 'kt')
        ],
        [  #7
            Continuous(20e9, 35e9, 'modulo_concreto'),
            Continuous(10e9, 20e9, 'modulo_madeira'),
            Continuous(1e6, 1e9, 'GXY'),
            Continuous(1e6, 1e9, 'GYZ'),
            Continuous(1e6, 1e9, 'GXZ'),
            # Continuous(150e9, 250e9, 'modulo_aco_a36'),

            # Continuous(0.25, 0.6, 'h_concreto'),

            Continuous(5e7, 5e8, 'kv'),
            Continuous(5e7, 5e8, 'kh'),
            Continuous(5e7, 5e8, 'kt')
        ],
        [   #8
            Continuous(20e9, 35e9, 'modulo_concreto'),
            Continuous(0.1, 0.49, 'poisson_concreto'),
            # Continuous(0.25, 0.6, 'h_concreto'),

            Continuous(10e9, 20e9, 'modulo_madeira'),
            Continuous(0.1, 0.49, 'poisson_madeira'),

            Continuous(150e9, 250e9, 'modulo_perfis'),
            Continuous(0.1, 0.49, 'poisson_perfis'),

            Continuous(150e9, 250e9, 'modulo_cordoalhas'),
            Continuous(0.1, 0.49, 'poisson_cordoalhas'),

            Continuous(1e7, 1e9, 'kv'),
            Continuous(1e7, 1e9, 'kh'),
            Continuous(1e7, 1e9, 'kt')
        ],
        [ #9
            Continuous(1e6, 1e9, 'GXY'),
            Continuous(1e6, 1e9, 'GYZ'),
            Continuous(1e6, 1e9, 'GXZ'),
            Continuous(0.02, 0.06, 'h_concreto')
        ],
        [   # 10
            Continuous(20e9, 35e9, 'modulo_concreto'),
            Continuous(0.1, 0.49, 'poisson_concreto'),
            Continuous(0.02, 0.06, 'h_concreto'),

            Continuous(10e9, 20e9, 'modulo_madeira'),

            Continuous(150e9, 250e9, 'modulo_cordoalhas'),

            Continuous(1e7, 1e9, 'kv'),
            Continuous(1e7, 1e9, 'kh'),

            Continuous(1e6, 1e9, 'GXY'),
            Continuous(1e6, 1e9, 'GYZ'),
            Continuous(1e6, 1e9, 'GXZ')
        ]
    ]

    num_runs = 1
    # cases = [1,2,3,4]
    cases = [10]

    for irun in range(1, num_runs+1):

        for i, case in enumerate(cases):

            input_dir = os.path.join(base_dir, f"Analise {case}")
            log_dir = os.path.join(input_dir, f"log")
            os.makedirs(log_dir, exist_ok=True)

            p = Process(target=run_PSO, args=(parameters[case-1], f"case{case}_run{irun}", input_dir, log_dir))
            p.start()
            p.join()

            # p = Process(target=run_BO, args=(parameters[case-1], f"case{case}_run{irun}", input_dir, log_dir, True))
            # p.start()
            # p.join()
            #
            # p = Process(target=run_BO, args=(parameters[case-1], f"case{case}_run{irun}", input_dir, log_dir, False))
            # p.start()
            # p.join()
            #
            # p = Process(target=run_GA, args=(parameters[case-1], f"case{case}_run{irun}", input_dir, log_dir))
            # p.start()
            # p.join()
            #
            # p = Process(target=run_BO_skopt, args=(parameters[case-1], f"case{case}_run{irun}", input_dir, log_dir))
            # p.start()
            # p.join()