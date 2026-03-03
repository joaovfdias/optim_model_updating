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

    base_dir = r"C:\Users\Thiago Artur\OneDrive\Documentos\2025.2\Problema 3\Py\Input\Analise 10"

    parameters = [
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

    num_runs = 4

    for irun in range(1, num_runs+1):

        input_dir = base_dir
        log_dir = os.path.join(input_dir, f"log")
        os.makedirs(log_dir, exist_ok=True)

        p = Process(target=run_PSO, args=(parameters, f"run{irun}", input_dir, log_dir))
        p.start()
        p.join()

        p = Process(target=run_GA, args=(parameters, f"run{irun}", input_dir, log_dir))
        p.start()
        p.join()

        p = Process(target=run_BO_skopt, args=(parameters, f"run{irun}", input_dir, log_dir))
        p.start()
        p.join()