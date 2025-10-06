from tests.trelicada.BO_trel_TestRun import BO_run
from tests.trelicada.GA_trel_TestRun import GA_run
from tests.trelicada.PSO_trel_TestRun import PSO_run
from optimization.parameter import *

from multiprocessing import Process


def run_BO(parameters, irun, input_dir, log_dir, tuning):
    BO_run(parameters, irun, input_dir, log_dir, tuning)

def run_GA(parameters, irun, input_dir, log_dir):
    GA_run(parameters, irun, input_dir, log_dir)

def run_PSO(parameters, irun, input_dir, log_dir):
    PSO_run(parameters, irun, input_dir, log_dir)

if __name__ == '__main__':

    input_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 2\input"
    log_dir = r"D:\Thiago Artur\OneDrive\Documentos\2025.2\Problema 2\log"

    parameters = [
        Continuous(180e9, 220e9, 'modulo_banz'),
        # Continuous(0.1, 0.49, 'poisson_banz'),
        # Continuous(7500, 8200, 'dens_banz'),

        Continuous(180e9, 220e9, 'modulo_diag'),
        # Continuous(0.1, 0.49, 'poisson_diag'),
        # Continuous(7500, 8200, 'dens_diag'),

        Continuous(180e9, 220e9, 'modulo_contrav'),
        # Continuous(0.1, 0.49, 'poisson_contrav'),
        # Continuous(7500, 8200, 'dens_contrav'),

        Continuous(40e5, 100e6, 'rigidez1'),
        Continuous(40e5, 100e6, 'rigidez2'),
        Continuous(40e5, 100e6, 'rigidez3'),
        Continuous(40e5, 100e6, 'rigidez4'),

        Continuous(400, 800, 'massa')
    ]

    num_runs = 3

    for irun in range(1, num_runs+1):

        p = Process(target=run_GA, args=(parameters, irun, input_dir, log_dir))
        p.start()
        p.join()

        p = Process(target=run_BO, args=(parameters, irun, input_dir, log_dir, True))
        p.start()
        p.join()

        p = Process(target=run_BO, args=(parameters, irun, input_dir, log_dir, False))
        p.start()
        p.join()

        p = Process(target=run_PSO, args=(parameters, irun, input_dir, log_dir))
        p.start()
        p.join()