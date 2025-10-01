from tests.trelicada.BO_trel_TestRun import BO_run
from tests.trelicada.GA_trel_TestRun import GA_run
from tests.trelicada.PSO_trel_TestRun import PSO_run
from optimization.parameter import *

from multiprocessing import Process


def run_BO(parameters, irun, tuning):
    BO_run(parameters, irun, tuning)

def run_GA(parameters, irun):
    GA_run(parameters, irun)

def run_PSO(parameters, irun):
    PSO_run(parameters, irun)

if __name__ == '__main__':

    parameters = [
        Continuous(150e9, 250e9, 'modulo_banz'),
        # Continuous(0.1, 0.49, 'poisson_banz'),
        # Continuous(7500, 8200, 'dens_banz'),

        Continuous(150e9, 250e9, 'modulo_diag'),
        # Continuous(0.1, 0.49, 'poisson_diag'),
        # Continuous(7500, 8200, 'dens_diag'),

        Continuous(150e9, 250e9, 'modulo_contrav'),
        # Continuous(0.1, 0.49, 'poisson_contrav'),
        # Continuous(7500, 8200, 'dens_contrav'),

        Continuous(1e5, 1e7, 'rigidez1'),
        Continuous(1e5, 1e7, 'rigidez2'),
        Continuous(1e5, 1e7, 'rigidez3'),
        Continuous(1e5, 1e7, 'rigidez4'),

        Continuous(400, 800, 'massa')
    ]

    num_runs = 3

    for irun in range(1, num_runs+1):

        p = Process(target=run_GA, args=(parameters, irun))
        p.start()
        p.join()

        p = Process(target=run_PSO, args=(parameters, irun))
        p.start()
        p.join()

        p = Process(target=run_BO, args=(parameters, irun, False))
        p.start()
        p.join()

        p = Process(target=run_BO, args=(parameters, irun, True))
        p.start()
        p.join()