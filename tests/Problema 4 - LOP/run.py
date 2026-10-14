from TuRBO_run import TuRBO_run
from BO_LOP_run import BO_run
from GA_LOP_run import GA_run
from PSO_LOP_run import PSO_run

from tests.indexador_2026 import *

from multiprocessing import Process
import os


def run_TuRBO(irun, parameters, base_dir, local_dir=None, log_dir=None, base_script_filename=None, noise=False, initial_points=None, evaluations=None, batch_size=4, acqf="ts"):
    TuRBO_run(irun, parameters, base_dir, local_dir, log_dir, base_script_filename, noise, initial_points, evaluations, batch_size, acqf)

def run_BO(irun, parameters, base_dir, local_dir=None, log_dir=None, base_script_filename=None, noise=False, initial_points=None, evaluations=None, acq_func=None, xi=0.01, kappa=1.96):
    BO_run(irun, parameters, base_dir, local_dir, log_dir, base_script_filename, noise, initial_points, evaluations, acq_func, xi, kappa)

def run_GA(irun, parameters, base_dir, local_dir=None, log_dir=None, base_script_filename=None, noise=False, population_size=None, generations=None, elitism_rate=0.10, crossover_rate=0.60, mutation_strength=0.10, selection_method='tournament'):
    GA_run(irun, parameters, base_dir, local_dir, log_dir, base_script_filename, noise,
               population_size, generations, elitism_rate, crossover_rate, mutation_strength,
               selection_method)

def run_PSO(irun, parameters, base_dir, local_dir=None, log_dir=None, base_script_filename=None, noise=False, population_size=None, iterations=None, w=0.6, w_rate=0.99, c1=2.05, c2=2.05, init_vel_ratio=0.2):
    PSO_run(irun, parameters, base_dir, local_dir, log_dir, base_script_filename, noise,
                population_size, iterations, w, w_rate, c1, c2, init_vel_ratio)


if __name__ == '__main__':

    Compiuter = "LEST 1"
    runs = 10
    Problemas = [4, 2, 3]
    diretorio = r"C:\Users\Thiago\OneDrive\Documentos\2026.2\Artigo (mestrado)\Novas rodadas"

    for Problema in Problemas:
        pb = indexar_problema(Problema)
        pc = indexar_device(Compiuter)

        base_dir = os.path.join(pc.base_path, f"Problema {Problema}")
        local_dir = os.path.join(pc.local_path, f"Problema {Problema}")
        log_dir = os.path.join(diretorio, f"Problema {Problema}")

        script_name = pb.script_filename
        noise = pb.noise
        parameters = pb.parameters

        for irun in range(1, runs+1):
            print(f"\nRunning TuRBO ({irun}/{runs}). . .")

            p = Process(target=run_TuRBO, args=(irun, parameters, base_dir, local_dir, log_dir, script_name, noise))
            p.start()
            p.join()