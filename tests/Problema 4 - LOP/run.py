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

rodar = {
    "GA": GA_run,
    "PSO": PSO_run,
    "BO": BO_run,
    "TuRBO": TuRBO_run,
}

GA_sets = [
    {
        "elitism_rate": 0.12,
        "crossover_rate": 0.87,
        "mutation_strength": 0.19,
        "selection_method": "tournament"
    },
    {
        "elitism_rate": 0.10,
        "crossover_rate": 0.60,
        "mutation_strength": 0.10,
        "selection_method": "tournament"
    },
    {
        "elitism_rate": 0.10,
        "crossover_rate": 0.75,
        "mutation_strength": 0.25,
        "selection_method": "tournament"
    }
]

PSO_sets = [
    {
        "w": 0.73,
        "w_rate": 0.96,
        "c1": 1.90,
        "c2": 1.32,
        "init_vel_ratio": 0.06
    },
    {
        "w": 0.60,
        "w_rate": 0.99,
        "c1": 2.05,
        "c2": 2.05,
        "init_vel_ratio": 0.20
    },
    {
        "w": 1.13,
        "w_rate": 0.96,
        "c1": 1.15,
        "c2": 1.43,
        "init_vel_ratio": 0.18
    }
]

BO_sets = [
    {
        "acq_func": "PI",
        "xi": 0.0752
    },
    {
        "acq_func": "gp_hedge",
        "xi": 0.01
    },
    {
        "acq_func": "EI",
        "xi": 0.0032
    }
]

TuRBO_sets = [
    {
        "batch_size": 4,
        "acq_func": "ts"
    }
]

if __name__ == '__main__':

    Compiuter = "LEST 1"
    runs = 5
    Problemas = [4, 3]
    ListaAlgoritmos = ["BO", "PSO", "GA"]
    diretorio = r"C:\Users\Thiago\OneDrive\Documentos\2026.2\Artigo (mestrado)\Novas rodadas"

    for Problema in Problemas:
        pb = indexar_problema(Problema)
        pc = indexar_device(Compiuter)

        base_dir = os.path.join(pc.base_path, f"Problema {Problema}")
        local_dir = os.path.join(pc.local_path, f"Problema {Problema}")
        log_base_dir = os.path.join(diretorio, f"Problema {Problema}")

        script_name = pb.script_filename
        noise = pb.noise
        parameters = pb.parameters

        for Algoritmo in ListaAlgoritmos:
            sets = globals()[f"{Algoritmo}_sets"]

            for idx_set, curr_set in enumerate(sets, start=1):

                for irun in range(1, runs+1):
                    print(f"\nStructure {Problema}: Running {Algoritmo} - Set {idx_set} ({irun}/{runs}). . .")

                    log_dir = os.path.join(log_base_dir, Algoritmo, f'Set {idx_set}')

                    p = Process(
                        target=rodar[Algoritmo],
                        args=(irun, parameters, base_dir, local_dir, log_dir, script_name, noise, None, None),
                        kwargs=curr_set
                    )
                    p.start()
                    p.join()