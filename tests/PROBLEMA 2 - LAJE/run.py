from GA_run_TEST2 import GA_run
from PSO_run_TEST2 import PSO_run
from optimization.parameter import *

from multiprocessing import Process


def run_GA(irun, base_dir, parameters, population_size, generations, elitism_rate, crossover_rate, mutation_strength):
    GA_run(irun, base_dir, parameters, population_size, generations, elitism_rate, crossover_rate, mutation_strength)

def run_PSO(irun, base_dir, parameters, population_size, iterations, w, w_rate, c1, c2, init_vel_ratio):
    PSO_run(irun, base_dir, parameters, population_size, iterations, w, w_rate, c1, c2, init_vel_ratio)

if __name__ == '__main__':

    base_dir = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"

    parameters = [
        Continuous(20e9, 35e9, 'modulo_viga_1'),
        Continuous(20e9, 35e9, 'modulo_viga_2'),
        Continuous(20e9, 35e9, 'modulo_centro'),
        Continuous(20e9, 35e9, 'modulo_borda_1'),
        Continuous(20e9, 35e9, 'modulo_borda_2'),
        Continuous(50e6, 50e8, 'rigidez1'),
        Continuous(50e6, 50e8, 'rigidez2'),
        Continuous(50e6, 50e8, 'rigidez3'),
        Continuous(50e6, 50e8, 'rigidez4')
    ]

    population_size = 90
    generations = iterations = 1500
    elitism_rate = 0.1
    crossover_rate = 0.6
    mutation_strength = 0.2

    num_runs = 5

    for irun in range(1, num_runs+1):

        # p = Process(target=run_PSO, args=(irun, base_dir, parameters, population_size, iterations, w, w_rate, c1, c2, init_vel_ratio))
        # p.start()
        # p.join()

        p = Process(target=run_GA, args=(irun, base_dir, parameters, population_size, generations, elitism_rate, crossover_rate, mutation_strength))
        p.start()
        p.join()
