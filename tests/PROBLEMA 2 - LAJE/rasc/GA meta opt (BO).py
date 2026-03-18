from GA_run_TEST2 import GA_run
from PSO_run_TEST2 import PSO_run

from optimization.parameter import *
from external.ansys.parser import Ansys

from skopt import gp_minimize

import os


# chamada dos algoritmos
def run_GA(irun, base_dir, parameters, population_size, generations, elitism_rate, crossover_rate, mutation_strength):
    GA_run(irun, base_dir, parameters, population_size, generations, elitism_rate, crossover_rate, mutation_strength)

def run_PSO(irun, base_dir, parameters, population_size, iterations, w, w_rate, c1, c2, init_vel_ratio):
    PSO_run(irun, base_dir, parameters, population_size, iterations, w, w_rate, c1, c2, init_vel_ratio)

def BO_GA_metaopt(GA_params: dict):
    # lembrar que o registro de individuals, scores e log acontece aqui no BO_skopt do pacote, mas lá é pelo sistema de classe
    # função avaliadora que vai variar por algoritmo, recebendo os parâmetros, realizando 3-5 rodadas e retornando o score
    pass

def BO_PSO_metaopt(PSO_params: dict):
    # função avaliadora que vai variar por algoritmo, recebendo os parâmetros, realizando 3-5 rodadas e retornando o score
    pass

def run_BO(evaluate, search_space, evaluations, seed_points, acq_func=None, xi=None, kappa=None):

    acq_func = acq_func or "EI"
    xi = xi or 0.01  # default
    kappa = kappa or 1.96  # default

    result = gp_minimize(evaluate, search_space, n_calls=evaluations,
                         n_initial_points=seed_points, initial_point_generator='lhs',
                         acq_func=acq_func, acq_optimizer="sampling", xi=xi, kappa=kappa)

    print(
        f"\nMelhor solução encontrada: [adaptar]")

    return result

# bloco de execução
if __name__ == '__main__':

    # parâmetros do Problema 2
    parameters = [
        Continuous(20e9,35e9, 'modulo_viga_1'),
        Continuous(20e9, 35e9, 'modulo_viga_2'),

        Continuous(20e9, 35e9, 'modulo_centro'),
        Continuous(20e9, 35e9, 'modulo_borda_1'),
        Continuous(20e9, 35e9, 'modulo_borda_2'),

        Continuous(50e6, 50e8, 'rigidez1'),
        Continuous(50e6, 50e8, 'rigidez2'),
        Continuous(50e6, 50e8, 'rigidez3'),
        Continuous(50e6, 50e8, 'rigidez4')
    ]

    # diretórios
    base_dir = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"
    log_base_dir = os.path.join(base_dir, 'meta-opt')

    # parâmetros do GA
    GA_parameters = [
        Continuous(0.00, 0.15,"elitism_rate"),
        Continuous(0.40, 0.90, "crossover_rate"),
        Continuous(0.01, 0.30, "mutation_strength")
    ]

    # parâmetros do PSO
    PSO_parameters = [
        Continuous(0.40,1.20,"w"),
        Continuous(0.900,0.999,"w_rate"),
        Continuous(1.00,2.50,"c1"),
        Continuous(1.00,2.50,"c2"),
        Continuous(0.05,0.50,"init_vel_ratio")
    ]

    # loop de execução: metaopt com Bayesiano


    # parar ao finalizar:
    Ansys.kill_ansys_process()