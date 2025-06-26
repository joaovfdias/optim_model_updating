from optimization.parameter import Continuous
from optimization.ga_optimizer import GAOptimizer as GA

import random


# função objetivo: |x^3 + 3y^2 + 42z|
def fitness_function(param):
    x, y, z = param

    # Frequências: uma lista de 5 valores
    frequencies = [random.uniform(0.1, 1.0) for _ in range(5)]

    # Modos: uma matriz 5x5 (5 nós x 5 modos)
    modes = [[random.uniform(0, 1.0), random.uniform(1, 2.0), random.uniform(2, 3.0), random.uniform(3, 4.0),
              random.uniform(4, 5.0)] for _ in range(5)]

    return abs(x ** 3 + y ** 3 + z ** 3 - 42), [frequencies, modes]


# definição dos parâmetros do algoritmo
parameters_keys = ['x', 'y', 'z']
parameters = [Continuous(-8, 8, 'x'), Continuous(-8, 12, 'y'), Continuous(-25, 25, 'z')]
population_size = 100
sampling_method = "random"
elitism_rate = 0.1
mutation_rate = 0.1
crossover_rate = 0.8
crossover_type = "uniform"

generations = 50

# chamada
rodada = GA(fitness_function, parameters, population_size, elitism_rate, crossover_rate, mutation_rate)
rodada.set_sampling_method(sampling_method)
best_individual = rodada.optimize()
