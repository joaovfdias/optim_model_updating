from optimization.parameter import *
from optimization.ga_optimizer import GA

import numpy as np


def ackley(params):
    a = 20
    b = 0.2
    c = 2 * np.pi
    params = np.array(params)
    n = len(params)
    sum_sq = np.sum(params ** 2)
    sum_cos = np.sum(np.cos(c * params))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.exp(1)

# parâmetros do modelo:
parameters = [Continuous(-32.768,32.768,f"v{i}") for i in range(5)]

# parâmetros do algoritmo:
elitism_rate = 0.10
crossover_rate = 0.40
mutation_strength = 0.30

population_size = 100
generations = 200

# declaração do otimizador:
rodada = GA(ackley, parameters, population_size, elitism_rate, crossover_rate, mutation_strength)

# ajuste do registro:
log = "full" # tipo de registro (True: simplificado, "full": todos os indivíduos)
log_title = "GA_ackley" # alterar nome do arquivo gerado, se quiser
log_dir = None # alterar diretório do registro, por padrão \log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
rodada.set_log(log_title, log_dir)

# chamada:
best = rodada.run(generations, log=log)