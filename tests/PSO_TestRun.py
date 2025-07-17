from optimization.parameter import Continuous
from optimization.pso_optimizer.pso_optimizer import PSO

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
w = 0.6 # inércia
w_rate = 0.99 # taxa de decaimento de inércia
c1 = 2.05 # governa a exploração da população
c2 = 2.05 # governa a convergência
init_vel_ratio = 0.2 # proporção do espaço de busca que pode ser empregado para velocidade inicial

population_size = 50
iteracoes = 100

# declarção do otimizador:
rodada = PSO(ackley, parameters, population_size, w, w_rate, c1, c2, init_vel_ratio)

# ajuste do registro:
log = "full" # tipo de registro (True: simplificado, "full": todos os indivíduos)
log_title = "PSO_ackley" # alterar nome do arquivo gerado, se quiser
log_dir = None # alterar diretório do registro, por padrão \log (lembre-se de usar o formato r"{caminho}" para declarar diretórios)
rodada.set_log(log_title, log_dir)

# chamada:
best = rodada.run(iteracoes, log=log)