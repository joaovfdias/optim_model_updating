
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
    return term1 + term2 + a + np.exp(1), {"Term1": term1, "Term2": term2}

# parâmetros do modelo:
parameters = [Continuous(-32.768,32.768,f"v{i}") for i in range(5)]

population_size = 50
iteracoes = 10 # 10 iterações só para demo

# rodada
rodada = PSO(ackley, parameters, population_size, w=0.6, w_rate=0.99, c1=2.05, c2=2.05, init_vel_ratio=0.2)
rodada.set_tolerance(fit_abs=2e-2, patience=10)
rodada.set_log(log_title="PSO_ackley", log_dir=None)
best = rodada.run(iterations=iteracoes, log="full")

# salva estado (com RNG state para poder retomar exatamente do ponto)
state_path = rodada.save_state(filename="storage_test_PSO.json.gz", fitness_spec=None, include_rng_state=True)
print("Estado salvo em:", state_path)
