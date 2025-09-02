
from optimization.optimizer import Optimizer
import json
import math

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

STATE_FILE = r"log\storage_test_PSO.json.gz"

# por enquanto é necessário passar a função objetivo (pois ela não está em fitness_spec)
opt = Optimizer.load_state(STATE_FILE, fitness_function=ackley)

# --------- checagens sucintas de integridade ---------
def almost_equal(a, b, tol=1e-12):
    return (a == b) or (a is None and b is None) or (abs(a - b) <= tol)

# 1) tamanhos
assert len(opt.populations) > 0, "Sem populações carregadas."
n = len(opt.populations[-1])
assert all(len(p) == n for p in opt.populations), "Populações com tamanhos diferentes."

# 2) indivíduo best da última população (param e fitness)
last_best = min(opt.populations[-1], key=lambda x: x.fitness)
print("Best (carregado):", last_best.fitness, last_best.param)

# 3) PSO: conferir que cada Partícula tem velocity e (opcionalmente) best pessoal
for idx, ind in enumerate(opt.populations[-1]):
    if ind.__class__.__name__ == "Particle":
        assert hasattr(ind, "velocity") and (ind.velocity is None or len(ind.velocity) == len(ind.param)), f"Velocidade inválida na partícula {idx}"
        # best pessoal pode ser None nas primeiras iterações; se existir, deve ter [param, fitness]
        if getattr(ind, "best", None) is not None:
            bp = ind.best
            assert isinstance(bp, list) and len(bp) == 2 and isinstance(bp[0], list), f"Formato de best inválido na partícula {idx}"

print("Reconstrução OK (tamanhos, best e velocidades conferidos).")

# 4) (opcional) continuar a partir do estado carregado e comparar com seguir direto do run_and_save:
# Se você salvou include_rng_state=True, dá para esperar a MESMA trajetória ao continuar:
continuado = opt.run(iterations=len(opt.populations) + 1, log=False)  # roda +1 iteração
print("Continuou do snapshot. Novo best:", continuado.fitness)
