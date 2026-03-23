from optimization.optimizer import Optimizer
from sensitivity.sensitivity import SensitivityAnalyzer

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

# Carregue um estado salvo ou use um otimizador já rodado:
opt = Optimizer.load_state(r"log\storage_test_PSO.json.gz", fitness_function=ackley)

# Rodar análise (threshold ou top_k)
sa = SensitivityAnalyzer.from_optimizer(opt)
res = sa.run(mode="threshold", threshold=0.3, max_pval=0.05, absolute=True, verbose=True)

# Exportar resultados
sa.export_scores_csv(res, "log/sensitivity/spearman_scores.csv")
sa.export_selection_json(res, opt.parameters, "log/sensitivity/selection.json")

# Obter subset de Parameters recomendados p/ calibrar
selected_params = sa.build_parameter_subset(opt.parameters, res.selected_idx)
print("Selecionados:", [p.key for p in selected_params])
