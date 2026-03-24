import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer

from optimization.turbo_optimizer.turbo import TuRBO
from tests.indexador_2026 import indexar_problema


class MetaTuRBO:
    def __init__(
            self,
            problem_ids,
            evaluations=120,
            batch_size=4,
            n_runs=5,
            normalize=True,
            seeds=(0, 1, 2, 3, 4),
    ):
        """
        Meta-otimização do TuRBO considerando múltiplos problemas e múltiplas rodadas.

        :param problem_ids: lista de IDs para indexar_problema()
        :param evaluations: budget reduzido do TuRBO
        :param batch_size: tamanho do batch (q)
        :param n_runs: número de rodadas por problema
        :param normalize: se True, normaliza por problema
        :param seeds: seeds utilizadas
        """

        self.problem_ids = problem_ids
        self.evaluations = evaluations
        self.batch_size = batch_size
        self.n_runs = n_runs
        self.normalize = normalize
        self.seeds = seeds

        # carregar problemas
        self.problems = [indexar_problema(pid) for pid in problem_ids]

        self.dim = len(self.problems[0]["parameters"])

        # espaço de busca
        self.space = [
            Real(0.4, 0.9, name="length"),
            Integer(2, 6, name="success_tol"),
            Integer(3, 10, name="failure_tol"),
            Integer(self.dim, 4 * self.dim, name="n_init"),
        ]

        # baseline para normalização
        self.baselines = None
        if self.normalize:
            self.baselines = self._compute_baselines()

    # ============================
    # BASELINE (normalização)
    # ============================
    def _compute_baselines(self):
        """
        Executa TuRBO padrão para obter escala de cada problema.
        """
        baselines = []

        print("\n[Meta] Computing baselines...")

        for prob in self.problems:
            scores = []

            for seed in self.seeds[:2]:  # menos seeds para baseline
                optimizer = TuRBO(
                    prob["fitness_function"],
                    prob["parameters"]
                )

                result = optimizer.run(
                    evaluations=80,
                    batch_size=self.batch_size,
                    seed=seed,
                    status=False,
                    log=False
                )

                scores.append(result["best_fitness"])

            baselines.append(np.mean(scores))

        return np.array(baselines)

    # ============================
    # FUNÇÃO OBJETIVO
    # ============================
    def objective(self, config):

        length, success_tol, failure_tol, n_init = config

        problem_scores = []

        for i, prob in enumerate(self.problems):

            run_scores = []

            for run_id in range(self.n_runs):

                seed = self.seeds[run_id % len(self.seeds)]

                optimizer = TuRBO(
                    prob["fitness_function"],
                    prob["parameters"],
                    initial_points=n_init
                )

                result = optimizer.run(
                    evaluations=self.evaluations,
                    batch_size=self.batch_size,
                    n_init=n_init,
                    seed=seed,
                    status=False,
                    log=False,
                    turbo_params={
                        "length": length,
                        "success_tol": int(success_tol),
                        "failure_tol": int(failure_tol),
                    }
                )

                run_scores.append(result["best_fitness"])

            run_scores = np.array(run_scores)

            # média e desvio
            mean = np.mean(run_scores)
            std = np.std(run_scores)

            # normalização (opcional)
            if self.normalize:
                mean = mean / (self.baselines[i] + 1e-12)
                std = std / (self.baselines[i] + 1e-12)

            # penalizar variabilidade
            score = mean + 0.2 * std

            problem_scores.append(score)

        return np.mean(problem_scores)

    # ============================
    # EXECUÇÃO
    # ============================
    def run(self, n_calls=20, n_initial_points=5, random_state=42):

        result = gp_minimize(
            func=self.objective,
            dimensions=self.space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
        )

        best_params = {
            "length": result.x[0],
            "success_tol": result.x[1],
            "failure_tol": result.x[2],
            "n_init": result.x[3],
        }

        return result, best_params