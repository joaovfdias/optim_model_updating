import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
import os
import csv
from datetime import datetime

from optimization.turbo_optimizer.turbo import TuRBO
from tests.indexador_2026 import indexar_problema, indexar_device

from TuRBO_run import TuRBO_run



class MetaTuRBO:
    def __init__(
            self,
            problem_ids,
            device,
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
        self.device = device
        self.evaluations = evaluations
        self.batch_size = batch_size
        self.n_runs = n_runs
        self.normalize = normalize
        self.seeds = seeds

        # carregar problemas
        self.problems = [indexar_problema(pid) for pid in problem_ids]

        self.dim = len(self.problems[0]["parameters"])

        self.iter_counter = 0
        self.log_dir = None

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

    # REGISTRO
    @staticmethod
    def log_meta_iteration(
            log_dir: str,
            iter_id: int,
            config: dict,
            score_mean: float,
            score_std: float | None = None,
            filename: str | None = None,
    ):
        """
        Registra uma iteração da meta-otimização em CSV.

        :param log_dir: diretório onde salvar o log
        :param iter_id: número da iteração
        :param config: dicionário com hiperparâmetros
        :param score_mean: valor médio da função objetivo
        :param score_std: desvio padrão (opcional)
        :param filename: nome do arquivo (opcional)
        """

        os.makedirs(log_dir, exist_ok=True)

        # define nome do arquivo (fixo por execução)
        if filename is None:
            filename = "meta_turbo_log.csv"

        filepath = os.path.join(log_dir, filename)

        file_exists = os.path.isfile(filepath)

        # campos do CSV
        fieldnames = [
            "iter",
            "length",
            "success_tol",
            "failure_tol",
            "n_init",
            "score_mean",
            "score_std",
            "timestamp"
        ]

        with open(filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # escreve header só uma vez
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "iter": iter_id,
                "length": config["length"],
                "success_tol": config["success_tol"],
                "failure_tol": config["failure_tol"],
                "n_init": config["n_init"],
                "score_mean": score_mean,
                "score_std": score_std,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    # ============================
    # FUNÇÃO OBJETIVO
    # ============================
    def objective(self, config):

        length, success_tol, failure_tol, n_init = config

        problem_scores = []

        for Problema in self.problem_ids:

            pb = indexar_problema(Problema)
            pc = indexar_device(self.device)

            base_dir = os.path.join(pc.base_path, f"Problema {Problema}")
            local_dir = os.path.join(pc.local_path, f"Problema {Problema}")

            log_dir = self.log_dir or os.path.join(base_dir, "log", "metaopt")

            script_name = pb.script_filename
            noise = pb.noise
            parameters = pb.parameters

            run_scores = []

            for i in range(self.n_runs):

                result = TuRBO_run(
                    irun=i,
                    parameters=parameters,
                    base_dir=base_dir,
                    local_dir=local_dir,
                    log_dir=os.path.join(log_dir, "runs"),
                    base_script_filename=script_name,
                    noise=noise,
                    initial_points=n_init,
                    evaluations=self.evaluations,
                    batch_size=self.batch_size,
                    acqf="ts",
                    turbo_params={
                        "length": length,
                        "success_tol": int(success_tol),
                        "failure_tol": int(failure_tol),
                    }
                )

                run_scores.append(result["best_fitness"])

            run_scores = np.array(run_scores)

            mean = np.mean(run_scores)
            std = np.std(run_scores)

            score = mean + 0.2 * std

            self.log_meta_iteration(
                log_dir=log_dir,
                iter_id=self.iter_counter,
                config={
                    "length": length,
                    "success_tol": int(success_tol),
                    "failure_tol": int(failure_tol),
                    "n_init": int(n_init),
                },
                score_mean=mean,
                score_std=std
            )

            problem_scores.append(score)

            self.iter_counter += 1

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