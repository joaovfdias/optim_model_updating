import os
import time
import csv
import shutil
import numpy as np
from datetime import datetime
from multiprocessing import Process, Queue

from skopt import gp_minimize
from skopt.space import Real

from optimization.parameter import Continuous
from external.parser import Ansys

from GA_run_TEST2 import GA_run
from PSO_run_TEST2 import PSO_run


class MetaOptimizerRunner:

    def __init__(self, base_dir, struct_params, n_repetitions=3):
        self.base_dir = base_dir
        self.struct_params = struct_params
        self.n_repetitions = n_repetitions

        self.W_FIT = 0.6
        self.W_CV = 0.3
        self.W_TIME = 0.1

        self.global_results = []

        self.log_dir = os.path.join(base_dir, 'meta_opt', 'populational logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # EXECUTORES DOS ALGORITMOS

    @staticmethod
    def _worker_ga(run_id, work_dir, struct_params, ga_args, result_queue):
        try:
            start_t = time.time()

            result = GA_run(
                irun=run_id,
                base_dir=work_dir,
                parameters=struct_params,
                population_size=ga_args['population'],
                generations=ga_args['generations'],
                elitism_rate=ga_args['elitism_rate'],
                crossover_rate=ga_args['crossover_rate'],
                mutation_strength=ga_args['mutation_strength']
            )

            end_t = time.time()
            fitness = result[0]

            result_queue.put({'fitness': fitness, 'time': end_t - start_t, 'success': True})

        except Exception:
            result_queue.put({'fitness': 1e6, 'time': 0.0, 'success': False})

    @staticmethod
    def _worker_pso(run_id, work_dir, struct_params, pso_args, result_queue):
        try:
            start_t = time.time()

            result = PSO_run(
                irun=run_id,
                base_dir=work_dir,
                parameters=struct_params,
                population_size=pso_args['population'],
                iterations=pso_args['generations'],
                w=pso_args['w'],
                w_rate=pso_args['w_rate'],
                c1=pso_args['c1'],
                c2=pso_args['c2'],
                init_vel_ratio=pso_args['init_vel_ratio']
            )

            end_t = time.time()
            fitness = result[0]

            result_queue.put({'fitness': fitness, 'time': end_t - start_t, 'success': True})

        except Exception:
            result_queue.put({'fitness': 1e6, 'time': 0.0, 'success': False})


    # SCORE

    @staticmethod
    def _norm_percentile(val, lst, p_low=10, p_high=90):
        if len(lst) < 5:
            return 0.5

        low = np.percentile(lst, p_low)
        high = np.percentile(lst, p_high)

        if high == low:
            return 0.5

        n = (val - low) / (high - low)
        return np.clip(n, 0.0, 1.0)

    def calculate_score(self, fits, times):
        valid_fits = [f for f in fits if np.isfinite(f) and f < 1e5]
        if not valid_fits:
            return 1.0, None, None, None

        log_fits = [np.log10(abs(f) + 1e-20) for f in valid_fits]

        avg_log_fit = np.mean(log_fits)
        std_log_fit = np.std(log_fits)

        cv = std_log_fit / (abs(avg_log_fit) + 1e-6)
        cv = np.log10(1.0 + cv)

        avg_time = np.mean(times)

        self.global_results.append({
            'log_fit': avg_log_fit,
            'cv': cv,
            'time': avg_time
        })

        all_log_fits = [x['log_fit'] for x in self.global_results]
        all_cvs = [x['cv'] for x in self.global_results]
        all_times = [x['time'] for x in self.global_results]

        n_fit = self._norm_percentile(avg_log_fit, all_log_fits)
        n_cv = self._norm_percentile(cv, all_cvs)
        n_time = self._norm_percentile(avg_time, all_times)

        score = (
            self.W_FIT * n_fit +
            self.W_CV * n_cv +
            self.W_TIME * n_time
        )

        return score, avg_log_fit, cv, avg_time


    # ESTAGNAÇÃO

    @staticmethod
    def detect_stagnation(history, window=3, tol=1e-3):
        if len(history) < window + 1:
            return False

        f_old = history[-(window + 1)]
        f_new = history[-1]

        rel_change = abs(f_new - f_old) / (abs(f_old) + 1e-12)
        return rel_change < tol

    def pretest_population_size(self, algo_type, populations, max_generations=10):
        results = []

        for pop in populations:
            stagnation_gens = []

            for rep in range(3):
                history = []

                for gen in range(1, max_generations + 1):

                    if algo_type == "GA":
                        result = GA_run(
                            irun=rep,
                            base_dir=self.base_dir,
                            parameters=self.struct_params,
                            population_size=pop,
                            generations=gen,
                            elitism_rate=0.05,
                            crossover_rate=0.7,
                            mutation_strength=0.1
                        )
                    else:
                        result = PSO_run(
                            irun=rep,
                            base_dir=self.base_dir,
                            parameters=self.struct_params,
                            population_size=pop,
                            iterations=gen,
                            w=0.7,
                            w_rate=0.98,
                            c1=1.5,
                            c2=1.5,
                            init_vel_ratio=0.2
                        )

                    history.append(result[0])

                    if self.detect_stagnation(history):
                        stagnation_gens.append(gen)
                        break

            if stagnation_gens:
                results.append((pop, int(np.mean(stagnation_gens))))

        return results


    # AVALIAÇÃO BO

    def _evaluate_batch(self, algo_type, param_dict):
        fits, times = [], []

        for i in range(1, self.n_repetitions + 1):

            q = Queue()

            if algo_type == "GA":
                p = Process(target=self._worker_ga,
                            args=(i, self.base_dir, self.struct_params, param_dict, q))
            else:
                p = Process(target=self._worker_pso,
                            args=(i, self.base_dir, self.struct_params, param_dict, q))

            p.start()
            res = q.get()
            p.join()

            # shutil.rmtree(run_dir, ignore_errors=True)

            if res['success']:
                fits.append(res['fitness'])
                times.append(res['time'])

        score, avg_fit, cv, avg_time = self.calculate_score(fits, times)
        self.log_step(algo_type, param_dict, score, avg_fit, cv, avg_time)

        return score


    # LOG

    def log_step(self, algo, params, score, avg_fit, cv, avg_time):
        filename = os.path.join(self.log_dir, f"meta_opt_{algo}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv")
        exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(['Timestamp', 'Score', 'Avg_LogFit', 'CV', 'Avg_Time'] + list(params.keys()))

            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                score, avg_fit, cv, avg_time
            ] + list(params.values()))


    # EXE

    def run_meta_optimization(self, algo_type, generations, population, n_calls=30):

        if algo_type == "GA":
            space = [
                Real(0.00, 0.15),
                Real(0.40, 0.90),
                Real(0.01, 0.30)
            ]

            def objective(x):
                params = {
                    'elitism_rate': x[0],
                    'crossover_rate': x[1],
                    'mutation_strength': x[2],
                    'population': population,
                    'generations': generations
                }
                return self._evaluate_batch("GA", params)

        else:
            space = [
                Real(0.40, 1.20),
                Real(0.90, 0.999),
                Real(1.00, 2.50),
                Real(1.00, 2.50),
                Real(0.05, 0.50)
            ]

            def objective(x):
                params = {
                    'w': x[0],
                    'w_rate': x[1],
                    'c1': x[2],
                    'c2': x[3],
                    'init_vel_ratio': x[4],
                    'population': population,
                    'generations': generations
                }
                return self._evaluate_batch("PSO", params)

        return gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=10,
            acq_func="EI",
            xi=0.01
            # random_state=42
        )



# MAIN

if __name__ == "__main__":

    struct_params = [
            Continuous(20e9, 35e9, 'modulo_viga_1'),
            Continuous(20e9, 35e9, 'modulo_viga_2'),
            Continuous(20e9, 35e9, 'modulo_centro'),
            # Continuous(20e9, 35e9, 'modulo_borda_1'),
            # Continuous(20e9, 35e9, 'modulo_borda_2'),

            Continuous(0.1, 0.40, 'poisson'),
            Continuous(2400, 2600, 'dens'),

            Continuous(50e6, 50e8, 'rigidez1'),
            Continuous(50e6, 50e8, 'rigidez2'),
            Continuous(50e6, 50e8, 'rigidez3'),
            Continuous(50e6, 50e8, 'rigidez4')
        ]

    BASE_DIR = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"

    runner = MetaOptimizerRunner(BASE_DIR, struct_params)

    pop_tests = runner.pretest_population_size("GA", [30, 60, 90, 120])
    population, generations = min(pop_tests, key=lambda x: x[1])

    runner.run_meta_optimization("GA", generations, population)
    runner.run_meta_optimization("PSO", generations, population)

    try:
        Ansys.kill_ansys_process()
    except:
        pass
