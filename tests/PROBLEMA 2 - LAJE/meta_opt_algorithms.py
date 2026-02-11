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
            fitness = result.fitness

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
            fitness = result.fitness

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
    def detect_stagnation(history, window=10, tol=1e-3):
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
                            elitism_rate=0.10,
                            crossover_rate=0.60,
                            mutation_strength=0.10
                        )
                    else:
                        result = PSO_run(
                            irun=rep,
                            base_dir=self.base_dir,
                            parameters=self.struct_params,
                            population_size=pop,
                            iterations=gen,
                            w=0.6,
                            w_rate=0.99,
                            c1=2.05,
                            c2=2.05,
                            init_vel_ratio=0.2
                        )

                    history.append(result.fitness)

                    if self.detect_stagnation(history):
                        stagnation_gens.append(gen)
                        break

            if stagnation_gens:
                results.append((pop, int(np.mean(stagnation_gens))))

        return results

    def pretest_generations(self, algo_type, fixed_population, max_generations=100, expected_fit=0.30, resume=None):
        """
        Roda o algoritmo com população fixa e incrementa gerações até detectar estagnação.
        Retorna a média de gerações onde a convergência ocorreu.
        :param resume: entra com nome do arquivo log caso se deseje retomar uma rodada inicial
        """

        pretest_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n Calibrando Gerações para {algo_type} (População Fixa: {fixed_population})")

        stagnation_points = []
        n_reps = 3  # Número de repetições para média estatística

        for rep in range(n_reps):

            history = []
            detected_gen = max_generations  # Valor padrão caso não estagne

            # Loop incremental de gerações
            # NOTA: Se o GA_run não tiver "resume", isso roda do zero a cada passo (lento).
            # Se tiver acesso ao histórico de uma rodada única, seria muito mais rápido.
            count_resume = 0
            pretest_log_path = os.path.join(self.base_dir, 'meta-opt', 'results', 'pretest-generations')
            os.makedirs(pretest_log_path, exist_ok=True)
            log_title = resume if rep == 0 else None
            for gen in range(10, max_generations + 1, 10):

                if rep==0: print("\n\n" + "-" * 50)
                print(f"\n\n Iniciando rodada com {gen} gerações" + f"\nRepetição {rep + 1}/{n_reps} . . .")

                previous_log = log_title
                log_title = f"GA_rep({rep})_gen({gen})_{pretest_timestamp}"

                # Executa o algoritmo com 'gen' gerações
                if algo_type == "GA":
                    result = GA_run(
                        irun=rep,
                        base_dir=self.base_dir,
                        parameters=self.struct_params,
                        population_size=fixed_population,
                        generations=gen,
                        elitism_rate=0.10,
                        crossover_rate=0.60,
                        mutation_strength=0.10,
                        log_data={"dir": pretest_log_path, "resume": previous_log, "title": log_title} # algoritmo retoma rodada anterior, exceto pela primeira (resume = 0)
                    )
                else:  # PSO
                    result = PSO_run(
                        irun=rep,
                        base_dir=self.base_dir,
                        parameters=self.struct_params,
                        population_size=fixed_population,
                        iterations=gen,
                        w=0.6, w_rate=0.99, c1=2.05, c2=2.05, init_vel_ratio=0.2,
                        log_data={"dir": pretest_log_path, "resume": previous_log, "title": log_title} # algoritmo retoma rodada anterior, exceto pela primeira (resume = 0)
                    )

                # Assume que result.fitness é o melhor valor encontrado até aquela geração
                val = result.fitness if not isinstance(result, float) else result
                history.append(val) # armazena o melhor fit de 10 em 10

                count_resume += 1

                # Verifica estagnação usando sua função estática
                if self.detect_stagnation(history, window=1, tol=1e-3):
                    print(f"Convergiu na geração {gen}. Fitness: {history[-1]}.")
                    if history[-1] < expected_fit:
                        detected_gen = gen
                        break
                    else:
                        print(f"Fitness alto ({history[-1]}), dando sequência.")
                        continue
            else:
                print(f"Não convergiu até {max_generations} gerações. Fitness alcançado: {history[-1]}.")

            stagnation_points.append(detected_gen)

        # Calcula a média de gerações necessárias
        optimal_gens = int(np.ceil(np.mean(stagnation_points)))

        # Opcional: Adiciona uma margem de segurança (+10% ou +5 gens)
        safe_gens = optimal_gens + 5

        print(f">>> Média de Estagnação: {optimal_gens} -> Definido: {safe_gens} gerações.")
        return safe_gens

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


        print("\n" + "-" * 50)
        print(f"\n\nRODANDO META-OTIMIZAÇÃO DO {algo_type} . . .")

        return gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=10,
            acq_func="PI",
            xi=0.075
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
    population = len(struct_params)*10
    population = 3

    # alterar com base na máquina:
    BASE_DIR = r"C:\Users\Thiago\OneDrive\Documentos\2025.2\Pesquisa\4. Rodadas e resultados\Teste 2 - hiperparametros"
    resume = "GA_rep(0)_gen(10)_20260210_152031"

    runner = MetaOptimizerRunner(BASE_DIR, struct_params)

    choice = "GA, PSO" # alterar conforme algoritmo desejado

    if "GA" in choice:
        # pop_tests = runner.pretest_population_size("GA", [30, 60, 90, 120])
        # population, generations = min(pop_tests, key=lambda x: x[1])

        # teste de gerações:
        optimal_generations = runner.pretest_generations("GA", population, max_generations=120, resume=None)
        runner.run_meta_optimization("GA", optimal_generations, population)

    if "PSO" in choice:
        # pop_tests = runner.pretest_population_size("PSO", [30, 60, 90, 120])
        # population, generations = min(pop_tests, key=lambda x: x[1])

        # teste de gerações:
        optimal_generations = runner.pretest_generations("PSO", population, max_generations=120, resume=None)
        runner.run_meta_optimization("PSO", optimal_generations, population)

    try:
        Ansys.kill_ansys_process()
    except:
        pass