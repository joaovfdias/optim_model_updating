import os
import time
import csv
import shutil
import numpy as np
from datetime import datetime
from multiprocessing import Process, Queue

# Imports do skopt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Imports do seu pacote
from optimization.parameter import Continuous
from external.parser import Ansys

# Importação dos Algoritmos
from GA_run_TEST2 import GA_run
from PSO_run_TEST2 import PSO_run


class MetaOptimizerRunner:
    def __init__(self, base_dir, struct_params, n_repetitions=3):
        """
        Classe para gerenciar a meta-otimização de GA e PSO.
        """
        self.base_dir = base_dir
        self.struct_params = struct_params
        self.n_repetitions = n_repetitions  # 3 a 5 rodadas

        # Pesos do Score
        self.W_FIT = 0.6
        self.W_CV = 0.3
        self.W_TIME = 0.1

        # Histórico Global para Normalização
        self.global_results = []

        # Diretórios
        self.log_dir = os.path.join(base_dir, 'meta_opt_logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


    # 1. FUNÇÕES WORKER (Isolada)

    @staticmethod
    def _worker_ga(run_id, work_dir, struct_params, ga_args, result_queue):
        """Worker isolado para o GA."""
        try:
            # Chama a função original do arquivo GA_run_TEST2
            # GA_run retorna o melhor fitness.
            # ADAPTAÇÃO: GA_run precisa retornar (best_fitness, time_elapsed)
            # Aqui assumo que GA_run foi adaptada para colocar na Queue ou retornar.


            start_t = time.time()

            # Chamada da sua função (ajuste os argumentos conforme sua definição real)
            # Dica: Passe um diretório único para cada worker evitar conflito de arquivo
            result_data = GA_run(
                irun=run_id,
                base_dir=work_dir,  # Pasta isolada
                parameters=struct_params,
                population_size=50,  # Fixo ou parametrizável?
                generations=100,  # Fixo ou parametrizável?
                elitism_rate=ga_args['elitism_rate'],
                crossover_rate=ga_args['crossover_rate'],
                mutation_strength=ga_args['mutation_strength']
            )

            end_t = time.time()

            # Se a sua GA_run retorna o fitness, ótimo. Senão, leia do arquivo.
            # Supondo que retorne um dict ou tupla:
            fitness = result_data[0]

            result_queue.put({'fitness': fitness, 'time': end_t - start_t, 'success': True})

        except Exception as e:
            print(f"[ERRO GA Worker] {e}")
            result_queue.put({'fitness': 1e6, 'time': 0, 'success': False})

    @staticmethod
    def _worker_pso(run_id, work_dir, struct_params, pso_args, result_queue):
        """Worker isolado para o PSO."""
        try:
            start_t = time.time()

            result_data = PSO_run(
                irun=run_id,
                base_dir=work_dir,
                parameters=struct_params,
                population_size=30,  # Fixo
                iterations=100,  # Fixo
                w=pso_args['w'],
                w_rate=pso_args['w_rate'],
                c1=pso_args['c1'],
                c2=pso_args['c2'],
                init_vel_ratio=pso_args['init_vel_ratio']
            )

            end_t = time.time()
            fitness = result_data[0]

            result_queue.put({'fitness': fitness, 'time': end_t - start_t, 'success': True})

        except Exception as e:
            print(f"[ERRO PSO Worker] {e}")
            result_queue.put({'fitness': 1e6, 'time': 0, 'success': False})


    # 2. SISTEMA DE SCORING

    def calculate_score(self, fits, times):
        """
        Calcula o score escalar para meta-otimização (BO) a partir de múltiplas
        execuções estocásticas de GA/PSO.

        O score combina:
          - Qualidade média da solução (fitness médio)
          - Robustez do algoritmo (variabilidade entre execuções)
          - Custo computacional (tempo médio)

        Estratégias adotadas:
          - Uso de log10 do fitness para lidar com escalas muito distintas
          - Coeficiente de variação baseado no fitness em escala log
          - Normalização online (running normalization) para BO
        """

        # -------------------------
        # 1. Limpeza de dados
        # -------------------------
        valid_fits = [f for f in fits if np.isfinite(f) and f < 1e5]
        if not valid_fits:
            # Penalidade máxima para falha total
            return 1e6, None, None, None

        # -------------------------
        # 2. Fitness em escala log
        # -------------------------
        log_fits = [np.log10(abs(f) + 1e-20) for f in valid_fits]

        avg_log_fit = np.mean(log_fits)
        std_log_fit = np.std(log_fits)

        # -------------------------
        # 3. Robustez (CV em escala log)
        # -------------------------
        cv = std_log_fit / (abs(avg_log_fit) + 1e-6)
        cv = np.log10(1.0 + cv)  # suavização para evitar explosões

        # -------------------------
        # 4. Tempo médio
        # -------------------------
        avg_time = np.mean(times)

        # -------------------------
        # 5. Armazenamento global (normalização dinâmica)
        # -------------------------
        self.global_results.append({
            'log_fit': avg_log_fit,
            'time': avg_time,
            'cv': cv
        })

        all_log_fits = [x['log_fit'] for x in self.global_results]
        all_times = [x['time'] for x in self.global_results]
        all_cvs = [x['cv'] for x in self.global_results]

        # -------------------------
        # 6. Normalização Min-Max
        # -------------------------
        def norm(val, lst):
            vmin, vmax = min(lst), max(lst)
            if vmax == vmin:
                return 0.5
            return (val - vmin) / (vmax - vmin)

        n_fit = norm(avg_log_fit, all_log_fits)
        n_time = norm(avg_time, all_times)
        n_cv = norm(cv, all_cvs)

        # -------------------------
        # 7. Score final (minimização)
        # -------------------------
        score = (
                self.W_FIT * n_fit +
                self.W_CV * n_cv +
                self.W_TIME * n_time
        )

        return score, avg_log_fit, cv, avg_time

    # -------------------------------------------------------------------------
    # 3. MÉTODOS DE AVALIAÇÃO (Chamados pelo skopt)
    # -------------------------------------------------------------------------

    def evaluate_ga_batch(self, params_list):
        """Função chamada pelo gp_minimize do GA."""
        # 1. Decodifica parâmetros (skopt manda lista)
        param_dict = {
            'elitism_rate': params_list[0],
            'crossover_rate': params_list[1],
            'mutation_strength': params_list[2]
        }

        print(f"\n>>> Avaliando GA: {param_dict}")

        fits, times = [], []

        # 2. Loop de Repetições (3 a 5 vezes)
        for i in range(1, self.n_repetitions + 1):
            # Cria diretório temporário único para evitar lock
            run_dir = os.path.join(self.base_dir, f"temp_GA_run_{i}")
            if not os.path.exists(run_dir): os.makedirs(run_dir)

            # Multiprocessing
            q = Queue()
            p = Process(target=self._worker_ga, args=(i, run_dir, self.struct_params, param_dict, q))
            p.start()
            res = q.get()
            p.join()

            # Limpa temp
            try:
                shutil.rmtree(run_dir)
            except:
                pass

            if res['success']:
                fits.append(res['fitness'])
                times.append(res['time'])
                print(f"    Run {i}: Fit={res['fitness']:.2e}, Time={res['time']:.1f}s")
            else:
                print(f"    Run {i}: FALHA")

        # 3. Calcula Score
        score, avg_fit, cv, avg_time = self.calculate_score(fits, times)

        # 4. Log
        self.log_step("GA", param_dict, score, avg_fit, cv, avg_time)

        return score

    def evaluate_pso_batch(self, params_list):
        """Função chamada pelo gp_minimize do PSO."""
        param_dict = {
            'w': params_list[0],
            'w_rate': params_list[1],
            'c1': params_list[2],
            'c2': params_list[3],
            'init_vel_ratio': params_list[4]
        }

        print(f"\n>>> Avaliando PSO: {param_dict}")

        fits, times = [], []

        for i in range(1, self.n_repetitions + 1):
            run_dir = os.path.join(self.base_dir, f"temp_PSO_run_{i}")
            if not os.path.exists(run_dir): os.makedirs(run_dir)

            q = Queue()
            p = Process(target=self._worker_pso, args=(i, run_dir, self.struct_params, param_dict, q))
            p.start()
            res = q.get()
            p.join()

            try:
                shutil.rmtree(run_dir)
            except:
                pass

            if res['success']:
                fits.append(res['fitness'])
                times.append(res['time'])
                print(f"    Run {i}: Fit={res['fitness']:.2e}")

        score, avg_fit, cv, avg_time = self.calculate_score(fits, times)
        self.log_step("PSO", param_dict, score, avg_fit, cv, avg_time)

        return score

    # -------------------------------------------------------------------------
    # 4. LOGGING
    # -------------------------------------------------------------------------
    def log_step(self, algo_name, params, score, avg_fit, cv, avg_time):
        filename = os.path.join(self.log_dir, f"meta_opt_{algo_name}_history.csv")
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            # Header dinâmico baseado nas chaves dos parâmetros
            param_keys = list(params.keys())
            if not file_exists:
                writer.writerow(['Timestamp', 'Score', 'Avg_Fit', 'CV', 'Avg_Time'] + param_keys)

            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                score, f"{avg_fit:.4e}", f"{cv:.4f}", f"{avg_time:.2f}"
                            ] + list(params.values()))

    # -------------------------------------------------------------------------
    # 5. EXECUÇÃO
    # -------------------------------------------------------------------------
    def run_meta_optimization(self, algo_type="GA", n_calls=50):

        print(f"\n=== INICIANDO META-OTIMIZAÇÃO: {algo_type} ===")

        if algo_type == "GA":
            space = [
                Real(0.00, 0.15, name="elitism_rate"),
                Real(0.40, 0.90, name="crossover_rate"),
                Real(0.01, 0.30, name="mutation_strength")
            ]
            objective = self.evaluate_ga_batch

        elif algo_type == "PSO":
            space = [
                Real(0.40, 1.20, name="w"),
                Real(0.900, 0.999, name="w_rate"),
                Real(1.00, 2.50, name="c1"),
                Real(1.00, 2.50, name="c2"),
                Real(0.05, 0.50, name="init_vel_ratio")
            ]
            objective = self.evaluate_pso_batch

        # Executa o BO (gp_minimize)
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=10,
            acq_func="EI",
            xi=0.01,
            random_state=42
        )

        print(f"\nMelhor {algo_type} Encontrado:")
        print(f"Score: {result.fun}")
        print(f"Params: {result.x}")
        return result


# =============================================================================
# BLOCO DE EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':

    # 1. Definição do Problema Estrutural (Laje/Ponte)
    struct_params = [
        Continuous(20e9, 35e9, 'modulo_viga_1'),
        Continuous(20e9, 35e9, 'modulo_viga_2'),
        Continuous(20e9, 35e9, 'modulo_centro'),
        Continuous(20e9, 35e9, 'modulo_borda_1'),
        Continuous(20e9, 35e9, 'modulo_borda_2'),
        Continuous(50e6, 50e8, 'rigidez1'),
        Continuous(50e6, 50e8, 'rigidez2'),
        Continuous(50e6, 50e8, 'rigidez3'),
        Continuous(50e6, 50e8, 'rigidez4')
    ]

    # 2. Configuração de Diretórios
    # DICA: Use caminhos curtos e sem OneDrive para evitar problemas de Lock do ANSYS
    BASE_DIR = r"C:\AnsysMetaOpt"

    # 3. Instancia o Gerenciador
    runner = MetaOptimizerRunner(BASE_DIR, struct_params, n_repetitions=3)

    # 4. Roda Meta-Otimização para GA
    runner.run_meta_optimization("GA", n_calls=30)

    # 5. Roda Meta-Otimização para PSO
    runner.run_meta_optimization("PSO", n_calls=30)

    # Limpeza Final
    try:
        Ansys.kill_ansys_process()
    except:
        pass