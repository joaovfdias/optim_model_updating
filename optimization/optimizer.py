from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import os
import csv
import time
from datetime import datetime
import numpy as np
from pyDOE import lhs
import pandas as pd
import io

from .individual import Individual
from .pso_optimizer.particle import Particle

from data.storage import dumps_json, loads_json, param_to_dict, param_from_dict, indiv_to_dict, indiv_from_dict, load_fitness_from_spec
import random


class Optimizer:
    def __init__(self, fitness_function, parameters, population_size):
        self.current_dir = os.getcwd() # definindo o diretório atual
        self.opttime = None
        self.inicio = time.time()

        self.stopping_criteria = False
        self.tolerance_flag = [0]*3

        self.fitness_function = fitness_function
        self.parameters = parameters
        self.parameters_keys = [param.key for param in parameters]
        self.population_size = population_size

        self.log_header = False
        self.logfilename = None # função set
        self.log_dir = None # função set
        self.log_path = None
        self.status = True

        self.log_history = None
        self.logged_time = 0
        self.logged_iteration = 0

        self.iter_label = "Iteração"
        self.sampling_method = "lhs"
        self.sampling_methods = {"random": self.random_initial_population, "lhs": self.LHS_initial_population}
        self.ind_type = Particle if self.__class__.__name__ == "PSO" else Individual # auxiliar do inicializador de população

        self.populations = []


    # funções 'set' que permitem ao usuário modificar valores padrão
    def set_sampling_method(self, sampling_method):
        """
                Permite ao usuário definir o tipo de metodo de amostragem, validando se o tipo é permitido.
        """
        if sampling_method in self.sampling_methods:
            self.sampling_method = sampling_method
        else:
            print(
                f"Método de amostragem '{sampling_method}' inválido. Tipos válidos: {list(self.sampling_methods.keys())}")
            return

    def initial_population(self):

        pop = self.sampling_methods[self.sampling_method]() # chama a função do metodo indicado
        self.evaluate_population(pop)

        return pop

    def random_initial_population(self):
        pop =   [ # alteração para criar "Individual" no caso do GA e "Partcile" no caso do PSO, evitando repetição da função nas classes
                self.ind_type([p.random_value() for p in self.parameters], self.fitness_function)
                for _ in range(self.population_size)
                ]
        return pop

    def LHS_initial_population(self):
        n_dim = len(self.parameters)
        n_samples = self.population_size

        samples = lhs(n_dim, samples=n_samples)

        lower_bounds = np.array([p.lower_bound for p in self.parameters])
        upper_bounds = np.array([p.upper_bound for p in self.parameters])

        scaled_samples = lower_bounds + samples * (upper_bounds - lower_bounds)

        pop =   [ # alteração para criar "Individual" no caso do GA e "Partcile" no caso do PSO, evitando repetição da função nas classes
                self.ind_type([float(value) for value in scaled_samples[i]], self.fitness_function)
                for i in range(self.population_size)
                ]
        return pop

    def resume_from_log(self, csv_path):
        """
        Retoma rodada de otimização com base em log no caminho indicado
        {Não funciona no Bayesiano até implementação própria}
        """
        self.log_history = csv_path

    def resume_initial_population(self):
        """
        Lê o log CSV e reconstrói todas as populações válidas
        """
        valid_lines = []
        with open(self.log_history, mode='r', newline='', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split(';')
                if not values[0]:  # interrompe se a primeira coluna (Iteration) estiver vazia
                    break
                valid_lines.append(line)

        # Criar DataFrame com as colunas desejadas
        df = pd.read_csv(io.StringIO("".join(valid_lines)), sep=';')

        expect_col = ['Iteration', 'Individual', 'Fitness'] + [param.key for param in self.parameters] + ['Time (s)']
        if self.__class__.__name__ == 'PSO':
            vel = [f'pso.v{i}' for i, param in enumerate(self.parameters, start=1)]
            expect_col.extend(vel) # inclui velocidades registradas de cada partícula

        missing = [col for col in expect_col if col not in df.columns]
        if missing:
            if missing == ['Time (s)']:
                print(f"\nColuna 'Time' ausente no histórico, tempo total não será registrado")
            else:
                raise ValueError(f"Coluna(s) ausente(s) no CSV: {missing}")

        # Agrupar por geração com base na coluna "Iteration"
        curr_pop = []
        curr_iteration = 0

        for _, row in df.iterrows():
            iteration = int(row["Iteration"])

            # Quando mudar de geração
            if iteration != curr_iteration:
                if curr_pop:
                    self.populations.append(curr_pop)
                    if len(curr_pop) != self.population_size:
                        raise ValueError(f"Tamanho de população definido pelo usuário ({self.population_size}) incompatível com população #{curr_iteration} registrada ({len(curr_pop)})")
                curr_pop = []
                curr_iteration = iteration

            # Adiciona indivíduo atual
            individuo = self.ind_type([row[param.key] for param in self.parameters], self.fitness_function)
            individuo.fitness = float(row["Fitness"])
            if self.__class__.__name__ == 'PSO':
                part_vel = [row[v] for v in vel]
                individuo.velocity = part_vel

            curr_pop.append(individuo)

        # Adiciona a última população, se completa
        if len(curr_pop) == self.population_size:
            self.populations.append(curr_pop)
        else:
            print(f"\nDescartada última população incompleta registrada")
            curr_pop -= 1

        if self.__class__.__name__ == "PSO": # para o PSO, é necessário reconstruir o histórico da última instância da partícula (Particle.best)
            for i in range(self.population_size): # para cada partícula da população
                particle_history = [population[i] for population in self.populations] # armazena a partícula na posição i ao longo das iterações
                particle_best = self.get_best_individual(particle_history)
                self.populations[-1][i].best = [particle_best.param, particle_best.fitness] # atualiza o histórico para a última população registrada para dar continuidade ao algoritmo
            self.best_particles.extend([self.get_best_individual(population) for population in self.populations])
            self.global_best = self.get_best_individual(self.best_particles)

        print(f"\nHistórico de [{len(self.populations)} {"população" if len(self.populations) == 1 else "populações"} de {len(self.populations[0])} indivíduos] reconstruído a partir do arquivo log: {self.log_history}"
              f"\nOtimização retomada a patir da {self.iter_label.lower()} {curr_iteration + 1}")

        self.create_log_path()
        header = valid_lines[0].strip().split(';')
        index_last_line = (len(self.populations) * len(self.populations[0])) # total de linhas válidas

        if 'Time (s)' in header: # Salvar o tempo acumulado anterior, se houver
            self.logged_time = float(df.iloc[index_last_line - 1]["Time (s)"]) # desconta o cabeçalho do index pois não é linha no dataframe

        else: # Adiciona a coluna Time (s), caso não exista, para compatibilizar com log atual
            insert_index = len(self.parameters) + 3
            header.insert(insert_index, 'Time (s)')
            new_valid_lines = [';'.join(header) + '\n']

            for line in valid_lines[1:index_last_line + 1]:
                values = line.strip().split(';')
                values.insert(insert_index, "")
                new_valid_lines.append(';'.join(values) + '\n')

            valid_lines = new_valid_lines

        # Armazenar o conteúdo completo das linhas válidas (copia no novo log)
        with open(self.log_path, mode='w', newline='', encoding='utf-8') as new_log:
            new_log.writelines(valid_lines[:index_last_line + 1])
        self.log_header = True

        self.logged_iteration = curr_iteration # usada para retomar as rodadas


    @staticmethod
    def evaluate_population(population):
        for individual in population:
            individual.evaluate()

    @staticmethod
    def get_best_individual(pop):
        return min(pop, key=lambda x: x.fitness)


    def set_log(self, log_title=None, log_dir=None):
        """

        :param log_title: nome do arquivo de log. por padrão: {nome_do_algoritmo}_{data_hora}
        :param log_dir: diretório em que log será salvo. por padrão, subpasta log no diretório de chamada
        :return:
        """
        self.logfilename = log_title
        self.log_dir = log_dir
        if not log_title:
            print(f"\nArquivo de registro mantido padrão: \"{self.__class__.__name__}_ddmmaa_HHMMSS\"")
        else:
            print(f"\nArquivo de registro alterado para: \"{self.logfilename}\"")

    def display_parameters(self, individual):
        return ', '.join(f'{k} = {v:.3g}' for k, v in zip([param.key for param in self.parameters], individual.param))

    def create_log_path(self):
        timestamp = self.opttime or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.logfilename}_{timestamp}" if self.logfilename else f"{self.__class__.__name__}_{timestamp}"
        self.logfilename = f"{filename}.csv"
        self.log_dir = self.log_dir or os.path.join(self.current_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.logfilename)

    def create_log(self, individual=None, full=False): # alterar dados recebidos para um dicionário, de forma a registrar as keys e values
        """
        função que cria uma planilha com cabeçalho relacionando os dados do problema.
        :param individual: indíviduo declarado da classe Individual (por padrão recebe o 1º da população inicial, só é necessário para quantificar modos e frequências)
        :param full: True caso for criar o registro completo com a função add_full_log, com Iteração e número do Indivíduo no cabeçalho; False (padrão) caso for usar "add_log" para registrar apenas o melhor indivíduo de dada iteração.
        """
        self.create_log_path()

        individual = individual or self.populations[0][0]

        header = ["Iteration", "Fitness"] + self.parameters_keys + ["Time (s)"]

        #best = self.__class__.__name__ == "BO"

        if self.__class__.__name__ == "BO":
            header.insert(1, "Global Best")

        if full:
            header.insert(1, "Individual")

        # verifica se a entrada de .data é um dicionário e adapta o espaço adequado para escalar, vetor ou matriz (2d)
        if isinstance(individual.data, dict):

            idata = individual.data
            dkeys = list(idata.keys())

            i = 0
            while i < (len(dkeys)):
                cdata = np.asarray(idata[dkeys[i]])

                try:
                    if cdata.ndim == 0: # escalar
                        header.append(f"{dkeys[i]}")

                    elif cdata.ndim == 1: # vetor (como frequências)
                        count = len(cdata)
                        header.extend([f"{dkeys[i]} #{j+1}" for j in range(count)])

                    elif cdata.ndim == 2: # matriz (como de modos)
                        for cdata_id in range(cdata.shape[0]): # para a quantidade de linhas (modos)
                            header.append(f"{dkeys[i]} #{cdata_id+1}")
                            header.extend([""] * (cdata.shape[1] - 1)) # cria o espaçamento para a quantidade de colunas (nós)

                    else:
                        raise ValueError(f"Formato do dado '{dkeys[i]}' (posição {i+1}) não compatível com registro.") # atualmente suporta até array 2d
                except ValueError as e:
                    print(f"[ERRO] {e}")

                i += 1

        else:
            # não registra dados caso o retorno da função não seja dict
            print("\nDados adicionais não registrados, é necessário que a segunda saída de 'fitness_function' seja um dicionário no formato {'Identificador do dado': Valor (escalar, vetor ou matriz)}")

        if self.__class__.__name__ == 'PSO':
            vel = [f'pso.v{i}' for i, param in enumerate(self.parameters, start=1)]
            header.extend(vel)

        with open(self.log_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(header)

        self.log_header = True

    def add_log(self, iteration, population, full=False):
        """
        adiciona informações da população inicial da planilha de registro. cria a planilha com o cabeçalho caso ainda não houver (self.log = True).
        :param iteration: iteração atual
        :param population: população atual
        :param full: False (padrão): adiciona as informações do melhor indivíduo de cada iteração; True: adiciona informações para cada indivíduo de population na planilha de registro.

        """

        if not self.log_header: # cria o header se não houver
            self.create_log(individual=population[0], full=full)

        if not full:
            population = [min(population, key=lambda p: p.fitness)] # reduz a população apenas ao melhor

        with open(self.log_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')

            for num, individual in enumerate(population, start=1):
                row = [iteration]
                if full:
                    row.append(num) # adiciona a numeração do indivíduo para o caso log full
                if not full and self.__class__.__name__ == "BO": # armazena o Global Best apenas no caso de amostragem Bayesiana
                    row.append(self.best.fitness)
                elapsed_time = self.logged_time + (individual.etime - self.inicio) # registra o tempo passado até imediatamente após a avaliação desse indivíduo, considerando o tempo acumulado do log
                row.extend([individual.fitness] + individual.param + [elapsed_time])

                # verifica se a entrada de .data é um dicionário e adapta o espaço adequado para escalar, vetor ou matriz (2d)
                if isinstance(individual.data, dict):

                    idata = individual.data
                    dkeys = list(idata.keys())

                    i = 0
                    while i < (len(dkeys)):
                        cdata = np.asarray(idata[dkeys[i]])

                        if cdata.ndim == 0:
                            row.append(cdata)

                        if cdata.ndim == 1:
                            row.extend(cdata)

                        if cdata.ndim == 2:
                            row.extend(cdata.flatten())

                        i += 1

                if self.__class__.__name__ == 'PSO':
                    row.extend(individual.velocity)

                writer.writerow(row)

    def log_time(self, fim):
        tempo = fim - self.inicio

        if not self.log_history:
            row = ["Time (s):", tempo]
        else: # caso o histórico tenha sido reconstruído de um log, indica separadamente o tempo da rodada anterior, atual e somatória
            row = [
                "Logged Time (s):", self.logged_time,
                "Current Run Time (s):", tempo,
                "Total Accumulated Time (s):", self.logged_time + tempo
            ]

        with open(self.log_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([])
            writer.writerow(row)

    def log_specs(self):
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')

            writer.writerow([])

            #writer.writerow([f"{self.__class__.__name__} parameters:",])

            algorithm_parameters = self.specs
            writer.writerow([f"{self.__class__.__name__} parameters:",""] + list(algorithm_parameters.keys()))
            writer.writerow(["values:",""] + list(algorithm_parameters.values()))

    @property
    def specs(self):
        pass


    def set_tolerance(self, fit_abs=None, fit_rel=None, param_rel=None, patience=1):
        """
        Critérios de parada
        :param fit_abs: define a tolerância do valor absoluto de fitness
        :param fit_tol: define a tolerância da diferença relativa entre melhores fitness de iterações consecutivas
        :param param_tol: define a tolerância da diferença entre valores dos parâmetros dos melhores indivíduos de iterações consecutivas (normalizada pelo espaço de busca)
        :param patience: define quantas vezes as tolerâncias podem ser superadas antes de interromper o algoritmo
        :return:
        """
        self.stopping_criteria = True
        self.fitness_abs_tol = fit_abs
        self.fitness_rel_tol = fit_rel
        self.parameters_rel_tol = param_rel
        self.patience = patience

    # criar uma função em otimizador que receba duas populações ou individuos e compare as diferenças, verificando se estão dentro da tolerância por uma quantidade consecutiva de iterações
    def tolerance(self, previous: List[Individual], current: List[Individual]) -> bool:
        # estrutura de chamada externa:
            #if self.tolerance(previous, current):
                #break

        if not self.stopping_criteria:
            return False

        previous = min(previous, key=lambda p: p.fitness)
        current = min(current, key=lambda p: p.fitness)

        fitness_diff = abs((previous.fitness - current.fitness) / previous.fitness)
        search_spaces = [param.search_space for param in self.parameters]
        parameters_diff = [abs(pp - cp) / ss for pp, cp, ss in zip(previous.param, current.param, search_spaces)]

        if self.fitness_abs_tol is not None:
            if current.fitness < self.fitness_abs_tol:
                self.tolerance_flag[0] += 1
                if self.tolerance_flag[0] >= self.patience:
                    print(
                        f"\nCritério de convergência atingido: fitness menor que {self.fitness_abs_tol} por {self.patience} iterações consecutivas.")
                    return True
            else:
                self.tolerance_flag[0] = 0

        if self.fitness_rel_tol is not None:
            if fitness_diff < self.fitness_rel_tol:
                self.tolerance_flag[1] += 1
                if self.tolerance_flag[1] >= self.patience:
                    print(
                        f"\nCritério de convergência atingido: valores de fitness entre iterações apresentaram diferença menor que {self.fitness_rel_tol*100}% por {self.patience} vezes consecutivas.")
                    return True
            else:
                self.tolerance_flag[1] = 0

        if self.parameters_rel_tol is not None:
            if all(diff < self.parameters_rel_tol for diff in parameters_diff):
                self.tolerance_flag[2] += 1
                if self.tolerance_flag[2] >= self.patience:
                    print(
                        f"\nCritério de convergência atingido: valores de parâmetros entre iterações apresentaram diferença menor que {self.parameters_rel_tol*100}% do intervalo de busca por {self.patience} vezes consecutivas.")
                    return True
            else:
                self.tolerance_flag[2] = 0

        return False

    def sync_time(self, stime):
        self.opttime = stime

    # --------- métodos para armazenamento de dados ---------

    def _algo_name(self):
        return self.__class__.__name__  # "GA" | "PSO" | "BO"

    def _common_state(self):
        return {
            "sampling_method": getattr(self, "sampling_method", None),
            "tolerance": {
                "fit_abs": getattr(self, "fitness_abs_tol", None),
                "fit_rel": getattr(self, "fitness_rel_tol", None),
                "param_rel": getattr(self, "parameters_rel_tol", None),
                "patience": getattr(self, "patience", None),
            },
            "logged_iteration": getattr(self, "logged_iteration", 0),
            "logged_time": getattr(self, "logged_time", 0.0),
        }

    def _algo_state(self):
        algo = self._algo_name()
        if algo == "GA":
            return {"GA": getattr(self, "specs", {})}
        if algo == "PSO":
            # coletar os hiperparâmetros e instantâneos úteis
            s = getattr(self, "specs", {})
            s_ps = {"PSO": dict(s)}
            s_ps["PSO"]["global_best"] = None if self.global_best is None else {
                "param": self.global_best.param, "fitness": self.global_best.fitness
            }
            return s_ps
        if algo == "BO":
            bo = {"BO": {}}
            cfg = getattr(self, "config", None)
            if cfg:
                from dataclasses import asdict
                bo["BO"]["config"] = asdict(cfg)
            bo["BO"]["bounds"] = getattr(self, "bounds", None).tolist() if getattr(self, "bounds", None) is not None else None
            bo["BO"]["history_X"] = [x.tolist() for x in getattr(self, "history_X", [])]
            bo["BO"]["history_y"] = list(getattr(self, "history_y", []))
            bo["BO"]["best"] = indiv_to_dict(self.best) if self.best else None
            # RGN local
            if hasattr(self, "_get_local_rng_state"):
                bo["BO"]["rng_state"] = self._get_local_rng_state()
            # snapshot opcional do kernel treinado
            gp = getattr(self, "gp", None)
            if gp and getattr(gp, "kernel_", None) is not None:
                bo["BO"]["gp_snapshot"] = {
                    "kernel_str": str(gp.kernel_),
                    "theta": gp.kernel_.theta.tolist(),
                    "params": gp.kernel_.get_params()
                }
            return bo
        return {}

    def save_state(self, filename: str | None = None, fitness_spec: dict | None = None, include_rng_state: bool = True):
        now = time.time()
        algo = self._algo_name()

        if filename is None:
            filename = f"{algo}_state_{time.strftime('%Y-%m-%d_%H-%M-%SZ', time.gmtime(now))}.json.gz"

        outdir = "log"
        os.makedirs(outdir, exist_ok=True)
        filepath = os.path.join(outdir, filename)

        # RNG simples (sementes) para reproduzir amostragem
        rng = {
            "py_random_seed": getattr(self, "_py_random_seed", None),
            "numpy_seed": getattr(self, "_np_random_seed", None),
            "py_random_state": None,
            "numpy_state": None
        }
        if include_rng_state:
            try:
                rng["py_random_state"] = list(random.getstate())
            except Exception:
                pass
            try:
                rng_state = np.random.get_state()
                rng["numpy_state"] = [rng_state[0], rng_state[1].tolist(), *rng_state[2:]]
            except Exception:
                pass

        # populações
        pops = []
        for it_idx, pop in enumerate(self.populations):
            pops.append({
                "iteration": it_idx,
                "individuals": [indiv_to_dict(ind) for ind in pop]
            })

        run = {
            "version": 1,
            "algorithm": algo,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "host_cwd": getattr(self, "current_dir", None),
            "iter_label": getattr(self, "iter_label", "Iteração"),
            "population_size": self.population_size,
            "rng": rng,
            "parameters": [param_to_dict(p) for p in self.parameters],
            "fitness_spec": fitness_spec or None,
            "optimizer": {
                "common": self._common_state(),
                **self._algo_state()
            },
            "timeline": {
                "start_time": float(getattr(self, "inicio", now)),
                "now_time": float(now)
            },
            "populations": pops,
            "notes": None
        }

        # Mensagens de status
        total_iters = len(self.populations)
        total_inds = sum(len(pop) for pop in self.populations)
        print(f"[save_state] Salvando estado ({algo}) em: {filepath}")
        print(
            f"[save_state] Iterações: {total_iters} | Indivíduos totais: {total_inds} | RNG state incluso: {bool(rng.get('py_random_state') or rng.get('numpy_state'))}")

        dumps_json(run, filepath)
        print(f"[save_state] Arquivo gravado.")
        return filepath  # retorna caminho completo

    @classmethod
    def load_state(cls, filename: str, fitness_function=None, snapshot=False):  # recriação do estado de otimização
        print(f"[load_state] Carregando estado de: {filename}")
        d = loads_json(filename)
        algo = d["algorithm"]
        print(f"[load_state] Algoritmo detectado: {algo}")

        # 1) parâmetros
        params = [param_from_dict(p) for p in d["parameters"]]
        print(f"[load_state] Parâmetros: {len(params)}")

        # 2) fitness
        if fitness_function is None:
            fitness_function = load_fitness_from_spec(d.get("fitness_spec"))
        if fitness_function is None:
            # fallback: função dummy que impede evaluate() até ser substituída
            def _stub(_):
                raise RuntimeError("Defina 'fitness_function' ao carregar o estado.")

            fitness_function = _stub
            print("[load_state] Aviso: fitness_function não fornecida; usando stub que força erro ao avaliar.")

        # 3) instanciar otimizador correto
        if algo == "GA":
            from .ga_optimizer import GA
            opt = GA(fitness_function, params, d["population_size"], **{})
        elif algo == "PSO":
            from .pso_optimizer.pso_optimizer import PSO
            pso_cfg = d["optimizer"]["PSO"]
            opt = PSO(fitness_function, params, d["population_size"],
                      w=pso_cfg.get("inertia weight (w)") or pso_cfg.get("w") or 0.6,
                      w_rate=pso_cfg.get("inertia decay rate") or pso_cfg.get("w_rate") or 0.99,
                      c1=pso_cfg.get("cognitive coefficient (c1)") or pso_cfg.get("c1") or 2.0,
                      c2=pso_cfg.get("social coefficient (c2)") or pso_cfg.get("c2") or 2.0,
                      init_vel_ratio=pso_cfg.get("initial velocity ratio") or pso_cfg.get("init_vel_ratio") or 0.2)
        elif algo == "BO":
            from .bo_optimizer.bayesian import BO, BOConfig
            bo_block = d["optimizer"]["BO"]
            bo_cfg = bo_block.get("config", {})
            cfg = BOConfig(**bo_cfg)

            init_pts = d.get("population_size") \
                       or bo_cfg.get("init_points") \
                       or cfg.init_points \
                       or cfg.computed_init_points(len(params))  # fallback seguro

            opt = BO(fitness_function, params, initial_points=init_pts, config=cfg)

            # bounds, history
            if bo_block.get("bounds"):
                opt.bounds = np.array(bo_block["bounds"], dtype=float)
                opt._fit_scaler()  # garantir scaler consistente com bounds carregado
            opt.history_X = [np.asarray(x, dtype=float) for x in bo_block.get("history_X", [])]
            opt.history_y = [float(y) for y in bo_block.get("history_y", [])]

            # RGN local
            if bo_block.get("rng_state") is not None and hasattr(opt, "_set_local_rng_state"):
                opt._set_local_rng_state(bo_block["rng_state"])

            print(f"[load_state] BO: history_X={len(opt.history_X)} pontos | history_y={len(opt.history_y)}")

        else:
            raise ValueError(f"Unsupported algorithm '{algo}' in archive")

        # 4) campos comuns
        opt.iter_label = d.get("iter_label", opt.iter_label)
        if algo != "BO": opt.population_size = d.get("population_size", opt.population_size)
        opt.sampling_method = d["optimizer"]["common"].get("sampling_method", opt.sampling_method)
        tol = d["optimizer"]["common"].get("tolerance", {})
        if any(v is not None for v in tol.values()):
            opt.set_tolerance(fit_abs=tol.get("fit_abs"), fit_rel=tol.get("fit_rel"),
                              param_rel=tol.get("param_rel"), patience=tol.get("patience") or 1)
        opt.logged_iteration = d["optimizer"]["common"].get("logged_iteration", 0)
        opt.logged_time = d["optimizer"]["common"].get("logged_time", 0.0)

        # 5) reconstruir populações
        opt.populations = []
        for entry in d["populations"]:
            inds = [indiv_from_dict(i, fitness_function) for i in entry["individuals"]]
            opt.populations.append(inds)
        print(
            f"[load_state] Populações carregadas: {len(opt.populations)} (tam última: {len(opt.populations[-1]) if opt.populations else 0})")

        # 6) reconstruções específicas
        if algo == "PSO":
            # recomputar histórico de best por partícula
            best_particles = []
            for pop in opt.populations:
                best_particles.append(min(pop, key=lambda x: x.fitness))
            opt.best_particles = best_particles
            opt.global_best = min(best_particles, key=lambda x: x.fitness)
            gb = opt.global_best
            print(f"[load_state] PSO: global_best fitness={getattr(gb, 'fitness', None)}")

        if algo == "BO":
            best_dict = bo_block.get("best")
            if best_dict is not None:
                opt.best = indiv_from_dict(best_dict, fitness_function)

            # reconstruir GP (opcional): refit com history, respeitando config.random_state
            try:
                snap = bo_block.get("gp_snapshot") if snapshot else None
                if snap and "params" in snap:
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    k0 = opt.config.kernel or opt._default_kernel(np.asarray(opt.history_y, dtype=float))
                    k0.set_params(**snap["params"])
                    # Reconstrói o GP com os parâmetros salvos
                    opt.gp = GaussianProcessRegressor(
                        kernel=k0,
                        alpha=opt.config.alpha,
                        normalize_y=opt.config.normalize_y,
                        optimizer=None,  # congela hiperparâmetros na inicialização
                        n_restarts_optimizer=0,
                        random_state=opt.rng,
                    )
                    opt.gp.fit(np.vstack(opt.history_X), np.array(opt.history_y))
                else:
                    # fallback: refit normal
                    if opt.history_X:
                        opt._fit_gp()
                print("[load_state] BO: GP refit concluído.")

            except Exception as e:
                print(f"[load_state] BO: falha ao refazer fit do GP: {e}")
                pass

        print("[load_state] OK ✔ Estado reconstituído.")
        return opt

    def analyze_sensitivity(self, df=None, **kwargs):
        # df: DataFrame opcional com histórico/log; se None, você pode passar df externamente
        from .sensitivity import SensitivityAnalyzer
        sa = SensitivityAnalyzer(minimize=True)
        if df is None:
            raise ValueError("Passe um DataFrame 'df' com parâmetros + métricas (+ Fitness opcional).")
        selected = sa.workflow(
            df,
            fitness_col=kwargs.get("fitness_col", "Fitness"),
            param_keys_hint=[p.key for p in self.parameters],
            strategy=kwargs.get("strategy", "max_abs"),
            tau=kwargs.get("tau", None),
            topk=kwargs.get("topk", None),
            show_plot=kwargs.get("show_plot", True),
            interactive=kwargs.get("interactive", True),
        )
        return selected, sa


# subclasse para funções comuns a algoritmos populacionais
class PopulationBased(Optimizer):
    def __init__(self, fitness_function, parameters, population_size):
        self.global_best = None
        super().__init__(fitness_function, parameters, population_size)

    def run(self, iterations=100, status=True, log=True):
        """
        :param itera: número de iterações a serem executadas
        :param status: por padrão mostra o andamento das soluções a cada iteração, False para não mostrar
        :param log: define o registro dos resultados em planilha. True (padrão): registra os melhores indivíduos de cada iteração, "full": registra todos os indivíduos de todas as iterações. False: não cria registro.
        :return: retorna a melhor partícula encontrada, da qual é possível obter o fitness (.fitness), parâmetros (.param) e dados modais (.data)
        """
        self.inicio = time.time()
        self.status = status

        if not self.log_history: # caso não tenha sido indicada reconstrução a partir de log anterior, gera e registra a população inicial normalmente
            self.populations.append(self.initial_population())
            full = log == "full"
            if log:
                self.add_log(0, self.populations[-1], full=full)
            if self.status:
                print(f"\nPopulação Inicial: Melhor Fitness = {self.get_best_individual(self.populations[-1]).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(self.populations[-1]))}")

        else:
            self.resume_initial_population() # reconstroi as populações iniciais com base em log anterior
            full = True # demanda registro completo

        for iteration in range(self.logged_iteration, iterations):

            new_pop = self.opt_step(iteration)

            self.evaluate_population(new_pop)
            self.populations.append(new_pop)

            if log:
                self.add_log(iteration+1, new_pop, full=full)

            if self.status:
                print(f"{self.iter_label} {iteration + 1}: Melhor Fitness = {self.get_best_individual(self.populations[-1]).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(self.populations[-1]))}")

            if self.tolerance(self.populations[-2], self.populations[-1]): # critério de parada, determinado com a função set_tolerance
                print("Execução interrompida.")
                break

        fim = time.time()
        if log:
            self.log_specs()
            self.log_time(fim)
            print(f"\nRegistro salvo em: {self.log_path}")

        self.global_best = self.global_best or self.get_best_individual(self.populations[-1])

        print(f"\nMelhor solução encontrada: Fitness = {self.global_best.fitness:.4g}, Parâmetros: {self.display_parameters(self.global_best)}")

        return self.global_best # retorna o melhor indivíduo final

    def opt_step(self, iteration): # definida nos algoritmos específicos
        pass