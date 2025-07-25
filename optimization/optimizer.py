import os
import csv
import time
from contextlib import contextmanager
from datetime import datetime
import numpy as np
from pyDOE import lhs
import threading
import sys

from .individual import Individual
from .pso_optimizer.particle import Particle


class Optimizer:
    def __init__(self, fitness_function, parameters, population_size):
        self.current_dir = os.getcwd() # definindo o diretório atual
        self.opttime = None

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

        self.sampling_method = "lhs"
        self.sampling_methods = {"random": self.random_initial_population, "lhs": self.LHS_initial_population}
        self.algorithms = {"GA": Individual, "PSO": Particle} # auxiliar do inicializador de população

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

        self.populations = [self.initial_population()]

    def initial_population(self):

        pop = self.sampling_methods[self.sampling_method]() # chama a função do metodo indicado
        self.evaluate_population(pop)

        return pop

    def random_initial_population(self):
        pop =   [ # alteração para criar "Individual" no caso do GA e "Partcile" no caso do PSO, evitando repetição da função nas classes
                self.algorithms[self.__class__.__name__]([p.random_value() for p in self.parameters], self.fitness_function)
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
                self.algorithms[self.__class__.__name__]([float(value) for value in scaled_samples[i]], self.fitness_function)
                for i in range(self.population_size)
                ]
        return pop


    def evaluate_population(self, population):
        for i, individual in enumerate(population, start=1):
            with self.display_process(f"Avaliando indivíduo {i}/{self.population_size}"):
                individual.evaluate()

    @staticmethod
    def get_best_individual(pop):
        return min(pop, key=lambda x: x.fitness)

    @contextmanager
    def display_process(self, message):
        stop = False

        def animate_dots():
            dots = ["", ".", "..", "...", "..", ".", ""]
            while not stop:
                for d in dots:
                    if stop:
                        break
                    sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
                    sys.stdout.write(f"\r{message}{d} ")
                    sys.stdout.flush()
                    time.sleep(0.5)

        t = threading.Thread(target=animate_dots)
        t.start()

        try:
            yield
        finally:
            stop = True
            t.join()
            sys.stdout.write("\033[2K\r")  # limpa conteúdo da animação
            sys.stdout.flush()

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

    def create_log(self, individual=None, full=False): # alterar dados recebidos para um dicionário, de forma a registrar as keys e values
        """
        função que cria uma planilha com cabeçalho relacionando os dados do problema.
        :param individual: indíviduo declarado da classe Individual (por padrão recebe o 1º da população inicial, só é necessário para quantificar modos e frequências)
        :param full: True caso for criar o registro completo com a função add_full_log, com Iteração e número do Indivíduo no cabeçalho; False (padrão) caso for usar "add_log" para registrar apenas o melhor indivíduo de dada iteração.
        """
        timestamp = self.opttime or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.logfilename}_{timestamp}" if self.logfilename else f"{self.__class__.__name__}_{timestamp}"
        self.logfilename = f"{filename}.csv"
        self.log_dir = self.log_dir or os.path.join(self.current_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.logfilename)

        individual = individual or self.populations[0][0]

        header = ["Iteration", "Fitness"] + self.parameters_keys

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
                row.extend([individual.fitness] + individual.param)

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

                writer.writerow(row)

    def log_time(self, fim):
        tempo = fim - self.inicio
        row = ["Time (s):", tempo]

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
    def tolerance(self, previous, current):
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
                        f"\nCritério de convergência atingido: fitness menor que {self.fitness_abs_tol} por {self.patience} iterações consecutivas. \nExecução interrompida.")
                    return True
            else:
                self.tolerance_flag[0] = 0

        if self.fitness_rel_tol is not None:
            if fitness_diff < self.fitness_rel_tol:
                self.tolerance_flag[1] += 1
                if self.tolerance_flag[1] >= self.patience:
                    print(
                        f"\nCritério de convergência atingido: valores de fitness entre iterações apresentaram diferença menor que {self.fitness_rel_tol*100}% por {self.patience} vezes consecutivas. \nExecução interrompida.")
                    return True
            else:
                self.tolerance_flag[1] = 0

        if self.parameters_rel_tol is not None:
            if all(diff < self.parameters_rel_tol for diff in parameters_diff):
                self.tolerance_flag[2] += 1
                if self.tolerance_flag[2] >= self.patience:
                    print(
                        f"\nCritério de convergência atingido: valores de parâmetros entre iterações apresentaram diferença menor que {self.parameters_rel_tol*100}% do intervalo de busca por {self.patience} vezes consecutivas. \nExecução interrompida.")
                    return True
            else:
                self.tolerance_flag[2] = 0

        return False

    def sync_time(self, stime):
        self.opttime = stime


# subclasse para funções comuns a algoritmos populacionais
class PopulationBased(Optimizer):
    def __init__(self, fitness_function, parameters, population_size):
        self.iter_label = "Iteração"
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
        self.populations = [self.initial_population()]

        full = log == "full"
        if log:
            self.create_log(full=full)
            self.add_log(0, self.populations[-1], full=full)

        if status:
            print(f"\nPopulação Inicial: Melhor Fitness = {self.get_best_individual(self.populations[-1]).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(self.populations[-1]))}")

        for iteration in range(iterations):

            new_pop = self.opt_step(iteration)

            self.evaluate_population(new_pop)
            self.populations.append(new_pop)

            if log:
                self.add_log(iteration+1, new_pop, full=full)

            if status:
                print(f"{self.iter_label} {iteration + 1}: Melhor Fitness = {self.get_best_individual(self.populations[-1]).fitness:.4g}, Parâmetros: {self.display_parameters(self.get_best_individual(self.populations[-1]))}")

            if self.tolerance(self.populations[-2], self.populations[-1]): # critério de parada, determinado com a função set_tolerance
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