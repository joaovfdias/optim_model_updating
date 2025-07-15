import os
import csv
import time
from datetime import datetime
import numpy as np
from pyDOE import lhs

from .individual import Individual
from .pso_optimizer.particle import Particle


class Optimizer:
    def __init__(self, fitness_function, parameters, population_size):
        self.inicio = time.time() # passar depois para a função run dos respectivos algoritmos
        self.current_dir = os.getcwd() # definindo o diretório atual
        self.opttime = None

        self.stopping_criteria = False
        self.fitness_tolerance = 0
        self.parameters_tolerance = 0
        self.patience = 10
        self.consecutive_iterations = 0

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
        self.inicio = time.time()

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

    def time_log(self, fim):
        tempo = fim - self.inicio
        row = ["Time (s)", tempo]

        with open(self.log_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([])
            writer.writerow(row)


    def set_tolerance(self, fit_tol=None, param_tol=None, patience=None):
        """
        Critérios de parada
        :param fit_tol: define a tolerância na diferença do melhor fitness entre iterações consecutivas
        :param param_tol: define a tolerância na diferença entre parâmetros nos melhores indivíduos de iterações consecutivas
        :param patience: define quantas vezes as tolerâncias podem ser superadas antes de interromper o algoritmo
        :return:
        """
        self.stopping_criteria = True
        self.fitness_tolerance = fit_tol or self.fitness_tolerance
        self.parameters_tolerance = param_tol or self.parameters_tolerance
        self.patience = patience or self.patience

    # criar uma função em otimizador que receba duas populações ou individuos e compare as diferenças, verificando se estão dentro da tolerância por uma quantidade consecutiva de iterações
    def tolerance(self, previous, current):
        # estrutura de chamada externa:
            #if self.tolerance(previous, current):
                #break

        if not self.stopping_criteria:
            return False

        previous = min(previous, key=lambda p: p.fitness)
        current = min(current, key=lambda p: p.fitness)

        fitness_diff = abs(previous.fitness - current.fitness)
        parameters_diff = max(abs(pp - cp) for pp, cp in zip(previous.param, current.param))

        if fitness_diff < self.fitness_tolerance or parameters_diff < self.parameters_tolerance:
            self.consecutive_iterations += 1
            if self.consecutive_iterations >= self.patience:
                print("\nCritério de convergência atingido: execução interrompida.")
                return True

        else:
            self.consecutive_iterations = 0

        return False

    def sync_time(self, time):
        self.opttime = time