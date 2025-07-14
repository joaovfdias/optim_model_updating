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

        self.stopping_criteria = False
        self.fitness_tolerance = 0
        self.parameters_tolerance = 0
        self.patience = 10
        self.consecutive_iterations = 0

        self.fitness_function = fitness_function
        self.parameters = parameters
        self.parameters_keys = [param.key for param in parameters]
        self.population_size = population_size

        self.log_header=False
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
            print(f"Arquivo de registro mantido padrão: \"{self.__class__.__name__}_ddmmaa_HHMMSS\"\n")
        else:
            print(f"Arquivo de registro alterado para: \"{self.logfilename}\"\n")

    def display_parameters(self, individual):
        return ', '.join(f'{k} = {v:.3g}' for k, v in zip([param.key for param in self.parameters], individual.param))

    def create_log(self, individual=None, full=False):
        """
        função que cria uma planilha com cabeçalho relacionando os dados do problema.
        :param individual: indíviduo declarado da classe Individual (por padrão recebe o 1º da população inicial, só é necessário para quantificar modos e frequências)
        :param full: True caso for criar o registro completo com a função add_full_log, com Iteração e número do Indivíduo no cabeçalho; False (padrão) caso for usar "add_log" para registrar apenas o melhor indivíduo de dada iteração.
        """
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = self.logfilename or f"{self.__class__.__name__}_{timestamp}"
        self.logfilename = f"{filename}.csv"
        self.log_dir = self.log_dir or os.path.join(self.current_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, self.logfilename)

        individual = individual or self.populations[0][0]

        header = ["Iteration", "Fitness"] + self.parameters_keys

        #best = self.__class__.__name__ == "BO"

        if self.__class__.__name__ == "BO":
            header.insert(2, "Best")

        if full:
            header.insert(1, "Individual")
        # alterar para criar a coluna de freqs e modos apenas se elas estiverem preenchidas em .data
        if individual.data:
            freq_count = len(individual.data[0])
            header.extend([f"Freq. #{i+1}" for i in range(freq_count)])

            if len(individual.data) > 1:
                node_count = len(individual.data[1][0])
                mode_count = len(individual.data[1])

                for mode_id in range(mode_count):
                        header.append(f"Mode #{mode_id+1}")
                        header.extend([""] * (node_count - 1)) # espaçamento para alinhar os modos ao número de nós

        with open(self.log_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(header)

        # df = pd.DataFrame(columns=header)
        # with pd.ExcelWriter(self.log_path, engine='openpyxl', mode='w') as writer:
        #     df.to_excel(writer, index=False, sheet_name="Data")

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

            individual = min(population, key=lambda p: p.fitness) # melhor indivíduo da população atual

            row = [iteration, individual.fitness]

            if self.__class__.__name__ == "BO":
                best_individual = min(self.populations, key=lambda p: p.fitness)
                row.append(best_individual.fitness)

            row.extend(individual.param)

            if individual.data:
                frequencies = individual.data[0]
                row.extend(frequencies)
                if len(individual.data) > 1:
                    modes = individual.data[1]
                    row.extend(modes.flatten())

            with open(self.log_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(row)

            # with pd.ExcelWriter(self.log_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            #     # Carrega o conteúdo atual da aba
            #     existing_data = pd.read_excel(self.log_path, sheet_name="Data")
            #     # Cria um novo DataFrame com a linha
            #     new_row = pd.DataFrame([row], columns=existing_data.columns)
            #     # Concatena os dados
            #     updated_data = pd.concat([existing_data, new_row], ignore_index=True)
            #     # Salva novamente na aba específica
            #     updated_data.to_excel(writer, index=False, sheet_name="Data")

        else:

            with open(self.log_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')

                for i, individual in enumerate(population):

                    row = [iteration, i+1, individual.fitness]
                    row.extend(individual.param)
                    if individual.data:
                        frequencies = individual.data[0]
                        row.extend(frequencies)
                        if len(individual.data) > 1:
                            modes = individual.data[1]
                            row.extend(modes.flatten())

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