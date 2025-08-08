from ..optimizer import Optimizer
from ..individual import Individual

from skopt.space import Real
from skopt import gp_minimize

import csv
import time
from datetime import datetime
import numpy as np

class BO(Optimizer):
    def __init__(self, fitness_function, parameters, initial_points):
        """
        Otimização Bayesiana utilizando a função "gp_minimize" da biblioteca "scikit-optimize"
        :param fitness_function: função objetivo a ser otimizada (função que recebe lista de valores dos parâmetros e retorna: fitness, [dados]
        :param parameters: lista de objetos da classe Parameters declarados com limites inferior e superior e nome
        :param initial_points: equivalente a "n_initial_points" na função "gp_minimize", referente ao número de avaliações executadas antes da aproximação da função com "base_estimator"
        """
        super().__init__(fitness_function, parameters, population_size=initial_points)

        self.status = False
        self.log = True

        self.populations = []

        self.initial_evaluations = initial_points
        self.search_space = [Real(p.lower_bound, p.upper_bound, name=p.key) for p in parameters]

        self.sampling_method = 'random'
        self.sampling_methods = ['random', 'lhs']


    def set_sampling_method(self, sampling_method):
        if sampling_method in self.sampling_methods:
            self.sampling_method = sampling_method
        else:
            print(
                f"Método de amostragem '{sampling_method}' inválido. Tipos válidos: {list(self.sampling_methods)}")
            return


    def evaluate_model(self, params):
        self.populations.append(Individual(params, self.fitness_function))
        self.populations[-1].evaluate()
        fitness = self.populations[-1].fitness
        self.best = min(self.populations, key=lambda p:p.fitness)

        if self.status:
            it = max((len(self.populations) - self.initial_evaluations), 0) # mantém IT=0 para avaliações iniciais printadas, começa a contar quando GP assume
            if it > 0:
                print(f"Avaliação {it}: Global Best = {self.best.fitness}, Fitness = {self.populations[-1].fitness}, Parâmetros: {self.display_parameters(self.populations[-1])}")
            else: # altera a mensagem caso esteja nos pontos iniciais ainda
                print(f"Avaliação Inicial {len(self.populations)}: Global Best = {self.best.fitness}, Fitness = {self.populations[-1].fitness}, Parâmetros: {self.display_parameters(self.populations[-1])}")

        if self.log:
            it = max((len(self.populations) - self.initial_evaluations), 0)
            self.add_log(it, [self.populations[-1]])

        return fitness


    # Otimizar com scikit-optimize
    def run(self, evaluations, acq_func=None, xi=None, kappa=None, status=True, log=True):

        acq_func = acq_func or "EI"
        xi = xi or 0.01 # default
        kappa = kappa or 1.96 # default

        self.inicio = time.time()
        self.status = status
        self.log = log
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        if acq_func in ["EI","PI"]:
            self.logfilename = self.logfilename or f"BO_{timestamp}_acq_fun={acq_func}_xi={xi}"
        elif acq_func=="LCB":
            self.logfilename = self.logfilename or f"BO_{timestamp}_acq_fun={acq_func}_kappa={kappa}"
        else:
            self.logfilename = self.logfilename or f"BO_{timestamp}_acq_fun={acq_func}"

        result = gp_minimize(self.evaluate_model, self.search_space, n_calls=evaluations, n_initial_points=self.initial_evaluations, initial_point_generator=self.sampling_method, acq_func=acq_func, acq_optimizer="sampling", xi=xi, kappa=kappa)

        best_individual = self.get_best_individual(self.populations)

        gp_final = result.models[-1]
        kernel = gp_final.kernel_

        # Função para encontrar o componente com length_scale
        def extract_length_scales(kernel):
            if hasattr(kernel, "length_scale"):
                return kernel.length_scale

            for attr in ("k1", "k2"):
                if hasattr(kernel, attr):
                    try:
                        ls = extract_length_scales(getattr(kernel, attr))
                        if ls is not None:
                            return ls
                    except Exception as e:
                        print(f"Erro ao extrair length scales de {attr}: {e}")

            return None

        length_scales = extract_length_scales(kernel)
        print("Length-scales:", length_scales)

        fim = time.time()
        if self.log:
            self.log_time(fim)
            self.add_log_specs(result.specs, length_scales)
            print(f"\nRegistro salvo em: {self.log_path}")

        print(f"\nMelhor solução encontrada: Fitness = {best_individual.fitness}, Parâmetros: {self.display_parameters(best_individual)}")

        return result


    def add_log_specs(self, specs_dictionary, length_scales):
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')

            # Adiciona uma linha vazia
            writer.writerow([])

            # Adiciona o título "Specifications"
            writer.writerow(["Specifications"])

            # Adiciona as keys em uma linha
            writer.writerow(specs_dictionary.keys())

            # Adiciona os valores correspondentes em outra linha
            writer.writerow(specs_dictionary.values())

            # comprimentos de escala e sensibilidade
            writer.writerow([])
            writer.writerow(["Length scales:"] + [length_scales])

            sensitivities = 1 / np.array(length_scales)
            relative_sensitivities = sensitivities / np.sum(sensitivities)

            writer.writerow(["Relative sensitivities:"] + [relative_sensitivities])


    # salvar os resultados em log [sem uso]
    @staticmethod
    def save_log_BO(filename, result):
        timestamp = datetime.now().strftime("%d%m%Y_%H%M")
        filename = filename or f"BayesianOpt_{timestamp}.csv"

        header = ["Iteration", "Fitness", "x", "y", "z"]
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(header)

            for i, (fitness, params) in enumerate(zip(result.func_vals, result.x_iters)):
                row = [i + 1, fitness] + list(params)
                writer.writerow(row)

        print(f"Log saved as {filename}")