from pyDOE import lhs
from .individual import Individual
import numpy as np


class Optimizer():
    def __init__(self, fitness_function, parameters):
        self.sampling_methods = None
        self.population_size = None

        self.fitness_function = fitness_function
        self.parameters = parameters
        self.history = []

    def optimize(self):
        raise NotImplementedError('Use subclasses')

    def create_log(self):
        raise NotImplementedError('Implmentation pending')

    def set_sampling_method(self, sampling_method):
        """
                Permite ao usuário definir o tipo de metodo de amostragem, validando se o tipo é permitido.
        """
        self.sampling_methods = {"random": self.random_initial_population, "lhs": self.lhs_initial_population}
        if sampling_method in self.sampling_methods:
            self.sampling_method = sampling_method
        else:
            print(
                f"Método de amostragem '{sampling_method}' inválido. Tipos válidos: {list(self.sampling_methods.keys())}")
            return

    @staticmethod
    def evaluate_population(population):
        for individual in population:
            individual.evaluate()

    def initial_population(self):
        return self.sampling_methods[self.sampling_method]()
        # chama a função do metodo indicado

    def random_initial_population(self):
        pop = [Individual([p.random_value() for p in self.parameters], self.fitness_function) for _ in
                range(self.population_size)]
        self.evaluate_population(pop)
        return pop


    def lhs_initial_population(self):
        n_dim = len(self.parameters)
        n_samples = self.population_size

        samples = lhs(n_dim, samples=n_samples)

        lower_bounds = np.array([p.lower_bound for p in self.parameters])
        upper_bounds = np.array([p.upper_bound for p in self.parameters])

        scaled_samples = lower_bounds + samples * (upper_bounds - lower_bounds)

        pop = [
            Individual([float(value) for value in scaled_samples[i]], self.fitness_function)
            for i in range(self.population_size)
        ]
        self.evaluate_population(pop)
        return pop
