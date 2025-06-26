from ..optimizer import Optimizer
from ..individual import Individual
from .mutation import mutate_individual

class GAOptimizer(Optimizer):
    def __init__(self, fitness_function, parameters, population_size, elitism_rate=0.1, crossover_rate = 0.8, mutation_rate = 0.05):
        super().__init__(fitness_function,parameters)
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

