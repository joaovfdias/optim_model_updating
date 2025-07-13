from ..optimizer import Optimizer
from ..individual import Individual
from optimization.ga_optimizer.operators.mutation import mutate
from ..parameter import *
import random


class GAOptimizer(Optimizer):
    def __init__(self, fitness_function, parameters, population_size, elitism_rate=0.1, crossover_rate = 0.8, mutation_rate = 0.05, max_generations = 100):
        super().__init__(fitness_function,parameters)
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_type = self.generate_mut_type()
        self.max_generations = max_generations
        self.crossover_type = "uniform"

    def optimize(self, status = True, log = True):
        initial_population = self.initial_population()
        self.history.append(initial_population)
        best_individual = None
        for gen in range(self.max_generations):
            pop = self.history[-1]
            new_population = self.evolve_population(pop)
            self.history.append(new_population)
            best_individual = self.get_best_individual(new_population)
        print(best_individual)
        return best_individual

    def generate_mut_type(self):
        """atributes the mutation type as uniform to all continuous parameters, as bitflip to binary and as random to state parameters"""
        mut_type = []
        for par in self.parameters:
            if type(par) == Continuous:
                mut_type.append("gaussian")
            elif type(par) == Binary:
                mut_type.append("bitflip")
            elif type(par) == State:
                mut_type.append("random")
        return mut_type

    def get_best_individual(self, pop):
        return min(pop, key=lambda x: x.fitness)

    def evolve_population(self, pop):
        elite_size = int(self.elitism_rate*self.population_size)
        pop.sort(key=lambda x: x.fitness)
        new_population = pop[:elite_size]

        while (len(new_population) < self.population_size):
            #parent selection
            parent1 = self.select_parent(pop)
            parent2 = self.select_parent(pop)
            while parent1 == parent2:
                parent2 = self.select_parent(pop)

            #crossover
            if random.random() < self.crossover_rate:
                child1 = self.crossover(parent1, parent2)
                if Individual.compareIndividuals(child1, parent1) or Individual.compareIndividuals(child1, parent2):
                    child1 = mutate(child1, self.parameters, self.mutation_type, self.mutation_rate, mut_strength=0.5)
            else:
                child1= mutate(parent1, self.parameters, self.mutation_type, self.mutation_rate, mut_strength=0.5)

            new_population.extend([child1])
        self.evaluate_population(new_population)
        return new_population[:self.population_size]

def crossover(self, parent1, parent2):
    raise NotImplementedError

def select_parent(self):
    raise NotImplementedError