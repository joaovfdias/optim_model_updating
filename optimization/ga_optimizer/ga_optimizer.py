from ..optimizer import Optimizer
from ..individual import Individual
from .mutation import mutate
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
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = mutate(parent1, self.parameters, self.mutation_type, self.mutation_rate, mut_strength=0.5), mutate(parent2, self.parameters, self.mutation_type, self.mutation_rate, mut_strength=0.5)

            new_population.extend([child1, child2])

            return new_population[:self.population_size]

    def crossover(self,parent1, parent2):
        crossover_methods = {"one_point": self.one_point_crossover, "two_point": self.two_point_crossover,
                             "uniform": self.uniform_crossover}

        return crossover_methods[self.crossover_type](parent1, parent2)

    def one_point_crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(self.parameters) - 1)
        child1_param = parent1.param[:crossover_point] + parent2.param[crossover_point:]
        child2_param = parent2.param[:crossover_point] + parent1.param[crossover_point:]
        return Individual(child1_param, self.fitness_function), Individual(child2_param, self.fitness_function)

    def two_point_crossover(self, parent1, parent2):
        point1, point2 = sorted(random.sample(range(1, len(self.parameters) - 1), 2))
        child1_param = parent1.param[:point1] + parent2.param[point1:point2] + parent1.param[point2:]
        child2_param = parent2.param[:point1] + parent1.param[point1:point2] + parent2.param[point2:]
        return Individual(child1_param, self.fitness_function), Individual(child2_param, self.fitness_function)

    def uniform_crossover(self, parent1, parent2):
        child1_param = []
        child2_param = []
        for i in range(len(self.parameters)):
            if random.random() < 0.5:
                child1_param.append(parent1.param[i])
                child2_param.append(parent2.param[i])
            else:
                child1_param.append(parent2.param[i])
                child2_param.append(parent1.param[i])
        return Individual(child1_param, self.fitness_function), Individual(child2_param, self.fitness_function)

    def select_parent(self, population, type="roulette_wheel", truncation_rate=0.3):
        """selects two parent individuals to pass through the crossover process"""
        selection_method = {
            "roulette_wheel": self.roulette_wheel,
            "tournament": self.tournament_selection,
            "random": self.random_selection,
            "truncation": self.truncation_selection,
            "ranking": self.ranking_selection,
        }  # selection methods avaiable
        return selection_method[type](population, truncation_rate)

    def roulette_wheel(self, population, truncation_rate):
        """Selects randonly one individual from the population. The probability of an individual to be chosen is as high as its fit"""
        fitnesses = [population[i].fitness for i in range(len(population))]
        total = sum(fitnesses)
        inversion = [total - fit for fit in fitnesses]  # for the probability to be higher for smaller fitness_values
        total_inversion = sum(inversion)
        cumulative_probabilities = []
        cumulative_weight = 0
        for fitness in fitnesses:
            cumulative_weight += (total - fitness) / total_inversion
            cumulative_probabilities.append(cumulative_weight)
        rand = random.random()
        for i, individual in enumerate(population):
            if rand <= cumulative_probabilities[i]:
                return individual
        raise IndexError ('Individual not found in roulette wheel method.')


    def tournament_selection(self, pop, truncation_rate):
        """Selects randomly 3 individuals from the population and returns the one with best fit"""
        competitors = [pop[random.randint(0, len(pop) - 1)] for _ in range(3)]  # selects 3 randomly
        return min(competitors, key=lambda x: x.fitness)

    def random_selection(self, pop, truncation_rate):
        """Parents are selected randomly"""
        return random.choice(pop)

    def truncation_selection(self, pop, truncation_rate):
        """selects randomly one individual in the elite, considering the given truncation rate"""
        elite = sorted(pop, key=lambda x: x.fitness)[:(int(len(pop) * truncation_rate))]
        return elite[random.randint(0, len(elite) - 1)]

    def ranking_selection(self, pop, truncation_rate):
        """selects randomly one individual considering higher probability for better fit, but with controled probability variation"""
        sort_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
        probabilities = 0
        cumulated_probabilities = []
        rand = random.random()
        ranking = [individual for individual in sort_pop]
        for i in range(len(ranking)):
            probabilities += (i + 1) / len(ranking)
            cumulated_probabilities.append(probabilities)
        for i, individual in enumerate(ranking):
            if rand <= cumulated_probabilities[i]:
                return individual
        raise IndexError('Individual not found in ranking selection method.')
