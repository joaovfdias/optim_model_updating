import random


def crossover(self, parent1, parent2):
    crossover_methods = {"one_point": self.one_point_crossover, "two_point": self.two_point_crossover,
                         "uniform": self.uniform_crossover}

    return crossover_methods[self.crossover_type](parent1, parent2)


def one_point_crossover(self, parent1, parent2):
    crossover_point = random.randint(1, len(self.parameters) - 1)
    child1_param = parent1.param[:crossover_point] + parent2.param[crossover_point:]
    return Individual(child1_param, self.fitness_function)


def two_point_crossover(self, parent1, parent2):
    point1, point2 = sorted(random.sample(range(1, len(self.parameters) - 1), 2))
    child1_param = parent1.param[:point1] + parent2.param[point1:point2] + parent1.param[point2:]
    return Individual(child1_param, self.fitness_function)


def uniform_crossover(self, parent1, parent2):
    child1_param = []
    child2_param = []
    for i in range(len(self.parameters)):
        if random.random() < 0.5:
            child1_param.append(parent1.param[i])
        else:
            child1_param.append(parent2.param[i])
    return Individual(child1_param, self.fitness_function)