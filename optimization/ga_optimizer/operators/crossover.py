import random

from ...individual import Individual


def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1.param) - 1)
    if random.random() > 0.5:
        child_param = parent1.param[:crossover_point] + parent2.param[crossover_point:]
    else:
        child_param = parent2.param[:crossover_point] + parent1.param[crossover_point:]
    return Individual(child_param, parent1.fitness_function)

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(1, len(parent1.param) - 1), 2))
    if random.random() > 0.5:
        child_param = parent1.param[:point1] + parent2.param[point1:point2] + parent1.param[point2:]
    else:
        child_param = parent2.param[:point1] + parent1.param[point1:point2] + parent2.param[point2:]
    return Individual(child_param, parent1.fitness_function)

def uniform_crossover(parent1, parent2):
    child_param = []
    for i in range(len(parent1.param)):
        if random.random() < 0.5:
            child_param.append(parent1.param[i])
        else:
            child_param.append(parent2.param[i])
    return Individual(child_param, parent1.fitness_function)


crossover_methods = {
                    "one_point": one_point_crossover,
                    "two_point": two_point_crossover,
                    "uniform": uniform_crossover
                    }

def crossover(parent1, parent2, crossover_type):
    # chama o metodo de crossover correspondente à entrada do usuário (ou padrão)
    return crossover_methods[crossover_type](parent1, parent2)