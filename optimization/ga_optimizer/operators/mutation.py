import random

from ...individual import Individual


def mutate(individual, param, mutation_type, mutation_rate, mutation_strength):
    """mutates according with the mutation type, mutation rate and mutation strength given to each gene"""
    # Avaiable mutation options
    options =   {
                "gaussian": mutate_gaussian,
                "uniform": mutate_uniform,
                "bitflip": mutate_bitflib,
                "random": mutate_random
                }
    new = []
    # Mutation process
    for i, value in enumerate(individual.param):
        new.append(options[mutation_type[i]](value, param[i], mutation_rate[i], mutation_strength))
    return Individual(new, individual.fitness_function)


def mutate_gaussian(value, bounds, rate, strength):
    """Mutates a gene in a chromossome according with a probability mut_rate in a given gaussian deviation"""
    mut = random.uniform(-strength, strength)
    new_param = round(value * (1 + mut), 8) if random.random() < rate else value
    while bounds.check_bounds(new_param) != 1:
        mut = random.uniform(-strength, strength)
        new_param = round(value * (1 + mut), 8)
    return new_param

def mutate_uniform(value, bounds, rate, strength):
    """Mutates a gene, according with a probability mut_rate, to a random value inside the given limits"""
    new_param = bounds.random_value() if random.random() < rate else value
    return new_param

def mutate_bitflib(value, bounds, rate, strength):
    """Mutates a gene, according with a probability mut_rate, to a random value inside the given limits"""
    new_param = int(not (value)) if random.random() < rate else value
    return new_param

def mutate_random(value, bounds, rate, strength):
    new_param = bounds.random_value() if random.random() < rate else value
    return new_param