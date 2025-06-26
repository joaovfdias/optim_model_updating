import random
from ..individual import Individual
from ..parameter import *


def mutate(individual, param, mut_type, mut_rate, mut_strength=0.5):
    """mutates according with the mutation type, mutation rate and mutation strength given to each gene"""
    # Avaiable mutation options
    options = {
        "gaussian": mutate_gaussian,
        "uniform": mutate_uniform,
        "bitflip": mutate_bitflib,
        "random": mutate_random
    }
    new = []
    # Mutation process
    for i, value in enumerate(individual.param):
        new.append(options[mut_type[i]](value, param[i], mut_rate, mut_strength))
    return Individual(new, individual.fitness_function)


def mutate_gaussian(value, bounds, mut_rate, mut_strength):
    """Mutates a gene in a chromossome according with a probability mut_rate in a given gaussian deviation"""
    dev = random.uniform(-mut_strength, mut_strength)
    new_param = round(value + (dev * value), 2) if random.random() < mut_rate else value
    while bounds.check_bounds(new_param) != 1:
        dev = random.uniform(-mut_strength, mut_strength)
        new_param = round(value + (dev * value), 2)
    return new_param


def mutate_uniform(value, bounds, mut_rate, mut_stregth):
    """Mutates a gene, according with a probability mut_rate, to a random value inside the given limits"""
    new_param = bounds.random_value() if random.random() < mut_rate else value
    return new_param


def mutate_bitflib(value, bounds, mut_rate, mut_strength):
    """Mutates a gene, according with a probability mut_rate, to a random value inside the given limits"""
    new_param = int(not (value)) if random.random() < mut_rate else value
    return new_param


def mutate_random(value, bounds, mut_rate, mut_strength):
    new_param = bounds.random_value() if random.random() < mut_rate else value
    return new_param
